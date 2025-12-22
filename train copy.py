import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import time
import socket
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint

from streamingflow.config_cz import get_parser, get_cfg
from streamingflow.datas.dataloaders import prepare_dataloaders
from streamingflow.trainer_dsec import TrainingModule


def get_latest_checkpoint(folder_path):
    import glob
    import re
    import os

    ckpt_files = glob.glob(os.path.join(folder_path, '*.ckpt'))

    pattern = re.compile(r'epoch=(\d+).*\.ckpt')

    max_epoch = -1
    max_file_path = None

    for file_path in ckpt_files:
        match = pattern.match(os.path.basename(file_path))
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                max_file_path = file_path

    if max_file_path:
        print(f"The path to the .ckpt file with the highest epoch number is: {max_file_path}")
    else:
        print("No .ckpt files with the naming convention epoch_{}* were found.")
    
    return max_file_path


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    trainloader, valloader = prepare_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # 加载预训练权重；仅加载在当前模型中存在且形状兼容的参数
        ckpt = torch.load(cfg.PRETRAINED.PATH, map_location='cpu')
        if 'state_dict' not in ckpt:
            raise RuntimeError(f"Checkpoint at {cfg.PRETRAINED.PATH} has no 'state_dict' key.")

        pretrained_state = ckpt['state_dict']
        current_state = model.state_dict()

        filtered_state = {}
        for k, v in pretrained_state.items():
            # 跳过 decoder，以避免任务头不一致的问题（沿用原有逻辑）
            if 'decoder' in k:
                continue
            # 跳过 frustum：它由当前配置计算得到，形状依赖分辨率，不能直接从旧 ckpt 复制
            if k.endswith('model.frustum') or '.frustum' in k:
                continue
            if k not in current_state:
                continue
            if current_state[k].shape != v.shape:
                # 严格避免形状不匹配导致的隐性错误
                continue
            filtered_state[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded pretrained weights from {cfg.PRETRAINED.PATH}")
        if missing_keys:
            print(f"Missing keys (not loaded, kept init): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint (ignored): {len(unexpected_keys)}")

    save_dir = os.path.join(
        cfg.LOG_DIR, cfg.TAG
    )

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=-1,
        save_last=False,
        every_n_epochs=1,
        mode='max'
    )

    latest_ckpt = get_latest_checkpoint(save_dir)
    
    # 检查 checkpoint 是否存在且有效
    if latest_ckpt is not None:
        if not os.path.exists(latest_ckpt):
            print(f"WARNING: Checkpoint file not found: {latest_ckpt}")
            print("Starting training from scratch (no checkpoint resume).")
            latest_ckpt = None
        else:
            # 尝试加载 checkpoint 并检查是否包含 NaN 权重
            try:
                print(f"Checking checkpoint file: {latest_ckpt}")
                ckpt = torch.load(latest_ckpt, map_location='cpu')
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                    has_nan_in_ckpt = False
                    for key, value in state_dict.items():
                        if isinstance(value, torch.Tensor):
                            if torch.isnan(value).any() or torch.isinf(value).any():
                                print(f"WARNING: NaN/Inf found in checkpoint parameter: {key}")
                                has_nan_in_ckpt = True
                    if has_nan_in_ckpt:
                        print("WARNING: Checkpoint contains NaN/Inf weights! Will not resume from this checkpoint.")
                        print("Starting training from scratch.")
                        latest_ckpt = None
                    else:
                        print(f"Checkpoint validation passed: {latest_ckpt}")
                else:
                    print(f"WARNING: Checkpoint file does not contain 'state_dict' key. Will not resume.")
                    latest_ckpt = None
            except Exception as e:
                print(f"WARNING: Failed to load checkpoint file: {e}")
                print("Starting training from scratch.")
                latest_ckpt = None
    else:
        print("No checkpoint found. Starting training from scratch.")

    # 根据GPU数量自动选择训练模式
    num_gpus = len(cfg.GPUS) if isinstance(cfg.GPUS, list) else 1
    if num_gpus > 1:
        accelerator = 'ddp'
        plugins = DDPPlugin(find_unused_parameters=False)
    else:
        accelerator = None
        plugins = None

    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator=accelerator,
        precision=cfg.PRECISION,
        sync_batchnorm=True if num_gpus > 1 else False,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary='full',
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        plugins=plugins,
        profiler='simple',
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=latest_ckpt
    )
    
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()
