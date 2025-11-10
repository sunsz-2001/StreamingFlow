"""
事件分支推理自检脚本。

两种使用方式：
1. **真实数据推理**（默认）：
   ```bash
   python tests/test_event_branch.py --config-file configs/dsec_event.yaml --split val --num-batches 2
   ```
   - 需提前准备好数据集（例如 DSEC）并在配置中设定 `DATASET.NAME`、`DATAROOT`、`MODEL.MODALITY.*` 等。
   - 仅做前向推理，不更新参数。

2. **合成数据快速自检**：
   ```bash
   python tests/test_event_branch.py --synthetic
   ```
   - 不依赖实际数据，用随机事件流验证 EvRT 编码 → LSS → BEV 全链路。

运行前请确认：
    * 已 `pip install -e ./evrt-detr`（事件编码依赖）。
    * 其它依赖（efficientnet_pytorch 等）就绪。
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from streamingflow.config import get_cfg, get_parser  # noqa: E402
from streamingflow.datas.dataloaders import prepare_dataloaders  # noqa: E402
from streamingflow.models.streamingflow import streamingflow  # noqa: E402
from streamingflow.utils.event_tensor import EventTensorizer  # noqa: E402


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def move_to_device(value, device):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(list(value), device))
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


def infer_batch_size(tensors):
    for key in [
        'image', 'event', 'intrinsics', 'future_egomotion', 'points', 'padded_voxel_points'
    ]:
        value = tensors.get(key)
        if torch.is_tensor(value):
            return value.shape[0]
    return 1


def prepare_model_inputs(batch, device, cfg):
    keys = [
        "image",
        "intrinsics",
        "extrinsics",
        "future_egomotion",
        "padded_voxel_points",
        "camera_timestamp",
        "points",
        "lidar_timestamp",
        "target_timestamp",
        "event",
    ]
    prepared = {k: move_to_device(batch.get(k), device) for k in keys}

    batch_size = infer_batch_size(prepared)

    seq = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)

    if prepared.get("future_egomotion") is None:
        prepared["future_egomotion"] = torch.zeros(batch_size, seq, 6, device=device)

    if prepared.get("target_timestamp") is None:
        prepared["target_timestamp"] = torch.zeros(batch_size, seq, device=device)

    if prepared.get("intrinsics") is None:
        prepared["intrinsics"] = torch.eye(3, device=device).view(1, 1, 1, 3, 3).repeat(batch_size, seq, 1, 1, 1)

    if prepared.get("extrinsics") is None:
        prepared["extrinsics"] = torch.eye(4, device=device).view(1, 1, 1, 4, 4).repeat(batch_size, seq, 1, 1, 1)

    if prepared.get("image") is None and cfg.MODEL.MODALITY.USE_CAMERA:
        h, w = cfg.IMAGE.FINAL_DIM
        prepared["image"] = torch.zeros(batch_size, seq, len(cfg.IMAGE.NAMES), 3, h, w, device=device)

    if prepared.get("event") is None and cfg.MODEL.MODALITY.USE_EVENT:
        channels = getattr(cfg.MODEL.EVENT, 'IN_CHANNELS', 0)
        if channels <= 0:
            channels = 2 * getattr(cfg.MODEL.EVENT, 'BINS', 10)
        h, w = cfg.IMAGE.FINAL_DIM
        prepared["event"] = torch.zeros(batch_size, seq, len(cfg.IMAGE.NAMES), channels, h, w, device=device)
    else:
        event = prepared.get("event")
        if isinstance(event, list) or isinstance(event, tuple):
            raise ValueError("事件输入需要整理为张量或支持的 dict 结构。")
        if torch.is_tensor(event):
            cur_seq = event.shape[1]
            if cur_seq < seq:
                pad = torch.zeros(
                    (batch_size, seq - cur_seq) + tuple(event.shape[2:]),
                    device=device,
                    dtype=event.dtype,
                )
                prepared["event"] = torch.cat([event, pad], dim=1)
            elif cur_seq > seq:
                prepared["event"] = event[:, :seq]
    return prepared


def describe_outputs(name, obj):
    if torch.is_tensor(obj):
        print(f"[Result] {name}: {tuple(obj.shape)}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            describe_outputs(f"{name}.{key}", value)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            describe_outputs(f"{name}[{idx}]", value)
    else:
        print(f"[Result] {name}: {type(obj)}")


def run_dataset_inference(cfg, args):
    device = resolve_device(args.device)
    cfg = cfg.clone()
    cfg.BATCHSIZE = max(1, args.batch_size)
    cfg.N_WORKERS = max(0, args.num_workers)

    print(f"[Info] 使用设备: {device}")
    print(f"[Info] DATASET.NAME={cfg.DATASET.NAME}, BATCHSIZE={cfg.BATCHSIZE}, N_WORKERS={cfg.N_WORKERS}")

    trainloader, valloader = prepare_dataloaders(cfg)
    loader = trainloader if args.split == "train" else valloader

    try:
        model = streamingflow(cfg).to(device)
    except ImportError as exc:
        raise SystemExit(
            "\n[Error] 导入 EvRT-DETR 失败，请先 `pip install -e ./evrt-detr`。\n"
            f"原始异常：{exc}"
        )

    model.eval()
    iterator = iter(loader)

    for step in range(args.num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            print("[Warn] 达到数据集末尾，提前结束。")
            break

        raw_voxel_count = batch.get('event_voxel_count')
        if torch.is_tensor(raw_voxel_count):
            voxel_count_value = int(raw_voxel_count[0].item())
        elif isinstance(raw_voxel_count, (list, tuple)):
            voxel_count_value = int(raw_voxel_count[0])
        elif raw_voxel_count is None:
            voxel_count_value = 0
        else:
            voxel_count_value = int(raw_voxel_count)

        inputs = prepare_model_inputs(batch, device, cfg)

        if cfg.MODEL.MODALITY.USE_EVENT and inputs.get("event") is None:
            raise RuntimeError("当前 batch 缺少事件张量，无法测试事件分支。")
        if cfg.MODEL.MODALITY.USE_CAMERA and inputs.get("image") is None:
            raise RuntimeError("USE_CAMERA=True 但 batch['image'] 不存在。")

        event_bev_inspect = None
        encoder_feats_shape = None
        voxel_file_count = 0
        if cfg.MODEL.MODALITY.USE_EVENT:
            with torch.no_grad():
                modality_probe = model.calculate_birds_eye_view_features(
                    inputs.get("intrinsics"),
                    inputs.get("extrinsics"),
                    inputs.get("future_egomotion"),
                    image=inputs.get("image") if model.use_camera else None,
                    event=inputs.get("event"),
                )
            event_probe = modality_probe.get("event") if modality_probe else None
            if not (isinstance(event_probe, dict) and torch.is_tensor(event_probe.get("bev"))):
                raise RuntimeError("事件 BEV 探测失败，请检查输入事件张量是否有效。")
            event_bev_inspect = event_probe["bev"].detach().cpu()

            # 直接跑一次 encoder forward 以检查输出形状
            frames = model._prepare_event_frames(inputs.get("event"), cfg.TIME_RECEPTIVE_FIELD, device=device)
            frames = frames[:, :cfg.TIME_RECEPTIVE_FIELD]
            b, s, n, c, h, w = frames.shape
            packed = frames.view(b, s * n, c, h, w)
            with torch.no_grad():
                encoder_feats, _ = model.event_encoder_forward(packed)
            encoder_feats_shape = tuple(encoder_feats.shape)
            print(f"[Event] Input frames shape: {tuple(frames.shape)}")

            event_input = inputs.get("event")
            if torch.is_tensor(event_input):
                channels = event_input.shape[3]
            else:
                channels = 0
            voxel_file_count = voxel_count_value

        with torch.no_grad():
            outputs = model(
                inputs.get("image"),
                inputs.get("intrinsics"),
                inputs.get("extrinsics"),
                inputs.get("future_egomotion"),
                padded_voxel_points=inputs.get("padded_voxel_points"),
                camera_timestamp=inputs.get("camera_timestamp"),
                points=inputs.get("points"),
                lidar_timestamp=inputs.get("lidar_timestamp"),
                target_timestamp=inputs.get("target_timestamp"),
                event=inputs.get("event"),
            )

        print(f"\n[Info] Batch {step} 推理完成，输出结构：")
        describe_outputs("output", outputs)

        if event_bev_inspect is not None:
            print(f"[Event] BEV shape: {tuple(event_bev_inspect.shape)}")
        if encoder_feats_shape is not None:
            print(f"[Event] Encoder output shape: {encoder_feats_shape}")
        if cfg.MODEL.MODALITY.USE_EVENT:
            print(f"[Event] Voxel files stacked: {voxel_file_count}, Channels: {channels}")


def build_test_cfg(use_camera=True):
    cfg = get_cfg()
    cfg.TIME_RECEPTIVE_FIELD = 2
    cfg.N_FUTURE_FRAMES = 0

    cfg.MODEL.MODALITY.USE_CAMERA = False
    cfg.MODEL.MODALITY.USE_EVENT = True
    cfg.MODEL.MODALITY.USE_LIDAR = False
    cfg.MODEL.MODALITY.USE_RADAR = False

    cfg.MODEL.ENCODER.NAME = "efficientnet-b0"
    cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION = True
    cfg.MODEL.ENCODER.OUT_CHANNELS = 64
    cfg.MODEL.ENCODER.DOWNSAMPLE = 8

    if not use_camera:
        cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION = False

    cfg.MODEL.EVENT.BINS = 4
    cfg.MODEL.EVENT.IN_CHANNELS = 0
    cfg.MODEL.EVENT.FUSION_TYPE = "independent"
    cfg.MODEL.EVENT.BEV_FUSION = "sum"
    cfg.MODEL.EVENT.MULTISCALE_FUSION = "sum"
    cfg.MODEL.EVENT.FREEZE_BACKBONE = False
    cfg.MODEL.EVENT.PRETRAINED = False
    cfg.MODEL.EVENT.NORMALIZE = False
    cfg.MODEL.EVENT.USE_DEPTH_HEAD = True
    cfg.MODEL.EVENT.DEPTH_BINS = int((cfg.LIFT.D_BOUND[1] - cfg.LIFT.D_BOUND[0]) / cfg.LIFT.D_BOUND[2])
    cfg.MODEL.EVENT.DEPTH_HEAD_CHANNELS = 64

    cfg.MODEL.TEMPORAL_MODEL.NAME = "identity"
    cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE = False
    cfg.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS = cfg.MODEL.ENCODER.OUT_CHANNELS

    cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED = False
    cfg.SEMANTIC_SEG.HDMAP.ENABLED = False
    cfg.INSTANCE_SEG.ENABLED = False
    cfg.INSTANCE_FLOW.ENABLED = False
    cfg.PLANNING.ENABLED = False
    cfg.LIFT.GT_DEPTH = False
    cfg.GEN.GEN_DEPTH = False

    cfg.IMAGE.FINAL_DIM = (128, 256)
    cfg.IMAGE.ORIGINAL_HEIGHT = cfg.IMAGE.FINAL_DIM[0]
    cfg.IMAGE.ORIGINAL_WIDTH = cfg.IMAGE.FINAL_DIM[1]
    cfg.IMAGE.RESIZE_SCALE = 1.0
    cfg.IMAGE.TOP_CROP = 0

    return cfg


def make_mock_batch(cfg, device):
    torch.manual_seed(0)

    B = 1
    S = cfg.TIME_RECEPTIVE_FIELD
    N = len(cfg.IMAGE.NAMES[:3])
    C_img = 3
    H, W = cfg.IMAGE.FINAL_DIM

    image = torch.randn(B, S, N, C_img, H, W, device=device)
    intrinsics = torch.eye(3, device=device).view(1, 1, 1, 3, 3).repeat(B, S, N, 1, 1)
    extrinsics = torch.eye(4, device=device).view(1, 1, 1, 4, 4).repeat(B, S, N, 1, 1)
    future_ego = torch.zeros(B, S, 6, device=device)

    events_nested = []
    base_t = torch.linspace(0.0, 1.0, steps=cfg.MODEL.EVENT.BINS + 1)
    for _ in range(B):
        seq_list = []
        for s in range(S):
            cam_list = []
            for _ in range(N):
                num_events = 200
                xs = torch.rand(num_events) * (cfg.IMAGE.ORIGINAL_WIDTH - 1)
                ys = torch.rand(num_events) * (cfg.IMAGE.ORIGINAL_HEIGHT - 1)
                ts = torch.rand(num_events) * (base_t[-1] - base_t[0]) + base_t[0] + s * 0.05
                polarity = torch.where(torch.rand(num_events) > 0.5, torch.ones(num_events), -torch.ones(num_events))
                cam_list.append(torch.stack([xs, ys, ts, polarity], dim=1))
            seq_list.append(cam_list)
        events_nested.append(seq_list)

    tensorizer = EventTensorizer(cfg)
    event_frames = tensorizer.prepare_frames(events_nested, device=device, max_seq_len=S)
    print(f"[Synthetic] Event frames shape: {tuple(event_frames.shape)}, sum={event_frames.sum().item():.2f}")

    return {
        "image": image,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "future_egomotion": future_ego,
        "event_nested": events_nested,
    }


def run_synthetic_self_check(device):
    for use_camera in (True, False):
        print(f"\n===== Synthetic: USE_CAMERA={use_camera}, USE_EVENT=True =====")
        cfg = build_test_cfg(use_camera=use_camera)
        cfg.IMAGE.NAMES = cfg.IMAGE.NAMES[:3]

        batch = make_mock_batch(cfg, device)
        if not use_camera:
            batch["image"] = torch.zeros_like(batch["image"])

        try:
            model = streamingflow(cfg).to(device)
        except ImportError as exc:
            raise SystemExit(
                "\n[Error] 导入 EvRT-DETR 失败，请先 `pip install -e ./evrt-detr`。\n"
                f"原始异常：{exc}"
            )

        model.eval()

        with torch.no_grad():
            modality_outputs = model.calculate_birds_eye_view_features(
                batch["intrinsics"],
                batch["extrinsics"],
                batch["future_egomotion"],
                image=batch["image"] if use_camera else None,
                event=batch["event_nested"],
            )

        camera_data = modality_outputs.get("camera")
        event_data = modality_outputs.get("event")
        if camera_data is not None:
            print(f"[Synthetic] Camera BEV shape: {tuple(camera_data['bev'].shape)}")
        if event_data is not None:
            print(f"[Synthetic] Event BEV shape: {tuple(event_data['bev'].shape)}")

        target_timestamp = torch.zeros(batch["image"].size(0), cfg.TIME_RECEPTIVE_FIELD, device=device)

        with torch.no_grad():
            outputs = model(
                batch["image"],
                batch["intrinsics"],
                batch["extrinsics"],
                batch["future_egomotion"],
                target_timestamp=target_timestamp,
                event=batch["event_nested"],
            )

        describe_outputs("synthetic_output", outputs)


def parse_args():
    parser = get_parser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='推理使用的划分（train/val）。')
    parser.add_argument('--num-batches', type=int, default=1, help='连续推理的 batch 数量。')
    parser.add_argument('--batch-size', type=int, default=1, help='仅 dataset 模式下 DataLoader 的 batch size。')
    parser.add_argument('--num-workers', type=int, default=0, help='仅 dataset 模式下 DataLoader 的 worker 数。')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='推理设备。')
    parser.add_argument('--synthetic', action='store_true', help='使用合成数据而非真实数据集。')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.synthetic:
        device = resolve_device(args.device)
        run_synthetic_self_check(device)
        return

    cfg = get_cfg(args)
    run_dataset_inference(cfg, args)


if __name__ == "__main__":
    main()
