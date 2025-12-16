"""
事件分支推理自检脚本。

========================================
快速运行指令
========================================

1. **真实数据推理**（推荐，快速验证）：
   ```bash
   python tests/test_event_branch.py \
     --config-file streamingflow/configs/dsec_event.yaml \
     --split val \
     --num-batches 1 \
     --batch-size 1
   ```

2. **合成数据快速自检**（无需真实数据）：
   ```bash
   python tests/test_event_branch.py --synthetic
   ```

3. **速度基准测试**（完整性能测试）：
   ```bash
   python tests/test_event_branch.py \
     --config-file streamingflow/configs/dsec_event.yaml \
     --benchmark \
     --num-batches 100 \
     --num-warmup 10 \
     --save-benchmark results.csv
   ```

4. **仅测试编码器速度**（编码器专项测试）：
   ```bash
   python tests/test_event_branch.py \
     --config-file streamingflow/configs/dsec_event.yaml \
     --benchmark \
     --benchmark-encoder-only \
     --num-batches 50
   ```

5. **流式数据格式测试**：
   ```bash
   python tests/test_event_branch.py \
     --config-file streamingflow/configs/dsec_event.yaml \
     --split val \
     --num-batches 2 \
     --test-flow-data
   ```

6. **完整NODE测试**（包含未来预测）：
   ```bash
   python tests/test_event_branch.py \
     --config-file streamingflow/configs/dsec_event_with_node.yaml \
     --split val \
     --num-batches 1 \
     --batch-size 1
   ```
   注意：需要配置文件中 `N_FUTURE_FRAMES > 0` 才能启用NODE模块。

========================================
详细使用说明
========================================

1. **真实数据推理**（默认）：
   - 需提前准备好数据集（例如 DSEC）并在配置中设定 `DATASET.NAME`、`DATAROOT`、`MODEL.MODALITY.*` 等。
   - 仅做前向推理，不更新参数。
   - 验证 100Hz 频率配置、输入/输出尺寸、事件编码器和 BEV 输出信息。

2. **合成数据快速自检**：
   - 不依赖实际数据，用随机事件流验证 EvRT 编码 → LSS → BEV 全链路。
   - 适合快速验证模型结构是否正确。

3. **速度基准测试**：
   - 测试数据加载速度、预处理速度和模型推理速度。
   - 输出详细的性能统计（平均值、中位数、标准差、百分位数等）。
   - 可选保存结果为 CSV 文件，方便后续分析。
   - 使用 CUDA Event 进行精确的 GPU 时间测量。

4. **编码器专项测试**：
   - 只测试 Event 编码器到 BEV 投影前的速度。
   - 不包括 BEV 投影、时序模型、解码器。
   - 用于分析编码器性能瓶颈。

5. **流式数据格式测试**：
   - 测试 `get_data_flow` 方法生成的流式数据格式。
   - 需要在配置文件中设置 `DATASET.USE_FLOW_DATA=True` 以启用流式数据生成。
   - 输出流式数据的详细结构信息，包括每个时间窗口的事件和 LiDAR 数据。

6. **完整NODE测试**：
   - 测试包含 Neural ODE (NODE) 未来预测模块的完整流程。
   - 需要在配置文件中设置 `N_FUTURE_FRAMES > 0`（例如 6 或 10）。
   - 测试流程包括：
     * Event编码器（EvRT）
     * BEV投影（LSS）
     * 时序模型（temporal_model）
     * **NODE未来预测模块**（FuturePredictionODE）
     * Decoder
   - 验证输出序列长度是否正确（past frames + future frames）。

========================================
命令行参数说明
========================================

必需参数：
  --config-file PATH      配置文件路径（例如：streamingflow/configs/dsec_event.yaml）

可选参数：
  --split {train,val}     数据集划分（默认：train）
  --num-batches N        推理的 batch 数量（默认：1）
  --batch-size N         DataLoader 的 batch size（默认：1）
  --num-workers N        DataLoader 的 worker 数（默认：0）
  --device {auto,cuda,cpu}  推理设备（默认：auto）

模式选择：
  --synthetic            使用合成数据模式（不依赖真实数据集）
  --benchmark            启用速度基准测试模式
  --benchmark-encoder-only  只测试编码器速度（需配合 --benchmark）
  --test-flow-data       测试流式数据格式

基准测试参数：
  --num-warmup N         预热次数（默认：5）
  --save-benchmark PATH  保存基准测试结果为 CSV 文件

========================================
流式数据格式说明
========================================

`get_data_flow` 将事件序列分割为多个时间窗口，每个窗口包含：
  - `flow_events`: 该窗口的事件数据列表
  - `flow_lidar`: 对应的 LiDAR 点云数据列表
  - `events_stmp`: 事件时间戳列表
  - `lidar_stmp`: LiDAR 时间戳列表
  - `curr_time_stmp`: 当前时间戳

返回的 `data_dict['flow_data']` 是一个列表，每个元素代表一个时间窗口。

========================================
运行前准备
========================================

1. 安装事件编码依赖：
   ```bash
   pip install -e ./evrt-detr
   ```

2. 确保其它依赖就绪（efficientnet_pytorch 等）

3. 配置数据集路径和参数（在配置文件中）：
   - `DATASET.NAME`: 数据集名称
   - `DATAROOT`: 数据集根目录
   - `MODEL.MODALITY.*`: 模态配置
   - `DATASET.USE_FLOW_DATA`: 是否启用流式数据（可选）
"""

import sys
import time
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


def _get_tensor_batch_size(value):
    """从张量或字典中提取 batch 大小"""
    if torch.is_tensor(value):
        return value.shape[0]
    if isinstance(value, dict):
        # 支持 event 字典格式 {'frames': tensor}
        if 'frames' in value and torch.is_tensor(value['frames']):
            return value['frames'].shape[0]
    return None


def infer_batch_size(tensors):
    """推断 batch 大小，支持张量和字典格式"""
    for key in [
        'image', 'event', 'intrinsics', 'future_egomotion', 'points', 'padded_voxel_points'
    ]:
        value = tensors.get(key)
        batch_size = _get_tensor_batch_size(value)
        if batch_size is not None:
            return batch_size
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

    # 先处理 event 数据以确定正确的 batch_size
    event = prepared.get("event")
    batch_size = infer_batch_size(prepared)
    
    # 如果 event 是字典格式，从中提取实际的 batch_size
    if event is not None:
        if isinstance(event, dict) and 'frames' in event:
            event_frames = event['frames']
            if torch.is_tensor(event_frames):
                actual_batch_size = event_frames.shape[0]
                if actual_batch_size != batch_size:
                    batch_size = actual_batch_size
        elif torch.is_tensor(event):
            actual_batch_size = event.shape[0]
            if actual_batch_size != batch_size:
                batch_size = actual_batch_size
        elif isinstance(event, (list, tuple)):
            raise ValueError("事件输入需要整理为张量或支持的 dict 结构。")

    seq = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)

    # 使用正确的 batch_size 创建所有默认值
    if prepared.get("future_egomotion") is None:
        prepared["future_egomotion"] = torch.zeros(batch_size, seq, 6, device=device)

    if prepared.get("target_timestamp") is None:
        n_future = getattr(cfg, 'N_FUTURE_FRAMES', 0)
        if n_future > 0:
            target_time = n_future * 0.1
            prepared["target_timestamp"] = torch.full((batch_size,), target_time, device=device)
        else:
            prepared["target_timestamp"] = torch.zeros(batch_size, device=device)

    if prepared.get("camera_timestamp") is None:
        timestamps = torch.arange(-seq + 1, 1, dtype=torch.float32, device=device) * 0.1
        prepared["camera_timestamp"] = timestamps.unsqueeze(0).repeat(batch_size, 1)
    
    if prepared.get("lidar_timestamp") is None:
        timestamps = torch.arange(-seq + 1, 1, dtype=torch.float32, device=device) * 0.1
        prepared["lidar_timestamp"] = timestamps.unsqueeze(0).repeat(batch_size, 1)

    if prepared.get("intrinsics") is None:
        prepared["intrinsics"] = torch.eye(3, device=device).view(1, 1, 1, 3, 3).repeat(batch_size, seq, 1, 1, 1)

    if prepared.get("extrinsics") is None:
        prepared["extrinsics"] = torch.eye(4, device=device).view(1, 1, 1, 4, 4).repeat(batch_size, seq, 1, 1, 1)

    if prepared.get("image") is None and cfg.MODEL.MODALITY.USE_CAMERA:
        h, w = cfg.IMAGE.FINAL_DIM
        prepared["image"] = torch.zeros(batch_size, seq, len(cfg.IMAGE.NAMES), 3, h, w, device=device)

    # 处理 event 数据的序列长度
    if event is None and cfg.MODEL.MODALITY.USE_EVENT:
        channels = getattr(cfg.MODEL.EVENT, 'IN_CHANNELS', 0)
        if channels <= 0:
            channels = 2 * getattr(cfg.MODEL.EVENT, 'BINS', 10)
        h, w = cfg.IMAGE.FINAL_DIM
        prepared["event"] = torch.zeros(batch_size, seq, len(cfg.IMAGE.NAMES), channels, h, w, device=device)
    elif event is not None:
        if isinstance(event, dict) and 'frames' in event:
            # 字典格式：处理序列长度
            event_frames = event['frames']
            if torch.is_tensor(event_frames):
                cur_seq = event_frames.shape[1]
                if cur_seq < seq:
                    pad = torch.zeros(
                        (batch_size, seq - cur_seq) + tuple(event_frames.shape[2:]),
                        device=device,
                        dtype=event_frames.dtype,
                    )
                    prepared["event"] = {'frames': torch.cat([event_frames, pad], dim=1)}
                elif cur_seq > seq:
                    prepared["event"] = {'frames': event_frames[:, :seq]}
                else:
                    prepared["event"] = event
        elif torch.is_tensor(event):
            # 张量格式：处理序列长度
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


def log_stage_shape(stage, payload):
    prefix = f"[Event][{stage}]"
    if torch.is_tensor(payload):
        print(f"{prefix} Tensor shape={tuple(payload.shape)}, dtype={payload.dtype}, device={payload.device}")
    elif isinstance(payload, dict):
        keys = list(payload.keys())
        print(f"{prefix} Dict keys={keys}")
    elif isinstance(payload, (list, tuple)):
        length = len(payload)
        exemplar = payload[0] if length > 0 else None
        if torch.is_tensor(exemplar):
            print(f"{prefix} {type(payload).__name__} len={length}, first shape={tuple(exemplar.shape)}")
        else:
            print(f"{prefix} {type(payload).__name__} len={length}")
    else:
        print(f"{prefix} {type(payload)}")


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

    # 检查100Hz频率配置
    from streamingflow.datas.DSECData import DatasetDSEC
    if cfg.DATASET.NAME == 'dsec':
        # 创建临时数据集实例以检查配置
        temp_dataset = DatasetDSEC(cfg, cfg, is_train=(args.split == "train"))
        event_speed = temp_dataset.event_speed
        num_speed = temp_dataset.num_speed
        print(f"\n[100Hz 频率验证]")
        print(f"  event_speed: {event_speed} Hz")
        print(f"  num_speed: {num_speed}")
        print(f"  data_split_interval: {num_speed//(1000//event_speed)}")
        print(f"  event_num per window: {event_speed//10}")
        if event_speed != 100:
            print(f"  [警告] event_speed 不是 100Hz，当前为 {event_speed}Hz")
        else:
            print(f"  [✓] event_speed 配置正确: 100Hz")

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

        raw_event_payload = batch.get("event")
        log_stage_shape("Stage0-raw-loader", raw_event_payload)

        inputs = prepare_model_inputs(batch, device, cfg)

        prepared_event_payload = inputs.get("event")
        log_stage_shape("Stage1-after-prepare_inputs", prepared_event_payload)

        if cfg.MODEL.MODALITY.USE_EVENT and inputs.get("event") is None:
            raise RuntimeError("当前 batch 缺少事件张量，无法测试事件分支。")
        if cfg.MODEL.MODALITY.USE_CAMERA and inputs.get("image") is None:
            raise RuntimeError("USE_CAMERA=True 但 batch['image'] 不存在。")

        event_bev_inspect = None
        encoder_feats_shape = None
        voxel_file_count = 0
        frames = None
        event_input = None
        channels = 0
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
            log_stage_shape("Stage2-after-frame-normalize", frames)
            frames = frames[:, :cfg.TIME_RECEPTIVE_FIELD]
            b, s, n, c, h, w = frames.shape
            
            # 获取期望的通道数（直接使用，不再stacking）
            expected_channels = getattr(cfg.MODEL.EVENT, "IN_CHANNELS", 0)
            if expected_channels <= 0:
                bins = getattr(cfg.MODEL.EVENT, "BINS", 10)
                expected_channels = 2 * bins
            if c != expected_channels:
                raise ValueError(f"事件帧通道数C={c}与期望通道数{expected_channels}不匹配。")
            
            # 直接reshape: [B, S, N, C, H, W] -> [B*S*N, C, H, W]
            event_reshaped = frames.view(b * s * n, c, h, w)
            
            with torch.no_grad():
                encoder_feats, _ = model.event_encoder_forward(event_reshaped)
            encoder_feats_shape = tuple(encoder_feats.shape)
            print(f"[Event][Stage3-after-encoder] Reshaped input shape: {tuple(event_reshaped.shape)}")
            print(f"[Event][Stage3-after-encoder] Encoder output shape: {encoder_feats_shape}")

            event_input = inputs.get("event")
            if torch.is_tensor(event_input):
                channels = event_input.shape[3]
            elif isinstance(event_input, dict) and 'frames' in event_input and torch.is_tensor(event_input['frames']):
                channels = event_input['frames'].shape[3]
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

        if isinstance(outputs, dict):
            for key in ["segmentation", "pedestrian", "hdmap", "instance_center", "instance_offset"]:
                tensor = outputs.get(key)
                if torch.is_tensor(tensor):
                    print(f"[Model][Stage4-final-output] {key}: {tuple(tensor.shape)}")

        # Check if NODE is enabled
        n_future = getattr(cfg, 'N_FUTURE_FRAMES', 0)
        if n_future > 0:
            print(f"\n[Info] NODE 已启用: N_FUTURE_FRAMES={n_future}")
            if inputs.get("target_timestamp") is not None:
                target_ts = inputs.get("target_timestamp")
                if torch.is_tensor(target_ts):
                    print(f"[NODE] target_timestamp shape: {tuple(target_ts.shape)}, values: {target_ts.cpu().numpy()}")
            # Check output sequence length (should include past + future frames)
            if isinstance(outputs, dict) and "segmentation" in outputs:
                seg_shape = outputs["segmentation"].shape
                expected_seq = cfg.TIME_RECEPTIVE_FIELD + n_future
                if len(seg_shape) >= 2 and seg_shape[1] == expected_seq:
                    print(f"[✓] NODE 输出序列长度正确: {seg_shape[1]} (期望: {expected_seq} = {cfg.TIME_RECEPTIVE_FIELD} past + {n_future} future)")
                else:
                    print(f"[!] NODE 输出序列长度: {seg_shape[1] if len(seg_shape) >= 2 else 'N/A'} (期望: {expected_seq})")

        print(f"\n[Info] Batch {step} 推理完成，输出结构：")
        describe_outputs("output", outputs)

        if event_bev_inspect is not None:
            print(f"[Event] BEV shape: {tuple(event_bev_inspect.shape)}")
            # 验证BEV输出尺寸
            # BEV输出尺寸由bev_dimension决定，格式为 (batch, s, n_cameras, channels, H, W)
            # 但实际输出可能是 (batch*s, n_cameras, channels, H, W) 或 (batch, s, channels, H, W)
            actual_bev_shape = tuple(event_bev_inspect.shape)
            expected_channels = model.encoder_out_channels
            expected_h = model.bev_dimension[0].item()
            expected_w = model.bev_dimension[1].item()
            
            # 检查关键维度是否正确
            channels_match = actual_bev_shape[-3] == expected_channels
            h_match = actual_bev_shape[-2] == expected_h
            w_match = actual_bev_shape[-1] == expected_w
            
            if channels_match and h_match and w_match:
                print(f"[✓] BEV 输出尺寸正确: {actual_bev_shape}")
                print(f"    通道数: {actual_bev_shape[-3]} (期望: {expected_channels})")
                print(f"    空间尺寸: {actual_bev_shape[-2]}x{actual_bev_shape[-1]} (期望: {expected_h}x{expected_w})")
            else:
                print(f"[✗] BEV 输出尺寸不匹配!")
                print(f"    实际: {actual_bev_shape}")
                print(f"    期望通道数: {expected_channels}, 实际: {actual_bev_shape[-3]}")
                print(f"    期望空间尺寸: {expected_h}x{expected_w}, 实际: {actual_bev_shape[-2]}x{actual_bev_shape[-1]}")
        if encoder_feats_shape is not None and frames is not None:
            print(f"[Event] Encoder output shape: {encoder_feats_shape}")
            # 验证编码器输出尺寸
            # 注意：编码器输入已将时间维度S堆叠到通道维度(S*C)，因此输出形状为[B*N, C, H', W']，不包含时间维度
            b, s, n, c, h, w = frames.shape
            expected_encoder_shape = (b * n, model.encoder_out_channels, h // model.encoder_downsample, w // model.encoder_downsample)
            if encoder_feats_shape == expected_encoder_shape:
                print(f"[✓] Event 编码器输出尺寸正确: {encoder_feats_shape}")
            else:
                print(f"[✗] Event 编码器输出尺寸不匹配!")
                print(f"    期望: {expected_encoder_shape}")
                print(f"    实际: {encoder_feats_shape}")
        if cfg.MODEL.MODALITY.USE_EVENT:
            print(f"[Event] Voxel files stacked: {voxel_file_count}, Channels: {channels}")
            # 验证输入事件帧尺寸
            if torch.is_tensor(event_input):
                print(f"[Event] 输入事件张量 shape: {tuple(event_input.shape)}")
                expected_event_shape = (cfg.BATCHSIZE, cfg.TIME_RECEPTIVE_FIELD, len(cfg.IMAGE.NAMES), channels, 
                                      cfg.IMAGE.FINAL_DIM[0], cfg.IMAGE.FINAL_DIM[1])
                actual_event_shape = tuple(event_input.shape)
                if actual_event_shape == expected_event_shape:
                    print(f"[✓] 输入事件张量尺寸正确: {actual_event_shape}")
                else:
                    print(f"[✗] 输入事件张量尺寸不匹配!")
                    print(f"    期望: {expected_event_shape}")
                    print(f"    实际: {actual_event_shape}")
        
        # 检查流式数据格式
        if args.test_flow_data:
            flow_data = batch.get('flow_data')
            if flow_data is not None:
                print(f"\n[Flow Data] 检测到流式数据格式")
                if isinstance(flow_data, list):
                    print(f"[Flow Data] 流式数据窗口数量: {len(flow_data)}")
                    # 验证100Hz频率下的窗口数量
                    temp_dataset_for_flow = None
                    if cfg.DATASET.NAME == 'dsec':
                        from streamingflow.datas.DSECData import DatasetDSEC
                        temp_dataset_for_flow = DatasetDSEC(cfg, cfg, is_train=(args.split == "train"))
                        expected_windows = temp_dataset_for_flow.num_speed // (1000 // temp_dataset_for_flow.event_speed)
                        if len(flow_data) == expected_windows:
                            print(f"[✓] 流式数据窗口数量正确: {len(flow_data)} (期望: {expected_windows})")
                        else:
                            print(f"[✗] 流式数据窗口数量不匹配: {len(flow_data)} (期望: {expected_windows})")
                    
                    for i, flow_window in enumerate(flow_data):
                        if isinstance(flow_window, dict):
                            print(f"  窗口 {i}:")
                            # 验证每个窗口的事件数量
                            flow_events = flow_window.get('flow_events', [])
                            if isinstance(flow_events, (list, tuple)) and len(flow_events) > 0:
                                expected_event_num = temp_dataset_for_flow.event_speed // 10 if temp_dataset_for_flow else None
                                if expected_event_num and len(flow_events) == expected_event_num:
                                    print(f"    [✓] flow_events 数量正确: {len(flow_events)} (期望: {expected_event_num})")
                                elif expected_event_num:
                                    print(f"    [✗] flow_events 数量不匹配: {len(flow_events)} (期望: {expected_event_num})")
                                else:
                                    print(f"    flow_events: {len(flow_events)} 个事件")
                            
                            for key, value in flow_window.items():
                                if isinstance(value, (list, tuple)):
                                    print(f"    {key}: list/tuple with {len(value)} elements")
                                    if len(value) > 0:
                                        first_elem = value[0]
                                        if torch.is_tensor(first_elem):
                                            print(f"      First element shape: {tuple(first_elem.shape)}")
                                        elif isinstance(first_elem, np.ndarray):
                                            print(f"      First element shape: {first_elem.shape}")
                                elif torch.is_tensor(value):
                                    print(f"    {key}: tensor {tuple(value.shape)}")
                                elif isinstance(value, np.ndarray):
                                    print(f"    {key}: array {value.shape}")
                                else:
                                    print(f"    {key}: {type(value).__name__}")
                else:
                    print(f"[Flow Data] 流式数据格式: {type(flow_data)}")
            else:
                print(f"\n[Flow Data] 未检测到流式数据格式（flow_data 为 None）")
                print(f"[Flow Data] 提示: 在配置文件中设置 DATASET.USE_FLOW_DATA=True 以启用流式数据格式")


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

        # target_timestamp shape: [batch_size], each element is a scalar target time
        n_future = getattr(cfg, 'N_FUTURE_FRAMES', 0)
        if n_future > 0:
            target_time = n_future * 0.1  # Assuming 0.1s per frame
            target_timestamp = torch.full((batch["image"].size(0),), target_time, device=device)
            print(f"[Synthetic] NODE enabled: N_FUTURE_FRAMES={n_future}, target_timestamp={target_time:.2f}s")
        else:
            target_timestamp = torch.zeros(batch["image"].size(0), device=device)

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


def benchmark_collate_fn(batch):
    """
    自定义 collate 函数，用于基准测试。
    处理字符串类型和变长字段（如点云数据）。
    
    注意：当 USE_LIDAR=False 时，'points' 字段不会出现在数据中。
    """
    from torch.utils.data.dataloader import default_collate
    
    # 变长字段列表（这些字段在不同样本间大小可能不同，保持为 list）
    # - 点云数据：只有当 USE_LIDAR=True 时才会有 'points' 字段
    # - 标注框数据：不同帧中物体数量不同（gt_boxes, gt_obj_ids 等）
    variable_length_keys = {
        'points', 'padded_voxel_points',  # 点云相关
        'gt_boxes', 'gt_len', 'gt_obj_ids', 'gt_boxes_prosed',  # 标注框相关
        'gt_names', 'num_lidar_pts', 'num_radar_pts',  # 其他可能的变长字段
    }
    
    # 过滤掉字符串字段，分离变长字段
    filtered_batch = []
    variable_length_data = {key: [] for key in variable_length_keys}
    
    for sample in batch:
        if isinstance(sample, dict):
            filtered_sample = {}
            for key, value in sample.items():
                # 跳过字符串和其他不支持的类型
                if isinstance(value, str):
                    continue
                # 跳过包含字符串的 numpy 数组
                if isinstance(value, np.ndarray) and value.dtype.kind in ('U', 'S', 'O'):
                    continue
                
                # 变长字段单独处理
                if key in variable_length_keys:
                    variable_length_data[key].append(value)
                else:
                    filtered_sample[key] = value
            
            filtered_batch.append(filtered_sample)
        else:
            filtered_batch.append(sample)
    
    # 使用默认 collate 处理固定长度字段
    try:
        collated = default_collate(filtered_batch) if filtered_batch else {}
        
        # 将变长字段作为 list 添加回去
        for key, values in variable_length_data.items():
            if values and any(v is not None for v in values):
                collated[key] = values  # 保持为 list，不 stack
        
        return collated
    except Exception as e:
        print(f"[Warn] Collate 失败: {e}")
        if filtered_batch:
            print(f"[Debug] Batch keys: {filtered_batch[0].keys() if isinstance(filtered_batch[0], dict) else 'N/A'}")
            # 打印形状信息帮助调试
            if isinstance(filtered_batch[0], dict):
                for key, value in filtered_batch[0].items():
                    if torch.is_tensor(value) or isinstance(value, np.ndarray):
                        print(f"  {key}: {type(value).__name__} {getattr(value, 'shape', 'N/A')}")
        raise


def run_encoder_benchmark_test(cfg, args):
    """
    运行 Event 编码器专项基准测试
    只测试：数据加载 → Event 编码器 → 深度估计 → 特征扩展（到 BEV 投影前）
    不包括：BEV 投影、时序模型、解码器
    """
    device = resolve_device(args.device)
    cfg = cfg.clone()
    cfg.BATCHSIZE = max(1, args.batch_size)
    cfg.N_WORKERS = max(0, args.num_workers)

    print("="*80)
    print("Event 编码器专项速度基准测试")
    print("测试范围：Event 编码器 → 深度估计 → 特征扩展（BEV 投影前）")
    print("="*80)
    print(f"[Info] 使用设备: {device}")
    print(f"[Info] DATASET.NAME={cfg.DATASET.NAME}, BATCHSIZE={cfg.BATCHSIZE}")
    print(f"[Info] 测试批次数: {args.num_batches}, 预热次数: {args.num_warmup}")

    # 准备数据加载器
    from torch.utils.data import DataLoader
    from streamingflow.datas.dataloaders import prepare_dataloaders
    
    trainloader, valloader, train_dataset, val_dataset = prepare_dataloaders(cfg, return_dataset=True)
    dataset = train_dataset if args.split == "train" else val_dataset
    
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCHSIZE,
        shuffle=False,
        num_workers=cfg.N_WORKERS,
        collate_fn=benchmark_collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # 创建模型
    try:
        model = streamingflow(cfg).to(device)
    except ImportError as exc:
        raise SystemExit(
            "\n[Error] 导入 EvRT-DETR 失败，请先 `pip install -e ./evrt-detr`。\n"
            f"原始异常：{exc}"
        )

    model.eval()
    iterator = iter(loader)

    # 计时记录
    data_loading_times = []
    preprocessing_times = []
    encoder_times = []  # 编码器时间
    depth_estimation_times = []  # 深度估计时间
    feature_expansion_times = []  # 特征扩展时间
    total_encoder_pipeline_times = []  # 整个编码器管道时间
    total_times = []

    print(f"\n[1/2] 预热阶段 ({args.num_warmup} 次)...")
    # 预热阶段
    for _ in range(args.num_warmup):
        try:
            batch = next(iterator)
            inputs = prepare_model_inputs(batch, device, cfg)
            event_in = inputs.get("event")
            
            if event_in is None:
                raise ValueError("Event 数据为空，无法测试编码器")
            
            # 准备输入参数
            intrinsics = inputs.get("intrinsics")
            extrinsics = inputs.get("extrinsics")
            b, s, n = intrinsics.shape[:3]
            
            with torch.no_grad():
                # 1. 准备 event frames
                event_frames = model._prepare_event_frames(event_in, s, device=device)
                event_frames = event_frames[:, :s]
                
                # 2. 直接reshape: [B, S, N, C, H, W] -> [B*S*N, C, H, W]
                expected_channels = getattr(cfg.MODEL.EVENT, "IN_CHANNELS", 0)
                if expected_channels <= 0:
                    bins = getattr(cfg.MODEL.EVENT, "BINS", 10)
                    expected_channels = 2 * bins
                b, s, n, c, h, w = event_frames.shape
                if c != expected_channels:
                    raise ValueError(f"事件帧通道数C={c}与期望通道数{expected_channels}不匹配。")
                event_reshaped = event_frames.view(b * s * n, c, h, w)
                
                # 3. Event 编码器前向传播
                event_feats, event_depth_logits = model.event_encoder_forward(event_reshaped)
                
                # 4. 恢复形状: [B*S*N, ...] -> [B, S, N, ...]
                event_feats = event_feats.view(b, s, n, *event_feats.shape[1:])
                
                # 5. 调整深度 bins
                if event_depth_logits is not None:
                    event_depth_logits_out = event_depth_logits.view(b, s, n, *event_depth_logits.shape[1:])
                    event_depth_logits_out = model._resize_event_depth_bins(
                        event_depth_logits_out, model.depth_channels
                    )
                else:
                    event_depth_logits_out = torch.zeros(
                        (b, s, n, model.depth_channels, event_feats.shape[-2], event_feats.shape[-1]),
                        device=event_feats.device,
                        dtype=event_feats.dtype,
                    )
                
                # 6. 深度分布计算
                event_depth_prob = event_depth_logits_out.softmax(dim=3)
                
                # 7. 特征+深度扩展
                event_volume = model._expand_features_with_depth(event_feats, event_depth_prob)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

    print(f"\n[2/2] 正式测试阶段 ({args.num_batches} 次)...")
    # 正式测试阶段
    for step in range(args.num_batches):
        t_start_total = time.time()
        
        # 测量数据加载时间
        t_start_data = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            print("[Warn] 达到数据集末尾，重新开始。")
            iterator = iter(loader)
            batch = next(iterator)
        t_data = time.time() - t_start_data

        # 测量预处理时间
        t_start_preproc = time.time()
        inputs = prepare_model_inputs(batch, device, cfg)
        event_in = inputs.get("event")
        intrinsics = inputs.get("intrinsics")
        extrinsics = inputs.get("extrinsics")
        b, s, n = intrinsics.shape[:3]
        t_preproc = time.time() - t_start_preproc

        # 测量编码器管道时间（细粒度）
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
            # 1. Event 编码器
            start_encoder = torch.cuda.Event(enable_timing=True)
            end_encoder = torch.cuda.Event(enable_timing=True)
            
            # 2. 深度估计
            start_depth = torch.cuda.Event(enable_timing=True)
            end_depth = torch.cuda.Event(enable_timing=True)
            
            # 3. 特征扩展
            start_expansion = torch.cuda.Event(enable_timing=True)
            end_expansion = torch.cuda.Event(enable_timing=True)
            
            # 完整管道
            start_pipeline = torch.cuda.Event(enable_timing=True)
            end_pipeline = torch.cuda.Event(enable_timing=True)
            
            start_pipeline.record()
            
            with torch.no_grad():
                # 准备 event frames
                from streamingflow.utils.network import pack_sequence_dim, unpack_sequence_dim
                event_frames = model._prepare_event_frames(event_in, s, device=device)
                event_frames = event_frames[:, :s]
                event_packed = pack_sequence_dim(event_frames)
                
                # Event 编码器
                start_encoder.record()
                event_feats, event_depth_logits = model.event_encoder_forward(event_packed)
                end_encoder.record()
                
                # Unpack
                event_feats = unpack_sequence_dim(event_feats, b, s)
                
                # 深度估计
                start_depth.record()
                if event_depth_logits is not None:
                    event_depth_logits_out = unpack_sequence_dim(event_depth_logits, b, s)
                    event_depth_logits_out = model._resize_event_depth_bins(
                        event_depth_logits_out, model.depth_channels
                    )
                else:
                    event_depth_logits_out = torch.zeros(
                        (b, s, n, model.depth_channels, event_feats.shape[-2], event_feats.shape[-1]),
                        device=event_feats.device,
                        dtype=event_feats.dtype,
                    )
                event_depth_prob = event_depth_logits_out.softmax(dim=3)
                end_depth.record()
                
                # 特征扩展
                start_expansion.record()
                event_volume = model._expand_features_with_depth(event_feats, event_depth_prob)
                end_expansion.record()
            
            end_pipeline.record()
            torch.cuda.synchronize()
            
            t_encoder = start_encoder.elapsed_time(end_encoder) / 1000.0
            t_depth = start_depth.elapsed_time(end_depth) / 1000.0
            t_expansion = start_expansion.elapsed_time(end_expansion) / 1000.0
            t_pipeline = start_pipeline.elapsed_time(end_pipeline) / 1000.0
        else:
            # CPU 模式
            t_start_pipeline = time.time()
            
            with torch.no_grad():
                from streamingflow.utils.network import pack_sequence_dim, unpack_sequence_dim
                event_frames = model._prepare_event_frames(event_in, s, device=device)
                event_frames = event_frames[:, :s]
                event_packed = pack_sequence_dim(event_frames)
                
                t_start_encoder = time.time()
                event_feats, event_depth_logits = model.event_encoder_forward(event_packed)
                t_encoder = time.time() - t_start_encoder
                
                event_feats = unpack_sequence_dim(event_feats, b, s)
                
                t_start_depth = time.time()
                if event_depth_logits is not None:
                    event_depth_logits_out = unpack_sequence_dim(event_depth_logits, b, s)
                    event_depth_logits_out = model._resize_event_depth_bins(
                        event_depth_logits_out, model.depth_channels
                    )
                else:
                    event_depth_logits_out = torch.zeros(
                        (b, s, n, model.depth_channels, event_feats.shape[-2], event_feats.shape[-1]),
                        device=event_feats.device,
                        dtype=event_feats.dtype,
                    )
                event_depth_prob = event_depth_logits_out.softmax(dim=3)
                t_depth = time.time() - t_start_depth
                
                t_start_expansion = time.time()
                event_volume = model._expand_features_with_depth(event_feats, event_depth_prob)
                t_expansion = time.time() - t_start_expansion
            
            t_pipeline = time.time() - t_start_pipeline

        t_total = time.time() - t_start_total

        # 记录时间（毫秒）
        data_loading_times.append(t_data * 1000)
        preprocessing_times.append(t_preproc * 1000)
        encoder_times.append(t_encoder * 1000)
        depth_estimation_times.append(t_depth * 1000)
        feature_expansion_times.append(t_expansion * 1000)
        total_encoder_pipeline_times.append(t_pipeline * 1000)
        total_times.append(t_total * 1000)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  进度: {step + 1}/{args.num_batches}")

    # 统计结果
    data_loading_times = np.array(data_loading_times)
    preprocessing_times = np.array(preprocessing_times)
    encoder_times = np.array(encoder_times)
    depth_estimation_times = np.array(depth_estimation_times)
    feature_expansion_times = np.array(feature_expansion_times)
    total_encoder_pipeline_times = np.array(total_encoder_pipeline_times)
    total_times = np.array(total_times)

    print("\n" + "="*80)
    print("Event 编码器专项速度基准测试结果")
    print("="*80)
    print(f"{'指标':<40} {'平均值':>12} {'中位数':>12} {'标准差':>12}")
    print("-"*80)
    print(f"{'数据加载时间 (ms/batch)':<40} {data_loading_times.mean():>12.2f} {np.median(data_loading_times):>12.2f} {data_loading_times.std():>12.2f}")
    print(f"{'数据预处理时间 (ms/batch)':<40} {preprocessing_times.mean():>12.2f} {np.median(preprocessing_times):>12.2f} {preprocessing_times.std():>12.2f}")
    print("-"*80)
    print(f"{'Event 编码器时间 (ms/batch)':<40} {encoder_times.mean():>12.2f} {np.median(encoder_times):>12.2f} {encoder_times.std():>12.2f}")
    print(f"{'深度估计时间 (ms/batch)':<40} {depth_estimation_times.mean():>12.2f} {np.median(depth_estimation_times):>12.2f} {depth_estimation_times.std():>12.2f}")
    print(f"{'特征扩展时间 (ms/batch)':<40} {feature_expansion_times.mean():>12.2f} {np.median(feature_expansion_times):>12.2f} {feature_expansion_times.std():>12.2f}")
    print("-"*80)
    print(f"{'编码器管道总时间 (ms/batch)':<40} {total_encoder_pipeline_times.mean():>12.2f} {np.median(total_encoder_pipeline_times):>12.2f} {total_encoder_pipeline_times.std():>12.2f}")
    print(f"{'端到端总时间 (ms/batch)':<40} {total_times.mean():>12.2f} {np.median(total_times):>12.2f} {total_times.std():>12.2f}")
    print("-"*80)
    
    # 吞吐量 (每秒处理的样本数) = (每秒批次数) * (每批样本数)
    encoder_throughput = (1000.0 / total_encoder_pipeline_times.mean()) * cfg.BATCHSIZE if total_encoder_pipeline_times.mean() > 0 else 0
    # 延迟 (每个样本的平均处理时间)
    encoder_latency_per_sample = total_encoder_pipeline_times.mean() / cfg.BATCHSIZE if cfg.BATCHSIZE > 0 else 0
    
    # Event frames 吞吐量 (考虑时序长度和相机数量)
    time_receptive_field = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)
    num_cameras = len(getattr(cfg.IMAGE, 'NAMES', ['camera_0']))
    frames_per_sample = time_receptive_field * num_cameras
    encoder_fps = encoder_throughput * frames_per_sample
    encoder_latency_per_frame = encoder_latency_per_sample / frames_per_sample if frames_per_sample > 0 else 0
    
    print(f"{'编码器吞吐量 (samples/s)':<40} {encoder_throughput:>12.2f}")
    print(f"{'编码器延迟 (ms/sample)':<40} {encoder_latency_per_sample:>12.2f}")
    print(f"{'编码器吞吐量 (frames/s, FPS)':<40} {encoder_fps:>12.2f}")
    print(f"{'编码器延迟 (ms/frame)':<40} {encoder_latency_per_frame:>12.2f}")
    print(f"  (TIME_RECEPTIVE_FIELD={time_receptive_field}, cameras={num_cameras}, frames/sample={frames_per_sample})")
    print("="*80)
    
    # 批处理效率分析
    print(f"\n批处理效率分析 (batch_size={cfg.BATCHSIZE}):")
    if cfg.BATCHSIZE > 1:
        print(f"  单个样本平均延迟: {encoder_latency_per_sample:.2f} ms")
        print(f"  并行效率: 如果串行处理 {cfg.BATCHSIZE} 个样本需要 {encoder_latency_per_sample * cfg.BATCHSIZE:.2f} ms")
        print(f"            实际批处理只需要 {total_encoder_pipeline_times.mean():.2f} ms")
        speedup = (encoder_latency_per_sample * cfg.BATCHSIZE) / total_encoder_pipeline_times.mean()
        print(f"  加速比: {speedup:.2f}x (理想值: {cfg.BATCHSIZE}.00x)")
    else:
        print(f"  单个样本延迟: {encoder_latency_per_sample:.2f} ms")
        print(f"  (提示: 增大 batch_size 可以提高吞吐量)")

    # 百分位数分析
    print(f"\n编码器管道时间分布:")
    p25, p50, p75, p90, p95, p99 = np.percentile(total_encoder_pipeline_times, [25, 50, 75, 90, 95, 99])
    print(f"  P25:  {p25:>8.2f} ms")
    print(f"  P50:  {p50:>8.2f} ms (中位数)")
    print(f"  P75:  {p75:>8.2f} ms")
    print(f"  P90:  {p90:>8.2f} ms")
    print(f"  P95:  {p95:>8.2f} ms")
    print(f"  P99:  {p99:>8.2f} ms")

    # 组件占比分析
    print(f"\n编码器管道各组件占比:")
    encoder_pct = (encoder_times.mean() / total_encoder_pipeline_times.mean()) * 100
    depth_pct = (depth_estimation_times.mean() / total_encoder_pipeline_times.mean()) * 100
    expansion_pct = (feature_expansion_times.mean() / total_encoder_pipeline_times.mean()) * 100
    print(f"  Event 编码器: {encoder_pct:>6.2f}%  ({encoder_times.mean():.2f} ms)")
    print(f"  深度估计:     {depth_pct:>6.2f}%  ({depth_estimation_times.mean():.2f} ms)")
    print(f"  特征扩展:     {expansion_pct:>6.2f}%  ({feature_expansion_times.mean():.2f} ms)")


def run_benchmark_test(cfg, args):
    """运行速度基准测试"""
    device = resolve_device(args.device)
    cfg = cfg.clone()
    cfg.BATCHSIZE = max(1, args.batch_size)
    cfg.N_WORKERS = max(0, args.num_workers)

    print("="*80)
    print("Event 分支速度基准测试")
    print("="*80)
    print(f"[Info] 使用设备: {device}")
    print(f"[Info] DATASET.NAME={cfg.DATASET.NAME}, BATCHSIZE={cfg.BATCHSIZE}")
    print(f"[Info] 测试批次数: {args.num_batches}, 预热次数: {args.num_warmup}")

    # 准备数据加载器（使用自定义 collate_fn）
    from torch.utils.data import DataLoader
    
    # 获取数据集对象（return_dataset=True 时返回 4 个值）
    trainloader, valloader, train_dataset, val_dataset = prepare_dataloaders(cfg, return_dataset=True)
    
    # 选择训练集或验证集
    dataset = train_dataset if args.split == "train" else val_dataset
    
    # 创建 DataLoader，使用自定义 collate_fn
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCHSIZE,
        shuffle=False,
        num_workers=cfg.N_WORKERS,
        collate_fn=benchmark_collate_fn,  # 使用自定义 collate 函数
        pin_memory=True if device.type == 'cuda' else False,
    )

    # 创建模型
    try:
        model = streamingflow(cfg).to(device)
    except ImportError as exc:
        raise SystemExit(
            "\n[Error] 导入 EvRT-DETR 失败，请先 `pip install -e ./evrt-detr`。\n"
            f"原始异常：{exc}"
        )

    model.eval()
    iterator = iter(loader)

    # 计时记录
    data_loading_times = []
    preprocessing_times = []
    inference_times = []
    total_times = []

    print(f"\n[1/2] 预热阶段 ({args.num_warmup} 次)...")
    # 预热阶段
    for _ in range(args.num_warmup):
        try:
            batch = next(iterator)
            inputs = prepare_model_inputs(batch, device, cfg)
            with torch.no_grad():
                _ = model(
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
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

    print(f"\n[2/2] 正式测试阶段 ({args.num_batches} 次)...")
    # 正式测试阶段
    for step in range(args.num_batches):
        # 测量数据加载时间
        t_start_total = time.time()
        t_start_data = time.time()
        
        try:
            batch = next(iterator)
        except StopIteration:
            print("[Warn] 达到数据集末尾，重新开始。")
            iterator = iter(loader)
            batch = next(iterator)
        
        t_data = time.time() - t_start_data

        # 测量预处理时间
        t_start_preproc = time.time()
        inputs = prepare_model_inputs(batch, device, cfg)
        t_preproc = time.time() - t_start_preproc

        # 测量推理时间
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                _ = model(
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
            end_event.record()
            torch.cuda.synchronize()
            t_inference = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
        else:
            t_start_inference = time.time()
            with torch.no_grad():
                _ = model(
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
            t_inference = time.time() - t_start_inference

        t_total = time.time() - t_start_total

        # 记录时间（毫秒）
        data_loading_times.append(t_data * 1000)
        preprocessing_times.append(t_preproc * 1000)
        inference_times.append(t_inference * 1000)
        total_times.append(t_total * 1000)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  进度: {step + 1}/{args.num_batches}")

    # 统计结果
    data_loading_times = np.array(data_loading_times)
    preprocessing_times = np.array(preprocessing_times)
    inference_times = np.array(inference_times)
    total_times = np.array(total_times)

    print("\n" + "="*80)
    print("速度基准测试结果")
    print("="*80)
    print(f"{'指标':<40} {'平均值':>12} {'中位数':>12} {'标准差':>12}")
    print("-"*80)
    print(f"{'数据加载时间 (ms/batch)':<40} {data_loading_times.mean():>12.2f} {np.median(data_loading_times):>12.2f} {data_loading_times.std():>12.2f}")
    print(f"{'数据预处理时间 (ms/batch)':<40} {preprocessing_times.mean():>12.2f} {np.median(preprocessing_times):>12.2f} {preprocessing_times.std():>12.2f}")
    print(f"{'模型推理时间 (ms/batch)':<40} {inference_times.mean():>12.2f} {np.median(inference_times):>12.2f} {inference_times.std():>12.2f}")
    print(f"{'总时间 (ms/batch)':<40} {total_times.mean():>12.2f} {np.median(total_times):>12.2f} {total_times.std():>12.2f}")
    print("-"*80)
    
    # 吞吐量 (每秒处理的样本数) = (每秒批次数) * (每批样本数)
    inference_throughput = (1000.0 / inference_times.mean()) * cfg.BATCHSIZE if inference_times.mean() > 0 else 0
    total_throughput = (1000.0 / total_times.mean()) * cfg.BATCHSIZE if total_times.mean() > 0 else 0
    # 延迟 (每个样本的平均处理时间)
    inference_latency_per_sample = inference_times.mean() / cfg.BATCHSIZE if cfg.BATCHSIZE > 0 else 0
    total_latency_per_sample = total_times.mean() / cfg.BATCHSIZE if cfg.BATCHSIZE > 0 else 0
    
    # Event frames 吞吐量 (考虑时序长度和相机数量)
    time_receptive_field = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)
    num_cameras = len(getattr(cfg.IMAGE, 'NAMES', ['camera_0']))
    frames_per_sample = time_receptive_field * num_cameras
    inference_fps = inference_throughput * frames_per_sample
    total_fps = total_throughput * frames_per_sample
    inference_latency_per_frame = inference_latency_per_sample / frames_per_sample if frames_per_sample > 0 else 0
    total_latency_per_frame = total_latency_per_sample / frames_per_sample if frames_per_sample > 0 else 0
    
    print(f"{'推理吞吐量 (samples/s)':<40} {inference_throughput:>12.2f}")
    print(f"{'推理延迟 (ms/sample)':<40} {inference_latency_per_sample:>12.2f}")
    print(f"{'推理吞吐量 (frames/s, FPS)':<40} {inference_fps:>12.2f}")
    print(f"{'推理延迟 (ms/frame)':<40} {inference_latency_per_frame:>12.2f}")
    print("-"*80)
    print(f"{'端到端吞吐量 (samples/s)':<40} {total_throughput:>12.2f}")
    print(f"{'端到端延迟 (ms/sample)':<40} {total_latency_per_sample:>12.2f}")
    print(f"{'端到端吞吐量 (frames/s, FPS)':<40} {total_fps:>12.2f}")
    print(f"{'端到端延迟 (ms/frame)':<40} {total_latency_per_frame:>12.2f}")
    print(f"  (TIME_RECEPTIVE_FIELD={time_receptive_field}, cameras={num_cameras}, frames/sample={frames_per_sample})")
    print("="*80)
    
    # 批处理效率分析
    print(f"\n批处理效率分析 (batch_size={cfg.BATCHSIZE}):")
    if cfg.BATCHSIZE > 1:
        print(f"  单个样本平均延迟: {inference_latency_per_sample:.2f} ms")
        print(f"  并行效率: 如果串行处理 {cfg.BATCHSIZE} 个样本需要 {inference_latency_per_sample * cfg.BATCHSIZE:.2f} ms")
        print(f"            实际批处理只需要 {inference_times.mean():.2f} ms")
        speedup = (inference_latency_per_sample * cfg.BATCHSIZE) / inference_times.mean()
        print(f"  加速比: {speedup:.2f}x (理想值: {cfg.BATCHSIZE}.00x)")
    else:
        print(f"  单个样本延迟: {inference_latency_per_sample:.2f} ms")
        print(f"  (提示: 增大 batch_size 可以提高吞吐量)")
    print("="*80)

    # 百分位数分析
    print(f"\n推理时间分布:")
    p25, p50, p75, p90, p95, p99 = np.percentile(inference_times, [25, 50, 75, 90, 95, 99])
    print(f"  P25:  {p25:>8.2f} ms")
    print(f"  P50:  {p50:>8.2f} ms (中位数)")
    print(f"  P75:  {p75:>8.2f} ms")
    print(f"  P90:  {p90:>8.2f} ms")
    print(f"  P95:  {p95:>8.2f} ms")
    print(f"  P99:  {p99:>8.2f} ms")

    # 保存结果
    if args.save_benchmark:
        try:
            import pandas as pd
            results = {
                'data_loading_mean': data_loading_times.mean(),
                'data_loading_std': data_loading_times.std(),
                'preprocessing_mean': preprocessing_times.mean(),
                'preprocessing_std': preprocessing_times.std(),
                'inference_mean': inference_times.mean(),
                'inference_std': inference_times.std(),
                'total_mean': total_times.mean(),
                'total_std': total_times.std(),
                'fps': 1000.0 / inference_times.mean(),
                'throughput': 1000.0 / total_times.mean(),
                'batch_size': cfg.BATCHSIZE,
                'num_batches': args.num_batches,
                'device': str(device),
            }
            df = pd.DataFrame([results])
            df.to_csv(args.save_benchmark, index=False)
            print(f"\n[Info] 结果已保存到: {args.save_benchmark}")
        except ImportError:
            print("\n[Warn] pandas 未安装，无法保存CSV文件")

    return {
        'data_loading': data_loading_times,
        'preprocessing': preprocessing_times,
        'inference': inference_times,
        'total': total_times,
    }


def parse_args():
    parser = get_parser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='推理使用的划分（train/val）。')
    parser.add_argument('--num-batches', type=int, default=1, help='连续推理的 batch 数量。')
    parser.add_argument('--batch-size', type=int, default=1, help='仅 dataset 模式下 DataLoader 的 batch size。')
    parser.add_argument('--num-workers', type=int, default=0, help='仅 dataset 模式下 DataLoader 的 worker 数。')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='推理设备。')
    parser.add_argument('--synthetic', action='store_true', help='使用合成数据而非真实数据集。')
    parser.add_argument('--benchmark', action='store_true', help='启用速度基准测试模式。')
    parser.add_argument('--benchmark-encoder-only', action='store_true', help='只测试 Event 编码器到 BEV 投影前的速度（不包括 BEV 投影、时序模型、解码器）。')
    parser.add_argument('--num-warmup', type=int, default=5, help='速度测试预热次数。')
    parser.add_argument('--save-benchmark', type=str, default=None, help='基准测试结果保存路径（CSV格式）。')
    parser.add_argument('--test-flow-data', action='store_true', help='测试流式数据格式（flow_data）。需要在配置文件中设置 DATASET.USE_FLOW_DATA=True。')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.synthetic:
        device = resolve_device(args.device)
        run_synthetic_self_check(device)
        return

    cfg = get_cfg(args)
    
    if args.benchmark_encoder_only:
        run_encoder_benchmark_test(cfg, args)
    elif args.benchmark:
        run_benchmark_test(cfg, args)
    else:
        run_dataset_inference(cfg, args)


if __name__ == "__main__":
    main()
