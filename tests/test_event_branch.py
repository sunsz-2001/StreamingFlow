"""
快速自检脚本：验证事件分支从原始事件流→张量化→编码→BEV 融合的完整路径。

运行方式：
    python tests/test_event_branch.py

需提前确保：
    1) 已 `pip install -e ./evrt-detr`（或确保包可导入）；
    2) efficientnet_pytorch 等依赖安装完毕。
"""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from streamingflow.config import get_cfg  # noqa: E402
from streamingflow.models.streamingflow import streamingflow  # noqa: E402
from streamingflow.utils.event_tensor import EventTensorizer  # noqa: E402


def build_test_cfg():
    cfg = get_cfg()

    # 简化尺寸，保持编码器兼容
    cfg.TIME_RECEPTIVE_FIELD = 2
    cfg.N_FUTURE_FRAMES = 0

    cfg.MODEL.MODALITY.USE_CAMERA = True
    cfg.MODEL.MODALITY.USE_EVENT = True
    cfg.MODEL.MODALITY.USE_LIDAR = False
    cfg.MODEL.MODALITY.USE_RADAR = False

    cfg.MODEL.ENCODER.NAME = "efficientnet-b0"
    cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION = True
    cfg.MODEL.ENCODER.OUT_CHANNELS = 64
    cfg.MODEL.ENCODER.DOWNSAMPLE = 8

    cfg.MODEL.EVENT.BINS = 4
    cfg.MODEL.EVENT.IN_CHANNELS = 0  # 自动使用 2 * bins
    cfg.MODEL.EVENT.FUSION_TYPE = "concat"
    cfg.MODEL.EVENT.MULTISCALE_FUSION = "sum"
    cfg.MODEL.EVENT.FREEZE_BACKBONE = False
    cfg.MODEL.EVENT.PRETRAINED = False
    cfg.MODEL.EVENT.NORMALIZE = False

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

    # 将图像 resize / crop 逻辑设置为恒等，方便事件对齐
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
    N = len(cfg.IMAGE.NAMES[:3])  # 取前三个相机即可
    C_img = 3
    H, W = cfg.IMAGE.FINAL_DIM

    image = torch.randn(B, S, N, C_img, H, W, device=device)

    intrinsics = torch.eye(3, device=device).view(1, 1, 1, 3, 3).repeat(B, S, N, 1, 1)
    extrinsics = torch.eye(4, device=device).view(1, 1, 1, 4, 4).repeat(B, S, N, 1, 1)
    future_ego = torch.zeros(B, S, 6, device=device)

    # 构造原始事件流（嵌套 list）：[B][S][N] -> (E, 4)
    events_nested = []
    base_t = torch.linspace(0.0, 1.0, steps=cfg.MODEL.EVENT.BINS + 1)
    for b in range(B):
        seq_list = []
        for s in range(S):
            cam_list = []
            for n in range(N):
                num_events = 200
                xs = torch.rand(num_events) * (cfg.IMAGE.ORIGINAL_WIDTH - 1)
                ys = torch.rand(num_events) * (cfg.IMAGE.ORIGINAL_HEIGHT - 1)
                ts = torch.rand(num_events) * (base_t[-1] - base_t[0]) + base_t[0] + s * 0.05
                polarity = torch.where(torch.rand(num_events) > 0.5, torch.ones(num_events), -torch.ones(num_events))
                event_tensor = torch.stack([xs, ys, ts, polarity], dim=1)
                cam_list.append(event_tensor)
            seq_list.append(cam_list)
        events_nested.append(seq_list)

    # 也演示一次张量化输出
    tensorizer = EventTensorizer(cfg)
    event_frames = tensorizer.prepare_frames(events_nested, device=device, max_seq_len=S)
    print(f"[Step] Event frames (tensorizer) shape: {event_frames.shape}, sum: {event_frames.sum().item():.2f}")

    return {
        "image": image,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "future_egomotion": future_ego,
        "event_nested": events_nested,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_test_cfg()

    # 只保留前 N 个相机名称以匹配假数据
    cfg.IMAGE.NAMES = cfg.IMAGE.NAMES[:3]

    batch = make_mock_batch(cfg, device)

    try:
        model = streamingflow(cfg).to(device)
    except ImportError as exc:
        raise SystemExit(
            "\n[Error] 导入 EvRT-DETR 失败，请先 `pip install -e ./evrt-detr`。\n"
            f"原始异常：{exc}"
        )

    model.eval()

    with torch.no_grad():
        # 先测试直接调用内部函数（避免额外 copy）
        bev_feat, depth, cam_front = model.calculate_birds_eye_view_features(
            batch["image"],
            batch["intrinsics"],
            batch["extrinsics"],
            batch["future_egomotion"],
            event=batch["event_nested"],
        )

    print(f"[Step] BEV tensor shape: {bev_feat.shape}")
    if depth is not None:
        print(f"[Step] Depth logits shape: {depth.shape}")
    print(f"[Step] Cam front feature: {None if cam_front is None else cam_front.shape}")

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

    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(f"[Output] {key}: {tuple(value.shape)}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"[Output] {key}.{sub_key}: {tuple(sub_value.shape)}")

    print("\n事件分支自检完成，可在日志中核对各阶段形状。")


if __name__ == "__main__":
    main()
