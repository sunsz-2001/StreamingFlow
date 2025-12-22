import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from streamingflow.datas.dataloaders import prepare_dataloaders
from streamingflow.trainer import TrainingModule
from streamingflow.utils.network import preprocess_batch
from streamingflow.config import get_cfg, get_parser
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d


def to_device(batch: Dict, device: torch.device) -> Dict:
    """Move required tensors to device."""
    preprocess_batch(batch, device)
    return batch


def gather_batch_outputs(model, batch, device, cfg=None, decode: bool = True) -> Tuple[List, List]:
    """Run forward pass and optionally decode predictions."""
    event = batch.get("event")
    
    # 处理 points 和 lidar_timestamp：如果使用 flow_data 格式，需要从 flow_data 中提取
    points = batch.get("points")
    lidar_timestamp = batch.get("lidar_timestamp")
    
    if cfg is not None:
        use_flow_data = getattr(cfg.DATASET, 'USE_FLOW_DATA', False)
        use_lidar = getattr(cfg.MODEL.MODALITY, 'USE_LIDAR', False)
        
        if use_flow_data and use_lidar and 'flow_data' in batch:
            # 从 flow_data 中提取点云数据（类似 trainer.py 中的逻辑）
            flow_data_list = batch['flow_data']
            points = []
            lidar_timestamps_list = []
            
            for sample_idx, sample_flow_data in enumerate(flow_data_list):
                if isinstance(sample_flow_data, list) and len(sample_flow_data) > 0:
                    first_window = sample_flow_data[0]
                    if isinstance(first_window, dict) and 'flow_lidar' in first_window:
                        flow_lidar = first_window['flow_lidar']
                        if isinstance(flow_lidar, list) and len(flow_lidar) > 0:
                            first_lidar = flow_lidar[0]
                            receptive_field = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)
                            
                            if isinstance(first_lidar, (list, tuple)):
                                # 多时间步点云格式
                                sample_points = []
                                for t in range(min(len(first_lidar), receptive_field)):
                                    pc_t = first_lidar[t]
                                    if isinstance(pc_t, np.ndarray):
                                        pc_t = torch.from_numpy(pc_t)
                                    elif not torch.is_tensor(pc_t):
                                        raise TypeError(
                                            f"Expected point cloud to be Tensor or np.ndarray, but got {type(pc_t)}"
                                        )
                                    else:
                                        pc_t = pc_t.to(dtype=torch.float32)
                                    sample_points.append(pc_t)
                                # 如果时间步不足，用最后一个点云填充
                                while len(sample_points) < receptive_field:
                                    sample_points.append(sample_points[-1].clone() if torch.is_tensor(sample_points[-1]) else sample_points[-1])
                                points.append(sample_points)
                            else:
                                # 单个点云格式（TIME_RECEPTIVE_FIELD=1）
                                point_cloud = first_lidar
                                if isinstance(point_cloud, np.ndarray):
                                    point_cloud = torch.from_numpy(point_cloud)
                                elif not torch.is_tensor(point_cloud):
                                    raise TypeError(
                                        f"Expected point cloud to be Tensor or np.ndarray, but got {type(point_cloud)}"
                                    )
                                else:
                                    point_cloud = point_cloud.to(dtype=torch.float32)
                                points.append(point_cloud)
                            
                            # 提取时间戳
                            if 'lidar_stmp' in first_window:
                                lidar_stmp = first_window['lidar_stmp']
                                if isinstance(lidar_stmp, (list, tuple)) and len(lidar_stmp) > 0:
                                    lidar_timestamps_list.append(lidar_stmp[0])
                                else:
                                    lidar_timestamps_list.append(lidar_stmp if not isinstance(lidar_stmp, (list, tuple)) else lidar_stmp[0])
                            else:
                                lidar_timestamps_list.append(0.0)
                        else:
                            # 空点云处理
                            receptive_field = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)
                            if receptive_field > 1:
                                points.append([torch.zeros(0, 4, dtype=torch.float32)] * receptive_field)
                            else:
                                points.append(torch.zeros(0, 4, dtype=torch.float32))
                            lidar_timestamps_list.append(0.0)
                    else:
                        # 无效的窗口格式
                        receptive_field = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)
                        if receptive_field > 1:
                            points.append([torch.zeros(0, 4, dtype=torch.float32)] * receptive_field)
                        else:
                            points.append(torch.zeros(0, 4, dtype=torch.float32))
                        lidar_timestamps_list.append(0.0)
                else:
                    # 空的 sample_flow_data
                    receptive_field = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)
                    if receptive_field > 1:
                        points.append([torch.zeros(0, 4, dtype=torch.float32)] * receptive_field)
                    else:
                        points.append(torch.zeros(0, 4, dtype=torch.float32))
                    lidar_timestamps_list.append(0.0)
            
            # 将时间戳列表转换为张量 [B, T]
            if len(lidar_timestamps_list) > 0:
                receptive_field = getattr(cfg, 'TIME_RECEPTIVE_FIELD', 1)
                if receptive_field == 1:
                    lidar_timestamp = torch.tensor(lidar_timestamps_list, device=device, dtype=torch.float32).unsqueeze(1)  # [B, 1]
                else:
                    # 对于多时间步，需要从 flow_data 中提取所有时间步的时间戳
                    lidar_timestamp_list_2d = []
                    for sample_idx, sample_flow_data in enumerate(flow_data_list):
                        if isinstance(sample_flow_data, list) and len(sample_flow_data) > 0:
                            first_window = sample_flow_data[0]
                            if 'lidar_stmp' in first_window:
                                lidar_stmp = first_window['lidar_stmp']
                                if isinstance(lidar_stmp, (list, tuple)):
                                    lidar_timestamp_list_2d.append(lidar_stmp[:receptive_field])
                                else:
                                    lidar_timestamp_list_2d.append([lidar_stmp] * receptive_field)
                            else:
                                lidar_timestamp_list_2d.append([0.0] * receptive_field)
                        else:
                            lidar_timestamp_list_2d.append([0.0] * receptive_field)
                    lidar_timestamp = torch.tensor(lidar_timestamp_list_2d, device=device, dtype=torch.float32)  # [B, T]
    
    output = model(
        batch.get("image"),
        batch.get("intrinsics"),
        batch.get("extrinsics"),
        batch.get("future_egomotion"),
        batch.get("padded_voxel_points"),
        batch.get("camera_timestamp"),
        points,
        lidar_timestamp,
        batch.get("target_timestamp"),
        event=event,
    )

    if not decode:
        return [], []

    preds_raw = output["detection"]
    decoded = model.decoder.detection_head.get_bboxes(preds_raw, batch["metas"])
    # decoded: [[boxes, scores, labels]] - 外层是layer，内层是batch
    # 当前batch_size=1，所以 decoded[0] = [boxes, scores, labels]
    preds = []
    for layer_result in decoded:  # 通常只有1个layer
        boxes, scores, labels = layer_result
        preds.append((boxes.tensor.detach().cpu(), scores.detach().cpu(), labels.detach().cpu()))

    gts = []
    for gt_boxes, gt_labels in zip(batch["gt_bboxes_3d"], batch["gt_labels_3d"]):
        gts.append((gt_boxes.tensor.cpu(), gt_labels.cpu()))
    return preds, gts


def compute_ap_per_class(
    preds: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    gts: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    iou_thr: float,
) -> Dict[str, float]:
    """Compute per-class AP with 3D IoU matching (single IoU threshold)."""
    ap_results = {}
    for cid in range(num_classes):
        flat_preds = []
        for sid, boxes, scores, labels in preds:
            mask = labels == cid
            if mask.sum() == 0:
                continue
            for box, score in zip(boxes[mask][:, :7], scores[mask]):
                flat_preds.append((sid, box, float(score)))
        flat_preds.sort(key=lambda x: x[2], reverse=True)

        gt_used = {sid: torch.zeros(len(gts[sid][0]), dtype=torch.bool) for sid in gts}
        num_gt = sum((gts[sid][1] == cid).sum().item() for sid in gts)
        if num_gt == 0:
            ap_results[f"class_{cid}"] = 0.0
            continue

        tp, fp = [], []
        for sid, box, _ in flat_preds:
            gt_boxes, gt_labels = gts.get(sid, (None, None))
            if gt_boxes is None:
                fp.append(1)
                tp.append(0)
                continue
            mask = gt_labels == cid
            if mask.sum() == 0:
                fp.append(1)
                tp.append(0)
                continue
            overlaps = bbox_overlaps_3d(
                box.unsqueeze(0), gt_boxes[mask][:, :7], coordinate="lidar"
            )[0]
            max_iou, max_idx = overlaps.max(0)
            # 转换为标量
            max_iou = max_iou.item() if isinstance(max_iou, torch.Tensor) else float(max_iou)
            max_idx = max_idx.item() if isinstance(max_idx, torch.Tensor) else max_idx
            # 获取原始gt_boxes中的索引
            gt_indices = mask.nonzero().flatten()
            gt_idx = gt_indices[max_idx].item()
            if max_iou >= iou_thr and not gt_used[sid][gt_idx]:
                tp.append(1)
                fp.append(0)
                gt_used[sid][gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        if len(tp) == 0:
            ap_results[f"class_{cid}"] = 0.0
            continue
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (num_gt + 1e-6)
        precision = tp / np.maximum(tp + fp, 1e-6)
        # VOC style 11-point interpolation
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            prec = precision[recall >= thr]
            ap += prec.max() if prec.size else 0
        ap /= 11
        ap_results[f"class_{cid}"] = float(ap)
    ap_results["mAP"] = float(np.mean([v for v in ap_results.values()]))
    return ap_results


def export_and_eval(_cfg_path: str, checkpoint: str, dataroot: str, iou_thr: float, score_thr: float = 0.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 从配置文件加载配置（而不是从检查点）
    parser = get_parser()
    args = parser.parse_args(['--config-file', _cfg_path])
    cfg = get_cfg(args)
    cfg.DATASET.DATAROOT = dataroot
    cfg.BATCHSIZE = 1
    cfg.GPUS = "[1]" if device.type == "cuda" else "[]"
    
    # 打印配置信息，确认 use_lidar 和 use_event 状态
    print(f"[Config] USE_LIDAR: {cfg.MODEL.MODALITY.USE_LIDAR}")
    print(f"[Config] USE_EVENT: {cfg.MODEL.MODALITY.USE_EVENT}")
    print(f"[Config] USE_CAMERA: {cfg.MODEL.MODALITY.USE_CAMERA}")
    
    # 加载检查点并更新模型配置
    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt.get("hyper_parameters")
    if hparams is None:
        raise KeyError("Checkpoint is missing 'hyper_parameters'; cannot build model.")
    trainer = TrainingModule(hparams)
    trainer.eval().to(device)
    
    # 更新模型的配置（使用配置文件中的配置，而不是检查点中的配置）
    trainer.model.cfg = cfg
    trainer.model.use_lidar = cfg.MODEL.MODALITY.USE_LIDAR
    trainer.model.use_event = cfg.MODEL.MODALITY.USE_EVENT
    trainer.model.use_camera = cfg.MODEL.MODALITY.USE_CAMERA
    
    # 如果启用lidar但模型中没有lidar相关模块，需要重新初始化
    if cfg.MODEL.MODALITY.USE_LIDAR and not hasattr(trainer.model, 'encoders'):
        print("[Warning] Model doesn't have lidar encoders, but USE_LIDAR=True. This may cause errors.")

    _, valloader = prepare_dataloaders(cfg)

    # Warm-up forward to initialize lazy LiDAR modules before re-loading weights.
    val_iter = iter(valloader)
    try:
        first_batch = next(val_iter)
    except StopIteration:
        print("[Warning] Validation loader is empty, nothing to evaluate.")
        return

    first_batch = to_device(first_batch, device)
    with torch.no_grad():
        _ = gather_batch_outputs(trainer.model, first_batch, device, cfg=cfg, decode=False)

    incompatible = trainer.load_state_dict(ckpt.get("state_dict", {}), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[Checkpoint] Missing keys: {len(incompatible.missing_keys)}, "
            f"unexpected keys: {len(incompatible.unexpected_keys)}"
        )

    preds_all = []
    gts_all = {}
    class_names = ["Vehicle", "Cyclist", "Pedestrian"]
    
    # 统计信息
    total_preds = 0
    total_gts = 0
    pred_scores_sum = []
    pred_counts_per_class = {0: 0, 1: 0, 2: 0}  # Vehicle, Cyclist, Pedestrian
    gt_counts_per_class = {0: 0, 1: 0, 2: 0}  # Vehicle, Cyclist, Pedestrian

    def iter_batches():
        yield first_batch
        for batch in val_iter:
            yield to_device(batch, device)

    total_batches = None
    try:
        total_batches = len(valloader)
    except TypeError:
        total_batches = None

    for batch in tqdm(iter_batches(), desc="Exporting predictions", total=total_batches):
        with torch.no_grad():
            preds, gts = gather_batch_outputs(trainer.model, batch, device, cfg=cfg)
        for idx, (pred, gt) in enumerate(zip(preds, gts)):
            # 生成样本ID：使用 sequence_name 和 frame_id（如果存在）
            seq_name = batch.get('sequence_name', ['unknown'])[idx] if 'sequence_name' in batch else f'seq_{idx}'
            frame_id = batch.get('frame_id', [idx])[idx] if 'frame_id' in batch else idx
            sample_id = f"{seq_name}_{frame_id}"
            boxes, scores, labels = pred
            
            # 应用score threshold过滤低质量预测
            if score_thr > 0:
                mask = scores >= score_thr
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]
            
            # 统计信息
            total_preds += len(boxes)
            total_gts += len(gt[0])
            if len(scores) > 0:
                pred_scores_sum.extend(scores.tolist())
            for label in labels.tolist():
                if label in pred_counts_per_class:
                    pred_counts_per_class[label] += 1
            for label in gt[1].tolist():
                if label in gt_counts_per_class:
                    gt_counts_per_class[label] += 1
            
            preds_all.append((sample_id, boxes[:, :7], scores, labels))
            gts_all[sample_id] = (gt[0][:, :7], gt[1])
    
    # 打印统计信息
    print(f"\n=== 数据统计 ===")
    print(f"总预测数: {total_preds}")
    print(f"总GT数: {total_gts}")
    if len(pred_scores_sum) > 0:
        print(f"预测分数范围: [{min(pred_scores_sum):.4f}, {max(pred_scores_sum):.4f}], 平均: {np.mean(pred_scores_sum):.4f}")
    print(f"预测类别分布: Vehicle={pred_counts_per_class[0]}, Cyclist={pred_counts_per_class[1]}, Pedestrian={pred_counts_per_class[2]}")
    print(f"GT类别分布: Vehicle={gt_counts_per_class[0]}, Cyclist={gt_counts_per_class[1]}, Pedestrian={gt_counts_per_class[2]}")
    print(f"总样本数: {len(gts_all)}")
    print("=" * 50)

    ap_results = compute_ap_per_class(preds_all, gts_all, num_classes=len(class_names), iou_thr=iou_thr)
    print(f"mAP@{iou_thr}: {ap_results['mAP']:.4f}")
    for idx, name in enumerate(class_names):
        print(f"{name} AP: {ap_results[f'class_{idx}']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export DSEC detection predictions and compute mAP (IoU-based).")
    parser.add_argument("--config-file", default="streamingflow/configs/dsec_event_lidar.yaml", help="Path to config file. Use dsec_event_lidar.yaml for event+lidar fusion, or dsec_event.yaml for event-only.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.ckpt).")
    parser.add_argument("--dataroot", required=True, help="Path to DSEC dataset root.")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for AP.")
    parser.add_argument("--score-thr", type=float, default=0.0, help="Score threshold for filtering predictions (0.0 means no filtering).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_and_eval(args.config_file, args.checkpoint, args.dataroot, args.iou_thr, args.score_thr)

