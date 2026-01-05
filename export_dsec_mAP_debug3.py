import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from streamingflow.datas.dataloaders import prepare_dataloaders
from streamingflow.trainer_dsec import TrainingModule
from streamingflow.utils.network import preprocess_batch
from streamingflow.config import get_cfg, get_parser
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d


def to_device(batch: Dict, device: torch.device) -> Dict:
    preprocess_batch(batch, device)
    return batch


def process_flow_data(batch, cfg):
    """Process flow_data from batch."""
    flow_data = []
    flow_batch = batch['flow_data']

    for flow_frame in range(len(flow_batch[0])):
        temp_flow_data = {
            'intrinsics': [], 'extrinsics': [], 'flow_lidar': [],
            'flow_events': [], 'events_stmp': [], 'lidar_stmp': [], 'target_timestamp': [],
        }

        for bs_idx in range(len(flow_batch)):
            tmp = flow_batch[bs_idx][flow_frame]
            for key in temp_flow_data.keys():
                if key == 'flow_events':
                    temp_flow_data[key].append(torch.stack(tmp[key], dim=0))
                else:
                    temp_flow_data[key].append(tmp[key])

        for key in temp_flow_data.keys():
            if type(temp_flow_data[key][0]) in (np.ndarray, list):
                try:
                    temp_flow_data[key] = np.array(temp_flow_data[key])
                except:
                    pass
            elif type(temp_flow_data[key][0]) == torch.Tensor:
                temp_flow_data[key] = torch.stack(temp_flow_data[key], dim=0)

        flow_data.append(temp_flow_data)
    return flow_data


def gather_batch_outputs(model, batch, device, cfg=None, decode: bool = True) -> Tuple[List, List, Dict]:
    """Run forward pass and decode predictions."""
    event = None

    use_lidar = False

    use_event = False

    points, lidar_timestamp, intrinsics, extrinsics, camera_timestamps, target_timestamp = None, None, None, None, None, None

    if cfg is not None:
        use_flow_data = getattr(cfg.DATASET, 'USE_FLOW_DATA', False)

        use_lidar = getattr(cfg.MODEL.MODALITY, 'USE_LIDAR', False)

        use_event = getattr(cfg.MODEL.MODALITY, 'USE_EVENT', False)

        if use_flow_data:
            if 'flow_data' not in batch:
                raise ValueError("USE_FLOW_DATA is True but batch['flow_data'] is missing.")
            flow_data = process_flow_data(batch, cfg)
            intrinsics = flow_data[0]['intrinsics']
            extrinsics = flow_data[0]['extrinsics']
            camera_timestamps = flow_data[0]['events_stmp']
            target_timestamp = flow_data[0]['target_timestamp']
            if use_lidar:
                points = flow_data[0]['flow_lidar']
                lidar_timestamp = flow_data[0]['lidar_stmp']
            event = flow_data[0].get("flow_events")
            if torch.is_tensor(event) and event.dim() == 5:
                event = event.unsqueeze(2)  # [B, S, 1, C, H, W]
        else:
            points = batch.get("points")
            lidar_timestamp = batch.get("lidar_timestamp")
            intrinsics = batch.get("intrinsics")
            extrinsics = batch.get("extrinsics")
            camera_timestamps = batch.get("camera_timestamp")
            target_timestamp = batch.get("target_timestamp")
            event = batch.get("event")
    else:
        points = batch.get("points")
        lidar_timestamp = batch.get("lidar_timestamp")
        intrinsics = batch.get("intrinsics")
        extrinsics = batch.get("extrinsics")
        camera_timestamps = batch.get("camera_timestamp")
        target_timestamp = batch.get("target_timestamp")
        event = batch.get("event")

    if intrinsics is None or extrinsics is None or camera_timestamps is None or target_timestamp is None:
        raise ValueError("Missing required camera metadata (intrinsics/extrinsics/timestamps) for gather_batch_outputs.")
    if use_lidar and (points is None or lidar_timestamp is None):
        raise ValueError("USE_LIDAR is True but points/lidar_timestamp is missing in batch.")
    if use_event and event is None:
        raise ValueError("USE_EVENT is True but event input is missing in batch.")

    def to_device_event(event, device):
        if event is None:
            return None
        if torch.is_tensor(event):
            return event.float().to(device)
        if isinstance(event, dict) and "frames" in event and torch.is_tensor(event["frames"]):
            event = dict(event)
            event["frames"] = event["frames"].float().to(device)
        return event

    def to_device_tensor(data, device):
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(device)
        elif isinstance(data, torch.Tensor):
            return data.float().to(device)
        return data

    intrinsics = to_device_tensor(intrinsics, device)
    extrinsics = to_device_tensor(extrinsics, device)
    camera_timestamps = to_device_tensor(camera_timestamps, device)
    target_timestamp = to_device_tensor(target_timestamp, device)
    lidar_timestamp = to_device_tensor(lidar_timestamp, device)
    event = to_device_event(event, device)

    output = model(
        batch.get("image"), intrinsics, extrinsics, batch.get("future_egomotion"),
        batch.get("padded_voxel_points"), camera_timestamps, points, lidar_timestamp,
        target_timestamp, event=event, metas=batch.get("metas"),
    )

    if not decode:
        return [], [], output

    decoded = model.decoder.detection_head.get_bboxes(output["detection"], batch["metas"])
    preds = []
    for layer_result in decoded:
        boxes, scores, labels = layer_result
        preds.append((boxes.tensor.detach().cpu(), scores.detach().cpu(), labels.detach().cpu()))

    gts = []
    idx = 0
    for gt_boxes, gt_labels in zip(batch["gt_bboxes_3d"], batch["gt_labels_3d"]):
        gts.append((gt_boxes.tensor.cpu(), gt_labels.cpu()))
        idx+=1
        if idx >= batch['gt_len'][0][0]:
            break
        
    return preds, gts, output


def compute_ap_per_class(preds, gts, num_classes: int, iou_thr: float) -> Dict[str, float]:
    """Compute per-class AP with 3D IoU matching."""
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
                fp.append(1); tp.append(0); continue
            mask = gt_labels == cid
            if mask.sum() == 0:
                fp.append(1); tp.append(0); continue
            overlaps = bbox_overlaps_3d(box.unsqueeze(0), gt_boxes[mask][:, :7], coordinate="lidar")[0]
            max_iou, max_idx = overlaps.max(0)
            max_iou = max_iou.item() if isinstance(max_iou, torch.Tensor) else float(max_iou)
            max_idx = max_idx.item() if isinstance(max_idx, torch.Tensor) else max_idx
            gt_indices = mask.nonzero().flatten()
            gt_idx = gt_indices[max_idx].item()
            if max_iou >= iou_thr and not gt_used[sid][gt_idx]:
                tp.append(1); fp.append(0); gt_used[sid][gt_idx] = True
            else:
                tp.append(0); fp.append(1)

        if len(tp) == 0:
            ap_results[f"class_{cid}"] = 0.0
            continue
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall = tp / (num_gt + 1e-6)
        precision = tp / np.maximum(tp + fp, 1e-6)
        ap = sum(precision[recall >= thr].max() if precision[recall >= thr].size else 0 for thr in np.linspace(0, 1, 11)) / 11
        ap_results[f"class_{cid}"] = float(ap)
    ap_results["mAP"] = float(np.mean([v for v in ap_results.values()]))
    return ap_results


def save_dsec_visualization(batch, output, preds, gts, idx, save_dir, cfg):
    # event_img     | lidar_img | event_bev_img，
    # lidar_bev_img | pred_img  | gt_img
    from streamingflow.utils.visualisation import (
        plot_event_frame, plot_lidar_bev, plot_boxes_bev, plot_bev_feature
    )

    os.makedirs(save_dir, exist_ok=True)
    bev_range = (cfg.LIFT.X_BOUND[0], cfg.LIFT.X_BOUND[1],
                 cfg.LIFT.Y_BOUND[0], cfg.LIFT.Y_BOUND[1])
    resolution = cfg.LIFT.X_BOUND[2]

    # 提取数据
    flow_data = batch['flow_data'][0][0]

    def _last_item(value):
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return _last_item(value[-1])
        return value

    events = _last_item(flow_data.get('flow_events'))
    if torch.is_tensor(events):
        events = events.cpu().numpy()
    elif isinstance(events, np.ndarray):
        events = events
    else:
        raise TypeError(f"Unexpected flow_events type: {type(events)}")

    points = _last_item(flow_data.get('flow_lidar'))
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    elif isinstance(points, np.ndarray):
        points = points
    else:
        raise TypeError(f"Unexpected flow_lidar type: {type(points)}")

    event_img = plot_event_frame(events)
    lidar_img = plot_lidar_bev(points, bev_range, resolution)
    event_bev_img = plot_bev_feature(output['event_bev'][0, -1].cpu().numpy())
    lidar_bev_img = plot_bev_feature(output['lidar_states'][0, -1].cpu().numpy())

    pred_boxes = preds[0][0].numpy()
    gt_boxes = gts[0][0].numpy()
    pred_img = plot_boxes_bev(pred_boxes, bev_range, resolution, (0, 255, 0))
    gt_img = plot_boxes_bev(gt_boxes, bev_range, resolution, (255, 0, 0))

    target_h, target_w = event_img.shape[:2]
    lidar_img = cv2.resize(lidar_img, (target_w, target_h))
    event_bev_img = cv2.resize(event_bev_img, (target_w, target_h))
    lidar_bev_img = cv2.resize(lidar_bev_img, (target_w, target_h))
    pred_img = cv2.resize(pred_img, (target_w, target_h))
    gt_img = cv2.resize(gt_img, (target_w, target_h))

    row1 = np.concatenate([event_img, lidar_img, event_bev_img], axis=1)
    row2 = np.concatenate([lidar_bev_img, pred_img, gt_img], axis=1)
    result = np.concatenate([row1, row2], axis=0)

    cv2.imwrite(f"{save_dir}/{idx:04d}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def export_and_eval(_cfg_path: str, checkpoint: str, dataroot: str, iou_thr: float,
                    score_thr: float = 0.0, visualize: bool = False,
                    vis_interval: int = 10, vis_save_path: str = "dsec_visualize"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = get_parser()
    args = parser.parse_args(['--config-file', _cfg_path])
    cfg = get_cfg(args)
    cfg.DATASET.DATAROOT = dataroot
    cfg.BATCHSIZE = 1
    cfg.GPUS = "[1]" if device.type == "cuda" else "[]"

    print(f"[Config] USE_LIDAR: {cfg.MODEL.MODALITY.USE_LIDAR}, USE_EVENT: {cfg.MODEL.MODALITY.USE_EVENT}, USE_CAMERA: {cfg.MODEL.MODALITY.USE_CAMERA}")

    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt.get("hyper_parameters")
    if hparams is None:
        raise KeyError("Checkpoint is missing 'hyper_parameters'")
    trainer = TrainingModule(hparams)
    trainer.model.cfg = cfg
    trainer.model.use_lidar = cfg.MODEL.MODALITY.USE_LIDAR
    trainer.model.use_event = cfg.MODEL.MODALITY.USE_EVENT
    trainer.model.use_camera = cfg.MODEL.MODALITY.USE_CAMERA

    _, valloader = prepare_dataloaders(cfg)

    # 加载训练好的模型权重
    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint is missing 'state_dict'")

    needs_lidar_init = any(k.startswith("model.temporal_model_lidar") for k in state_dict)
    if needs_lidar_init and trainer.model.temporal_model_lidar is None:
        trainer.eval().to(device)
        init_batch = next(iter(valloader))
        init_batch = to_device(init_batch, device)
        with torch.no_grad():
            gather_batch_outputs(trainer.model, init_batch, device, cfg=cfg, decode=False)

    trainer.load_state_dict(state_dict)
    trainer.eval().to(device)

    preds_all, gts_all = [], {}
    class_names = ["Vehicle", "Cyclist", "Pedestrian"]
    total_preds, total_gts = 0, 0
    pred_counts = {0: 0, 1: 0, 2: 0}
    gt_counts = {0: 0, 1: 0, 2: 0}

    for batch_idx, batch in enumerate(tqdm(valloader, desc="Evaluating")):
        batch = to_device(batch, device)
        with torch.no_grad():
            preds, gts, output = gather_batch_outputs(trainer.model, batch, device, cfg=cfg)

        vis_preds = preds
        if score_thr > 0:
            vis_preds = []
            for boxes, scores, labels in preds:
                mask = scores >= score_thr
                vis_preds.append((boxes[mask], scores[mask], labels[mask]))
        if visualize and batch_idx % vis_interval == 0:
            save_dsec_visualization(batch, output, vis_preds, gts, batch_idx, vis_save_path, cfg)

        for idx, (pred, gt) in enumerate(zip(preds, gts)):
            seq_name = batch.get('sequence_name', ['unknown'])[idx] if 'sequence_name' in batch else f'seq_{idx}'
            frame_id = batch.get('frame_id', [idx])[idx] if 'frame_id' in batch else idx
            sample_id = f"{seq_name}_{frame_id}"
            boxes, scores, labels = pred

            if score_thr > 0:
                mask = scores >= score_thr
                boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

            total_preds += len(boxes)
            total_gts += len(gt[0])
            for label in labels.tolist():
                if label in pred_counts: pred_counts[label] += 1
            for label in gt[1].tolist():
                if label in gt_counts: gt_counts[label] += 1

            preds_all.append((sample_id, boxes[:, :7], scores, labels))
            gts_all[sample_id] = (gt[0][:, :7], gt[1])

    print(f"\n=== Statistics ===")
    print(f"Predictions: {total_preds}, GT: {total_gts}")
    print(f"Pred dist: Vehicle={pred_counts[0]}, Cyclist={pred_counts[1]}, Pedestrian={pred_counts[2]}")
    print(f"GT dist: Vehicle={gt_counts[0]}, Cyclist={gt_counts[1]}, Pedestrian={gt_counts[2]}")

    ap_results = compute_ap_per_class(preds_all, gts_all, num_classes=len(class_names), iou_thr=iou_thr)
    print(f"\nmAP@{iou_thr}: {ap_results['mAP']:.4f}")
    for idx, name in enumerate(class_names):
        print(f"{name} AP: {ap_results[f'class_{idx}']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSEC mAP evaluation")
    parser.add_argument("--config-file", default="/home/user/sunsz/StreamingFlow/streamingflow/configs/dsec_event_lidar.yaml")
    parser.add_argument("--checkpoint", default='/home/user/sunsz/StreamingFlow/logs/dsec_event_lidar_eval/epoch=34-step=18374.ckpt')
    parser.add_argument("--dataroot", default='/media/switcher/sda/datasets/dsec/')
    parser.add_argument("--iou-thr", type=float, default=0.1)
    parser.add_argument("--score-thr", type=float, default=0.01)
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--vis-interval", type=int, default=10, help="Visualize every N batches")
    parser.add_argument("--vis-save-path", default="dsec_visualize", help="Save directory")
    args = parser.parse_args()
    export_and_eval(args.config_file, args.checkpoint, args.dataroot, args.iou_thr,
                    args.score_thr, args.visualize, args.vis_interval, args.vis_save_path)







