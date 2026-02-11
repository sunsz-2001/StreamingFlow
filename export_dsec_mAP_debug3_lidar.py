import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from streamingflow.datas.dataloaders import prepare_dataloaders
from streamingflow.trainer_dsec_debug_lidar import TrainingModule_lidar
from streamingflow.utils.network import preprocess_batch
from streamingflow.config_debug_lidar import get_cfg, get_parser
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d

import open3d as o3d

def to_device(batch: Dict, device: torch.device) -> Dict:
    preprocess_batch(batch, device)
    return batch


def gather_batch_outputs(model, batch, device, cfg=None, decode: bool = True) -> Tuple[List, List, Dict]:
    """Run forward pass and decode predictions."""
    points = batch.get("points")

    output = model(points=points, metas=batch.get("metas"))


    decoded = model.decoder.detection_head.get_bboxes(output["detection"], batch["metas"])
    preds = []
    for layer_result in decoded:
        boxes, scores, labels = layer_result
        preds.append((boxes.tensor.detach().cpu(), scores.detach().cpu(), labels.detach().cpu()))

    gts = []
    # idx = 0
    for gt_boxes, gt_labels in zip(batch["gt_bboxes_3d"], batch["gt_labels_3d"]):
        gts.append((gt_boxes.tensor.cpu(), gt_labels.cpu()))
        # idx+=1
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

def compute_ap_per_class_fixed(preds, gts, num_classes: int, iou_thr: float) -> Dict[str, float]:
    """Compute per-class AP with 3D IoU matching."""
    ap_results = {}
    
    for cid in range(num_classes):
        # Step 1: 收集该类别的所有预测
        flat_preds = []
        for sid, boxes, scores, labels in preds:
            mask = labels == cid
            if mask.sum() == 0:
                continue
            for box, score in zip(boxes[mask][:, :7], scores[mask]):
                flat_preds.append((sid, box, float(score)))
        flat_preds.sort(key=lambda x: x[2], reverse=True)

        # Step 2: 为每个场景初始化当前类别的真实框使用标记
        gt_used = {}
        for sid in gts:
            gt_boxes, gt_labels = gts[sid]
            mask = gt_labels == cid
            num_gt_class = mask.sum().item()
            gt_used[sid] = torch.zeros(num_gt_class, dtype=torch.bool)
        
        # 计算该类别真实框总数
        num_gt = sum((gts[sid][1] == cid).sum().item() for sid in gts)
        if num_gt == 0:
            ap_results[f"class_{cid}"] = 0.0
            continue

        # Step 3: 按置信度遍历预测框
        tp, fp = [], []
        for sid, box, _ in flat_preds:
            gt_boxes, gt_labels = gts.get(sid, (None, None))
            
            # 场景不存在或没有真实框
            if gt_boxes is None:
                fp.append(1)
                tp.append(0)
                continue
            
            # 筛选当前类别的真实框
            mask = gt_labels == cid
            if mask.sum() == 0:
                fp.append(1)
                tp.append(0)
                continue
            
            # 计算与当前类别真实框的IoU
            gt_boxes_class = gt_boxes[mask][:, :7]
            overlaps = bbox_overlaps_3d(box.unsqueeze(0), gt_boxes_class, coordinate="lidar")[0]
            max_iou, max_idx = overlaps.max(0)
            max_iou = max_iou.item() if isinstance(max_iou, torch.Tensor) else float(max_iou)
            max_idx = max_idx.item() if isinstance(max_idx, torch.Tensor) else max_idx
            
            # 检查是否匹配成功
            if max_iou >= iou_thr and not gt_used[sid][max_idx]:
                tp.append(1)
                fp.append(0)
                gt_used[sid][max_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        # Step 4: 计算AP
        if len(tp) == 0:
            ap_results[f"class_{cid}"] = 0.0
            continue
            
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / (num_gt + 1e-6)
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)
        
        # 11点插值法
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            mask = recall >= thr
            if mask.any():
                ap += np.max(precision[mask])
        
        ap_results[f"class_{cid}"] = ap / 11
    
    # Step 5: 计算mAP（排除可能的非class键）
    class_aps = [v for k, v in ap_results.items() if k.startswith('class_')]
    ap_results["mAP"] = float(np.mean(class_aps)) if class_aps else 0.0
    
    return ap_results

def vis_point(pts, preds, gts, win_name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
    pred_boxes = preds[0][0].numpy()
    gt_boxes = gts[0][0].numpy()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1200,height=800)
    opt = vis.get_render_option()
    opt.background_color = [0,0,0]
    opt.point_size = 1.5
    vis.add_geometry(pcd)
    def box7d2obb(box,color):
        if len(box) == 7:
            x,y,z,dx,dy,dz,yaw = box
        if len(box) == 9:
            x,y,z,dx,dy,dz,yaw,_,_ = box
        center = np.array([x,y,z])
        extent = np.array([dx,dy,dz])
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])
        obb = o3d.geometry.OrientedBoundingBox(center,R,extent)
        obb.color = color
        return obb
    
    pred_obbs = []
    if len(pred_boxes) >0:
        for box in pred_boxes:
            obb = box7d2obb(box[:7],[0,1,0])
            pred_obbs.append(obb)
            vis.add_geometry(obb)
    gt_obbs = []
    if len(gt_boxes) >0:
        for box in gt_boxes:
            obb = box7d2obb(box[:7],[1,0,0])
            vis.add_geometry(obb)
            
            gt_obbs.append(obb)
    
    vis.run()
    vis.destroy_window()
    

def save_dsec_visualization(batch, output, preds, gts, idx, save_dir, cfg):
    # event_img     | lidar_img | event_bev_img，
    # lidar_bev_img | pred_img  | gt_img
    points = batch.get("points")[-1][-1]
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    elif isinstance(points, np.ndarray):
        points = points
    else:
        return 
    if True and len(points)!=0:
        vis_point(points,preds,gts, batch['sequence_name'][-1])




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
    trainer = TrainingModule_lidar(hparams)
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
        if visualize and batch_idx % vis_interval == 0 and batch_idx>0:  
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
    # parser.add_argument("--config-file", default="/home/user/sunsz/StreamingFlow/streamingflow/configs/evwaymo_lidar.yaml")
    parser.add_argument("--config-file", default="/home/user/sunsz/StreamingFlow/streamingflow/configs/dsec_lidar.yaml")
    parser.add_argument("--checkpoint", default='/home/user/sunsz/StreamingFlow/logs/dsec_lidar_ep100_2/epoch=99-step=26199.ckpt')
    # parser.add_argument("--checkpoint", default='/home/user/sunsz/StreamingFlow/logs/dsec_lidar_ep100/epoch=99-step=26199.ckpt')
    # parser.add_argument("--dataroot", default='/media/switcher/sda/datasets/evwaymo/')
    parser.add_argument("--dataroot", default='/media/switcher/sda/datasets/dsec/')
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--visualize", default=False, action="store_true", help="Enable visualization")
    parser.add_argument("--vis-interval", type=int, default=30, help="Visualize every N batches")
    parser.add_argument("--vis-save-path", default="dsec_visualize", help="Save directory")
    args = parser.parse_args()
    export_and_eval(args.config_file, args.checkpoint, args.dataroot, args.iou_thr,
                    args.score_thr, args.visualize, args.vis_interval, args.vis_save_path)







