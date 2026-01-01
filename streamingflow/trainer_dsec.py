import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from streamingflow.config import get_cfg
from streamingflow.models.streamingflow import streamingflow
from streamingflow.losses import SpatialRegressionLoss, SegmentationLoss, HDmapLoss, DepthLoss
from streamingflow.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from streamingflow.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features
from streamingflow.utils.instance import predict_instance_segmentation_and_trajectories
from streamingflow.utils.visualisation import visualise_output
from streamingflow.utils.data_utils import voxelize_occupy


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # capture hyperparameters in a lightning-compatible way
        self.save_hyperparameters(hparams)

        # pytorch lightning does not support saving YACS CfgNone
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg
        self.n_classes = len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
        self.hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        if cfg.DATASET.NAME == 'lyft':
            self.is_lyft = True
        else:
            self.is_lyft = False
        # Model
        self.model = streamingflow(cfg)

        self.losses_fn = nn.ModuleDict()

        # Semantic segmentation
        self.losses_fn['segmentation'] = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
            use_top_k=self.cfg.SEMANTIC_SEG.VEHICLE.USE_TOP_K,
            top_k_ratio=self.cfg.SEMANTIC_SEG.VEHICLE.TOP_K_RATIO,
            future_discount=self.cfg.FUTURE_DISCOUNT,
        )
        self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.metric_vehicle_val = IntersectionOverUnion(self.n_classes)

        # Pedestrian segmentation
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            self.losses_fn['pedestrian'] = SegmentationLoss(
                class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.PEDESTRIAN.WEIGHTS),
                use_top_k=self.cfg.SEMANTIC_SEG.PEDESTRIAN.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.PEDESTRIAN.TOP_K_RATIO,
                future_discount=self.cfg.FUTURE_DISCOUNT,
            )
            self.model.pedestrian_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_pedestrian_val = IntersectionOverUnion(self.n_classes)

        # HD map
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            self.losses_fn['hdmap'] = HDmapLoss(
                class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.HDMAP.WEIGHTS),
                training_weights=self.cfg.SEMANTIC_SEG.HDMAP.TRAIN_WEIGHT,
                use_top_k=self.cfg.SEMANTIC_SEG.HDMAP.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.HDMAP.TOP_K_RATIO,
            )
            self.metric_hdmap_val = []
            for i in range(len(self.hdmap_class)):
                self.metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1))
            self.model.hdmap_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_hdmap_val = nn.ModuleList(self.metric_hdmap_val)

        # Depth
        if self.cfg.LIFT.GT_DEPTH:
            self.losses_fn['depths'] = DepthLoss()
            self.model.depths_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Instance segmentation
        if self.cfg.INSTANCE_SEG.ENABLED:
            self.losses_fn['instance_center'] = SpatialRegressionLoss(
                norm=2, future_discount=self.cfg.FUTURE_DISCOUNT
            )
            self.losses_fn['instance_offset'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            self.model.centerness_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.model.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        # Instance flow
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.losses_fn['instance_flow'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            self.model.flow_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Planning
        if self.cfg.PLANNING.ENABLED:
            self.metric_planning_val = PlanningMetric(self.cfg, self.cfg.N_FUTURE_FRAMES)
            self.model.planning_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Detection
        if getattr(self.cfg, 'DETECTION', None) and getattr(self.cfg.DETECTION, 'ENABLED', False):
            self.model.detection_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.training_step_count = 0

    def shared_step(self, batch, is_train):
        image = batch.get('image')
        flow_data = []
        flow_batch = batch['flow_data']
        for flow_frame in range(len(flow_batch[0])):
            temp_flow_data = {
            'intrinsics':[],
            'extrinsics':[],
            'flow_lidar':[],
            'flow_events':[],
            'events_stmp':[],
            'lidar_stmp':[],
            'target_timestamp':[],
            }
            for bs_idx in range(len(flow_batch)):
                tmp = flow_batch[bs_idx][flow_frame]
                
                for key in temp_flow_data.keys():
                    if key =='flow_events':
                        tmp_evs = torch.stack(tmp[key], dim=0)
                        temp_flow_data[key].append(tmp_evs)
                    else:
                        temp_flow_data[key].append(tmp[key])
            for key in temp_flow_data.keys():
                
                if type(temp_flow_data[key][0]) in (np.ndarray,list):
                    try:temp_flow_data[key] = np.array(temp_flow_data[key])
                    except:pass
                elif type(temp_flow_data[key][0]) == torch.Tensor:
                    temp_flow_data[key] = torch.stack(temp_flow_data[key], dim=0)
            flow_data.append(temp_flow_data)
        
        intrinsics = flow_data[0]['intrinsics']
        extrinsics = flow_data[0]['extrinsics']
        camera_timestamps = flow_data[0]['events_stmp']
        target_timestamp = flow_data[0]['target_timestamp']

        future_egomotion = batch['future_egomotion']
        # 只在规划任务启用且非lyft数据集时访问这些字段
        if not self.is_lyft and self.cfg.PLANNING.ENABLED:
            command = batch.get('command', 'FORWARD')
            trajs = batch.get('sample_trajectory', None)
            target_points = batch.get('target_point', None)
        else:
            command = None
            trajs = None
            target_points = None
        range_clouds = None
        radar_pointclouds = None  
        padded_voxel_points = None
        lidar_timestamps = None
        points = None
        if self.cfg.MODEL.MODALITY.USE_RADAR:
            radar_pointclouds = batch['radar_pointclouds']
        if self.cfg.MODEL.MODALITY.USE_LIDAR:
            points = flow_data[0]['flow_lidar']
            lidar_timestamps = flow_data[0]['lidar_stmp']
                
        
        # 获取 batch size：优先从 image，否则从 event 或其他可用字段
        if image is not None:
            B = len(image)
        elif self.cfg.MODEL.MODALITY.USE_EVENT and batch.get('event') is not None:
            event = flow_data[0]['flow_events']
            if torch.is_tensor(event) and event.dim() == 5:
                event = event.unsqueeze(2)  # [B, S, 1, C, H, W]

            if torch.is_tensor(event):
                B = event.shape[0]
            elif isinstance(event, dict) and 'frames' in event:
                B = event['frames'].shape[0]
            else:
                B = intrinsics[0].shape[0]
        else:
            B = intrinsics.shape[0]

        # 提取检测标签（如果启用检测）
        gt_bboxes_3d = None
        gt_labels_3d = None
        metas = None
        if getattr(self.cfg, 'DETECTION', None) and getattr(self.cfg.DETECTION, 'ENABLED', False):
            gt_bboxes_3d = batch.get('gt_bboxes_3d')
            gt_labels_3d = batch.get('gt_labels_3d')
            metas = batch.get('metas')
            if gt_bboxes_3d is None or gt_labels_3d is None:
                raise ValueError("DETECTION.ENABLED is True but gt_bboxes_3d or gt_labels_3d is missing in batch")
            if metas is None:
                raise ValueError("DETECTION.ENABLED is True but metas is missing in batch")
            
            # 确保gt_labels_3d是torch.Tensor格式并在正确的device上
            if gt_labels_3d is not None:
                device = next(self.model.parameters()).device
                converted_labels = []
                for v in gt_labels_3d:
                    if v is None:
                        converted_labels.append(torch.tensor([], dtype=torch.long, device=device))
                    elif isinstance(v, np.ndarray):
                        converted_labels.append(torch.from_numpy(v).long().to(device))
                    elif isinstance(v, torch.Tensor):
                        converted_labels.append(v.long().to(device))
                    else:
                        converted_labels.append(torch.tensor(v, dtype=torch.long, device=device))
                gt_labels_3d = converted_labels

        # Warp labels (仅用于分割任务)
        labels = self.prepare_future_labels(batch) if not (getattr(self.cfg, 'DETECTION', None) and getattr(self.cfg.DETECTION, 'ENABLED', False)) else {}

        # Forward pass
        # event = batch['event'] if self.cfg.MODEL.MODALITY.USE_EVENT else None

        output = self.model(
            image, intrinsics, extrinsics, future_egomotion, padded_voxel_points,camera_timestamps, points,lidar_timestamps,target_timestamp,
            image_hi=batch.get('image_hi'), intrinsics_hi=batch.get('intrinsics_hi'), extrinsics_hi=batch.get('extrinsics_hi'),
            camera_timestamp_hi=batch.get('camera_timestamp_hi'), event=event, metas=metas
        )

        #####
        # Loss computation
        #####
        loss = {}

        if is_train:
            # segmentation (only if not using detection task)
            if 'segmentation' in output and 'segmentation' in labels:
                segmentation_factor = 1 / (2 * torch.exp(self.model.segmentation_weight))
                loss['segmentation'] = segmentation_factor * self.losses_fn['segmentation'](
                    output['segmentation'], labels['segmentation'], self.model.receptive_field
                )
                loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight

            # Pedestrian (only if segmentation is enabled)
            if 'pedestrian' in output and 'pedestrian' in labels and self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                pedestrian_factor = 1 / (2 * torch.exp(self.model.pedestrian_weight))
                loss['pedestrian'] = pedestrian_factor * self.losses_fn['pedestrian'](
                    output['pedestrian'], labels['pedestrian'], self.model.receptive_field
                )
                loss['pedestrian_uncertainty'] = 0.5 * self.model.pedestrian_weight

            # hdmap loss (only if segmentation is enabled)
            if 'hdmap' in output and 'hdmap' in labels and self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                hdmap_factor = 1 / (2 * torch.exp(self.model.hdmap_weight))
                loss['hdmap'] = hdmap_factor * self.losses_fn['hdmap'](output['hdmap'], labels['hdmap'])
                loss['hdmap_uncertainty'] = 0.5 * self.model.hdmap_weight

            if self.cfg.INSTANCE_SEG.ENABLED:
                # instance center
                centerness_factor = 1 / (2 * torch.exp(self.model.centerness_weight))
                loss['instance_center'] = centerness_factor * self.losses_fn['instance_center'](
                    output['instance_center'], labels['centerness'], self.model.receptive_field
                )
                loss['centerness_uncertainty'] = 0.5 * self.model.centerness_weight

                # instance offset
                offset_factor = 1 / (2 * torch.exp(self.model.offset_weight))
                loss['instance_offset'] = offset_factor * self.losses_fn['instance_offset'](
                    output['instance_offset'], labels['offset'], self.model.receptive_field
                )
                loss['offset_uncertainty'] = 0.5 * self.model.offset_weight

            # depth loss
            if self.cfg.LIFT.GT_DEPTH:
              
                depths_factor = 1 / (2 * torch.exp(self.model.depths_weight))
                loss['depths'] = depths_factor * self.losses_fn['depths'](output['depth_prediction'], labels['depths'])
                loss['depths_uncertainty'] = 0.5 * self.model.depths_weight

            # instance flow
            if self.cfg.INSTANCE_FLOW.ENABLED:
                flow_factor = 1 / (2 * torch.exp(self.model.flow_weight))
                loss['instance_flow'] = flow_factor * self.losses_fn['instance_flow'](
                    output['instance_flow'], labels['flow'], self.model.receptive_field
                )
                loss['flow_uncertainty'] = 0.5 * self.model.flow_weight

            # Detection loss
            if getattr(self.cfg, 'DETECTION', None) and getattr(self.cfg.DETECTION, 'ENABLED', False):
                detection_losses = self.model.decoder.detection_head.loss(
                    gt_bboxes_3d, gt_labels_3d, output['detection']
                )
                # Check for NaN in detection losses
                for key, val in detection_losses.items():
                    if torch.isnan(val).any() or torch.isinf(val).any():
                        print(f"Warning: NaN/Inf detected in detection_losses['{key}']: {val}")
                        # Replace NaN/Inf with 0 to prevent propagation
                        detection_losses[key] = torch.where(
                            torch.isnan(val) | torch.isinf(val),
                            torch.zeros_like(val),
                            val
                        )
                
                # Clamp detection_weight to prevent exp overflow
                detection_weight_clamped = torch.clamp(self.model.detection_weight, min=-10.0, max=10.0)
                detection_factor = 1 / (2 * torch.exp(detection_weight_clamped))
                for key, val in detection_losses.items():
                    loss[f'detection_{key}'] = detection_factor * val
                loss['detection_uncertainty'] = 0.5 * self.model.detection_weight

            # Planning
            if self.cfg.PLANNING.ENABLED:
                receptive_field = self.model.receptive_field
                planning_factor = 1 / (2 * torch.exp(self.model.planning_weight))
                occupancy = torch.logical_or(labels['segmentation'][:, receptive_field:].squeeze(2),
                                             labels['pedestrian'][:, receptive_field:].squeeze(2))
                pl_loss, final_traj = self.model.planning(
                    cam_front=output['cam_front'].detach(),
                    trajs=trajs[:, :, 1:],
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    cost_volume=output['costvolume'][:, receptive_field:],
                    semantic_pred=occupancy,
                    hd_map=labels['hdmap'],
                    commands=command,
                    target_points=target_points
                )
                loss['planning'] = planning_factor * pl_loss
                loss['planning_uncertainty'] = 0.5 * self.model.planning_weight
                output = {**output, 'selected_traj': torch.cat(
                    [torch.zeros((B, 1, 3), device=final_traj.device), final_traj], dim=1)}
            else:
                if 'gt_trajectory' in labels:
                    output = {**output, 'selected_traj': labels['gt_trajectory']}

        # Metrics
        else:
            n_present = self.model.receptive_field
            
            # 检查是否使用检测模式
            is_detection_mode = getattr(self.cfg, 'DETECTION', None) and getattr(self.cfg.DETECTION, 'ENABLED', False)

            # semantic segmentation metric (only if not using detection task)
            seg_prediction = None
            pedestrian_prediction = None
            if not is_detection_mode and 'segmentation' in output and 'segmentation' in labels:
                seg_prediction = output['segmentation'].detach()
                seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
                self.metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

            # pedestrian segmentation metric (only if not using detection task and segmentation is enabled)
            if not is_detection_mode:
                if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED and 'pedestrian' in output and 'pedestrian' in labels:
                    pedestrian_prediction = output['pedestrian'].detach()
                    pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
                    self.metric_pedestrian_val(pedestrian_prediction[:, n_present - 1:],
                                               labels['pedestrian'][:, n_present - 1:])
                elif seg_prediction is not None:
                    pedestrian_prediction = torch.zeros_like(seg_prediction)

            # hdmap metric (only if not using detection task)
            if not is_detection_mode and 'segmentation' in output and 'segmentation' in labels:
                if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                    for i in range(len(self.hdmap_class)):
                        hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                        hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                        self.metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

            # instance segmentation metric (only if not using detection task)
            if not is_detection_mode and self.cfg.INSTANCE_SEG.ENABLED:
                pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                    output, compute_matched_centers=False
                )
                self.metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
                                         labels['instance'][:, n_present - 1:])

            # planning metric (only if not using detection task)
            if not is_detection_mode and self.cfg.PLANNING.ENABLED and seg_prediction is not None and pedestrian_prediction is not None:
                occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
                _, final_traj = self.model.planning(
                    cam_front=output['cam_front'].detach(),
                    trajs=trajs[:, :, 1:],
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    cost_volume=output['costvolume'][:, n_present:].detach(),
                    semantic_pred=occupancy[:, n_present:].squeeze(2),
                    hd_map=output['hdmap'].detach(),
                    commands=command,
                    target_points=target_points
                )
                occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                            labels['pedestrian'][:, n_present:].squeeze(2))
                self.metric_planning_val(final_traj, labels['gt_trajectory'][:, 1:], occupancy)
                output = {**output,
                        'selected_traj': torch.cat([torch.zeros((B, 1, 3), device=final_traj.device), final_traj],
                                                    dim=1)}
            else:
                if 'gt_trajectory' in labels:
                    output = {**output, 'selected_trajectory': labels['gt_trajectory']}

        return output, labels, loss

    def prepare_future_labels(self, batch):
        labels = {}

        segmentation_labels = batch['segmentation']
        # hdmap_labels = batch['hdmap']
        future_egomotion = batch['future_egomotion']
        gt_trajectory = batch['gt_trajectory']
        
        if not self.is_lyft:
            # present frame hd map gt
            # labels['hdmap'] = hdmap_labels[:, self.model.receptive_field - 1].long().contiguous()

            # gt trajectory
            labels['gt_trajectory'] = gt_trajectory

        # Past frames gt depth
        if self.cfg.LIFT.GT_DEPTH:
            depths = batch['depths']
            depth_labels = depths[:, :self.model.receptive_field, :, ::self.model.encoder_downsample,
                           ::self.model.encoder_downsample]
            depth_labels = torch.clamp(depth_labels, self.cfg.LIFT.D_BOUND[0], self.cfg.LIFT.D_BOUND[1] - 1) - \
                           self.cfg.LIFT.D_BOUND[0]
            depth_labels = depth_labels.long().contiguous()
            labels['depths'] = depth_labels

        # Warp labels to present's reference frame
        segmentation_labels_past = cumulative_warp_features(
            segmentation_labels[:, :self.model.receptive_field].float(),
            future_egomotion[:, :self.model.receptive_field],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()[:, :-1]


        segmentation_labels = cumulative_warp_features_reverse(
            segmentation_labels[:, (self.model.receptive_field - 1):].float(),
            future_egomotion[:, (self.model.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()
        
        
        labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)

        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_labels = batch['pedestrian']
            pedestrian_labels_past = cumulative_warp_features(
                pedestrian_labels[:, :self.model.receptive_field].float(),
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :-1]
            pedestrian_labels = cumulative_warp_features_reverse(
                pedestrian_labels[:, (self.model.receptive_field - 1):].float(),
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()
            labels['pedestrian'] = torch.cat([pedestrian_labels_past, pedestrian_labels], dim=1)

        # Warp instance labels to present's reference frame
        if self.cfg.INSTANCE_SEG.ENABLED:
            gt_instance = batch['instance']
            instance_center_labels = batch['centerness']
            instance_offset_labels = batch['offset']
            gt_instance_past = cumulative_warp_features(
                gt_instance[:, :self.model.receptive_field].float().unsqueeze(2),
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :-1, 0]
            gt_instance = cumulative_warp_features_reverse(
                gt_instance[:, (self.model.receptive_field - 1):].float().unsqueeze(2),
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :, 0]
            labels['instance'] = torch.cat([gt_instance_past, gt_instance], dim=1)

            instance_center_labels_past = cumulative_warp_features(
                instance_center_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_center_labels = cumulative_warp_features_reverse(
                instance_center_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['centerness'] = torch.cat([instance_center_labels_past, instance_center_labels], dim=1)

            instance_offset_labels_past = cumulative_warp_features(
                instance_offset_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_offset_labels = cumulative_warp_features_reverse(
                instance_offset_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['offset'] = torch.cat([instance_offset_labels_past, instance_offset_labels], dim=1)

        if self.cfg.INSTANCE_FLOW.ENABLED:
            instance_flow_labels = batch['flow']
            instance_flow_labels_past = cumulative_warp_features(
                instance_flow_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_flow_labels = cumulative_warp_features_reverse(
                instance_flow_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['flow'] = torch.cat([instance_flow_labels_past, instance_flow_labels], dim=1)

        return labels

    def visualise(self, labels, output, batch_idx, prefix='train'):
        # 如果是检测任务，跳过可视化（因为visualise_output只支持分割任务）
        if getattr(self.cfg, 'DETECTION', None) and getattr(self.cfg.DETECTION, 'ENABLED', False):
            return  # TODO: 未来可以实现检测任务的可视化
        
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

    def training_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1
        
        # DIAGNOSIS: 保存 loss 信息供 backward 后检查
        if not hasattr(self, '_first_step_loss_saved'):
            self._first_step_loss_saved = True
            self._last_loss_dict = loss.copy()
        
        for key, value in loss.items():
            self.logger.experiment.add_scalar('step_train_loss_' + key, value, global_step=self.training_step_count)
        if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
            self.visualise(labels, output, batch_idx, prefix='train')

        total_loss = sum(loss.values())
        return total_loss

    def on_after_backward(self):
        """DIAGNOSIS: 检查 backward 后的梯度状态，定位 NaN 来源"""
        if not hasattr(self, '_first_backward_checked'):
            self._first_backward_checked = True
            print("=" * 80)
            print("DIAGNOSIS: First training step - checking gradients AFTER backward...")
            print("=" * 80)
            
            # 显示 loss 信息
            if hasattr(self, '_last_loss_dict'):
                print("  Loss components:")
                for key, val in self._last_loss_dict.items():
                    if torch.is_tensor(val):
                        val_item = val.item() if val.numel() == 1 else f"shape={val.shape}"
                        print(f"    - {key}: {val_item}")
            
            # 检查所有模块的梯度
            nan_grad_modules = {}
            total_grad_norm = 0.0
            param_count = 0
            max_grad_norm = 0.0
            max_grad_param_name = None
            
            for module_name, module in self.model.named_modules():
                if module_name == '':
                    continue
                module_nan_count = 0
                module_param_count = 0
                module_grad_norm = 0.0
                
                for name, param in module.named_parameters(recurse=False):
                    if param.grad is not None:
                        module_param_count += 1
                        param_count += 1
                        param_grad_norm = param.grad.norm().item()
                        module_grad_norm += param_grad_norm
                        total_grad_norm += param_grad_norm
                        
                        if param_grad_norm > max_grad_norm:
                            max_grad_norm = param_grad_norm
                            max_grad_param_name = f"{module_name}.{name}"
                        
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            module_nan_count += 1
                            if module_name not in nan_grad_modules:
                                nan_grad_modules[module_name] = []
                            try:
                                grad_min = param.grad.min().item()
                                grad_max = param.grad.max().item()
                                grad_mean = param.grad.mean().item()
                            except:
                                grad_min = grad_max = grad_mean = float('nan')
                            nan_grad_modules[module_name].append({
                                'name': name,
                                'has_nan': torch.isnan(param.grad).any().item(),
                                'has_inf': torch.isinf(param.grad).any().item(),
                                'grad_norm': param_grad_norm,
                                'grad_stats': f"min={grad_min:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}"
                            })
                
                if module_nan_count > 0:
                    print(f"\n  ERROR: Module '{module_name}' has {module_nan_count}/{module_param_count} parameters with NaN/Inf gradients")
                    for param_info in nan_grad_modules[module_name][:3]:  # 只显示前3个
                        print(f"    - {param_info['name']}: NaN={param_info['has_nan']}, Inf={param_info['has_inf']}, norm={param_info['grad_norm']:.6f}")
                        print(f"      {param_info['grad_stats']}")
            
            print(f"\n  Total parameters with gradients: {param_count}")
            print(f"  Total gradient norm: {total_grad_norm:.6f}")
            print(f"  Max gradient norm: {max_grad_norm:.6f} (parameter: {max_grad_param_name})")
            print(f"  Modules with NaN/Inf gradients: {len(nan_grad_modules)}")
            
            if len(nan_grad_modules) > 0:
                print(f"\n  ERROR: Found NaN/Inf gradients in {len(nan_grad_modules)} modules:")
                for module_name in list(nan_grad_modules.keys())[:10]:  # 只显示前10个模块
                    print(f"    - {module_name}: {len(nan_grad_modules[module_name])} parameters")
                
                # 尝试推断是哪个 loss 项导致的
                print(f"\n  Potential loss sources:")
                if any('detection' in name or 'head' in name or 'decoder' in name for name in nan_grad_modules.keys()):
                    print(f"    - Detection loss components (detection_loss_heatmap, detection_layer_*_loss_cls, detection_layer_*_loss_bbox)")
                if any('event_encoder' in name or 'backbone' in name for name in nan_grad_modules.keys()):
                    print(f"    - Event encoder/backbone (affected by all loss components)")
            else:
                print(f"  ✓ All gradients are valid")
            
            # 检查梯度裁剪配置
            if hasattr(self.trainer, 'gradient_clip_val') and self.trainer.gradient_clip_val is not None:
                print(f"\n  Gradient clipping enabled: max_norm={self.trainer.gradient_clip_val}")
                if total_grad_norm > self.trainer.gradient_clip_val:
                    print(f"  WARNING: Total gradient norm ({total_grad_norm:.6f}) exceeds clip value!")
            else:
                print(f"\n  Gradient clipping: NOT ENABLED")
            
            print("=" * 80)
    

    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        scores = self.metric_vehicle_val.compute()
        self.log('step_val_seg_iou_dynamic', scores[1])
        if self.cfg.PLANNING.ENABLED:
            self.log('step_predicted_traj_x', output['selected_traj'][0, -1, 0])
            self.log('step_target_traj_x', labels['gt_trajectory'][0, -1, 0])
            self.log('step_predicted_traj_y', output['selected_traj'][0, -1, 1])
            self.log('step_target_traj_y', labels['gt_trajectory'][0, -1, 1])

        if batch_idx == 0:
            self.visualise(labels, output, batch_idx, prefix='val')

    def shared_epoch_end(self, step_outputs, is_train):
        if not is_train:
            scores = self.metric_vehicle_val.compute()
            print(scores)
            self.logger.experiment.add_scalar('epoch_val_all_seg_iou_dynamic', scores[1],
                                              global_step=self.training_step_count)
            self.metric_vehicle_val.reset()

            if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                scores = self.metric_pedestrian_val.compute()
                self.logger.experiment.add_scalar('epoch_val_all_seg_iou_pedestrian', scores[1],
                                                  global_step=self.training_step_count)
                self.metric_pedestrian_val.reset()

            if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                for i, name in enumerate(self.hdmap_class):
                    scores = self.metric_hdmap_val[i].compute()
                    self.logger.experiment.add_scalar('epoch_val_hdmap_iou_' + name, scores[1],
                                                      global_step=self.training_step_count)
                    self.metric_hdmap_val[i].reset()

            if self.cfg.INSTANCE_SEG.ENABLED:
                scores = self.metric_panoptic_val.compute()
                print(scores)
                for key, value in scores.items():
                    self.logger.experiment.add_scalar(f'epoch_val_all_ins_{key}_vehicle', value[1].item(),
                                                      global_step=self.training_step_count)
                self.metric_panoptic_val.reset()

            if self.cfg.PLANNING.ENABLED:
                scores = self.metric_planning_val.compute()
                for key, value in scores.items():
                    self.logger.experiment.add_scalar('epoch_val_plan_' + key, value.mean(),
                                                      global_step=self.training_step_count)
                self.metric_planning_val.reset()

        self.logger.experiment.add_scalar('epoch_segmentation_weight',
                                          1 / (2 * torch.exp(self.model.segmentation_weight)),
                                          global_step=self.training_step_count)
        if self.cfg.LIFT.GT_DEPTH:
            self.logger.experiment.add_scalar('epoch_depths_weight', 1 / (2 * torch.exp(self.model.depths_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            self.logger.experiment.add_scalar('epoch_pedestrian_weight',
                                              1 / (2 * torch.exp(self.model.pedestrian_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            self.logger.experiment.add_scalar('epoch_hdmap_weight', 1 / (2 * torch.exp(self.model.hdmap_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.INSTANCE_SEG.ENABLED:
            self.logger.experiment.add_scalar('epoch_centerness_weight',
                                              1 / (2 * torch.exp(self.model.centerness_weight)),
                                              global_step=self.training_step_count)
            self.logger.experiment.add_scalar('epoch_offset_weight', 1 / (2 * torch.exp(self.model.offset_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.logger.experiment.add_scalar('epoch_flow_weight', 1 / (2 * torch.exp(self.model.flow_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.PLANNING.ENABLED:
            self.logger.experiment.add_scalar('epoch_planning_weight', 1 / (2 * torch.exp(self.model.planning_weight)),
                                              global_step=self.training_step_count)

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, True)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, False)

    def configure_optimizers(self):
        """配置优化器，支持事件编码器的分组学习率"""
        # 检查是否需要为事件编码器设置不同的学习率
        event_lr = getattr(self.cfg.OPTIMIZER, 'EVENT_LR', None)
        use_event = getattr(self.cfg.MODEL.MODALITY, 'USE_EVENT', False)
        
        if use_event and event_lr is not None and self.model.event_encoder is not None:
            # 分组学习率：事件编码器使用 EVENT_LR，其他参数使用默认 LR
            event_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue  # 跳过冻结的参数（如冻结的backbone）
                if 'event_encoder' in name:
                    event_params.append(param)
                else:
                    other_params.append(param)
            
            param_groups = [
                {'params': other_params, 'lr': self.cfg.OPTIMIZER.LR},
                {'params': event_params, 'lr': event_lr},
            ]
            optimizer = torch.optim.Adam(
                param_groups, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
            )
        else:
            # 单一学习率：所有参数使用默认 LR
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(
                params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
            )

        return optimizer
