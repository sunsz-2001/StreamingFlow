"""
流式训练器 - 遍历所有时间窗口进行训练

关键原则：
1. 所有窗口参与训练：遍历 flow_data 中的每个时间窗口
2. 损失累积：每个窗口的损失累积后平均
3. 简洁实现：直接使用 forward，无状态注入
"""

import torch
import numpy as np
from streamingflow.trainer_dsec import TrainingModule as BaseTrainingModule


class StreamingTrainingModule(BaseTrainingModule):
    """
    流式训练模块 - 遍历所有时间窗口
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        print("[StreamingTrainer] Initialized - processing all flow_data windows")

    def shared_step(self, batch, is_train):
        """
        流式处理实现

        流程：
        1. 遍历所有时间窗口
        2. 每个窗口独立前向传播
        3. 累积所有窗口的损失
        """

        image = batch.get('image')
        flow_batch = batch['flow_data']
        future_egomotion = batch['future_egomotion']

        # 准备所有窗口的数据
        flow_data = self._prepare_flow_data(flow_batch)

        # 逐窗口处理
        total_loss = {}
        all_outputs = []
        all_labels = []

        for window_idx, flow_window in enumerate(flow_data):
            # 准备当前窗口的输入
            window_inputs = self._prepare_window_inputs(
                flow_window, image, future_egomotion, batch
            )

            # 前向传播
            output = self.model.forward(**window_inputs)

            all_outputs.append(output)

            # 计算损失（仅训练时）
            if is_train:
                window_labels = self.prepare_window_labels(batch, window_idx)
                all_labels.append(window_labels)

                window_loss = self._compute_window_loss(output, window_labels)

                # 累积损失
                for key, value in window_loss.items():
                    total_loss[key] = total_loss.get(key, 0.0) + value

        # 平均损失
        num_windows = len(flow_data)
        for key in total_loss:
            total_loss[key] = total_loss[key] / num_windows

        # 返回最后一个窗口的输出（用于评估）
        final_output = all_outputs[-1]
        final_labels = all_labels[-1] if all_labels else {}

        return final_output, final_labels, total_loss

    def _prepare_flow_data(self, flow_batch):
        """准备所有时间窗口的数据（复用原有逻辑）"""
        flow_data = []
        for flow_frame in range(len(flow_batch[0])):
            temp_flow_data = {
                'intrinsics': [],
                'extrinsics': [],
                'flow_lidar': [],
                'flow_events': [],
                'events_stmp': [],
                'lidar_stmp': [],
                'target_timestamp': [],
            }

            for bs_idx in range(len(flow_batch)):
                tmp = flow_batch[bs_idx][flow_frame]

                for key in temp_flow_data.keys():
                    if key == 'flow_events':
                        tmp_evs = torch.stack(tmp[key], dim=0)
                        temp_flow_data[key].append(tmp_evs)
                    else:
                        temp_flow_data[key].append(tmp[key])

            for key in temp_flow_data.keys():
                if isinstance(temp_flow_data[key][0], (np.ndarray, list)):
                    try:
                        temp_flow_data[key] = np.array(temp_flow_data[key])
                    except:
                        pass
                elif isinstance(temp_flow_data[key][0], torch.Tensor):
                    temp_flow_data[key] = torch.stack(temp_flow_data[key], dim=0)

            flow_data.append(temp_flow_data)

        return flow_data

    def _prepare_window_inputs(self, flow_window, image, future_egomotion, batch):
        """准备单个窗口的所有输入"""
        intrinsics = flow_window['intrinsics']
        extrinsics = flow_window['extrinsics']
        camera_timestamps = flow_window['events_stmp']
        lidar_timestamps = flow_window['lidar_stmp']
        target_timestamp = flow_window['target_timestamp']

        use_lidar = getattr(self.cfg.MODEL.MODALITY, 'USE_LIDAR', False)
        use_event = getattr(self.cfg.MODEL.MODALITY, 'USE_EVENT', False)

        # 处理点云
        points = None
        padded_voxel_points = None
        if use_lidar:
            points = flow_window['flow_lidar']
            if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI:
                padded_voxel_points = batch.get('padded_voxel_points')

        # 处理事件
        event = None
        if use_event:
            event = flow_window['flow_events']
            if torch.is_tensor(event) and event.dim() == 4:
                event = event.unsqueeze(2)  # [B, S, 1, C, H, W]

        return {
            'image': image,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'future_egomotion': future_egomotion,
            'padded_voxel_points': padded_voxel_points,
            'camera_timestamps': camera_timestamps,
            'points': points,
            'lidar_timestamps': lidar_timestamps,
            'target_timestamp': target_timestamp,
            'event': event,
            'metas': batch.get('metas'),
        }

    def prepare_window_labels(self, batch, _window_idx):
        """
        为特定窗口准备标签

        说明：
        - 所有窗口使用相同的标签（相对于batch的固定时间基准）
        - 标签通过cumulative_warp已经对齐到ego坐标系
        - 损失函数会根据receptive_field自动处理时间维度

        Args:
            batch: 完整的batch数据
            _window_idx: 当前窗口的索引（未使用，保留用于接口兼容性）

        Returns:
            window_labels: 标签字典
        """
        _ = _window_idx  # 接口兼容性，当前未使用

        # 获取完整的标签（所有窗口共享）
        full_labels = self.prepare_future_labels(batch)

        window_labels = {}

        # 检测标签
        if 'gt_bboxes_3d' in batch:
            window_labels['gt_bboxes_3d'] = batch['gt_bboxes_3d']
        if 'gt_labels_3d' in batch:
            window_labels['gt_labels_3d'] = batch['gt_labels_3d']

        # 分割标签（所有窗口使用相同标签）
        for key in ['segmentation', 'pedestrian', 'hdmap', 'centerness',
                    'offset', 'flow', 'depths']:
            if key in full_labels:
                window_labels[key] = full_labels[key]

        # Planning标签
        if self.cfg.PLANNING.ENABLED:
            if 'gt_trajectory' in batch:
                window_labels['gt_trajectory'] = batch['gt_trajectory']

        return window_labels

    def _compute_window_loss(self, output, labels):
        """计算单个窗口的所有损失"""
        loss = {}

        # 检测损失
        if getattr(self.cfg, 'DETECTION', None) and getattr(self.cfg.DETECTION, 'ENABLED', False):
            if 'detection' in output:
                gt_bboxes_3d = labels.get('gt_bboxes_3d')
                gt_labels_3d = labels.get('gt_labels_3d')

                if gt_bboxes_3d is not None and gt_labels_3d is not None:
                    detection_losses = self.model.decoder.detection_head.loss(
                        gt_bboxes_3d, gt_labels_3d, output['detection']
                    )
                    for key, val in detection_losses.items():
                        if not (torch.isnan(val) or torch.isinf(val)):
                            loss[f'det_{key}'] = val

        # 分割损失
        if 'segmentation' in output and 'segmentation' in labels:
            segmentation_factor = 1 / (2 * torch.exp(self.model.segmentation_weight))
            loss['segmentation'] = segmentation_factor * self.losses_fn['segmentation'](
                output['segmentation'], labels['segmentation'], self.model.receptive_field
            )
            loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight

        # Pedestrian损失
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            if 'pedestrian' in output and 'pedestrian' in labels:
                pedestrian_factor = 1 / (2 * torch.exp(self.model.pedestrian_weight))
                loss['pedestrian'] = pedestrian_factor * self.losses_fn['pedestrian'](
                    output['pedestrian'], labels['pedestrian'], self.model.receptive_field
                )
                loss['pedestrian_uncertainty'] = 0.5 * self.model.pedestrian_weight

        # HDMap损失
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            if 'hdmap' in output and 'hdmap' in labels:
                hdmap_factor = 1 / (2 * torch.exp(self.model.hdmap_weight))
                loss['hdmap'] = hdmap_factor * self.losses_fn['hdmap'](
                    output['hdmap'], labels['hdmap']
                )
                loss['hdmap_uncertainty'] = 0.5 * self.model.hdmap_weight

        # Instance分割损失
        if self.cfg.INSTANCE_SEG.ENABLED:
            if 'instance_center' in output and 'centerness' in labels:
                centerness_factor = 1 / (2 * torch.exp(self.model.centerness_weight))
                loss['instance_center'] = centerness_factor * self.losses_fn['instance_center'](
                    output['instance_center'], labels['centerness'], self.model.receptive_field
                )
                loss['centerness_uncertainty'] = 0.5 * self.model.centerness_weight

            if 'instance_offset' in output and 'offset' in labels:
                offset_factor = 1 / (2 * torch.exp(self.model.offset_weight))
                loss['instance_offset'] = offset_factor * self.losses_fn['instance_offset'](
                    output['instance_offset'], labels['offset'], self.model.receptive_field
                )
                loss['offset_uncertainty'] = 0.5 * self.model.offset_weight

        # 深度损失
        if self.cfg.LIFT.GT_DEPTH:
            if 'depth_prediction' in output and 'depths' in labels:
                depths_factor = 1 / (2 * torch.exp(self.model.depths_weight))
                loss['depths'] = depths_factor * self.losses_fn['depths'](
                    output['depth_prediction'], labels['depths']
                )
                loss['depths_uncertainty'] = 0.5 * self.model.depths_weight

        # Flow损失
        if self.cfg.INSTANCE_FLOW.ENABLED:
            if 'instance_flow' in output and 'flow' in labels:
                flow_factor = 1 / (2 * torch.exp(self.model.flow_weight))
                loss['instance_flow'] = flow_factor * self.losses_fn['instance_flow'](
                    output['instance_flow'], labels['flow'], self.model.receptive_field
                )
                loss['flow_uncertainty'] = 0.5 * self.model.flow_weight

        return loss

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        output, labels, loss = self.shared_step(batch, True)

        # 记录所有损失
        for key, value in loss.items():
            self.log(f'train_{key}', value, prog_bar=False)

        # 总损失
        total_loss = sum(loss.values())
        self.log('train_loss', total_loss, prog_bar=True)

        # 更新计数
        if hasattr(self, 'training_step_count'):
            self.training_step_count += 1

        return total_loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        output, labels, loss = self.shared_step(batch, False)

        # 计算评估指标（复用原有逻辑）
        if hasattr(self, 'metric_vehicle_val'):
            scores = self.metric_vehicle_val.compute()
            self.log('step_val_seg_iou_dynamic', scores[1])

        # Planning评估
        if self.cfg.PLANNING.ENABLED and 'selected_traj' in output:
            self.log('step_predicted_traj_x', output['selected_traj'][0, -1, 0])
            self.log('step_target_traj_x', labels['gt_trajectory'][0, -1, 0])

        # 可视化第一个batch
        if batch_idx == 0:
            self.visualise(labels, output, batch_idx, prefix='val')

        return output
