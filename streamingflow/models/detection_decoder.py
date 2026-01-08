import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet3d.models.builder import build_head
except ImportError as exc:
    raise ImportError(
        "Failed to import mmdet3d. Please install mmdet3d."
    ) from exc


class ConfigDict(dict):
    """支持属性访问的字典，兼容 mmdet3d 的配置访问方式"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def _dict_to_configdict(d):
    """递归将 dict 转换为 ConfigDict"""
    if isinstance(d, dict):
        result = ConfigDict()
        for k, v in d.items():
            result[k] = _dict_to_configdict(v)
        return result
    elif isinstance(d, list):
        return [_dict_to_configdict(item) for item in d]
    else:
        return d


class DetectionDecoder(nn.Module):
    """检测解码器：使用TransFusion head进行3D目标检测"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 构建TransFusion检测头配置
        head_cfg = self._build_head_config(cfg)
        self.detection_head = build_head(head_cfg)
    
    def _build_head_config(self, cfg):
        """构建TransFusion head的配置字典"""
        # 计算BEV网格参数
        bev_h = int((cfg.LIFT.X_BOUND[1] - cfg.LIFT.X_BOUND[0]) / cfg.LIFT.X_BOUND[2])
        bev_w = int((cfg.LIFT.Y_BOUND[1] - cfg.LIFT.Y_BOUND[0]) / cfg.LIFT.Y_BOUND[2])
        grid_size = [bev_w, bev_h]
        
        # 计算voxel_size（从LIFT配置推导）
        voxel_size = [cfg.LIFT.X_BOUND[2], cfg.LIFT.Y_BOUND[2], cfg.LIFT.Z_BOUND[2]]
        
        # point_cloud_range
        pc_range = [
            cfg.LIFT.X_BOUND[0], cfg.LIFT.Y_BOUND[0], cfg.LIFT.Z_BOUND[0],
            cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1], cfg.LIFT.Z_BOUND[1]
        ]
        
        out_size_factor = cfg.DETECTION.OUT_SIZE_FACTOR
        
        head_cfg = dict(
            type='TransFusionHead',
            num_proposals=cfg.DETECTION.NUM_PROPOSALS,
            auxiliary=cfg.DETECTION.AUXILIARY,
            in_channels=cfg.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS,
            hidden_channel=cfg.DETECTION.HIDDEN_CHANNEL,
            num_classes=cfg.DETECTION.NUM_CLASSES,
            use_sigmoid_cls=True,
            num_decoder_layers=cfg.DETECTION.NUM_DECODER_LAYERS,
            num_heads=cfg.DETECTION.NUM_HEADS,
            nms_kernel_size=cfg.DETECTION.NMS_KERNEL_SIZE,
            ffn_channel=cfg.DETECTION.FFN_CHANNEL,
            dropout=cfg.DETECTION.DROPOUT,
            bn_momentum=cfg.MODEL.BN_MOMENTUM,
            activation='relu',
            common_heads=dict(
                center=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
            ),
            num_heatmap_convs=2,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            bias='auto',
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean'),
            loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner3D',
                    cls_cost=dict(type='ClassificationCost', weight=1.),
                    reg_cost=dict(type='BBoxBEVL1Cost', weight=1.0),
                    iou_cost=dict(type='IoU3DCost', weight=1.0),
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar')
                ),
                grid_size=grid_size,
                point_cloud_range=pc_range,
                voxel_size=voxel_size,
                out_size_factor=out_size_factor,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                pos_weight=-1,
                gaussian_overlap=0.5,  # Default min_overlap for gaussian_radius
                min_radius=2,  # Minimum radius for heatmap gaussian
            ),
            test_cfg=dict(
                grid_size=grid_size,
                out_size_factor=out_size_factor,
                pc_range=pc_range,
                voxel_size=voxel_size,
                dataset=cfg.DETECTION.DATASET,
                nms_type='circle',  # 评估时跳过NMS，评估所有预测框
            ),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=pc_range,
                out_size_factor=out_size_factor,
                voxel_size=voxel_size,
                post_center_range=pc_range,  # 使用pc_range作为后处理范围，用于过滤超出范围的预测框
                score_threshold=0.0,
                code_size=10,
            ),
        )
        # 转换 train_cfg 和 test_cfg 为支持属性访问的 ConfigDict
        head_cfg['train_cfg'] = _dict_to_configdict(head_cfg['train_cfg'])
        head_cfg['test_cfg'] = _dict_to_configdict(head_cfg['test_cfg'])
        return head_cfg
    
    def forward(self, states, metas=None):
        """
        前向传播
        
        Args:
            states: [B, S, C, H, W] BEV特征序列，S为时间步数
            metas: list of dict，每个样本的元数据（可选）
        
        Returns:
            dict: 包含'detection'键，值为TransFusion head的输出
        """
        # 使用最后一帧进行检测（检测关注当前时刻）
        # if states.dim() == 5:
        #     bev_feat = states[:, -1]  # [B, C, H, W]
        # elif states.dim() == 4:
        #     bev_feat = states  # [B, C, H, W]
        # else:
        #     raise ValueError(f"Expected states to be 4D or 5D, got {states.dim()}D")
        bev_feat = states  # [B, C, H, W]
        
        # 下采样BEV特征到TransFusion期望的尺寸
        # TransFusion期望的feature map尺寸 = grid_size // out_size_factor
        out_size_factor = self.cfg.DETECTION.OUT_SIZE_FACTOR
        # 计算BEV网格尺寸
        bev_h = int((self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0]) / self.cfg.LIFT.X_BOUND[2])
        bev_w = int((self.cfg.LIFT.Y_BOUND[1] - self.cfg.LIFT.Y_BOUND[0]) / self.cfg.LIFT.Y_BOUND[2])
        # TransFusion期望的feature map尺寸
        target_h = bev_h // out_size_factor
        target_w = bev_w // out_size_factor
        
        # 如果当前尺寸与目标尺寸不一致，进行下采样
        if bev_feat.shape[2] != target_h or bev_feat.shape[3] != target_w:
            bev_feat = F.interpolate(
                bev_feat,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )  # [B, C, H, W] → [B, C, target_h, target_w]
        
        # Check bev_feat before passing to detection_head
        if torch.isnan(bev_feat).any() or torch.isinf(bev_feat).any():
            print(f"Warning: NaN/Inf in DetectionDecoder bev_feat before detection_head")
            print(f"  bev_feat stats: min={bev_feat.min()}, max={bev_feat.max()}, mean={bev_feat.mean()}")
            print(f"  bev_feat shape: {bev_feat.shape}")
            # Replace NaN/Inf with 0
            bev_feat = torch.where(
                torch.isnan(bev_feat) | torch.isinf(bev_feat),
                torch.zeros_like(bev_feat),
                bev_feat
            )
        
        # TransFusion head期望输入为list
        detection_output = self.detection_head.forward([bev_feat], metas)
        
        return {'detection': detection_output}


