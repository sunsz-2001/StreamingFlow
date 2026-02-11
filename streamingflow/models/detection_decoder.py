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
        grid_size = [bev_h, bev_w]
        
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
            loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.),
            loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean',loss_weight=.25),
            loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean',loss_weight=1.0),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner3D',
                    cls_cost=dict(type='ClassificationCost', weight=1.),
                    reg_cost=dict(type='BBoxBEVL1Cost', weight=.25),
                    iou_cost=dict(type='IoU3DCost', weight=.25),
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar')
                ),
                grid_size=grid_size,
                point_cloud_range=pc_range,
                voxel_size=voxel_size,
                out_size_factor=out_size_factor,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                pos_weight=-1,
                gaussian_overlap=0.1,  # Default min_overlap for gaussian_radius
                min_radius=2,  # Minimum radius for heatmap gaussian
            ),
            test_cfg=dict(
                grid_size=grid_size,
                out_size_factor=out_size_factor,
                pc_range=pc_range,
                voxel_size=voxel_size,
                dataset=cfg.DETECTION.DATASET,
                # nms_type='rotate',  # 评估时跳过NMS，评估所有预测框
                pre_maxsize=1000,
                post_maxsize=500,

                # Scale-NMS
                nms_type=None,
                # nms_type=['rotate', 'rotate'],
                # nms_thr=0.01,
                # nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                #                     1.1, 1.0, 1.0, 1.5, 3.5]]
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
        # out_size_factor = self.cfg.DETECTION.OUT_SIZE_FACTOR
        # # 计算BEV网格尺寸
        # bev_h = int((self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0]) / self.cfg.LIFT.X_BOUND[2])
        # bev_w = int((self.cfg.LIFT.Y_BOUND[1] - self.cfg.LIFT.Y_BOUND[0]) / self.cfg.LIFT.Y_BOUND[2])
        # # TransFusion期望的feature map尺寸
        # target_h = bev_h // out_size_factor
        # target_w = bev_w // out_size_factor
        
        # 如果当前尺寸与目标尺寸不一致，进行下采样
        # if bev_feat.shape[2] != target_h or bev_feat.shape[3] != target_w:
        #     bev_feat = F.interpolate(
        #         bev_feat,
        #         size=(target_h, target_w),
        #         mode='bilinear',
        #         align_corners=False
        #     )  # [B, C, H, W] → [B, C, target_h, target_w]
        
        # Check bev_feat before passing to detection_head
        # if torch.isnan(bev_feat).any() or torch.isinf(bev_feat).any():
        #     print(f"Warning: NaN/Inf in DetectionDecoder bev_feat before detection_head")
        #     print(f"  bev_feat stats: min={bev_feat.min()}, max={bev_feat.max()}, mean={bev_feat.mean()}")
        #     print(f"  bev_feat shape: {bev_feat.shape}")
        #     # Replace NaN/Inf with 0
        #     bev_feat = torch.where(
        #         torch.isnan(bev_feat) | torch.isinf(bev_feat),
        #         torch.zeros_like(bev_feat),
        #         bev_feat
        #     )
        
        # TransFusion head期望输入为list
        detection_output = self.detection_head.forward([bev_feat], metas)
        
        return {'detection': detection_output}


class DetectionDecoder_CP(nn.Module):
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
        point_cloud_range = [0, -51.2, -4.0, 32, 32, 4.0]
        bev_h = int((cfg.LIFT.X_BOUND[1] - cfg.LIFT.X_BOUND[0]) / cfg.LIFT.X_BOUND[2])
        bev_w = int((cfg.LIFT.Y_BOUND[1] - cfg.LIFT.Y_BOUND[0]) / cfg.LIFT.Y_BOUND[2])
        grid_size = [bev_w, bev_h]
        
        # 计算voxel_size（从LIFT配置推导）
        voxel_size = [0.1,0.1,0.2]
        
        # point_cloud_range
        pc_range = [-10, -42, -10.0, 61.2, 42, 10.0]
        
        out_size_factor = cfg.DETECTION.OUT_SIZE_FACTOR
        
        head_cfg = dict(
            type='CenterHead',
            in_channels=64,
            # tasks=[
            #     dict(num_class=1, class_names=['vehicle']),
            #     dict(num_class=2, class_names=['pedestrian', 'cyclist'])],
            tasks=[[0], [1, 2]],
            # tasks=[['vehicle'], ['pedestrian', 'cyclist']],
            common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            num_heatmap_convs=2,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            bias='auto',
            separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean'),
            share_conv_channel=64,
            train_cfg=dict(
                point_cloud_range=point_cloud_range,
                grid_size=[512, 640, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            test_cfg=dict(
                    pc_range=point_cloud_range[:2],
                    post_center_limit_range=pc_range,
                    max_per_img=500,
                    max_pool_nms=False,
                    min_radius=[4, 12, 10, 1, 0.85, 0.175],
                    score_threshold=0.01,
                    out_size_factor=8,
                    voxel_size=voxel_size[:2],
                    pre_max_size=1000,
                    post_max_size=500,

                    # Scale-NMS
                    nms_type=['rotate', 'rotate'],
                    nms_thr=0.01,
                    nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                        1.1, 1.0, 1.0, 1.5, 3.5]]),
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=pc_range,
                max_num=500,
                score_threshold=0.01,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=9),
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
      
        bev_feat = states  # [B, C, H, W]
        
        # TransFusion head期望输入为list
        detection_output = self.detection_head.forward([bev_feat], metas)
        
        return {'detection': detection_output}
