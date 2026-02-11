import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from mmcv.runner import auto_fp16, force_fp32

from streamingflow.models.encoder import Encoder
from streamingflow.models.event_encoder_evrt import EventEncoderEvRT
from streamingflow.models.temporal_model import TemporalModelIdentity, TemporalModel
from streamingflow.models.distributions import DistributionModule
from streamingflow.models.decoder import Decoder
from streamingflow.models.detection_decoder import DetectionDecoder
from streamingflow.models.planning_model import Planning
from streamingflow.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from streamingflow.utils.geometry import calculate_birds_eye_view_parameters, VoxelsSumming, pose_vec2mat
from streamingflow.utils.event_tensor import EventTensorizer

import yaml

from streamingflow.ode_modules.cores.future_predictor import FuturePredictionODE
import time

from mmdet3d.ops import bev_pool
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models.builder import build_backbone


def _validate_tensor(tensor, name, allow_inf=False):
    """Validate tensor for NaN/Inf values and raise error if found.
    
    Args:
        tensor: Tensor to validate
        name: Name for error message
        allow_inf: If True, only check for NaN, not Inf
    """
    if torch.isnan(tensor).any():
        stats = f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
        raise ValueError(f"NaN detected in {name}. Stats: {stats}, shape: {tensor.shape}")
    if not allow_inf and torch.isinf(tensor).any():
        stats = f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
        raise ValueError(f"Inf detected in {name}. Stats: {stats}, shape: {tensor.shape}")


class streamingflow_lidar(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        
        self.use_lidar = self.cfg.MODEL.MODALITY.USE_LIDAR



        self.bev_h = int((self.cfg.LIFT.X_BOUND[1]-self.cfg.LIFT.X_BOUND[0])/self.cfg.LIFT.X_BOUND[2])
        self.bev_w = int((self.cfg.LIFT.Y_BOUND[1]-self.cfg.LIFT.Y_BOUND[0])/self.cfg.LIFT.Y_BOUND[2])

        # Decoder
        if getattr(self.cfg, 'DETECTION', None) and getattr(self.cfg.DETECTION, 'ENABLED', False):
            self.decoder = DetectionDecoder(self.cfg)
        else:
            self.decoder = Decoder(
                in_channels=self.future_pred_in_channels,
                n_classes=len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
                n_present=self.receptive_field,
                n_hdmap=len(self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS),
                predict_gate = {
                    'perceive_hdmap': self.cfg.SEMANTIC_SEG.HDMAP.ENABLED,
                    'predict_pedestrian': self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED,
                    'predict_instance': self.cfg.INSTANCE_SEG.ENABLED,
                    'predict_future_flow': self.cfg.INSTANCE_FLOW.ENABLED,
                    'planning': self.cfg.PLANNING.ENABLED,
                }
            )

        if self.use_lidar:
            voxel_size = [float(v) for v in self.cfg.VOXEL.VOXEL_SIZE]
            area_extents = np.array(self.cfg.VOXEL.AREA_EXTENTS, dtype=np.float32)
            point_cloud_range = [
                float(area_extents[0][0]),
                float(area_extents[1][0]),
                float(area_extents[2][0]),
                float(area_extents[0][1]),
                float(area_extents[1][1]),
                float(area_extents[2][1]),
            ]
            x_range = point_cloud_range[3] - point_cloud_range[0]
            y_range = point_cloud_range[4] - point_cloud_range[1]
            z_range = point_cloud_range[5] - point_cloud_range[2]
            x_size = int(np.floor((x_range / voxel_size[0]) + 0.5))
            y_size = int(np.floor((y_range / voxel_size[1]) + 0.5))
            z_size = int(np.floor((z_range / voxel_size[2]) + 0.5))
            sparse_shape = [x_size, y_size, z_size]
            # encoders = {
            #     'lidar': {
            #         'voxelize': {
            #             'max_num_points': 20,
            #             'point_cloud_range': point_cloud_range,
            #             'voxel_size': voxel_size,
            #             'max_voxels': [90000, 120000],
            #         },
            #         'backbone': {
            #             'type': 'SparseEncoder',
            #             'in_channels': 4,
            #             'sparse_shape': sparse_shape,
            #             'output_channels': 128,
            #             'order': ['conv', 'norm', 'act'],
            #             'encoder_channels': [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
            #             'encoder_paddings': [[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
            #             'block_type': 'basicblock',
            #         },
            #     },
            #     # 'temporal_model': {
            #     #     'type': 'Temporal3DConvModel',
            #     #     'receptive_field': 3,
            #     #     'input_egopose': True,
            #     #     'in_channels': 256,
            #     #     'input_shape': [128, 128],
            #     #     'with_skip_connect': True,
            #     #     'start_out_channels': 256,
            #     #     'det_grid_conf': {
            #     #         'xbound': [0, 80, 0.6],
            #     #         'ybound': [-80, 80, 0.6],
            #     #         'zbound': [-2, 4, .0],
            #     #         'dbound': [1.0, 60.0, 1.0],
            #     #     },
            #     #     'grid_conf': {
            #     #         'xbound': [0, 75.2, 0.8],
            #     #         'ybound': [-75.2, 75.2, 0.8],
            #     #         'zbound': [-2, 4, 6.0],
            #     #         'dbound': [1.0, 60.0, 1.0],
            #     #     },
            #     # },
            # }
            encoders = {
                'lidar': {
                    'voxelize': {
                        'max_num_points': 20,
                        'point_cloud_range': point_cloud_range,
                        'voxel_size': voxel_size,
                        'max_voxels': [90000, 120000],
                    },
                    'backbone': {
                        'type': 'SparseEncoder',
                        'in_channels': 4,
                        'sparse_shape': sparse_shape,
                        'output_channels': 128,
                        'order': ['conv', 'norm', 'act'],
                        'encoder_channels': [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
                        'encoder_paddings': [[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
                        'block_type': 'basicblock',
                    },
                },
                # 'temporal_model': {
                #     'type': 'Temporal3DConvModel',
                #     'receptive_field': 3,
                #     'input_egopose': True,
                #     'in_channels': 256,
                #     'input_shape': [128, 128],
                #     'with_skip_connect': True,
                #     'start_out_channels': 256,
                #     'det_grid_conf': {
                #         'xbound': [0, 54.0, 0.6],
                #         'ybound': [-32, 32, 0.6],
                #         'zbound': [-5, 3, .0],
                #         'dbound': [1.0, 60.0, 1.0],
                #     },
                #     'grid_conf': {
                #         'xbound': [0, 51.2, 0.8],
                #         'ybound': [-32, 32, 0.8],
                #         'zbound': [-5, 3, 8.0],
                #         'dbound': [1.0, 60.0, 1.0],
                #     },
                # },
            }

            self.encoders = nn.ModuleDict()
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
            lidar_backbone_output_channels = encoders["lidar"]["backbone"]["output_channels"]

            encoder_channels = encoders["lidar"]["backbone"]["encoder_channels"]
            downsample_xy = 2 ** max(len(encoder_channels) - 1, 0)
            lidar_h = int(math.ceil(x_size / downsample_xy))
            lidar_w = int(math.ceil(y_size / downsample_xy))

            def _conv_out_size(n, pad):
                return int(math.floor((n + 2 * pad - 3) / 2 + 1))

            stage_paddings = encoders["lidar"]["backbone"]["encoder_paddings"]
            stride2_pads = [stage_paddings[0][-1], stage_paddings[1][-1], stage_paddings[2][-1]]
            z_out = z_size
            for pad in stride2_pads:
                if isinstance(pad, (list, tuple)):
                    pad_z = pad[-1]
                else:
                    pad_z = int(pad)
                z_out = _conv_out_size(z_out, pad_z)
            z_out = _conv_out_size(z_out, 0)
            z_out = max(1, z_out)

            actual_lidar_output_channels = lidar_backbone_output_channels * z_out



            self.lidar_channels = actual_lidar_output_channels
            self.lidar_h = lidar_h
            self.lidar_w = lidar_w
            projection_key = f'lidar_states_std_projection_{128}_to_{64}'
            self.standard_projections = nn.ModuleDict()
                    # if projection_key not in self.standard_projections:
            self.standard_projections[projection_key] = nn.Conv2d(128, 64, 1)
            # self.temporal_model_lidar = TemporalModel(
            #     actual_lidar_output_channels,
            #     self.receptive_field,
            #     input_shape=(lidar_h, lidar_w),
            #     start_out_channels=lidar_start_out_channels,
            #     extra_in_channels=self.cfg.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS,
            #     n_spatial_layers_between_temporal_layers=self.cfg.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS,
            #     use_pyramid_pooling=self.cfg.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING,
            # )
            self.temporal_model_lidar = TemporalModelIdentity(actual_lidar_output_channels,1)

        if self.cfg.PLANNING.ENABLED:
            self.planning = Planning(cfg, self.encoder_out_channels, 6, gru_state_size=self.cfg.PLANNING.GRU_STATE_SIZE)

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = None

        for k, res in enumerate(points):
            if target_device is not None:
                if not hasattr(res, 'device') or res.device != target_device:
                    try:
                        res = res.to(device=target_device)
                    except Exception:
                        if torch.cuda.is_available():
                            res = res.cuda()
            else:
                if hasattr(res, 'device') and res.device.type == 'cpu' and torch.cuda.is_available():
                    res = res.cuda()

            if res.numel() == 0 or res.shape[0] == 0:
                continue

            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None

            feats.append(f)
            padded_c = F.pad(c, (1, 0), mode="constant", value=k)
            coords.append(padded_c)

            if n is not None:
                sizes.append(n)

        if len(feats) == 0:
            # print("[WARNING] No valid voxels generated from any point cloud!")
            device = target_device if target_device is not None else (points[0].device if len(points) > 0 else torch.device('cuda'))
            return torch.zeros((0, 0), device=device), \
                   torch.zeros((0, 4), dtype=torch.int32, device=device), \
                   torch.zeros((0,), dtype=torch.int32, device=device)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)

        sparse_shape = self.encoders["lidar"]["backbone"].sparse_shape

        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def extract_lidar_features(self, x, batch_size=None) -> torch.Tensor:
        if batch_size is None:
            batch_size = len(x)
            
        feats, coords, sizes = self.voxelize(x)
        # save_cuda_tensor_as_npz(coords[:,1:], 'coor')
        # print('coor', coords[:,0].max(),coords[:,1].max(),coords[:,2].max(),)
        if coords.numel() == 0:
            # print("[WARNING] Empty coords, returning empty tensor")
            return torch.zeros((batch_size, self.lidar_channels, self.lidar_h, self.lidar_w),
                             device=feats.device, dtype=feats.dtype)

        # batch_size = coords[-1, 0] + 1
        # batch_size = int(coords[-1, 0].item()) + 1

        coords = coords.contiguous()
        # coords = coords[:, [0, 3, 2, 1]].contiguous()

        try:
            x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
            return x
        except RuntimeError as e:
            raise

    def extract_lidar_features_time_series(self, points, device):
        """Extract LiDAR features for time series data. Returns [B, T, C, H, W]."""
        if torch.is_tensor(points) or isinstance(points, np.ndarray):
            pts_batch = [[torch.from_numpy(points).to(dtype=torch.float32)] if isinstance(points, np.ndarray) else [points.to(dtype=torch.float32)]]
        elif isinstance(points, (list, tuple)) and len(points) > 0 and not isinstance(points[0], (list, tuple)):
            pts_batch = [[p.to(dtype=torch.float32) if torch.is_tensor(p) else torch.from_numpy(p).to(dtype=torch.float32)] for p in points]
        else:
            pts_batch = []
            for sample_points in points:
                if isinstance(sample_points, (list, tuple)):
                    sample_norm = []
                    for p in sample_points:
                        while isinstance(p, (list, tuple)) and len(p) > 0:
                            p = p[0]
                        if isinstance(p, np.ndarray):
                            p = torch.from_numpy(p)
                        if not torch.is_tensor(p):
                            raise TypeError(f"Expected point cloud to be Tensor or np.ndarray, but got {type(p)}")
                        sample_norm.append(p.to(dtype=torch.float32))
                    pts_batch.append(sample_norm)
                else:
                    p = sample_points
                    if isinstance(p, np.ndarray):
                        p = torch.from_numpy(p)
                    if not torch.is_tensor(p):
                        raise TypeError(f"Expected point cloud to be Tensor or np.ndarray, but got {type(p)}")
                    pts_batch.append([p.to(dtype=torch.float32)])

        B = len(pts_batch)
        features = []

        # for t in range(T):
        # pts_t = []
        for s in range(B):
            sample_pts = pts_batch[s]
            # if t < len(sample_pts):
            #     p = sample_pts[t]
            # elif len(sample_pts)==0:
            #     p = torch.tensor([]).to(device)
            # else:
            if len(sample_pts)!=0:
                p = [sample_pts[-1]]
                # print(p[0][:,0].max(),p[0][:,1].max())
                
                # pts_t.append(p.to(dtype=torch.float32))

                feat_t = self.extract_lidar_features(p, batch_size=1)
                features.append(feat_t)
            else:
                # feat_t = None
                pass
            # pts_t.append()
            

        # x = torch.stack(features, dim=1)
        return features

    def forward(self, points=None, metas=None):
        output = {}

        if self.use_lidar:
            if isinstance(points, list) and len(points) > 0:
                
                # 先检查第一个元素是否是列表（多时间步格式）
                first_point = points[0]
                if isinstance(first_point, (list, tuple)):
                    norm_points = []
                    for sample_idx, sample_points in enumerate(points):
                        if not isinstance(sample_points, (list, tuple)):
                            raise TypeError(
                                f"Expected points[{sample_idx}] to be list/tuple of point clouds, "
                                f"but got {type(sample_points)}"
                            )
                        norm_sample_points = []
                        for t, pc in enumerate(sample_points):
                            # Accept nested lists/tuples but DO NOT flatten time dimension.
                            # If an element is a list (e.g. [[tensor]]), unwrap inner-most tensor,
                            # but preserve that this sample has multiple time steps.
                            while isinstance(pc, (list, tuple)) and len(pc) > 0:
                                # drill down one level; if inner element is still list, stop
                                # to preserve explicit time-step lists above this level.
                                inner = pc[0]
                                if isinstance(inner, (list, tuple)):
                                    break
                                pc = inner

                            if isinstance(pc, np.ndarray):
                                pc = torch.from_numpy(pc)
                            if not torch.is_tensor(pc):
                                raise TypeError(
                                    f"Expected point cloud at sample[{sample_idx}][{t}] to be Tensor or np.ndarray, "
                                    f"but got {type(pc)}"
                                )
                            norm_sample_points.append(pc.to(dtype=torch.float32))
                        norm_points.append(norm_sample_points)
                    points = norm_points
                else:
                    norm_points = []
                    for idx, p in enumerate(points):
                        if isinstance(p, np.ndarray):
                            p = torch.from_numpy(p)
                        if not torch.is_tensor(p):
                            raise TypeError(
                                f"Expected points[{idx}] to be Tensor or np.ndarray, "
                                f"but got {type(p)}"
                            )
                        norm_points.append(p.to(dtype=torch.float32))
                    points = norm_points
            elif torch.is_tensor(points):
                # 单个样本的点云，封装为列表以统一接口
                points = [points.to(dtype=torch.float32)]
            else:
                raise TypeError(
                    f"Expected points to be list[Tensor], list[list[Tensor]], or Tensor, but got {type(points)}"
                )
            
            first_point = points[0]
            if isinstance(first_point, list):
                x = self.extract_lidar_features_time_series(points, first_point[0].device)  # [B, T, C, H, W]
            else:
                B = len(points)
                # voxelize + 稀疏卷积编码，返回 [B, C, H_det, W_det]
                feature = self.extract_lidar_features(points, batch_size=B)
                _, C, H_det, W_det = feature.shape
                x = feature.unsqueeze(1)  # [B, 1, C, H, W]

            lidar_states = self.temporal_model_lidar(x)


        lidar_states = torch.cat(lidar_states, dim=0)
        bev_output = self.decoder(lidar_states, metas)

        output = {**output, **bev_output}
        output["lidar_states"] = lidar_states

        return output

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        # Ensure intrinsics and extrinsics are float32 to match frustum dtype
        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()
        extrinsics = torch.inverse(extrinsics)

        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # save_cuda_tensor_as_npz(points[0].view(-1,3))
        
        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)
        # print('pmin',points[...,0].min(),points[...,1].min())
        # print('pmax',points[...,0].max(),points[...,1].max())
        
        # points = points[...,[1,0,2]]
        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def encoder_forward(self, x, cam_front_index=1):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w) # (9 * 6, 3, 224, 480)
        x, depth = self.encoder(x) # (9 * 6, 64, 28, 60)
        if self.cfg.PLANNING.ENABLED:
            cam_front = x.view(b, n, *x.shape[1:])[:, cam_front_index]
        else:
            cam_front = None

        if self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION:
            depth_prob = depth.softmax(dim=1)
            x = depth_prob.unsqueeze(1) * x.unsqueeze(2)  # outer product depth and features
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.depth_channels, 1, 1)

        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2) # channel dimension
        depth = depth.view(b, n, *depth.shape[1:])

        return x, depth, cam_front

    def event_encoder_forward(self, event_frames):
        if not self.use_event:
            raise RuntimeError("Event encoder requested but USE_EVENT is False.")
        # event_frames已经是[B*S*N, C, H, W]形状，直接传入编码器
        feats, depth_logits = self.event_encoder(event_frames)
        return feats, depth_logits

    def _resize_event_depth_bins(self, depth_logits, target_bins):
        if depth_logits.dim() == 4:
            b, d, h, w = depth_logits.shape
            depth_logits = depth_logits.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D, H, W]
        elif depth_logits.dim() == 5:
            if depth_logits.shape[2] == target_bins or depth_logits.shape[1] == target_bins:
                b = depth_logits.shape[0]
                if depth_logits.shape[1] == target_bins:
                    depth_logits = depth_logits.unsqueeze(1)  # 添加 S 维度
                else:
                    # [B, S, D, H, W]
                    depth_logits = depth_logits.unsqueeze(2)  # 添加 N 维度
        
        if depth_logits.dim() != 6:
            raise ValueError(f"depth_logits 必须是6维 [B, S, N, D, H, W]，但得到 {depth_logits.dim()} 维 {depth_logits.shape}")
        
        b, s, n, d, h, w = depth_logits.shape
        
        if d == target_bins:
            return depth_logits
        
        reshaped = depth_logits.contiguous().view(b * s * n, 1, d, h, w)
        resized = F.interpolate(reshaped, size=(target_bins, h, w), mode="trilinear", align_corners=False)
        resized = resized.view(b, s, n, target_bins, h, w)
        return resized

    def _expand_features_with_depth(self, feats, depth_prob):
        b, s, n, c, h, w = feats.shape
        depth_bins = depth_prob.shape[3]
        feats_flat = feats.contiguous().view(b * s * n, c, h, w)
        depth_flat = depth_prob.contiguous().view(b * s * n, depth_bins, h, w)
        volume_flat = depth_flat.unsqueeze(1) * feats_flat.unsqueeze(2)
        volume = volume_flat.view(b, s, n, c, depth_bins, h, w).permute(0, 1, 2, 4, 5, 6, 3)
        return volume

    def _normalize_event_frame_shape(self, frames):
        if frames.dim() == 5:
            frames = frames.unsqueeze(0)
        if frames.dim() != 6:
            raise ValueError("事件帧需为[B, S, N, C, H, W]或[S, N, C, H, W]格式")
        return frames.contiguous()

    def _stack_event_time_to_channels(self, event_frames, expected_channels):
        b, s, n, c, h, w = event_frames.shape
        stacked_channels = s * c
        
        if stacked_channels != expected_channels:
            raise ValueError(
                f"事件帧时间堆叠后的通道数 {stacked_channels} 与预期的 {expected_channels} 不符。"
                f"请检查序列长度 S={s} 和每帧通道数 C={c}。" 
            )
        
        # [B, S, N, C, H, W] -> [B, N, S, C, H, W] -> [B, N, S*C, H, W] -> [B*N, S*C, H, W]
        event_frames = event_frames.permute(0, 2, 1, 3, 4, 5).contiguous()
        event_frames = event_frames.view(b, n, stacked_channels, h, w)
        event_frames = event_frames.view(b * n, stacked_channels, h, w)
        
        return event_frames

    def _prepare_event_frames(self, event_input, seq_len, device):
        if torch.is_tensor(event_input):
            frames = event_input.to(device=device, dtype=torch.float32)
            frames = self._normalize_event_frame_shape(frames)
        elif isinstance(event_input, dict) and "frames" in event_input and torch.is_tensor(event_input["frames"]):
            frames = event_input["frames"].to(device=device, dtype=torch.float32)
            frames = self._normalize_event_frame_shape(frames)
        else:
            if self.event_tensorizer is None:
                self.event_tensorizer = EventTensorizer(self.cfg)
            frames = self.event_tensorizer.prepare_frames(event_input, device=device, max_seq_len=seq_len)
            frames = self._normalize_event_frame_shape(frames)

        if frames.shape[1] < seq_len:
            raise ValueError("Event frames length is shorter than required sequence length.")
        if frames.shape[1] > seq_len:
            frames = frames[:, :seq_len]
        return frames

    def fuse_camera_event_features(self, camera_feats, depth_logits, event_feats):
        """
        Args:
            camera_feats: (B, S, N, D, H, W, C_cam)
            depth_logits: (B, S, N, D, H, W)
            event_feats: (B, S, N, C_evt, H, W)
        """
        if self.event_fusion_type not in {"concat", "residual"}:
            return camera_feats
        depth_prob = depth_logits.softmax(dim=3)
        event_feats = event_feats.permute(0, 1, 2, 4, 5, 3).unsqueeze(3)  # -> (B,S,N,1,H,W,C_evt)
        event_feats = depth_prob.unsqueeze(-1) * event_feats  # broadcast over depth bins

        if self.event_fusion_type == "concat":
            fused = torch.cat([camera_feats, event_feats], dim=-1)
            fused = self.event_fusion(fused)
        else:  # residual
            gate = self.event_gate.view(1, 1, 1, 1, 1, 1, -1)
            fused = camera_feats + gate * event_feats
        return fused


    # @force_fp32()
    def get_cam_feats(self, x, d, depth_gt=None):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        # d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)

        if self.ret_depth:
            depth_gt = depth_gt.view(B * N, *depth_gt.shape[2:])
            self.depth_loss = self.get_depth_loss(depth_gt, depth)

        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    # @force_fp32()
    def bev_pool(self, geom_feats, x):

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # print('max',geom_feats[...,0].max(),geom_feats[...,1].max(),geom_feats[...,2].max())
        # print('min',geom_feats[...,0].min(),geom_feats[...,1].min(),geom_feats[...,2].min())
        # flatten indices
        geom_feats = ((geom_feats - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        # geom_feats[...,0]-=geom_feats[...,0].min()
        # geom_feats[...,1]-=geom_feats[...,1].min()
        # print('gmin',geom_feats[...,0].min(),geom_feats[...,1].min())
        # print('gmax',geom_feats[...,0].max(),geom_feats[...,1].max())
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # save_cuda_tensor_as_npz(geom_feats, '4')
        # print('max',geom_feats[...,0].max(),geom_feats[...,1].max(),geom_feats[...,2].max())
        # print('min',geom_feats[...,0].min(),geom_feats[...,1].min(),geom_feats[...,2].min())
        # geom_feats[...,0] -=geom_feats[...,0].min()
        # geom_feats[...,1] -=geom_feats[...,1].min()
        # filter out points that are outside box
        kept = (
                (geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.bev_dimension[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.bev_dimension[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.bev_dimension[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        x = bev_pool(x, geom_feats, B, self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1])
        # save_cuda_tensor_as_npz(geom_feats, '5')

        # collapse Z
        # final = torch.cat(x.unbind(dim=2), 1)

        return x, geom_feats

    def projection_to_birds_eye_view(self, x, geometry, future_egomotion):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, s, n_cameras, depth, height, width, channels
        batch, s, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, s, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        future_egomotion_mat = pose_vec2mat(future_egomotion)  # (3,3,4,4)
        rotation, translation = future_egomotion_mat[..., :3, :3], future_egomotion_mat[..., :3, 3]
        
        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            flow_b = x[b]
            flow_geo = geometry[b]

            #####  transform the 3D voxel to current frame  #####
            for t in range(s):
                if t != s - 1:
                    flow_geo_tmp = flow_geo[:t + 1]
                    rotation_b = rotation[b, t].view(1, 1, 1, 1, 1, 3, 3)
                    translation_b = translation[b, t].view(1, 1, 1, 1, 1, 3)
                    flow_geo_tmp = rotation_b.matmul(flow_geo_tmp.unsqueeze(-1)).squeeze(-1)
                    flow_geo_tmp += translation_b
                    flow_geo[:t + 1] = flow_geo_tmp

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=flow_b.device)

            for t in range(s):

                # flatten x
                x_b = flow_b[t]
                geometry_b = flow_geo[t]
                n_geo, d_geo, h_geo, w_geo, _ = geometry_b.shape
                n_x, d_x, h_x, w_x, _ = x_b.shape
                
                assert n_geo == n_x, f"Camera/event count mismatch: {n_geo} != {n_x}"
                assert d_geo == d_x, f"Depth channel mismatch: {d_geo} != {d_x}"

                # 只有当空间维度不匹配时才进行调整（事件分支的情况）
                # 图像分支通常不需要这个调整，因为 geometry 和 camera_volume 基于相同的图像配置
                if h_geo != h_x or w_geo != w_x:
                    # geometry_b [n, d, h_geo, w_geo, 3] 调整 [n, d, h_x, w_x, 3]
                    # 使用双线性插值：需要先 permute [n, d, 3, h_geo, w_geo]，然后插值到 [n, d, 3, h_x, w_x]
                    geometry_b_permuted = geometry_b.permute(0, 1, 4, 2, 3)  # [n, d, 3, h_geo, w_geo]
                    geometry_b_resized = F.interpolate(
                        geometry_b_permuted.reshape(n_geo * d_geo, 3, h_geo, w_geo),
                        size=(h_x, w_x),
                        mode='bilinear',
                        align_corners=False
                    )  # [n*d, 3, h_x, w_x]
                    geometry_b = geometry_b_resized.reshape(n_geo, d_geo, 3, h_x, w_x).permute(0, 1, 3, 4, 2)  # [n, d, h_x, w_x, 3]

                x_b, geometry_b = self.bev_pool(geometry_b.unsqueeze(0), x_b.unsqueeze(0))
 
                tmp_bev_feature = x_b[0].permute(1,2,3,0)
         
                bev_feature = bev_feature * self.discount + tmp_bev_feature
                
                tmp_bev_feature = bev_feature.permute((0, 3, 1, 2))
                tmp_bev_feature = tmp_bev_feature.squeeze(0)
                output[b, t] = tmp_bev_feature

        return output

    def calculate_birds_eye_view_features(
        self,
        intrinsics,
        extrinsics,
        future_egomotion,
        image=None,
        event=None,
    ):
        b, s, n = intrinsics.shape[:3]

        intrinsics_packed = pack_sequence_dim(intrinsics)
        extrinsics_packed = pack_sequence_dim(extrinsics)
        geometry = self.get_geometry(intrinsics_packed, extrinsics_packed)
        geometry = unpack_sequence_dim(geometry, b, s)
        outputs = {}

        camera_volume = None
        camera_depth_logits = None
        cam_front = None
        if image is not None and self.use_camera:
            image_packed = pack_sequence_dim(image)
            cam_feats, depth_logits, cam_front_raw = self.encoder_forward(image_packed)
            camera_volume = unpack_sequence_dim(cam_feats, b, s)
            camera_depth_logits = unpack_sequence_dim(depth_logits, b, s)
            cam_front = unpack_sequence_dim(cam_front_raw, b, s)[:, -1] if cam_front_raw is not None else None

        event_volume = None
        event_depth_logits_out = None
        event_feats = None
        if event is not None and self.use_event:
            event_frames = self._prepare_event_frames(event, s, device=intrinsics.device)
            event_frames = event_frames[:, :s]
            
            # 获取EVRT期望的输入通道数（直接使用配置的IN_CHANNELS，不再进行时间维度堆叠）
            expected_channels = getattr(self.cfg.MODEL.EVENT, "IN_CHANNELS", 0)
            if expected_channels <= 0:
                bins = getattr(self.cfg.MODEL.EVENT, "BINS", 10)
                expected_channels = 2 * bins
            
            b, s, n, c, h, w = event_frames.shape
            if c != expected_channels:
                raise ValueError(
                    f"事件帧的通道数 {c} 与预期的 {expected_channels} 不符。"
                    f"请检查配置 MODEL.EVENT.IN_CHANNELS 是否正确设置。"
                )
            
            # [B, S, N, C, H, W] -> [B*S*N, C, H, W]
            event_reshaped = event_frames.view(b * s * n, c, h, w)
            event_feats, event_depth_logits = self.event_encoder_forward(event_reshaped)
            # event_feats[B,F,64,30,40], event_depth_logits[B,F,48,30,40]
            # Validate event encoder outputs
            # _validate_tensor(event_feats, "event_feats from event_encoder_forward")
            # if event_depth_logits is not None:
            #     _validate_tensor(event_depth_logits, "event_depth_logits from event_encoder_forward")
            
            # 恢复形状: [B*S*N, C, H, W] -> [B, S, N, C, H, W]
            event_feats = event_feats.view(b, s, n, *event_feats.shape[1:])
            # event_feats[B,F,1,C,30,40]
            if event_depth_logits is not None:
                event_depth_logits_out = event_depth_logits.view(b, s, n, *event_depth_logits.shape[1:])
                event_depth_logits_out = self._resize_event_depth_bins(event_depth_logits_out, self.depth_channels)
            else:
                event_depth_logits_out = torch.zeros(
                    (b, s, n, self.depth_channels, event_feats.shape[-2], event_feats.shape[-1]),
                    device=event_feats.device,
                    dtype=event_feats.dtype,
                )
            
            # Validate inputs before processing
            # _validate_tensor(event_feats, "event_feats before _expand_features_with_depth")
            # _validate_tensor(event_depth_logits_out, "event_depth_logits_out before softmax")
            
            # Clamp logits to prevent overflow in softmax
            event_depth_logits_out = torch.clamp(event_depth_logits_out, min=-50.0, max=50.0)
            event_depth_prob = event_depth_logits_out.softmax(dim=3)
            # _validate_tensor(event_depth_prob, "event_depth_prob after softmax")
            
            event_volume = self._expand_features_with_depth(event_feats, event_depth_prob)
            # _validate_tensor(event_volume, "event_volume before BEV projection")
            
            bev_event = self.projection_to_birds_eye_view(event_volume, geometry, future_egomotion)
            # _validate_tensor(bev_event, "bev_event after BEV projection")
            outputs["event"] = {
                "bev": bev_event,
                "depth": event_depth_logits_out,
            }

        if camera_volume is not None:
            if (
                event_feats is not None
                and event_depth_logits_out is not None
                and self.event_fusion_type in {"concat", "residual"}
            ):
                camera_volume = self.fuse_camera_event_features(
                    camera_volume, event_depth_logits_out, event_feats
                )
            bev_camera = self.projection_to_birds_eye_view(camera_volume, geometry, future_egomotion)
            outputs["camera"] = {
                "bev": bev_camera,
                "depth": camera_depth_logits,
                "cam_front": cam_front,
            }

        return outputs

    def distribution_forward(self, present_features, min_log_sigma, max_log_sigma):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        def get_mu_sigma(mu_log_sigma):
            mu = mu_log_sigma[:, :, :self.latent_dim]
            log_sigma = mu_log_sigma[:, :, self.latent_dim:2*self.latent_dim]
            log_sigma = torch.clamp(log_sigma, min_log_sigma, max_log_sigma)
            sigma = torch.exp(log_sigma)
            if self.training:
                gaussian_noise = torch.randn((b, s, self.latent_dim), device=present_features.device)
            else:
                gaussian_noise = torch.zeros((b, s, self.latent_dim), device=present_features.device)
            sample = mu + sigma * gaussian_noise
            return mu, log_sigma, sample


        if self.cfg.PROBABILISTIC.METHOD == 'GAUSSIAN':
            mu_log_sigma = self.present_distribution(present_features)
            present_mu, present_log_sigma, present_sample = get_mu_sigma(mu_log_sigma)
            sample = present_sample

            # Spatially broadcast sample to the dimensions of present_features
            sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)
            

        elif self.cfg.PROBABILISTIC.METHOD == "BERNOULLI":
            present_log_prob = self.present_distribution(present_features)
            if self.training:
                bernoulli_noise = torch.randn((b, self.latent_dim, h, w), device=present_features.device)
            else:
                bernoulli_noise = torch.zeros((b, self.latent_dim, h, w), device=present_features.device)
            sample = torch.exp(present_log_prob) + bernoulli_noise

            sample = sample.view(b, s, self.latent_dim, h, w)


        elif self.cfg.PROBABILISTIC.METHOD == 'MIXGAUSSIAN':
            mu_log_sigma = self.present_distribution(present_features)
            present_mu1, present_log_sigma1, present_sample1 = get_mu_sigma(mu_log_sigma[:, :, :2*self.latent_dim])
            present_mu2, present_log_sigma2, present_sample2 = get_mu_sigma(mu_log_sigma[:, :, 2 * self.latent_dim : 4 * self.latent_dim])
            present_mu3, present_log_sigma3, present_sample3 = get_mu_sigma(mu_log_sigma[:, :, 4 * self.latent_dim : 6 * self.latent_dim])
            coefficient = mu_log_sigma[:, :, 6 * self.latent_dim:]
            coefficient = torch.softmax(coefficient, dim=-1)
            sample = present_sample1 * coefficient[:,:,0:1] + \
                     present_sample2 * coefficient[:,:,1:2] + \
                     present_sample3 * coefficient[:,:,2:3]

            # Spatially broadcast sample to the dimensions of present_features
            sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)

        else:
            raise NotImplementedError

        return sample

import open3d as o3d
def save_cuda_tensor_as_npz(tensor, name='1',ispcd=True):
    """
    将CUDA中的PyTorch Tensor转换为NumPy数组并保存为npz文件
    
    参数:
        tensor: 在CUDA上的PyTorch Tensor
        filename: 保存的文件名（可以是完整路径）
    """
    filename = '/home/user/图片/'+name+'.npz'
    # 确保Tensor在CUDA上
    if not tensor.is_cuda:
        raise ValueError("Tensor不在CUDA上，请先将Tensor移动到CUDA")
    
    # 1. 将CUDA Tensor移动到CPU
    tensor_cpu = tensor.cpu()
    
    # 2. 转换为NumPy数组
    numpy_array = tensor_cpu.numpy()
    
    # 3. 保存为npz文件
    np.savez(filename, array=numpy_array)
    if ispcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(numpy_array[:,:3])
        o3d.io.write_point_cloud('/home/user/图片/'+name+'.pcd', pcd)
    
    print(f"Tensor已保存为 {filename}")
    print(f"Tensor形状: {numpy_array.shape}")
    print(f"Tensor数据类型: {numpy_array.dtype}")
