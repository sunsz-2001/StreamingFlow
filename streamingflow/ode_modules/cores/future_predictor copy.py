"""基于 NNFOwithBayesianJumps 的未来预测模块。"""

import numpy as np
import torch
import torch.nn as nn

from ..utils.convolutions import Bottleneck, Block, DeepLabHead
from ..utils.temporal import SpatialGRU, Dual_GRU, BiGRU
from .bayesian_ode import NNFOwithBayesianJumps


class FuturePredictionODE(nn.Module):
    """衔接骨干编码器与 NNFOwithBayesianJumps 的桥接模块。

    主要职责：
      1. 汇集多模态隐特征（相机、LiDAR、可选的高频相机帧），并按时间对齐。
      2. 将对齐后的观测序列送入 `NNFOwithBayesianJumps`，在那里完成连续时间的
         预测 / 更新循环。
      3. 使用多层 SpatialGRU 与卷积头对返回的未来序列做精炼，使其符合下游
         BEV 解码器的输入格式。
    """

    def __init__(
        self,
        in_channels,
        latent_dim,
        n_future,
        cfg,
        mixture=True,
        n_gru_blocks=2,
        n_res_layers=1,
        delta_t=0.05,
    ):
        super().__init__()
        self.n_future = n_future
        self.n_spatial_gru = n_gru_blocks
        self.delta_t = delta_t
        self.gru_ode = NNFOwithBayesianJumps(
            input_size=in_channels,
            hidden_size=latent_dim,
            cfg=cfg,
            mixing=int(mixture),
        )

        spatial_grus = []
        res_blocks = []
        for i in range(self.n_spatial_gru):
            # 每个 SpatialGRU 都把序列维度当作时间，保持未来序列的空间一致性。
            spatial_grus.append(SpatialGRU(in_channels, in_channels))
            if i < self.n_spatial_gru - 1:
                res_blocks.append(nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)]))
            else:
                # 最后一层使用 DeepLab 头，将特征上采样/投影回解码器期望的 BEV 尺寸。
                res_blocks.append(DeepLabHead(in_channels, in_channels, 128))

        self.spatial_grus = nn.ModuleList(spatial_grus)
        self.res_blocks = nn.ModuleList(res_blocks)

    def forward(
        self,
        future_prediction_input,
        camera_states,
        lidar_states,
        camera_timestamp,
        lidar_timestamp,
        target_timestamp,
        camera_states_hi=None,
        camera_timestamp_hi=None,
    ):
        batch_outputs = []
        # 支持仅 LiDAR / 仅相机的场景：直接由输入推断 batch 大小。
        batch_size = future_prediction_input.shape[0]
        for bs in range(batch_size):
            obs_feature_with_time = {}
            
            # --- 时间归一化逻辑 ---
            # 1. 收集当前 batch 所有相关的绝对时间戳，找到 t0 (最小值)
            all_timestamps = []
            
            if camera_timestamp is not None:
                ts_bs = camera_timestamp[bs]
                # 兼容 Tensor, numpy array 和 list
                if torch.is_tensor(ts_bs):
                    all_timestamps.extend(ts_bs.detach().cpu().numpy().tolist())
                elif isinstance(ts_bs, np.ndarray):
                    all_timestamps.extend(ts_bs.tolist())
                else:
                    all_timestamps.extend(list(ts_bs))

            if lidar_timestamp is not None:
                ts_bs = lidar_timestamp[bs]
                if torch.is_tensor(ts_bs):
                    all_timestamps.extend(ts_bs.detach().cpu().numpy().tolist())
                elif isinstance(ts_bs, np.ndarray):
                    all_timestamps.extend(ts_bs.tolist())
                else:
                    all_timestamps.extend(list(ts_bs))
            
            # 找到起始时间 t0 (防止空列表错误)
            t0 = min(all_timestamps) if all_timestamps else 0.0
            
            # 定义时间缩放因子：时间戳已是相对秒，保持尺度以匹配 delta_t=0.05
            TIME_SCALE = 1.0

            if camera_states is not None:
                for index in range(camera_timestamp.shape[1]):
                    # 获取原始绝对时间戳
                    ts_raw = camera_timestamp[bs, index].item() if torch.is_tensor(camera_timestamp[bs, index]) else float(camera_timestamp[bs, index])
                    # 转换为相对秒：(t - t0) * 1e-6
                    ts_rel = (ts_raw - t0) * TIME_SCALE
                    obs_feature_with_time[ts_rel] = camera_states[bs, index].unsqueeze(0)
            
            if lidar_states is not None:
                for index in range(lidar_timestamp.shape[1]):
                    ts_raw = lidar_timestamp[bs, index].item() if torch.is_tensor(lidar_timestamp[bs, index]) else float(lidar_timestamp[bs, index])
                    ts_rel = (ts_raw - t0) * TIME_SCALE
                    obs_feature_with_time[ts_rel] = lidar_states[bs, index].unsqueeze(0)
            
            if camera_states_hi is not None and camera_timestamp_hi is not None:
                for index in range(camera_timestamp_hi.shape[1]):
                    ts_raw = camera_timestamp_hi[bs, index].item() if torch.is_tensor(camera_timestamp_hi[bs, index]) else float(camera_timestamp_hi[bs, index])
                    ts_rel = (ts_raw - t0) * TIME_SCALE
                    obs_feature_with_time[ts_rel] = camera_states_hi[bs, index].unsqueeze(0)

            # 不同传感器可能拥有相同时间戳；排序可确保按时间顺序送入 ODE。
            # 若同一时间戳出现多个观测，后插入的会覆盖先前的，起到简单的融合作用。
            # 现在字典的 key 是 Python 标量 (相对秒)，排序不会有设备问题
            obs = dict(sorted(obs_feature_with_time.items(), key=lambda v: v[0]))
            # 使用模型输入的 dtype 和 device，避免类型不一致。
            times = torch.tensor(list(obs.keys()), device=future_prediction_input.device, dtype=future_prediction_input.dtype)
            observations = torch.stack(list(obs.values()), dim=1)

            # 生成未来时间点：从当前时间(0)到target_timestamp，等间隔生成n_future个点
            target_time_raw = target_timestamp[bs]
            # 确保转为 Python 标量 float
            if torch.is_tensor(target_time_raw):
                target_time_raw = target_time_raw.detach().cpu().item()
            else:
                target_time_raw = float(target_time_raw)
            
            # 归一化 target_time: (target - t0) * 1e-6
            target_time_rel = (target_time_raw - t0) * TIME_SCALE
            
            # 生成 n_future 个等间隔的未来时间点
            if self.n_future > 1:
                # 使用相对的目标时间计算步长
                time_step = target_time_rel / self.n_future
                
                # 先在 CPU 上完成基础序列生成，再搬移到对应设备，彻底规避 Device 转换问题
                T = torch.arange(
                    1, self.n_future + 1, 
                    dtype=torch.float32, 
                    device='cpu'
                ).to(device=future_prediction_input.device, dtype=future_prediction_input.dtype) * time_step
            else:
                # 如果只有1个未来帧，直接使用相对的目标时间
                T = torch.tensor(
                    [target_time_rel], 
                    dtype=future_prediction_input.dtype, 
                    device=future_prediction_input.device
                )
            
            # 为当前 batch 提取对应的输入
            input_bs = future_prediction_input[bs:bs+1]  # [1, 1, C, H, W]
            
            _, auxilary_loss, predict_x = self.gru_ode(
                times=times,
                input=input_bs,
                obs=observations,
                delta_t=self.delta_t,
                T=T+1e-4,
            )
            batch_outputs.append(predict_x)

        x = torch.concat(batch_outputs, dim=0)

        hidden_state = x[:, 0]
        for i in range(self.n_spatial_gru):
            # 对预测序列做循环细化。以第一帧隐状态作为种子，可在 ODE 预测漂移时保持时间上下文。
            x = self.spatial_grus[i](x, hidden_state)
            b, s, c, h, w = x.shape
            print('FPODE:x.shape:',x.shape, x.device)
            x = self.res_blocks[i](x.view(b * s, c, h, w))
            x = x.view(b, s, c, h, w)
        return x, auxilary_loss
