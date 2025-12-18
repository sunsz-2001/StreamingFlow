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
            if camera_states is not None:
                for index in range(camera_timestamp.shape[1]):
                    # 将时间戳转换为 Python 标量，避免设备不匹配问题
                    ts = camera_timestamp[bs, index].item() if torch.is_tensor(camera_timestamp[bs, index]) else float(camera_timestamp[bs, index])
                    obs_feature_with_time[ts] = camera_states[bs, index].unsqueeze(0)
            if lidar_states is not None:
                for index in range(lidar_timestamp.shape[1]):
                    # 将时间戳转换为 Python 标量，避免设备不匹配问题
                    ts = lidar_timestamp[bs, index].item() if torch.is_tensor(lidar_timestamp[bs, index]) else float(lidar_timestamp[bs, index])
                    obs_feature_with_time[ts] = lidar_states[bs, index].unsqueeze(0)
            if camera_states_hi is not None and camera_timestamp_hi is not None:
                for index in range(camera_timestamp_hi.shape[1]):
                    # 将时间戳转换为 Python 标量，避免设备不匹配问题
                    ts = camera_timestamp_hi[bs, index].item() if torch.is_tensor(camera_timestamp_hi[bs, index]) else float(camera_timestamp_hi[bs, index])
                    obs_feature_with_time[ts] = camera_states_hi[bs, index].unsqueeze(0)

            # 不同传感器可能拥有相同时间戳；排序可确保按时间顺序送入 ODE。
            # 若同一时间戳出现多个观测，后插入的会覆盖先前的，起到简单的融合作用。
            # 现在字典的 key 是 Python 标量，排序不会有设备问题
            obs = dict(sorted(obs_feature_with_time.items(), key=lambda v: v[0]))
            # 使用模型输入的 dtype 和 device，避免类型不一致。
            times = torch.tensor(list(obs.keys()), device=future_prediction_input.device, dtype=future_prediction_input.dtype)
            observations = torch.stack(list(obs.values()), dim=1)

            # ODE 模块会输出与 `target_timestamp` 对齐的隐 BEV 序列；
            # `auxilary_loss` 包含在 NNFOwithBayesianJumps 内部估计的 KL 等正则项。
            # 生成未来时间点：从当前时间(0)到target_timestamp，等间隔生成n_future个点
            target_time = target_timestamp[bs]
            if torch.is_tensor(target_time):
                target_time = target_time.item()
            
            # 生成 n_future 个等间隔的未来时间点
            # 例如：n_future=6, target_time=0.6 -> [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            if self.n_future > 1:
                time_step = target_time / self.n_future
                T = torch.arange(
                    1, self.n_future + 1, 
                    dtype=future_prediction_input.dtype, 
                    device=future_prediction_input.device
                ) * time_step
            else:
                # 如果只有1个未来帧，直接使用target_time
                T = torch.tensor(
                    [target_time], 
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
                T=T,
            )
            batch_outputs.append(predict_x)

        x = torch.concat(batch_outputs, dim=0)

        hidden_state = x[:, 0]
        for i in range(self.n_spatial_gru):
            # 对预测序列做循环细化。以第一帧隐状态作为种子，可在 ODE 预测漂移时保持时间上下文。
            x = self.spatial_grus[i](x, hidden_state)
            b, s, c, h, w = x.shape
            x = self.res_blocks[i](x.view(b * s, c, h, w))
            x = x.view(b, s, c, h, w)

        return x, auxilary_loss
