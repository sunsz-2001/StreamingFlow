"""
Future Prediction ODE module.

This module integrates multi-modal historical encodings and feeds them into NNFOwithBayesianJumps
to generate future sequences.
"""

import torch
import torch.nn as nn

from ..utils.convolutions import Block, DeepLabHead
from ..utils.temporal import SpatialGRU
from .bayesian_ode import NNFOwithBayesianJumps


class FuturePredictionODE(nn.Module):
    """将多模态历史编码送入 NNFOwithBayesianJumps 并得到未来序列。"""

    def __init__(self, in_channels, latent_dim, n_future, cfg, mixture=True,
                 n_gru_blocks=2, n_res_layers=1, delta_t=0.05):
        """构造未来预测模块。

        Args:
            in_channels (int): 输入到 ODE 的通道数，通常等于 temporal 模块输出。
            latent_dim (int): NNFOwithBayesianJumps 内部使用的隐藏维度。
            n_future (int): 需要预测的未来帧数量。
            cfg (CfgNode): 完整配置，可读取模型参数。
            mixture (bool): 是否在变分推断时使用混合分布。
            n_gru_blocks (int): 后续空间 GRU 的层数。
            n_res_layers (int): 每层空间 GRU 后接的残差块数量。
            delta_t (float): 连续时间积分的基本步长。
        """
        super().__init__()
        self.n_spatial_gru = n_gru_blocks
        self.delta_t = delta_t
        self.gru_ode = NNFOwithBayesianJumps(
            input_size=in_channels, hidden_size=latent_dim, cfg=cfg, mixing=int(mixture)
        )

        # 构建若干层空间 GRU 和残差模块，用于进一步加工连续时间预测出的序列。
        spatial_grus = []
        res_blocks = []
        for i in range(self.n_spatial_gru):
            spatial_grus.append(SpatialGRU(in_channels, in_channels))
            if i < self.n_spatial_gru - 1:
                res_blocks.append(nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)]))
            else:
                # 最后一层使用 DeepLabHead，输出同通道数特征。
                res_blocks.append(DeepLabHead(in_channels, in_channels, 128))

        self.spatial_grus = nn.ModuleList(spatial_grus)
        self.res_blocks = nn.ModuleList(res_blocks)

    def forward(self, future_prediction_input, camera_states, lidar_states,
                camera_timestamp, lidar_timestamp, target_timestamp):
        """整合多模态历史观测，求解连续时间 ODE，输出未来潜空间序列。

        Args:
            future_prediction_input (Tensor): 当前时刻的潜空间状态，形状 ``[B, 1, C, H, W]``。
            camera_states (Tensor | None): 相机时间窗的隐藏特征 ``[B, T_cam, C, H, W]``。
            lidar_states (Tensor | None): LiDAR 时间窗的隐藏特征 ``[B, T_lidar, C, H, W]``。
            camera_timestamp / lidar_timestamp (Tensor): 对应时间戳，单位秒。
            target_timestamp (Tensor): 目标预测时间列表 ``[B, N_future]``。

        Returns:
            Tuple[Tensor, Tensor]:
                * x: 未来潜空间序列 ``[B, N_future, C, H, W]``
                * auxilary_loss: 由 ODE 计算得到的附加损失（供训练使用）
        """

        batch_outputs = []
        auxilary_loss = 0  # Initialize auxiliary loss
        for bs in range(camera_states.shape[0]):
            # 把相机、激光的历史特征按照时间戳合并，并排序，构成 irregular 观测序列。
            obs_feature_with_time = {}
            if camera_states is not None:
                for index in range(camera_timestamp.shape[1]):
                    obs_feature_with_time[camera_timestamp[bs, index]] = camera_states[bs, index].unsqueeze(0)
            if lidar_states is not None:
                for index in range(lidar_timestamp.shape[1]):
                    obs_feature_with_time[lidar_timestamp[bs, index]] = lidar_states[bs, index].unsqueeze(0)

            obs = dict(sorted(obs_feature_with_time.items(), key=lambda v: v[0]))
            times = torch.tensor(list(obs.keys()))
            observations = torch.stack(list(obs.values()), dim=1)

            # 交给 NNFOwithBayesianJumps 做连续时间推断。
            _, auxilary_loss, predict_x = self.gru_ode(
                times=times,
                input=future_prediction_input,
                obs=observations,
                delta_t=self.delta_t,
                T=target_timestamp[bs],
            )
            batch_outputs.append(predict_x)

        x = torch.concat(batch_outputs, dim=0)

        # 空间 GRU + 残差模块：让未来序列进一步融合历史上下文（与输入的第 0 帧对齐）。
        hidden_state = x[:, 0]
        for i in range(self.n_spatial_gru):
            x = self.spatial_grus[i](x, hidden_state)
            b, s, c, h, w = x.shape
            x = self.res_blocks[i](x.view(b * s, c, h, w))
            x = x.view(b, s, c, h, w)

        return x, auxilary_loss
