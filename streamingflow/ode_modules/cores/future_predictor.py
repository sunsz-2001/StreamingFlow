"""Future prediction module backed by NNFOwithBayesianJumps."""

import numpy as np
import torch
import torch.nn as nn

from ..utils.convolutions import Bottleneck, Block, DeepLabHead
from ..utils.temporal import SpatialGRU, Dual_GRU, BiGRU
from .bayesian_ode import NNFOwithBayesianJumps


class FuturePredictionODE(nn.Module):
    """将多模态历史编码送入 NNFOwithBayesianJumps 并得到未来序列。"""

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
            spatial_grus.append(SpatialGRU(in_channels, in_channels))
            if i < self.n_spatial_gru - 1:
                res_blocks.append(nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)]))
            else:
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
    ):
        batch_outputs = []
        # Support LiDAR-only or Camera-only by deriving batch size from input
        batch_size = future_prediction_input.shape[0]
        for bs in range(batch_size):
            obs_feature_with_time = {}
            if camera_states is not None:
                for index in range(camera_timestamp.shape[1]):
                    obs_feature_with_time[camera_timestamp[bs, index]] = camera_states[bs, index].unsqueeze(0)
            if lidar_states is not None:
                for index in range(lidar_timestamp.shape[1]):
                    obs_feature_with_time[lidar_timestamp[bs, index]] = lidar_states[bs, index].unsqueeze(0)

            obs = dict(sorted(obs_feature_with_time.items(), key=lambda v: v[0]))
            times = torch.tensor(list(obs.keys()), device=future_prediction_input.device, dtype=target_timestamp.dtype)
            observations = torch.stack(list(obs.values()), dim=1)

            _, auxilary_loss, predict_x = self.gru_ode(
                times=times,
                input=future_prediction_input,
                obs=observations,
                delta_t=self.delta_t,
                T=target_timestamp[bs],
            )
            batch_outputs.append(predict_x)

        x = torch.concat(batch_outputs, dim=0)

        hidden_state = x[:, 0]
        for i in range(self.n_spatial_gru):
            x = self.spatial_grus[i](x, hidden_state)
            b, s, c, h, w = x.shape
            x = self.res_blocks[i](x.view(b * s, c, h, w))
            x = x.view(b, s, c, h, w)

        return x, auxilary_loss
