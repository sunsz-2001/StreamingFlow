"""Observation update cell mirroring the original implementation."""

import math
import torch
import torch.nn as nn

from .gru_cells import DualGRUCell


class GRUObservationCell(nn.Module):
    """Implements discrete update based on received observations."""

    def __init__(self, input_size, hidden_size, min_log_sigma=-5.0, max_log_sigma=5.0, bias=True):
        super().__init__()
        self.gru_d = DualGRUCell(input_size, hidden_size, bias=bias)
        prep_hidden = hidden_size
        std = math.sqrt(2.0 / (4 + prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden
        self.var_eps = 1e-6
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

    def forward(self, state, p, X_obs):
        bs, C, h, w = X_obs.shape

        gru_input = X_obs
        state = self.gru_d(gru_input, state)

        loss = None
        return state, loss
