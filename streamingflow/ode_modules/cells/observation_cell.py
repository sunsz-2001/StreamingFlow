"""
Observation update cell for Bayesian jumps in ODE models.

This cell handles discrete updates when new observations arrive.
"""

import math
import torch
import torch.nn as nn
from .gru_cells import DualGRUCell


class GRUObservationCell(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, min_log_sigma=-5.0, max_log_sigma=5.0, bias=True):
        super().__init__()
        self.gru_d = DualGRUCell(input_size, hidden_size, bias=bias)
        prep_hidden = hidden_size

        ## prep layer and its initialization
        std = math.sqrt(2.0 / (4 + prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden
        self.var_eps = 1e-6
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

    def forward(self, state, p, X_obs):
        bs, C, h, w = X_obs.shape

        # DualGRUCell expects input: [B, 1, C, H, W] and state: [B, T, C, H, W]
        gru_input = X_obs.unsqueeze(1)  # [B, 1, C, H, W]

        # State needs time dimension for DualGRUCell
        if len(state.shape) == 4:  # [B, C, H, W]
            state = state.unsqueeze(1)  # [B, 1, C, H, W]

        state = self.gru_d(gru_input, state)

        loss = None
        return state, loss