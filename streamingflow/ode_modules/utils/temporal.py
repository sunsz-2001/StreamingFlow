"""
Independent temporal modules for ODE components.

This module contains temporal processing components needed by the ODE modules,
making ode_modules completely independent.
"""

import torch
import torch.nn as nn
from .convolutions import ConvBlock


class SpatialGRU(nn.Module):
    """Spatial GRU mirroring the original implementation used during training."""

    def __init__(self, input_size, hidden_size, gru_bias_init=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder = nn.Conv2d(hidden_size, input_size, kernel_size=1, bias=False)

    def forward(self, x, state=None):
        """Process a BEV feature sequence with a convolutional GRU."""

        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'

        outputs = []
        b, timesteps, c, h, w = x.size()
        rnn_state = torch.zeros(b, self.hidden_size, h, w, device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]
            rnn_state = self.gru_cell(x_t, rnn_state)
            outputs.append(self.conv_decoder(rnn_state))

        return torch.stack(outputs, dim=1)

    def gru_cell(self, x, state):
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))
        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


__all__ = ['SpatialGRU']
