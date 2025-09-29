"""
Independent temporal modules for ODE components.

This module contains temporal processing components needed by the ODE modules,
making ode_modules completely independent.
"""

import torch
import torch.nn as nn
from .convolutions import ConvBlock


class SpatialGRU(nn.Module):
    """Spatial GRU for processing spatial sequences."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GRU gates
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, 3, padding=1)
        self.conv_candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, input_sequence, hidden_state):
        """
        Args:
            input_sequence: [B, T, C, H, W]
            hidden_state: [B, C, H, W]
        Returns:
            output_sequence: [B, T, C, H, W]
        """
        batch_size, seq_len, channels, height, width = input_sequence.shape

        outputs = []
        h = hidden_state

        for t in range(seq_len):
            x = input_sequence[:, t]  # [B, C, H, W]

            # Concatenate input and hidden state
            combined = torch.cat([x, h], dim=1)

            # Compute gates
            gates = self.conv_gates(combined)
            reset_gate, update_gate = gates.chunk(2, dim=1)
            reset_gate = torch.sigmoid(reset_gate)
            update_gate = torch.sigmoid(update_gate)

            # Compute candidate hidden state
            reset_combined = torch.cat([x, reset_gate * h], dim=1)
            candidate = torch.tanh(self.conv_candidate(reset_combined))

            # Update hidden state
            h = (1 - update_gate) * h + update_gate * candidate
            outputs.append(h)

        return torch.stack(outputs, dim=1)


__all__ = ['SpatialGRU']