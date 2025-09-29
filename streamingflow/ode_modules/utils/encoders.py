"""
Utility functions and helper classes for ODE modules.
"""

import torch


def init_weights(m):
    """Initialize weights for linear layers."""
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)