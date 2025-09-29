"""
Abstract base classes for ODE modules.

These interfaces define the contracts that ODE-based predictors should follow,
enabling different implementations to be used interchangeably.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ODEPredictor(nn.Module, ABC):
    """Abstract base class for ODE-based predictors.

    This interface defines the contract for modules that perform continuous-time
    ODE-based prediction with discrete observation updates.
    """

    @abstractmethod
    def forward(self, times, initial_state, observations, target_times, **kwargs):
        """Perform ODE prediction with observation updates.

        Args:
            times (Tensor): Observation timestamps in ascending order.
            initial_state (Tensor): Initial state at the beginning of prediction.
            observations (Tensor): Sequence of observations aligned with times.
            target_times (Tensor): Target prediction timestamps.
            **kwargs: Additional configuration parameters.

        Returns:
            Tuple: (predictions, loss) where predictions are the forecasted states
                   at target_times and loss is any auxiliary training loss.
        """
        pass


class FuturePredictor(nn.Module, ABC):
    """Abstract base class for future sequence predictors.

    This interface defines the contract for modules that predict future sequences
    by integrating multi-modal historical observations.
    """

    @abstractmethod
    def forward(self, current_state, history_states, timestamps, target_times, **kwargs):
        """Predict future sequences from current and historical states.

        Args:
            current_state (Tensor): Current state to start prediction from.
            history_states (Dict or List): Historical states from different modalities.
            timestamps (Dict or List): Corresponding timestamps for historical states.
            target_times (Tensor): Target prediction timestamps.
            **kwargs: Additional configuration parameters.

        Returns:
            Tuple: (future_sequence, auxiliary_loss) where future_sequence contains
                   the predicted states at target_times and auxiliary_loss is any
                   additional training loss.
        """
        pass


class ODECell(nn.Module, ABC):
    """Abstract base class for ODE cells.

    ODE cells define the dynamics function f(x, t) in dx/dt = f(x, t).
    """

    @abstractmethod
    def forward(self, x, state):
        """Compute the time derivative at the current state.

        Args:
            x (Tensor): Input at current time.
            state (Tensor): Current hidden state.

        Returns:
            Tensor: Time derivative of the state.
        """
        pass