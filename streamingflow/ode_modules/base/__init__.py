"""Abstract base classes for ODE modules"""

from .interfaces import ODEPredictor, FuturePredictor

__all__ = ['ODEPredictor', 'FuturePredictor']