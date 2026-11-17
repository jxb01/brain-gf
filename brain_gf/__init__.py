"""Gradient-free associative brain (no backprop, no autograd).

Public entry points:
    FeatureReadoutBrain  - frozen random features + free linear readout, NLMS writes
    PureAttentionBrain   - multi-layer content-addressed memory (no Q/K), residual + decay
"""

from .gf_brain import FeatureReadoutBrain, PureAttentionBrain
from .numerics import r2, rolling_mse

__all__ = [
    "FeatureReadoutBrain",
    "PureAttentionBrain",
    "r2",
    "rolling_mse",
]
