"""Tier 3: Metric learning losses -- differentiable distance functions.

Loss functions that learn embedding spaces via backpropagation.
All losses return differentiable JAX arrays for gradient flow.
"""

from calibrax.metrics.learning._base import MetricLearningLoss, Reducer
from calibrax.metrics.learning.angular import ArcFaceLoss, CosFaceLoss
from calibrax.metrics.learning.contrastive import (
    ContrastiveLoss,
    NTXentLoss,
    TripletMarginLoss,
)
from calibrax.metrics.learning.miners import (
    HardNegativeMiner,
    MinedIndices,
    SemiHardMiner,
)
from calibrax.metrics.learning.proxy import ProxyAnchorLoss, ProxyNCALoss


__all__ = [
    "ArcFaceLoss",
    "CosFaceLoss",
    "ContrastiveLoss",
    "HardNegativeMiner",
    "MetricLearningLoss",
    "MinedIndices",
    "NTXentLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "Reducer",
    "SemiHardMiner",
    "TripletMarginLoss",
]
