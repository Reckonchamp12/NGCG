"""
NGCG — Neural-Guided Conjecture Generation for Conservation Laws
================================================================
A four-stage neural-symbolic pipeline for automatic discovery of
conservation laws from time-series trajectories of dynamical systems.

Stages:
  1. Neural Dynamics    — MLP one-step predictor (frozen after training)
  2. Variance Minimiser — Multi-restart C_θ(z) network
  3. Symbolic Extraction — PySR + polynomial Lasso + LV log-basis Lasso
  4. Verification Gate  — strict constancy + trajectory diversity test

Usage:
    from ngcg import NGCG
    model = NGCG(system="mass_spring", data_path="ngcg_data_clean.h5")
    result = model.fit()
"""

from .model import NGCG
from .metrics import (
    constancy_score,
    discovery_rate,
    false_discovery_rate,
    f1_score_ngcg,
    conservation_violation,
    trajectory_diversity,
)

__version__ = "1.0.0"
__all__ = [
    "NGCG",
    "constancy_score",
    "discovery_rate",
    "false_discovery_rate",
    "f1_score_ngcg",
    "conservation_violation",
    "trajectory_diversity",
]
