"""
physarum_labs — Differentiable Linear Programming via Physarum dynamics
=========================================================================

A clean, MIT-licensed PyTorch implementation of Physarum-inspired
differentiable solvers, unifying four lines of recent research:

  1. Meng, Ravi, Singh (AAAI 2021) — Differentiable LP layer via Physarum dynamics
  2. Solé & Pla-Mauri (arXiv Nov 2025) — Lagrangian variational framework
  3. Schick et al. (PRX Life 2026) — Peristaltic mechanism (substrate inspiration)
  4. Pietak & Levin (iScience 2025) — Regulatory Network Machine paradigm

Public API
----------
    from physarum_labs import (
        PhysarumLPLayer,             # differentiable LP layer
        VariationalPhysarumSolver,   # LP layer + Lagrangian regularization
        VariationalNetworkMachine,   # end-to-end graph/transport learner
    )

License: MIT
"""

from __future__ import annotations

from physarum_labs.lp import PhysarumLPLayer, VariationalPhysarumSolver
from physarum_labs.variational import VariationalNetworkMachine

__all__ = [
    "PhysarumLPLayer",
    "VariationalPhysarumSolver",
    "VariationalNetworkMachine",
]

__version__ = "0.2.0"
