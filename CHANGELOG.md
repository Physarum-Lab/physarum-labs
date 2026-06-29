# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-28

### Added
- `PhysarumLPLayer`: Differentiable linear programming via Physarum dynamics.
  Drop-in replacement for Sinkhorn-based optimal transport layers.
- `VariationalPhysarumSolver`: Wraps the base solver with Solé-Pla-Mauri 2025
  Lagrangian regularization (transport, sparsity, entropy terms).
- `VariationalNetworkMachine`: End-to-end demo module for graph routing,
  keypoint matching, and learned flow allocation.
- Three demos in `variational_network_machine.py`:
  - Maze: 4x4 grid routing
  - Matching: 8x10 keypoint matching
  - Learning Loop: Train affinities to match target distribution (KL drops 82.7%)
- CVXPY comparison: `cvxpy_comparison.py` validates Physarum solver against
  scipy.optimize.linprog on multiple problem sizes (max_iter=100 → <5% gap).
- 26 unit tests covering shapes, transport properties, gradients, convergence,
  batching, variational extension, edge cases, and nn.Module integration.

### Implementation Notes
- Independent of the SuperGlue codebase (which is under Magic Leap's
  non-commercial license). This reimplementation is MIT-licensed.
- Algorithm described in Meng, Ravi, Singh (AAAI 2021), arXiv:2004.14539.
- Variational interpretation from Solé & Pla-Mauri (Nov 2025),
  arXiv:2511.08531.
- Substrate inspiration from Schick et al. (PRX Life 2026),
  DOI: 10.1103/rv7g-d9kx.
- Substrate-agnostic framing from Pietak & Levin (iScience 2025),
  DOI: 10.1016/j.isci.2025.112536.

### Fixed
- Cost shift now uses min of un-augmented matrix (preserves cheap/expensive
  edge distinction).
- Random initialization uses wider range (0.5-1.0) so dynamics can move.
- Clamp moved to end of iteration (not during) to avoid killing dynamics.
- Ridge regularization added to linear system solve for stability.

### Known Limitations
- QP relaxation, not strict LP — small gap to true LP optimum even at high
  iteration counts.
- Convergence depends on problem size and cost distribution.
- Higher-dim batching uses a Python loop (could be vectorized for performance).

## [Unreleased]

### Planned
- Vectorized batched solver (eliminate Python loop).
- GPU optimization.
- Comparison to Sinkhorn and CVXPY for full benchmark.
- Blog post: "How a Slime Mold Solves Linear Programs"
- Examples: graph routing, image matching, optimal transport for ML.