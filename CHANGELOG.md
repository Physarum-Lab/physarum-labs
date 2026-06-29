# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-06-29

### Fixed
- `release.yml` was missing `contents: write` in the workflow permissions
  block, so the `softprops/action-gh-release@v2` step failed with `403
  Resource not accessible by integration` after a successful PyPI upload.
  Added `contents: write` alongside the existing `id-token: write`.

## [0.2.0] - 2026-06-29

### Changed
- **Package layout** — Source moved into a `physarum_labs/` package directory.
  Import paths change from `from physarum_lp import ...` to
  `from physarum_labs import ...`. The PyPI distribution name remains
  `physarum-labs` (unchanged).
  - `physarum_lp.py` → `physarum_labs/lp.py`
  - `variational_network_machine.py` → `physarum_labs/variational.py`
  - `cvxpy_comparison.py` → `examples/cvxpy_comparison.py` (no longer part
    of the installable package; ships in the source tree for documentation
    and benchmarking)
  - `tests/test_physarum_lp.py` → `tests/test_physarum_labs.py`
- **Public API** — `physarum_labs/__init__.py` re-exports `PhysarumLPLayer`,
  `VariationalPhysarumSolver`, and `VariationalNetworkMachine`. Users no
  longer need to know the internal module split.
- **Version bump to 0.2.0** — minor bump (per semver) because the import
  path is a breaking change for anyone who was importing from the source
  tree directly. PyPI install (`pip install physarum-labs`) is new, so
  there are no external consumers to migrate.

### Fixed
- `pyproject.toml` and `setup.py` previously referenced a non-existent
  `physarum_ai` package, which would have produced an empty wheel at
  build time. The new config points at the real `physarum_labs/` package
  directory, and `find_packages` excludes `tests/` and `examples/`.

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