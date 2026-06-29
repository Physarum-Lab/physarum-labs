# Contributing to physarum-labs

Thanks for your interest in contributing! This project implements
Physarum-inspired differentiable linear programming, following the work
of Meng, Ravi & Singh (AAAI 2021) and the variational interpretation
of Solé & Pla-Mauri (Nov 2025).

## Quick start

```bash
git clone https://github.com/Physarum-Lab/Physarum-Lab/physarum-labs.git
cd physarum-labs
pip install -e ".[dev]"
pytest tests/ -v
```

## Code style

- Python 3.8+
- Type hints encouraged
- Docstrings for public functions
- Black formatting (88 char line length)

## Areas where contributions are most welcome

### 1. Performance optimization
The current implementation uses a Python loop over batches for the inner
LP solve. This could be vectorized using batched matrix operations for
significant speedup on GPU.

### 2. CVXPY/SciPy benchmarks
Add more comparison points in `cvxpy_comparison.py`:
- Compare against Sinkhorn (optimal transport baseline)
- Compare against commercial solvers (Gurobi, CPLEX)
- Test on larger problems (10x10, 50x50)

### 3. Applications
New demos in `variational_network_machine.py`:
- Image feature matching (replace SuperGlue's Sinkhorn)
- Graph neural network with Physarum aggregation
- Reinforcement learning with Physarum-based planning

### 4. Theory
- Formal convergence proofs for the QP relaxation
- Connection to free energy principle (Friston)
- Extension to non-linear cost functions
- Stochastic variants (noisy Physarum dynamics)

### 5. Documentation
- More worked examples
- Tutorial notebooks (Jupyter)
- Video walkthroughs

## How to contribute

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest tests/ -v`)
5. Submit a pull request

## Reporting issues

- Use GitHub Issues
- Include a minimal reproducible example
- Mention your PyTorch version and OS

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.

## Contact

- GitHub: @alfie
- Telegram: @alvinchangtech

## Acknowledgments

This work builds on:
- Meng, Ravi, Singh (AAAI 2021) — algorithm
- Solé & Pla-Mauri (Nov 2025) — variational framework
- Schick et al. (PRX Life 2026) — substrate inspiration
- Pietak & Levin (iScience 2025) — substrate-agnostic framing
- Adamatzky (Bristol) — unconventional computing canon
- Tero, Kobayashi, Nakagaki — original Physarum network model

## Privacy

This repo previously exposed a local-config email (`alvin@localhost`) in this file. Scrubbed 2026-06-29 per privacy audit. Maintainer contact: open a GitHub issue or PR.
