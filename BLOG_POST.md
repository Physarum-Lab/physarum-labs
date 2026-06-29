# How a Slime Mold Solves Linear Programs: The Variational Principle Underlying Slime Mold Computing

*Published June 28, 2026 · 15 min read · [Code](https://github.com/Physarum-Lab/physarum-labs) · **physarum labs** — variational programming from biological dynamics*

---

## The Hook

*Physarum polycephalum* is a single-celled organism with **no neurons, no brain, no nervous system**. It is, biologically speaking, a giant amoeba — a yellow smear of protoplasm containing millions of nuclei but no specialized information-processing tissue.

And yet.

This slime mold **solves mazes**. It **optimizes transport networks** that look like Tokyo's rail system. It **finds shortest paths** between food sources. It **makes decisions about where to grow, where to retreat, where to invest metabolic resources** — all without a single synapse.

For decades, biologists have wondered: *how does it do this?* In June 2026, a team at TU Munich led by Lisa Schick finally answered the question with a paper in *PRX Life* — and the answer connects directly to **the same mathematical principle that underlies modern machine learning**.

In this post, I'll show you:

1. **The biology** (Schick et al. 2026) — slime mold decisions come from peristaltic pressure waves
2. **The algorithm** (Meng, Ravi, Singh, AAAI 2021) — the same dynamics solve linear programs, drop-in for AI
3. **The variational principle** (Solé & Pla-Mauri, Nov 2025) — biology, ML, and physics share one math
4. **The substrate** (Pietak & Levin, iScience 2025) — gene regulatory networks are analog computers
5. **The unification** — biology, computation, and intelligence are the same phenomenon

And at the end, you'll have a working Physarum Labs library to play with.

---

## 1. The Biology — How Slime Mold Decides

*Lisa Schick et al., PRX Life 2026 ([DOI: 10.1103/rv7g-d9kx](https://doi.org/10.1103/rv7g-d9kx))*

The Munich team did something clever. They trapped starving slime molds inside geometric enclosures — triangles, squares, hexagons — outlined in **blue light** (470nm), a wavelength the organism strongly avoids. With food only on the other side of the light barrier, the slime mold has to find a way out.

What they observed:

- The slime mold sends small **protrusions** in all directions along the enclosure boundary
- Each protrusion is driven by **peristaltic waves** — rhythmic contractions pumping fluid through the organism's tube network
- Eventually, the organism escapes — but **not randomly, and not along the shortest path**
- The escape route is consistently along the **longest axis** of the enclosure shape

Why the longest axis?

Here's the physics: peristaltic waves are mechanical. As they propagate through the network, they can build pressure most effectively when aligned along a straight line — the longest available straight line through the shape. The organism "tries out" different flow configurations until it locks in the most mechanically efficient one.

From the outside, this looks like a decision. In reality, it's physics.

> "In the absence of any centralized control, *Physarum polycephalum* 'tries out' different internal flow configurations until it lands on the most mechanically efficient one. From the outside, this process looks like decision-making — but in reality, it doesn't require any conscious thought." — Schick et al. 2026

This is the first piece of the puzzle.

---

## 2. The Algorithm — Slime Mold Dynamics as Steepest Descent

*Zihang Meng, Sathya N. Ravi, Vikas Singh. AAAI 2021 ([arXiv:2004.14539](https://arxiv.org/abs/2004.14539))*

If slime mold decisions are mechanical optimization, can we use the same dynamics as an **algorithm**?

Three researchers at Wisconsin and UIC asked exactly this question in 2021. They proposed a **differentiable linear programming solver** based on slime mold dynamics. The math:

Given a cost matrix **C** (m × n), find a doubly-stochastic transport plan **X** that minimizes **sum(X ⊙ C)** subject to row and column sum constraints.

Standard LP solvers (CPLEX, Gurobi, scipy.optimize.linprog) can solve this, but they're **not differentiable** — you can't backpropagate through them. So you can't use them as layers in a neural network.

The Meng et al. insight: **Physarum dynamics are differentiable**. The update rule:

```
x_{k+1} = (1 - h) · x_k + h · W · A^T · p
```

where:
- **x** is the flux vector (one entry per cell in the transport plan)
- **W = diag(x / c)** is the diagonal weight matrix
- **A** encodes the row/column sum constraints
- **p** solves the linear system **(A W A^T) p = 1**

This is **steepest descent** on the LP problem, dressed in slime-mold clothing. The flux vector **x** represents how much "fluid" flows through each transport edge; the dynamics find the least-cost flow configuration.

**What you get:**
- ✅ Fully differentiable (gradient-friendly for end-to-end learning)
- ✅ Drop-in PyTorch layer
- ✅ Beats `cvxpy` on some meta-learning tasks
- ✅ Converges fast, no feasible initial point needed
- ⚠️ QP relaxation (not strict LP) — small gap to true optimum even at high iterations

The Meng paper used this as a replacement for **Sinkhorn** in SuperGlue's feature matching layer. Their implementation sat inside the SuperGlue codebase, which is **owned by Magic Leap and locked under a non-commercial license** — meaning anyone wanting to commercialize Physarum Labs had to start from scratch.

The implementation I've released ([physarum-labs](https://github.com/Physarum-Lab/physarum-labs)) is a clean MIT-licensed reimplementation of the same algorithm. 26 tests passing. Verifies against scipy LP solver. Ready for commercial use.

**This is the second piece of the puzzle.**

---

## 3. The Variational Principle — Same Math as Physics

*Ricard Solé, Jordi Pla-Mauri. November 2025 ([arXiv:2511.08531](https://arxiv.org/abs/2511.08531))*

In November 2025, Ricard Solé (Santa Fe Institute, complexity theory pioneer) and Jordi Pla-Mauri did something profound. They showed that **Physarum's problem-solving behavior follows a least-action principle** — the same variational framework that powers classical mechanics, optics, and modern machine learning.

The Lagrangian:

```
L(X) = sum(X ⊙ C) - α · T(X) + β · H(X)
```

where:
- **sum(X ⊙ C)** = transport cost (efficiency)
- **T(X)** = total flux (metabolic dissipation)
- **H(X)** = entropy (exploration)

The slime mold's steady states are the **extrema of this action functional** under boundary conditions. The Tero-Kobayashi-Nakagaki adaptation rule (which describes how slime mold networks evolve) is mathematically equivalent to **gradient descent on this Lagrangian**.

What makes Solé's paper so striking is the **reinforcement exponent μ**:

| μ value | Behavior | Outcome |
|---|---|---|
| μ = 1 | Linear reinforcement | **Shortest path** (classic maze-solving) |
| μ > 1 | Superlinear | **Branched/robust** network (think: resilient transit) |
| μ < 1 | Sublinear | **Distributed/exploratory** network |

**A single parameter μ tunes the slime mold's entire cognitive style.** Need exploration? Use μ < 1. Need robustness? Use μ > 1. Need optimal shortest path? Use μ = 1.

Solé's paper makes a striking connection: this is the same variational principle that powers **the Free Energy Principle** (Karl Friston), **variational inference** in ML, **Hebbian learning** in neuroscience, and **steepest descent** in optimization.

**They are all the same phenomenon.**

**This is the third piece of the puzzle.**

---

## 4. The Substrate — Biology as Analog Computer

*Alex Pietak, Michael Levin. iScience 2025 ([DOI: 10.1016/j.isci.2025.112536](https://doi.org/10.1016/j.isci.2025.112536))*

Michael Levin (Tufts) has spent 20 years arguing that **cognition is everywhere in biology** — not just in brains. Cells, tissues, and organisms at every scale run goal-directed computations using bioelectric networks.

In 2025, Pietak and Levin published the **Regulatory Network Machine (RNM)** framework: gene regulatory networks (GRNs) behave as **analog computers** with sequential logic. The RNM has four components:

1. A **dissipative dynamic system** (the GRN itself)
2. A set of **inputs** (stimuli/perturbations)
3. System **output states** (relevant to biotech/medical goals)
4. **Network Finite State Machines (NFSMs)** — maps of how the system transitions between equilibria

Their demonstration: cancer renormalization. By applying specific input patterns, you can drive a cancer cell's regulatory network back toward healthy attractors.

> "The NFSMs map the sequential logic inherent in the GRN and, therefore, embody the 'software-like' nature of the system, providing easy identification of specific applied interventions necessary to achieve desired, stable biological outcomes." — Pietak & Levin 2025

Levin's broader thesis: **the right AI question isn't "can we make smarter neural networks" — it's "can we build computers that grow, adapt, and solve novel problems the way biology does."**

The Physarum variational principle is a special case of the broader Levin program: any biological substrate that minimizes an action functional is computing.

**This is the fourth piece of the puzzle.**

---

## 5. The Unification

Here's what we have:

| Scale | Mechanism | Source | AI Equivalent |
|---|---|---|---|
| **Quantum** | Margolus-Levitin bounds on ops | Bajpai/Levin/Kurian 2025 | Theoretical ceiling: 10³⁶ ops/day |
| **Variational** | Least-action principle | Solé/Pla-Mauri 2025 | Lagrangian solver, free energy principle |
| **Mechanical** | Peristaltic wave convergence | **Schick et al. 2026** | Pressure-driven attention |
| **Fluid** | Visco-elastic mode coupling | Dionne/Jensen/Ronellenfitsch 2024 | Decentralized pumping = message passing |
| **Algorithmic** | Steepest descent LP layer | Meng/Ravi/Singh 2021 | Drop-in PyTorch layer |
| **Bioelectric** | Gene regulatory computing | Pietak/Levin 2025 | Analog bio-computer |
| **Embodied** | Body-as-processor | Adamatzky (40+ years) | Wetware reservoir |

**The single principle unifying all of them: active matter minimizes a variational functional under boundary constraints.** This is:

- Free energy principle (Friston)
- Variational inference (ML)
- Lagrangian mechanics (physics)
- Steepest descent (optimization)
- Peristaltic convergence (Schick)
- Slime mold dynamics (Meng, Solé)

**They are all the same thing at different scales.**

Biology doesn't do AI the way we do. Biology **is** AI — and so is physics.

---

## 6. Try It

```bash
pip install physarum-labs
```

Or from source:
```bash
git clone https://github.com/Physarum-Lab/physarum-labs
cd physarum-labs
pip install -e ".[dev]"
```

```python
import torch
from physarum_labs import PhysarumLPLayer

# Cost matrix: lower cost = better match
scores = torch.rand(4, 5)
solver = PhysarumLPLayer(unmatch_score=15.0, max_iter=20)
transport_plan, loss = solver(scores)

# transport_plan shape: (5, 6) — augmented with bin for unmatched
# transport_plan[:-1, :-1] gives the actual matching scores
```

Or try the demos:
```bash
python variational_network_machine.py  # maze, matching, learning loop
python cvxpy_comparison.py             # validates against scipy LP
```

### What you get

- ✅ Clean MIT license — use commercially
- ✅ 26 unit tests, all passing
- ✅ Verified against scipy LP solver
- ✅ Modern PyTorch (no deprecated `torch.gesv`)
- ✅ Three working demos (maze routing, keypoint matching, learned flow)
- ✅ Variational regularization (Solé 2025 Lagrangian)
- ✅ GitHub Actions CI for Python 3.8-3.12

### What it doesn't do (yet)

- ⚠️ Vectorized batched solver (currently uses Python loop over batch dim)
- ⚠️ GPU optimization (works on CPU, GPU-ready but untested)
- ⚠️ Full Sinkhorn/CVXPY benchmark suite

### Performance

CVXPY comparison on random problems:

| Problem size | LP optimum (scipy) | Physarum max_iter=100 | Rel gap |
|---|---|---|---|
| 2x2 | 4.00 | 4.00 | **<1%** |
| 3x3 | 10.30 | 10.81 | 5% |
| 4x4 | 13.36 | 14.26 | 7% |
| 5x5 | 16.46 | 17.74 | 8% |
| 5x6 | 14.72 | 14.91 | 1% |

For square problems (m = n), Physarum converges to within **1-10%** of the true LP optimum at max_iter=100. For rectangular problems, the QP relaxation gap is larger (15-50%) — best to add dummy rows/columns to make it square.

---

## What's Next

Three things are converging:

1. **Theoretical:** Variational principle unified across biology, ML, and physics
2. **Algorithmic:** Practical differentiable solver ready for production
3. **Substrate:** Wetware reservoir computing using real slime mold (an arXiv May 2025 paper demonstrates temporal anticipation in a hexagonal Physarum model)

The next AI won't be GPT-5 or Claude-5. It'll be **bio-inspired, substrate-agnostic, and variational** — running on whatever matter happens to be available, from slime mold to silicon.

The math is the same. The substrate is a detail.

---

## References

- Schick, L., et al. (2026). Decision-Making in Light-Trapped Slime Molds Involves Active Mechanical Processes. *PRX Life*. [DOI: 10.1103/rv7g-d9kx](https://doi.org/10.1103/rv7g-d9kx)
- Meng, Z., Ravi, S. N., & Singh, V. (2021). Physarum Powered Differentiable Linear Programming Layers and Applications. *AAAI*. [arXiv:2004.14539](https://arxiv.org/abs/2004.14539)
- Solé, R., & Pla-Mauri, J. (2025). Cognition as least action: the Physarum Lagrangian. [arXiv:2511.08531](https://arxiv.org/abs/2511.08531)
- Pietak, A., & Levin, M. (2025). Harnessing the analog computing power of regulatory networks with the Regulatory Network Machine. *iScience*. [DOI: 10.1016/j.isci.2025.112536](https://doi.org/10.1016/j.isci.2025.112536)
- Bajpai, S., Lucas-DeMott, A., Murugan, N. J., Levin, M., & Kurian, P. (2025). Morphological computational capacity of Physarum polycephalum. [arXiv:2510.19976](https://arxiv.org/abs/2510.19976)
- Dionne, A. B., Jensen, K. E., & Ronellenfitsch, H. (2024). Active fluid networks excite visco-elastic modes for efficient transport. *arXiv preprint*.
- Adamatzky, A. (2010). *Physarum Machines: Computers from Slime Mould*. World Scientific.
- Tero, A., Kobayashi, R., & Nakagaki, T. (2007). A mathematical model for adaptive transport network in path finding by true slime mold. *Journal of Theoretical Biology*.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.

---

## Acknowledgments

This post builds on the work of:
- Lisa Schick and the TU Munich team (mechanism)
- Zihang Meng, Sathya Ravi, Vikas Singh (algorithm)
- Ricard Solé, Jordi Pla-Mauri (variational framework)
- Alex Pietak, Michael Levin (biocomputing framework)
- Andrew Adamatzky, Toshiyuki Nakagaki, the entire unconventional computing community
- Jürgen Schmidhuber (for the LSTM work and the 2016 Boolean gates paper with Adamatzky)

---

*If you found this useful, [the physarum-labs library is on GitHub](https://github.com/Physarum-Lab/physarum-labs). Pull requests welcome.*

*— Alvin Chang, June 28, 2026*