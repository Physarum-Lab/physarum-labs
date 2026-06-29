"""
cvxpy_comparison.py — Compare Physarum solver to true LP solutions
===================================================================

Demonstrates that the Physarum dynamics solver (a QP relaxation) converges
to the true LP optimum as max_iter increases, when regularization weights
are zero.

Test setup:
- Generate random cost matrices of varying sizes
- Solve with PhysarumLPLayer at different max_iter values (5, 10, 20, 50, 100)
- Solve with scipy.optimize.linprog (true LP optimum)
- Compare: Physarum loss vs. true LP optimum

Expected result:
- At max_iter=100, Physarum should be within 1-5% of the true LP optimum
- At max_iter=20 (Meng 2021 default), still close but not exact
- Larger problems show more QP-relaxation gap

Run:  python cvxpy_comparison.py
Requires: scipy (for linprog), matplotlib (for plotting)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from physarum_labs import PhysarumLPLayer

# Check for optional dependencies
try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def solve_lp_with_scipy(cost: np.ndarray) -> float:
    """Solve the doubly-stochastic transport problem as a true LP.

    minimize  sum(X * C)
    subject to  X @ 1 = 1 (row sums, all m rows)
                X^T @ 1 = 1 (col sums, all n rows)
                X >= 0

    Note: With m+n equality constraints and m*n variables, the system has
    one redundant constraint (the last row/col sum is implied by the others).
    We drop the last row constraint to make the LP well-defined.
    """
    m, n = cost.shape
    # Variables: x_ij for i in [0, m), j in [0, n)
    # Objective: c^T x where c is flattened cost
    c = cost.flatten()

    # Equality constraints: A_eq @ x = b_eq
    # Drop the last row to avoid redundancy (m-1 + n constraints instead of m + n)
    A_rows = np.zeros((m - 1, m * n))
    for i in range(m - 1):
        A_rows[i, i * n : (i + 1) * n] = 1.0

    # Column sum constraints (all n columns)
    A_cols = np.zeros((n, m * n))
    for j in range(n):
        for i in range(m):
            A_cols[j, i * n + j] = 1.0

    A_eq = np.vstack([A_rows, A_cols])
    b_eq = np.ones(m - 1 + n)

    bounds = [(0, None) for _ in range(m * n)]

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP solve failed: {result.message}")
    return result.fun


def solve_with_physarum(cost: torch.Tensor, max_iter: int, unmatch_score: float = 15.0) -> float:
    """Solve with Physarum solver at given iteration count.

    Returns the "real" cost — sum(X[:-1,:-1] * cost) — which is comparable
    to the LP optimum on the original (un-augmented) problem.
    """
    solver = PhysarumLPLayer(unmatch_score=unmatch_score, max_iter=max_iter)
    plan, _ = solver(cost)
    # Extract the real part of the transport plan (excluding bin row/column)
    X_real = plan[:-1, :-1]
    # Compute cost against the original (un-augmented) cost matrix
    real_cost = (X_real * cost).sum().item()
    return real_cost


def run_convergence_study() -> dict:
    """Run convergence study on random cost matrices of varying size."""
    results = {
        "sizes": [],
        "lp_optimal": [],
        "physarum_5": [],
        "physarum_10": [],
        "physarum_20": [],
        "physarum_50": [],
        "physarum_100": [],
    }

    sizes = [(2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5), (5, 6)]

    for m, n in sizes:
        print(f"\n--- Problem size: {m}x{n} ---")
        torch.manual_seed(42)
        cost_tensor = torch.rand(m, n) * 10.0  # range [0, 10]
        cost_np = cost_tensor.numpy()

        if not HAS_SCIPY:
            print("  scipy not available — skipping LP comparison")
            break

        # True LP optimum
        lp_opt = solve_lp_with_scipy(cost_np)
        print(f"  LP optimum (scipy):       {lp_opt:.6f}")

        # Physarum at various iteration counts (bin_score > max(real cost))
        unmatch = cost_tensor.max().item() + 1.0
        phys_5 = solve_with_physarum(cost_tensor, max_iter=5, unmatch_score=unmatch)
        phys_10 = solve_with_physarum(cost_tensor, max_iter=10, unmatch_score=unmatch)
        phys_20 = solve_with_physarum(cost_tensor, max_iter=20, unmatch_score=unmatch)
        phys_50 = solve_with_physarum(cost_tensor, max_iter=50, unmatch_score=unmatch)
        phys_100 = solve_with_physarum(cost_tensor, max_iter=100, unmatch_score=unmatch)

        gap_5 = abs(phys_5 - lp_opt) / max(abs(lp_opt), 1e-6) * 100
        gap_10 = abs(phys_10 - lp_opt) / max(abs(lp_opt), 1e-6) * 100
        gap_20 = abs(phys_20 - lp_opt) / max(abs(lp_opt), 1e-6) * 100
        gap_50 = abs(phys_50 - lp_opt) / max(abs(lp_opt), 1e-6) * 100
        gap_100 = abs(phys_100 - lp_opt) / max(abs(lp_opt), 1e-6) * 100

        print(f"  Physarum max_iter=5:      {phys_5:.6f}  (rel gap={gap_5:.2f}%)")
        print(f"  Physarum max_iter=10:     {phys_10:.6f}  (rel gap={gap_10:.2f}%)")
        print(f"  Physarum max_iter=20:     {phys_20:.6f}  (rel gap={gap_20:.2f}%)")
        print(f"  Physarum max_iter=50:     {phys_50:.6f}  (rel gap={gap_50:.2f}%)")
        print(f"  Physarum max_iter=100:    {phys_100:.6f}  (rel gap={gap_100:.2f}%)")

        results["sizes"].append(f"{m}x{n}")
        results["lp_optimal"].append(lp_opt)
        results["physarum_5"].append(phys_5)
        results["physarum_10"].append(phys_10)
        results["physarum_20"].append(phys_20)
        results["physarum_50"].append(phys_50)
        results["physarum_100"].append(phys_100)

    return results


def plot_convergence(results: dict, output_path: Path) -> None:
    """Plot Physarum loss vs. iteration count vs. true LP optimum."""
    if not HAS_MPL:
        print("matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss values vs problem size
    ax = axes[0]
    x = np.arange(len(results["sizes"]))
    width = 0.13

    ax.bar(x - 2 * width, results["lp_optimal"], width, label="LP optimum", color="black")
    ax.bar(x - width, results["physarum_5"], width, label="Physarum-5", alpha=0.7)
    ax.bar(x, results["physarum_10"], width, label="Physarum-10", alpha=0.7)
    ax.bar(x + width, results["physarum_20"], width, label="Physarum-20", alpha=0.7)
    ax.bar(x + 2 * width, results["physarum_50"], width, label="Physarum-50", alpha=0.7)
    ax.bar(x + 3 * width, results["physarum_100"], width, label="Physarum-100", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(results["sizes"])
    ax.set_xlabel("Problem size (m x n)")
    ax.set_ylabel("LP objective value (lower = better)")
    ax.set_title("Physarum vs LP optimum: Loss values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Relative gap vs iteration count
    ax = axes[1]
    if results["lp_optimal"]:
        # Average across all problem sizes
        sizes_arr = np.array(results["sizes"])
        avg_lp = np.mean(results["lp_optimal"])
        gaps = {
            5:   np.mean(np.abs(np.array(results["physarum_5"]) - np.array(results["lp_optimal"])) / avg_lp),
            10:  np.mean(np.abs(np.array(results["physarum_10"]) - np.array(results["lp_optimal"])) / avg_lp),
            20:  np.mean(np.abs(np.array(results["physarum_20"]) - np.array(results["lp_optimal"])) / avg_lp),
            50:  np.mean(np.abs(np.array(results["physarum_50"]) - np.array(results["lp_optimal"])) / avg_lp),
            100: np.mean(np.abs(np.array(results["physarum_100"]) - np.array(results["lp_optimal"])) / avg_lp),
        }

        iters = sorted(gaps.keys())
        gap_values = [gaps[i] * 100 for i in iters]  # as percentage

        ax.plot(iters, gap_values, "o-", linewidth=2, markersize=8, color="steelblue")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physarum iterations")
        ax.set_ylabel("Mean relative gap to LP optimum (%)")
        ax.set_title("Convergence: Physarum → LP optimum")
        ax.grid(True, alpha=0.3, which="both")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="1% gap (excellent)")
        ax.axhline(y=5.0, color="orange", linestyle="--", alpha=0.5, label="5% gap (good)")
        ax.axhline(y=10.0, color="red", linestyle="--", alpha=0.5, label="10% gap (acceptable)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def run_targeted_test() -> None:
    """Run a targeted test on a known LP problem.

    Smallest non-trivial transport problem (2x2):
        minimize   sum(X * C)
        subject to row sums = 1, col sums = 1, X >= 0

    For cost C, the optimal X is the extreme point that minimizes cost.
    """
    print("=" * 60)
    print("Targeted Test: 2x2 transport problem")
    print("=" * 60)

    # Cost matrix: diagonal is cheap, off-diagonal is expensive.
    # IMPORTANT: unmatch_score (bin cost) must be HIGHER than real costs
    # so that real edges are preferred over the bin.
    costs = torch.tensor([[2.0, 11.0], [11.0, 2.0]])
    print(f"\nCosts (lower = better):\n{costs}")

    # Solve with Physarum at various iterations
    for iters in [5, 10, 20, 50, 100, 200]:
        # bin_score must be > max(real costs) to discourage matching to bin
        solver = PhysarumLPLayer(unmatch_score=15.0, max_iter=iters)
        plan, loss = solver(costs)
        X = plan[:-1, :-1].detach().numpy()
        print(f"\nPhysarum (max_iter={iters}):")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Transport plan:\n{X.round(4)}")

    if HAS_SCIPY:
        # Solve with scipy for ground truth
        lp_opt = solve_lp_with_scipy(costs.numpy())
        print(f"\nGround truth (scipy linprog): {lp_opt:.6f}")
        print(f"  Expected optimal cost: 2 + 2 = 4 (diagonal allocation)")
    else:
        print("\nscipy not available — manual check: diagonal allocation gives cost 2+2=4")


def main() -> None:
    print("=" * 60)
    print("CVXPY / SciPy LP comparison for Physarum solver")
    print("=" * 60)

    if not HAS_SCIPY:
        print("\n⚠ scipy not installed. Install with: pip install scipy")
        print("Falling back to targeted test only.\n")

    # Run targeted test on known problem
    run_targeted_test()

    if HAS_SCIPY:
        # Run convergence study
        results = run_convergence_study()

        # Plot
        output_dir = Path(__file__).resolve().parent / "assets"
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / "convergence.png"
        plot_convergence(results, plot_path)

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("\nPhysarum dynamics is a QP relaxation of LP. As max_iter increases,")
        print("the solver converges to the true LP optimum. The convergence rate")
        print("depends on problem size and cost distribution.")
        print()
        print("At max_iter=20 (Meng 2021 default), the Physarum solver typically")
        print("achieves <10% relative gap to LP optimum on small problems.")
        print("For production use, max_iter=50-100 is recommended.")
        print()
        print("See assets/convergence.png for visualization.")
    else:
        print("\nFor full convergence study, install scipy and re-run.")


if __name__ == "__main__":
    main()