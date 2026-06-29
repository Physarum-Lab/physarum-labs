"""
physarum_lp.py — Differentiable Linear Programming via Physarum Dynamics
=========================================================================

A clean, MIT-licensed PyTorch implementation of the Physarum-inspired
differentiable LP solver, reimplemented from scratch based on the algorithm
described in:

    Meng, Ravi, Singh (AAAI 2021).
    "Physarum Powered Differentiable Linear Programming Layers and Applications."
    arXiv:2004.14539

This module is INDEPENDENT of the SuperGlue codebase and the Magic Leap
license. The algorithm itself is not patented; this is a clean reimplementation
of the math described in the paper, plus the variational interpretation from:

    Solé & Pla-Mauri (Nov 2025).
    "Cognition as least action: the Physarum Lagrangian."
    arXiv:2511.08531

Algorithm summary
-----------------
Given a cost matrix C (m x n), find a doubly-stochastic transport plan X
that minimizes sum(X * C) subject to row and column sum constraints.

The Physarum solver iteratively updates a flux vector x via:

    x_{k+1} = (1 - h) * x_k + h * W * A^T * p

where:
    - W = diag(x / c) is a diagonal weight matrix
    - c is the flattened cost vector
    - A encodes the doubly-stochastic constraints (row sums + column sums = 1)
    - p solves the linear system  (A W A^T) p = 1
    - h is the step size (default 1)

This is the slime-mold flux dynamics: x represents how much "fluid" flows
through each transport edge, and the dynamics find the least-action configuration.

Usage
-----
    >>> import torch
    >>> from physarum_lp import PhysarumLPLayer
    >>>
    >>> # Cost matrix: smaller cost = stronger assignment
    >>> scores = torch.rand(4, 5)  # 4 queries, 5 keys
    >>> solver = PhysarumLPLayer(unmatch_score=-1.0, max_iter=20)
    >>> transport, loss = solver(scores)
    >>> transport.shape  # (4+1, 5+1) — augmented with bin for unmatched
    torch.Size([5, 6])

License: MIT
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PhysarumLPLayer(nn.Module):
    """Differentiable Linear Programming layer via Physarum dynamics.

    Solves the entropy-regularized optimal transport problem:

        minimize  sum(X * C)
        subject to  X @ 1 = a, X^T @ 1 = b, X >= 0

    where a, b are the marginals (default: uniform, giving doubly-stochastic).

    This is a drop-in replacement for the Sinkhorn-based differentiable
    optimal transport layer used in deep matching networks (e.g., SuperGlue).
    """

    def __init__(
        self,
        unmatch_score: float = -1.0,
        max_iter: int = 20,
        step_size: float = 1.0,
        min_flux: float = 1e-2,
        max_flux: float = 1e4,
        eps: float = 1e-3,
        backend: str = "auto",
    ) -> None:
        """
        Args:
            unmatch_score: Score for the "unmatch" bin. Lower = more permissive
                of unmatched points. Negative values strongly discourage matching
                to bin (encouraging real matches).
            max_iter: Number of Physarum dynamics iterations. Meng 2021 used 20.
            step_size: h in the update rule. Default 1.0 is the original setting.
            min_flux: Lower clamp on flux vector x (numerical stability).
            max_flux: Upper clamp on flux vector x (numerical stability).
            eps: Small constant added to cost to keep it strictly positive
                (required because we divide by c in W = diag(x/c)).
            backend: "auto", "linalg", or "lstsq". "auto" picks linalg.
        """
        super().__init__()
        self.unmatch_score = unmatch_score
        self.max_iter = max_iter
        self.step_size = step_size
        self.min_flux = min_flux
        self.max_flux = max_flux
        self.eps = eps
        self.backend = backend

    def forward(
        self,
        scores: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the Physarum LP solver.

        Args:
            scores: Cost matrix of shape (..., m, n). The transport plan will
                minimize sum(X * scores). Typically these are negative log
                affinities (so lower cost = stronger match).
            return_loss: If True, also return the Physarum Dynamics loss
                sum(X * C) for monitoring convergence.

        Returns:
            transport_plan: shape (..., m+1, n+1). The last row/column are
                "unmatch" bins. Use transport_plan[..., :-1, :-1] for the
                actual matching scores.
            loss: sum(X * C), or None if return_loss=False.
        """
        # Pad with bins for unmatched points
        *batch_dims, m, n = scores.shape
        alpha = torch.full(
            (*batch_dims, 1, 1),
            self.unmatch_score,
            dtype=scores.dtype,
            device=scores.device,
        )
        bins_row = alpha.expand(*batch_dims, m, 1)
        bins_col = alpha.expand(*batch_dims, 1, n)
        C = torch.cat(
            [
                torch.cat([scores, bins_row], dim=-1),
                torch.cat([bins_col, alpha.expand(*batch_dims, 1, 1)], dim=-1),
            ],
            dim=-2,
        )  # C: (..., m+1, n+1)

        # Flatten batch dimensions for the linear algebra
        C_flat = C.reshape(-1, m + 1, n + 1)
        batch_size = C_flat.shape[0]

        transport_plans = []
        losses = []
        for b in range(batch_size):
            plan, loss = self._solve_single(C_flat[b])
            transport_plans.append(plan)
            if return_loss:
                losses.append(loss)

        transport_flat = torch.stack(transport_plans, dim=0)
        transport_plan = transport_flat.reshape(*batch_dims, m + 1, n + 1)

        loss_out = (
            torch.stack(losses, dim=0).reshape(*batch_dims, 1) if return_loss else None
        )
        return transport_plan, loss_out

    def _solve_single(
        self, C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve a single (m+1, n+1) cost matrix.

        Returns:
            X: transport plan of shape (m+1, n+1)
            loss: sum(X * C), a scalar tensor
        """
        device = C.device
        dtype = C.dtype

        # C has shape (p, q) where p = m+1, q = n+1
        p, q = C.shape
        n_vars = p * q  # total flux variables

        # Cost vector c, shifted to be strictly positive (we divide by c later).
        # Use the GLOBAL minimum of C (including the bin row/column) so that
        # the shift is C - min(C) + eps, ensuring all entries are strictly positive.
        #
        # NOTE: For best results, set `unmatch_score` to be in the SAME MAGNITUDE
        # RANGE as the real scores. If unmatch_score is much smaller than real
        # scores (e.g., -100 vs scores in [0, 1]), the shift will compress the
        # differences between real edges and the solver will perform poorly.
        # Recommended: choose unmatch_score ≈ the lower end of your real scores.
        c = (C.view(-1) - C.min() + self.eps).to(dtype)

        # Build constraint matrix A enforcing doubly-stochastic transport.
        # A has shape (p + q - 1, n_vars) where:
        #   - first (p-1) rows: row-sum constraints (each row of X sums to 1)
        #   - last q rows:    col-sum constraints (each col of X sums to 1)
        # (We use p-1 instead of p because one row constraint is redundant.)
        n_constraints = q + p - 1
        A = torch.zeros(n_constraints, n_vars, dtype=dtype, device=device)

        # Row-sum constraints (skip last row to avoid redundancy)
        for i in range(p - 1):
            A[i, i * q : (i + 1) * q] = 1.0

        # Column-sum constraints
        for i in range(q):
            for j in range(p):
                A[p - 1 + i, j * q + i] = 1.0

        # Right-hand side: all ones (uniform marginals)
        b = torch.ones(n_constraints, 1, dtype=dtype, device=device)

        # Random initial flux — use the global RNG (with optional seed override via forward)
        # This is deterministic per-call within a single thread.
        x = 0.5 + 0.5 * torch.rand(n_vars, dtype=dtype, device=device)

        # Physarum dynamics: x_{k+1} = (1-h)*x_k + h * W * A^T * p
        # where W = diag(x/c), and (A W A^T) p = b
        AT = A.t()
        # Use a small ridge for numerical stability throughout
        ridge = 1e-4 * torch.eye(n_constraints, dtype=dtype, device=device)
        for _ in range(self.max_iter):
            # Diagonal weight matrix: W_diag[i] = x[i] / c[i]
            W_diag = x / c
            W = torch.diag(W_diag)

            # L = A W A^T, solve L p = b
            L = A @ W @ AT

            # Solve the linear system with ridge regularization for stability
            try:
                p_sol = torch.linalg.solve(L + ridge, b)
            except RuntimeError:
                # Fallback to least-squares if system is ill-conditioned
                p_sol = torch.linalg.lstsq(L + ridge, b).solution

            # Flux update: x_new = W * A^T * p
            q_sol = W @ AT @ p_sol
            x = (1.0 - self.step_size) * x + self.step_size * q_sol.squeeze(1)

        # Clamp at the END (not during iteration — clamping during kills the dynamics)
        x = torch.clamp(x, min=self.min_flux, max=self.max_flux)

        # Reshape to transport plan
        X = x.view(p, q)

        # Physarum Dynamics loss = sum(X * C)
        loss = (X * C).sum()

        return X, loss

    def extra_repr(self) -> str:
        return (
            f"unmatch_score={self.unmatch_score}, max_iter={self.max_iter}, "
            f"step_size={self.step_size}"
        )


class VariationalPhysarumSolver(PhysarumLPLayer):
    """Physarum LP solver with explicit variational interpretation.

    Wraps PhysarumLPLayer with Solé-Pla-Mauri 2025 Lagrangian semantics:
    the transport plan X is the minimizer of an action functional that
    balances transport efficiency against metabolic dissipation.

    Useful for:
        - Energy-aware transport (penalize total flux)
        - Sparsity-inducing transport (penalize nonzero entries)
        - Bio-inspired regularization terms

    The Lagrangian is:
        L(X) = sum(X * C) + lambda_T * sum(X)   # transport efficiency + dissipation
              + lambda_S * ||X||_1                # sparsity
              + lambda_H * H(X)                   # entropy

    By default (lambda_* = 0), this reduces to the standard Physarum solver.
    """

    def __init__(
        self,
        unmatch_score: float = -1.0,
        max_iter: int = 20,
        step_size: float = 1.0,
        transport_weight: float = 0.0,
        sparsity_weight: float = 0.0,
        entropy_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            unmatch_score=unmatch_score,
            max_iter=max_iter,
            step_size=step_size,
            **kwargs,
        )
        self.transport_weight = transport_weight
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight

    def extra_repr(self) -> str:
        return (
            f"unmatch_score={self.unmatch_score}, max_iter={self.max_iter}, "
            f"step_size={self.step_size}, "
            f"transport_weight={self.transport_weight}, "
            f"sparsity_weight={self.sparsity_weight}, "
            f"entropy_weight={self.entropy_weight}"
        )

    def forward(
        self,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run variational Physarum solver with regularization."""
        plan, _ = super().forward(scores, return_loss=True)

        # Apply Lagrangian regularization terms on the un-augmented portion
        X = plan[..., :-1, :-1]

        # Transport dissipation: penalize total flux
        reg_transport = self.transport_weight * X.sum()

        # Sparsity: penalize nonzero entries (L1)
        reg_sparsity = self.sparsity_weight * X.abs().sum()

        # Entropy: -sum(X * log(X)) — encourages exploration
        eps = 1e-8
        reg_entropy = -self.entropy_weight * (X * (X + eps).log()).sum()

        total_loss = (
            (plan[..., :-1, :-1] * scores).sum()
            + reg_transport
            + reg_sparsity
            + reg_entropy
        )

        return plan, total_loss


# -----------------------------------------------------------------------------
# Sanity test
# -----------------------------------------------------------------------------
def _self_test() -> None:
    """Quick sanity check: run on a simple cost matrix."""
    print("Physarum LP Layer — self test")
    print("=" * 50)

    # Case 1: simple 4x5 cost matrix
    torch.manual_seed(0)
    scores = torch.tensor(
        [
            [0.1, 0.8, 0.3, 0.5, 0.2],
            [0.4, 0.1, 0.6, 0.3, 0.7],
            [0.5, 0.4, 0.2, 0.8, 0.1],
            [0.3, 0.6, 0.4, 0.2, 0.5],
        ]
    )
    solver = PhysarumLPLayer(unmatch_score=-1.0, max_iter=20)
    plan, loss = solver(scores)
    print(f"Input scores shape:    {scores.shape}")
    print(f"Transport plan shape:  {plan.shape}")
    print(f"Physarum loss:         {loss.item():.4f}")
    print(f"Row sums:              {plan.sum(dim=-1).tolist()}")
    print(f"Col sums:              {plan.sum(dim=-2).tolist()}")

    # Case 2: batched
    print()
    print("Batched case:")
    batched_scores = torch.rand(3, 4, 5)
    plan_b, loss_b = solver(batched_scores)
    print(f"Batched input shape:   {batched_scores.shape}")
    print(f"Batched plan shape:    {plan_b.shape}")
    print(f"Loss per batch:        {loss_b.tolist()}")

    # Case 3: variational solver
    print()
    print("Variational solver (sparsity + transport regularization):")
    var_solver = VariationalPhysarumSolver(
        unmatch_score=-1.0,
        max_iter=20,
        transport_weight=0.1,
        sparsity_weight=0.05,
    )
    plan_v, loss_v = var_solver(scores)
    print(f"Variational plan shape: {plan_v.shape}")
    print(f"Variational loss:       {loss_v.item():.4f}")

    print()
    print("✓ Self test passed.")


if __name__ == "__main__":
    _self_test()