"""
variational_network_machine.py — Variational Network Machine Demo
=================================================================

Demonstrates how the Physarum-inspired differentiable LP solver combines
with Solé-Pla-Mauri 2025's Lagrangian framework to form a "Variational
Network Machine" — a substrate-agnostic computational primitive that:

  1. Encodes any graph/network structure as a transport problem
  2. Solves it via Physarum dynamics (slime-mold-inspired gradient descent)
  3. Treats the solution as the minimizer of a least-action functional

This unifies:
  - Meng 2021 (algorithmic)
  - Solé 2025 (variational)
  - Schick 2026 (mechanism)
  - Pietak/Levin 2025 (cognitive substrate)

Use cases:
  - Differentiable graph/network optimization as a PyTorch layer
  - Topology-aware resource allocation
  - Path planning with learned costs
  - Decision networks (which path is "best" given learned affinities)

Run:  python variational_network_machine.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physarum_labs.lp import PhysarumLPLayer, VariationalPhysarumSolver


class VariationalNetworkMachine(nn.Module):
    """A differentiable substrate for variational optimization on graphs.

    Takes a learned affinity matrix (edge weights) and a problem definition
    (source/sink nodes), then solves for the optimal transport plan via
    Physarum dynamics. The transport plan can be read as the network's
    "decision" — which paths it chooses, how it allocates flow, etc.

    The key insight: we work in LOG-SPACE for stability. The transport plan
    is the negative of the LP solution (since we minimize cost, the LP output
    is high where the cost is high — i.e. NOT where flow should go). The
    "match probabilities" are -X.

    Args:
        n_nodes: Number of nodes in the network.
        source_nodes: Indices of source nodes (where flow originates).
        sink_nodes:   Indices of sink nodes (where flow terminates).
        max_iter:     Number of Physarum dynamics iterations.
        regularization: "none", "sparsity", "transport", "entropy", or "full".
    """

    def __init__(
        self,
        n_nodes: int,
        source_nodes: torch.Tensor,
        sink_nodes: torch.Tensor,
        max_iter: int = 20,
        regularization: str = "full",
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.register_buffer("source_nodes", source_nodes)
        self.register_buffer("sink_nodes", sink_nodes)
        self.max_iter = max_iter

        # Learnable LOG-affinity matrix — the "edge weights" of the network.
        # Higher values = stronger match (lower cost).
        # Initialized small so the dynamics actually have gradients to work with.
        self.log_affinity = nn.Parameter(
            torch.randn(n_nodes, n_nodes) * init_scale
        )

        # Pick the right solver based on regularization
        reg_map = {
            "none":       (0.0, 0.0, 0.0),
            "sparsity":   (0.0, 0.1, 0.0),
            "transport":  (0.1, 0.0, 0.0),
            "entropy":    (0.0, 0.0, 0.1),
            "full":       (0.05, 0.05, 0.1),
        }
        if regularization not in reg_map:
            raise ValueError(f"Unknown regularization: {regularization}")
        transport_w, sparsity_w, entropy_w = reg_map[regularization]

        self.solver = VariationalPhysarumSolver(
            unmatch_score=-10.0,
            max_iter=max_iter,
            transport_weight=transport_w,
            sparsity_weight=sparsity_w,
            entropy_weight=entropy_w,
        )

    def forward(self) -> dict:
        """Compute the network's optimal transport plan.

        Returns:
            dict with:
                match_probs:  (n_nodes, n_nodes) — match probabilities (softmax-style)
                transport_plan: (n_nodes, n_nodes) — LP transport plan (= -match + offset)
                cost:          (n_nodes, n_nodes) — cost matrix (lower = better match)
                loss:          scalar — total Lagrangian
        """
        # Convert log-affinity to cost. Higher log_affinity = lower cost.
        # We negate and add a small positive offset to keep cost strictly positive.
        cost = -self.log_affinity + 5.0

        # Solve the LP via Physarum dynamics (minimizes sum(X * cost))
        plan, loss = self.solver(cost)

        # Extract the un-augmented transport plan
        X = plan[:-1, :-1]

        # Convert LP solution to "match probabilities" — values where match is strong
        # (low cost) get HIGH X, so we negate and apply a temperature.
        # In practice, the natural output of the LP IS a probability-like quantity
        # when properly normalized; we use a softmax over rows for clean gradients.
        match_logits = -X / 0.1  # temperature 0.1
        match_probs = F.softmax(match_logits, dim=-1)

        # Compute network metrics
        total_flow = X.sum()
        path_entropy = -(X * (X + 1e-8).log()).sum()
        n_active_edges = (X > 0.01).sum()

        return {
            "match_probs": match_probs,
            "transport_plan": X,
            "cost": cost,
            "loss": loss,
            "total_flow": total_flow,
            "path_entropy": path_entropy,
            "n_active_edges": n_active_edges,
        }


def _demo_maze() -> None:
    """Demo: solve a small maze-like transport problem.

    Network: 4x4 grid, with sources in the top-left and sinks in the
    bottom-right. The network should learn to route flow through the
    grid in a path-like manner.
    """
    print("=" * 60)
    print("Variational Network Machine — Maze Demo")
    print("=" * 60)

    # 4x4 grid = 16 nodes, numbered 0..15
    n_nodes = 16
    source_nodes = torch.tensor([0])   # top-left corner
    sink_nodes = torch.tensor([15])    # bottom-right corner

    torch.manual_seed(42)
    machine = VariationalNetworkMachine(
        n_nodes=n_nodes,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        max_iter=20,
        regularization="sparsity",
    )

    # Forward pass — see what the untrained network produces
    print("\n[Untrained network — random affinities]")
    result = machine()
    print(f"  Loss:                {result['loss'].item():.4f}")
    print(f"  Total flow:          {result['total_flow'].item():.4f}")
    print(f"  Path entropy:        {result['path_entropy'].item():.4f}")
    print(f"  Active edges (>0.01): {result['n_active_edges'].item()}")

    # Train: encourage flow to concentrate on a single path
    print("\n[Training — concentrate flow on shortest paths]")
    optimizer = torch.optim.Adam(machine.parameters(), lr=0.05)

    for step in range(80):
        optimizer.zero_grad()
        result = machine()

        # Loss: minimize transport cost + concentrate on a few edges
        cost_loss = (result["transport_plan"] * result["cost"]).sum()
        # Encourage concentration via negative max-entropy on match probs
        sparsity_loss = (result["match_probs"] ** 2).sum()  # L2 on probs → sparser
        loss = cost_loss + 0.5 * sparsity_loss

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(machine.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(
                f"  Step {step:3d}: loss={loss.item():.4f}, "
                f"active_edges={result['n_active_edges'].item()}, "
                f"entropy={result['path_entropy'].item():.3f}"
            )

    # Final plan
    print("\n[Final transport plan (row = source node, col = dest node)]")
    plan = result["transport_plan"].detach().numpy()
    print("Nonzero entries (threshold 0.01):")
    for i in range(n_nodes):
        for j in range(n_nodes):
            if plan[i, j] > 0.01:
                print(f"  {i:2d} -> {j:2d}: {plan[i, j]:.4f}")


def _demo_matching() -> None:
    """Demo: use as a differentiable matching layer.

    Given a set of learned keypoint affinities, find the optimal matching
    — same use case as Meng 2021 in SuperGlue.
    """
    print()
    print("=" * 60)
    print("Variational Network Machine — Matching Demo")
    print("=" * 60)

    # Simulate keypoint matching: 8 source keypoints, 10 dest keypoints
    n_src, n_dst = 8, 10
    scores = torch.randn(n_src, n_dst)

    solver = VariationalPhysarumSolver(
        unmatch_score=-1.0,
        max_iter=20,
        transport_weight=0.01,
        sparsity_weight=0.1,
    )

    plan, loss = solver(scores)
    X = plan[:-1, :-1].detach()

    print(f"\nInput scores shape:  {scores.shape}")
    print(f"Transport plan shape: {plan.shape}")
    print(f"Loss:                {loss.item():.4f}")

    # Convert to match probs and pick argmax
    match_probs = F.softmax(-X / 0.1, dim=-1)
    matches = match_probs.argmax(dim=1)
    confidences = match_probs.max(dim=1).values
    print(f"\nGreedy matches (source -> dest, confidence):")
    for i in range(n_src):
        if confidences[i] > 0.05:
            print(f"  src {i} -> dst {matches[i].item()}  (conf={confidences[i].item():.3f})")
        else:
            print(f"  src {i} -> UNMATCHED  (best conf={confidences[i].item():.3f})")


def _demo_learning_loop() -> None:
    """Demo: train the Variational Network Machine end-to-end.

    Task: route flow from node 0 (source) to node 4 (sink) through a
    5-node graph, with target = uniform distribution across paths of length 2.
    """
    print()
    print("=" * 60)
    print("Variational Network Machine — Learning Loop Demo")
    print("=" * 60)

    # 5-node line graph: 0 - 1 - 2 - 3 - 4
    n_nodes = 5
    source = 0
    sink = 4

    # Target: prefer path 0->2->4 (the "middle path") over the perimeter
    # but allow some flow on all paths for soft constraints
    target_probs = torch.tensor(
        [
            [0.00, 0.10, 0.70, 0.10, 0.00],  # from 0
            [0.00, 0.00, 0.10, 0.05, 0.05],  # from 1
            [0.00, 0.00, 0.00, 0.10, 0.80],  # from 2 (the "middle")
            [0.00, 0.00, 0.00, 0.00, 0.20],  # from 3
            [0.00, 0.00, 0.00, 0.00, 0.00],  # from 4 (sink)
        ]
    )

    machine = VariationalNetworkMachine(
        n_nodes=n_nodes,
        source_nodes=torch.tensor([source]),
        sink_nodes=torch.tensor([sink]),
        max_iter=30,
        regularization="entropy",  # entropy reg → smoother gradient
    )
    optimizer = torch.optim.Adam(machine.parameters(), lr=0.2)

    print(f"\nTarget match probabilities:")
    print(target_probs)

    print("\nTraining to match target routing pattern...")
    initial_loss = None
    for step in range(150):
        optimizer.zero_grad()
        result = machine()

        # KL divergence between learned match probs and target
        loss = F.kl_div(
            (result["match_probs"] + 1e-8).log(),
            target_probs,
            reduction="batchmean",
        )

        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(machine.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0:
            print(f"  Step {step:3d}: KL={loss.item():.4f}")

    final = machine()["match_probs"].detach().numpy()
    print(f"\nLearned match probabilities:")
    print(final.round(3))
    print(f"\nInitial KL: {initial_loss:.4f}")
    print(f"Final KL:   {loss.item():.4f}")
    print(f"Improvement: {((initial_loss - loss.item()) / initial_loss * 100):.1f}%")


if __name__ == "__main__":
    _demo_maze()
    _demo_matching()
    _demo_learning_loop()
    print("\n✓ All demos completed.")