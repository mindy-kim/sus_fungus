from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fungus_warmstart import fungus_solve
from learning.vrp_flow_data import (
    GraphBatch,
    VRPGraphExample,
    collate_graph_batch,
    load_graph_example,
)
from learning.vrp_flow_matching import sample_edge_probabilities
from learning.vrp_flow_model import EdgeFlowMatchingModel
from learning.vrp_warm_start import generate_candidate_orders
from vrpinstance import VRPInstance


def choose_torch_device(requested: str | torch.device | None = None) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_fm_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[EdgeFlowMatchingModel, dict]:
    device = choose_torch_device(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fm_config = payload["config"]
    model = EdgeFlowMatchingModel(
        node_feature_dim=fm_config["node_feature_dim"],
        edge_feature_dim=fm_config["edge_feature_dim"],
        hidden_dim=fm_config["hidden_dim"],
        num_layers=fm_config["layers"],
        time_dim=fm_config["time_dim"],
        dropout=fm_config["dropout"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, fm_config


def load_fm_example_batch(
    instance_path: str | Path,
    fm_config: dict,
    device: torch.device,
    *,
    record: dict[str, Any] | None = None,
    instance_id: str | None = None,
) -> tuple[VRPGraphExample, GraphBatch]:
    instance_path = Path(instance_path)
    resolved_record = dict(record or {})
    resolved_record.setdefault("path", instance_path.name)
    example = load_graph_example(
        data_dir=instance_path.parent,
        split=None,
        instance_id=instance_id or instance_path.stem,
        record=resolved_record,
        knn_k=fm_config["knn_k"],
    )
    return example, collate_graph_batch([example]).to(device)


def build_sampling_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(seed)
    return generator


def solve_with_fm_checkpoint(
    instance_path: str | Path,
    checkpoint_path: str | Path,
    *,
    solver_config: dict[str, Any] | None = None,
    num_samples: int = 4,
    max_orders: int = 6,
    integration_steps: int = 32,
    distance_weight: float = 0.35,
    seed: int = 0,
    use_fungus: bool = True,
    fungus_time_budget: float = 2.0,
    device: str | torch.device | None = None,
) -> tuple[str, float, int, int, bool]:
    model, fm_config = load_fm_model(checkpoint_path, device=device)
    return fm_solve(
        instance_path=instance_path,
        model=model,
        fm_config=fm_config,
        solver_config=solver_config,
        num_samples=num_samples,
        max_orders=max_orders,
        integration_steps=integration_steps,
        distance_weight=distance_weight,
        seed=seed,
        use_fungus=use_fungus,
        fungus_time_budget=fungus_time_budget,
    )


def fm_solve(
    instance_path: str | Path,
    model: EdgeFlowMatchingModel,
    fm_config: dict,
    solver_config: dict[str, Any] | None = None,
    num_samples: int = 4,
    max_orders: int = 6,
    integration_steps: int = 32,
    distance_weight: float = 0.35,
    seed: int = 0,
    use_fungus: bool = True,
    fungus_time_budget: float = 2.0,
) -> tuple[str, float, int, int, bool]:
    instance_path = Path(instance_path)
    device = next(model.parameters()).device
    example, batch = load_fm_example_batch(instance_path, fm_config, device)
    generator = build_sampling_generator(device, seed)
    solver = VRPInstance(str(instance_path), config=solver_config)

    best_solution: str | None = None
    best_objective = float("inf")
    fm_successes = 0
    fm_failures = 0

    for _ in range(num_samples):
        edge_probs = sample_edge_probabilities(
            model,
            batch,
            steps=integration_steps,
            noise_scale=fm_config["noise_scale"],
            generator=generator if device.type != "mps" else None,
        ).cpu()

        orders = generate_candidate_orders(
            example,
            edge_probabilities=edge_probs,
            max_orders=max_orders,
            distance_weight=distance_weight,
        )

        for order in orders:
            try:
                solution, objective = solver.solve_from_order(order)
                fm_successes += 1
            except RuntimeError:
                # The FM ordering couldn't be split into feasible routes
                # (happens with undertrained models that cluster high-demand
                # customers together). Skip and try the next candidate.
                fm_failures += 1
                continue
            if objective < best_objective:
                best_objective = objective
                best_solution = solution

    if use_fungus:
        fungus_candidate = _solve_with_fungus_candidate(
            instance_path=instance_path,
            solver_config=solver_config,
            time_budget_s=fungus_time_budget,
            seed=seed,
        )
        if fungus_candidate is not None:
            fg_solution, fg_objective = fungus_candidate
            if fg_objective < best_objective:
                best_objective = fg_objective
                best_solution = fg_solution

    used_fallback = best_solution is None
    if used_fallback:
        best_solution, best_objective = solver.solve()

    return best_solution, best_objective, fm_successes, fm_failures, used_fallback


def _solve_with_fungus_candidate(
    instance_path: Path,
    solver_config: dict[str, Any] | None,
    time_budget_s: float,
    seed: int,
) -> tuple[str, float] | None:
    try:
        solution, objective, _ = fungus_solve(
            instance_path=instance_path,
            solver_config=solver_config,
            time_budget_s=time_budget_s,
            seed=seed,
        )
    except Exception:
        return None
    return solution, objective
