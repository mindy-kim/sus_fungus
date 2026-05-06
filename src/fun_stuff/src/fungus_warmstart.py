from __future__ import annotations

import sys
from math import hypot
from pathlib import Path
from typing import Any

# Expose repo root so that fungus/, cvrp/, and models/ are importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cvrp.core import CVRPInstance  # noqa: E402
from fungus.solver import solve_fungus  # noqa: E402
from vrpinstance import VRPInstance         # noqa: E402  (also adds vendor/ for numpy)


def fungus_solve(
    instance_path: str | Path,
    solver_config: dict[str, Any] | None = None,
    time_budget_s: float = 2.0,
    seed: int = 0,
) -> tuple[str, float, bool]:
    instance_path = Path(instance_path)
    cvrp = CVRPInstance.from_file(instance_path)
    solver = VRPInstance(str(instance_path), config=solver_config)

    fungus_sol = solve_fungus(cvrp, time_budget_s=time_budget_s, seed=seed)
    routes = [list(r) for r in fungus_sol.routes if r]

    served = {customer for route in routes for customer in route}
    unserved = set(range(1, cvrp.num_nodes)) - served
    used_repair = bool(unserved)

    if used_repair:
        routes = _cheapest_insertion(routes, unserved, cvrp)

    try:
        solution, objective = solver.solve_from_routes(routes)
        return solution, objective, used_repair
    except (RuntimeError, ValueError):
        solution, objective = solver.solve()
        return solution, objective, True


def _cheapest_insertion(
    routes: list[list[int]],
    unserved: set[int],
    cvrp: CVRPInstance,
) -> list[list[int]]:
    loads = [sum(cvrp.demands[c] for c in r) for r in routes]

    for customer in sorted(unserved):
        demand = cvrp.demands[customer]
        best_delta = float("inf")
        best_route = -1
        best_pos = 0

        for ri, route in enumerate(routes):
            if loads[ri] + demand > cvrp.vehicle_capacity:
                continue
            for pos in range(len(route) + 1):
                prev = 0 if pos == 0 else route[pos - 1]
                after = 0 if pos == len(route) else route[pos]
                delta = (
                    hypot(
                        cvrp.xs[prev] - cvrp.xs[customer],
                        cvrp.ys[prev] - cvrp.ys[customer],
                    )
                    + hypot(cvrp.xs[customer] - cvrp.xs[after],
                            cvrp.ys[customer] - cvrp.ys[after])
                    - hypot(
                        cvrp.xs[prev] - cvrp.xs[after],
                        cvrp.ys[prev] - cvrp.ys[after],
                    )
                )
                if delta < best_delta:
                    best_delta = delta
                    best_route = ri
                    best_pos = pos

        if best_route == -1:
            routes.append([customer])
            loads.append(demand)
        else:
            routes[best_route].insert(best_pos, customer)
            loads[best_route] += demand

    return [r for r in routes if r]
