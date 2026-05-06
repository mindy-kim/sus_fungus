from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import TypeAlias

# Type alias used by fungus/solver.py
Routes: TypeAlias = list[list[int]]


@dataclass
class CVRPInstance:
    name: str
    num_nodes: int
    num_vehicles: int
    vehicle_capacity: int
    demands: list[int]
    xs: list[float]
    ys: list[float]

    @property
    def customers(self) -> range:
        return range(1, self.num_nodes)

    @classmethod
    def from_file(cls, path: str | Path) -> CVRPInstance:
        path = Path(path)
        tokens = path.read_text(encoding="utf-8").split()
        it = iter(tokens)
        num_nodes       = int(next(it))
        num_vehicles    = int(next(it))
        vehicle_capacity = int(next(it))
        demands, xs, ys = [], [], []
        for _ in range(num_nodes):
            demands.append(int(next(it)))
            xs.append(float(next(it)))
            ys.append(float(next(it)))
        return cls(
            name=path.stem,
            num_nodes=num_nodes,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            demands=demands,
            xs=xs,
            ys=ys,
        )

    def routes_objective(self, routes: list[list[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += hypot(self.xs[route[0]] - self.xs[0],
                           self.ys[route[0]] - self.ys[0])
            for a, b in zip(route, route[1:]):
                total += hypot(self.xs[a] - self.xs[b],
                               self.ys[a] - self.ys[b])
            total += hypot(self.xs[route[-1]] - self.xs[0],
                           self.ys[route[-1]] - self.ys[0])
        return total
