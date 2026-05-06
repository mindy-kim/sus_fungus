import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def parse_solution_routes(solution: str) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for token in map(int, solution.split()):
        if token == 0:
            if current:
                routes.append(current)
                current = []
            continue
        current.append(token)
    if current:
        routes.append(current)
    return routes


def routes_to_undirected_edges(routes: list[list[int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for route in routes:
        if not route:
            continue
        edges.add(_canonical_edge(0, route[0]))
        edges.add(_canonical_edge(route[-1], 0))
        for left, right in zip(route, route[1:]):
            edges.add(_canonical_edge(left, right))
    return edges


@dataclass
class VRPGraphExample:
    instance_id: str
    split: str | None
    instance_path: Path
    num_nodes: int
    num_customers: int
    num_vehicles: int
    vehicle_capacity: float
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_labels: torch.Tensor | None
    candidate_edges: list[tuple[int, int]]
    coords: torch.Tensor
    demands: torch.Tensor
    depot_distances: torch.Tensor
    max_distance: float
    target_routes: list[list[int]] | None
    customer_order: list[int] | None
    family: str | None = None
    family_variant: str | None = None


@dataclass
class GraphBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_labels: torch.Tensor | None
    node_batch: torch.Tensor
    edge_batch: torch.Tensor
    num_graphs: int
    instance_ids: list[str]
    num_nodes_per_graph: list[int]
    num_edges_per_graph: list[int]
    candidate_edges: list[list[tuple[int, int]]]
    coords: list[torch.Tensor]
    demands: list[torch.Tensor]
    vehicle_capacities: torch.Tensor
    vehicle_counts: torch.Tensor
    target_routes: list[list[list[int]] | None]
    max_distances: list[float]

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            edge_labels=None if self.edge_labels is None else self.edge_labels.to(device),
            node_batch=self.node_batch.to(device),
            edge_batch=self.edge_batch.to(device),
            num_graphs=self.num_graphs,
            instance_ids=self.instance_ids,
            num_nodes_per_graph=self.num_nodes_per_graph,
            num_edges_per_graph=self.num_edges_per_graph,
            candidate_edges=self.candidate_edges,
            coords=self.coords,
            demands=self.demands,
            vehicle_capacities=self.vehicle_capacities.to(device),
            vehicle_counts=self.vehicle_counts.to(device),
            target_routes=self.target_routes,
            max_distances=self.max_distances,
        )


def _resolve_instance_path(
    data_dir: Path,
    split: str | None,
    instance_id: str,
    record: dict[str, Any] | None,
) -> Path:
    if split is not None:
        direct = data_dir / split / f"{instance_id}.vrp"
        if direct.exists():
            return direct

    if record is not None and "path" in record:
        candidate = data_dir / Path(record["path"]).name
        if candidate.exists():
            return candidate
        candidate = data_dir / Path(record["path"])
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve instance path for {instance_id}.")


def _load_instance(instance_path: Path) -> tuple[int, int, int, torch.Tensor, torch.Tensor]:
    with instance_path.open("r", encoding="utf-8") as handle:
        tokens = handle.read().split()

    iterator = iter(tokens)
    num_nodes = int(next(iterator))
    num_vehicles = int(next(iterator))
    vehicle_capacity = int(next(iterator))

    demands = torch.zeros(num_nodes, dtype=torch.float32)
    coords = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for idx in range(num_nodes):
        demands[idx] = float(next(iterator))
        coords[idx, 0] = float(next(iterator))
        coords[idx, 1] = float(next(iterator))

    return num_nodes, num_vehicles, vehicle_capacity, demands, coords


def _build_candidate_edges(
    coords: torch.Tensor,
    positive_edges: set[tuple[int, int]],
    knn_k: int,
) -> list[tuple[int, int]]:
    num_nodes = coords.size(0)
    num_customers = max(0, num_nodes - 1)
    edges = set(positive_edges)

    for customer in range(1, num_nodes):
        edges.add((0, customer))

    if num_customers > 1 and knn_k > 0:
        customer_coords = coords[1:]
        distance_matrix = torch.cdist(customer_coords, customer_coords, p=2)
        distance_matrix.fill_diagonal_(float("inf"))
        neighbor_count = min(knn_k, num_customers - 1)
        nearest = torch.topk(distance_matrix, k=neighbor_count, largest=False).indices
        for offset, neighbors in enumerate(nearest.tolist(), start=1):
            for neighbor_offset in neighbors:
                neighbor = neighbor_offset + 1
                edges.add(_canonical_edge(offset, neighbor))

    return sorted(edges)


def load_graph_example(
    data_dir: str | Path,
    split: str | None,
    instance_id: str,
    record: dict[str, Any] | None = None,
    knn_k: int = 16,
) -> VRPGraphExample:
    base_dir = Path(data_dir)
    instance_path = _resolve_instance_path(base_dir, split, instance_id, record)
    num_nodes, num_vehicles, vehicle_capacity, demands, coords = _load_instance(instance_path)
    num_customers = num_nodes - 1

    route_targets = None
    customer_order = None
    family = None
    family_variant = None
    positive_edges: set[tuple[int, int]] = set()
    if record is not None and record.get("solution"):
        route_targets = parse_solution_routes(record["solution"])
        customer_order = [customer for route in route_targets for customer in route]
        positive_edges = routes_to_undirected_edges(route_targets)
        family = record.get("family")
        family_variant = record.get("family_variant")

    candidate_edges = _build_candidate_edges(coords, positive_edges, knn_k=knn_k)
    edge_index = torch.tensor(candidate_edges, dtype=torch.long).t().contiguous()

    coord_min = coords.min(dim=0).values
    coord_max = coords.max(dim=0).values
    coord_span = torch.clamp(coord_max - coord_min, min=1e-6)
    normalized_coords = (coords - coord_min) / coord_span

    depot_delta = coords - coords[0]
    depot_distances = torch.linalg.norm(depot_delta, dim=1)
    max_distance = max(float(depot_distances.max().item()), 1e-6)
    angles = torch.atan2(depot_delta[:, 1], depot_delta[:, 0])

    capacity = max(float(vehicle_capacity), 1.0)
    vehicle_ratio = num_vehicles / max(1.0, float(num_customers))
    customer_scale = num_customers / 400.0

    node_features = torch.stack(
        (
            (torch.arange(num_nodes) == 0).to(torch.float32),
            normalized_coords[:, 0],
            normalized_coords[:, 1],
            demands / capacity,
            depot_distances / max_distance,
            torch.sin(angles),
            torch.cos(angles),
            torch.full((num_nodes,), vehicle_ratio, dtype=torch.float32),
            torch.full((num_nodes,), customer_scale, dtype=torch.float32),
        ),
        dim=1,
    )
    node_features[0, 5] = 0.0
    node_features[0, 6] = 0.0

    edge_features = []
    edge_labels = []
    for left, right in candidate_edges:
        delta = coords[left] - coords[right]
        distance = float(torch.linalg.norm(delta).item())
        savings = float(depot_distances[left] + depot_distances[right] - distance)
        angle_delta = float(angles[left] - angles[right])
        edge_features.append(
            [
                1.0 if left == 0 or right == 0 else 0.0,
                distance / max_distance,
                savings / max_distance,
                abs(float(depot_distances[left] - depot_distances[right])) / max_distance,
                math.cos(angle_delta),
                abs(math.sin(angle_delta)),
                float(demands[left] + demands[right]) / capacity,
                abs(float(demands[left] - demands[right])) / capacity,
            ]
        )
        if record is not None:
            edge_labels.append(1.0 if (left, right) in positive_edges else 0.0)

    edge_feature_tensor = torch.tensor(edge_features, dtype=torch.float32)
    edge_label_tensor = None
    if edge_labels:
        edge_label_tensor = torch.tensor(edge_labels, dtype=torch.float32)

    return VRPGraphExample(
        instance_id=instance_id,
        split=split,
        instance_path=instance_path,
        num_nodes=num_nodes,
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=float(vehicle_capacity),
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_feature_tensor,
        edge_labels=edge_label_tensor,
        candidate_edges=candidate_edges,
        coords=coords,
        demands=demands,
        depot_distances=depot_distances,
        max_distance=max_distance,
        target_routes=route_targets,
        customer_order=customer_order,
        family=family,
        family_variant=family_variant,
    )


class VRPGraphDataset(Dataset[VRPGraphExample]):
    def __init__(
        self,
        data_dir: str | Path = "data",
        split: str = "train",
        knn_k: int = 16,
        limit: int | None = None,
        use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.knn_k = knn_k
        labels_path = self.data_dir / split / "labels.jsonl"
        with labels_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle]
        if limit is not None:
            self.records = self.records[:limit]
        self._cache: dict[int, VRPGraphExample] | None = {} if use_cache else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> VRPGraphExample:
        if self._cache is not None and index in self._cache:
            return self._cache[index]

        record = self.records[index]
        example = load_graph_example(
            data_dir=self.data_dir,
            split=self.split,
            instance_id=record["instance_id"],
            record=record,
            knn_k=self.knn_k,
        )
        if self._cache is not None:
            self._cache[index] = example
        return example


def collate_graph_batch(examples: list[VRPGraphExample]) -> GraphBatch:
    node_features = []
    edge_features = []
    edge_labels = []
    edge_indices = []
    node_batch = []
    edge_batch = []
    candidate_edges = []
    coords = []
    demands = []
    num_nodes_per_graph = []
    num_edges_per_graph = []
    vehicle_capacities = []
    vehicle_counts = []
    target_routes = []
    max_distances = []
    instance_ids = []

    node_offset = 0
    has_labels = all(example.edge_labels is not None for example in examples)

    for graph_idx, example in enumerate(examples):
        node_count = example.node_features.size(0)
        edge_count = example.edge_features.size(0)

        node_features.append(example.node_features)
        edge_features.append(example.edge_features)
        edge_indices.append(example.edge_index + node_offset)
        node_batch.append(torch.full((node_count,), graph_idx, dtype=torch.long))
        edge_batch.append(torch.full((edge_count,), graph_idx, dtype=torch.long))

        if has_labels:
            edge_labels.append(example.edge_labels)

        candidate_edges.append(example.candidate_edges)
        coords.append(example.coords)
        demands.append(example.demands)
        num_nodes_per_graph.append(node_count)
        num_edges_per_graph.append(edge_count)
        vehicle_capacities.append(example.vehicle_capacity)
        vehicle_counts.append(example.num_vehicles)
        target_routes.append(example.target_routes)
        max_distances.append(example.max_distance)
        instance_ids.append(example.instance_id)

        node_offset += node_count

    label_tensor = torch.cat(edge_labels) if has_labels and edge_labels else None
    return GraphBatch(
        node_features=torch.cat(node_features, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat(edge_features, dim=0),
        edge_labels=label_tensor,
        node_batch=torch.cat(node_batch),
        edge_batch=torch.cat(edge_batch),
        num_graphs=len(examples),
        instance_ids=instance_ids,
        num_nodes_per_graph=num_nodes_per_graph,
        num_edges_per_graph=num_edges_per_graph,
        candidate_edges=candidate_edges,
        coords=coords,
        demands=demands,
        vehicle_capacities=torch.tensor(vehicle_capacities, dtype=torch.float32),
        vehicle_counts=torch.tensor(vehicle_counts, dtype=torch.float32),
        target_routes=target_routes,
        max_distances=max_distances,
    )
