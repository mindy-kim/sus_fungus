import math

import torch

from .vrp_flow_data import VRPGraphExample


def _build_edge_lookup(
    example: VRPGraphExample,
    edge_probabilities: torch.Tensor,
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    lookup: dict[tuple[int, int], float] = {}
    depot_scores: dict[int, float] = {}
    for (left, right), score in zip(example.candidate_edges, edge_probabilities.tolist()):
        lookup[(left, right)] = float(score)
        if left == 0 and right != 0:
            depot_scores[right] = float(score)
        elif right == 0 and left != 0:
            depot_scores[left] = float(score)
    return lookup, depot_scores


def _pair_score(
    example: VRPGraphExample,
    lookup: dict[tuple[int, int], float],
    depot_scores: dict[int, float],
    left: int,
    right: int,
    distance_weight: float,
) -> float:
    edge_key = (left, right) if left < right else (right, left)
    affinity = lookup.get(edge_key, 0.0)
    distance = torch.linalg.norm(example.coords[left] - example.coords[right]).item()
    normalized_distance = distance / max(example.max_distance, 1e-6)
    depot_bonus = 0.25 * (depot_scores.get(left, 0.0) + depot_scores.get(right, 0.0))
    return affinity + depot_bonus - (distance_weight * normalized_distance)


def greedy_chain_order(
    example: VRPGraphExample,
    edge_probabilities: torch.Tensor,
    seed_customer: int,
    distance_weight: float = 0.35,
) -> list[int]:
    if seed_customer < 1 or seed_customer >= example.num_nodes:
        raise ValueError(f"Invalid seed customer: {seed_customer}")

    lookup, depot_scores = _build_edge_lookup(example, edge_probabilities)
    remaining = set(range(1, example.num_nodes))
    remaining.remove(seed_customer)
    chain = [seed_customer]

    while remaining:
        left_endpoint = chain[0]
        right_endpoint = chain[-1]
        best_customer = None
        best_side = None
        best_score = -math.inf

        for customer in remaining:
            left_score = _pair_score(
                example,
                lookup,
                depot_scores,
                customer,
                left_endpoint,
                distance_weight=distance_weight,
            )
            if left_score > best_score:
                best_score = left_score
                best_customer = customer
                best_side = "left"

            right_score = _pair_score(
                example,
                lookup,
                depot_scores,
                right_endpoint,
                customer,
                distance_weight=distance_weight,
            )
            if right_score > best_score:
                best_score = right_score
                best_customer = customer
                best_side = "right"

        if best_customer is None or best_side is None:
            break

        if best_side == "left":
            chain.insert(0, best_customer)
        else:
            chain.append(best_customer)
        remaining.remove(best_customer)

    if remaining:
        chain.extend(sorted(remaining))
    return chain


def generate_candidate_orders(
    example: VRPGraphExample,
    edge_probabilities: torch.Tensor,
    max_orders: int = 8,
    distance_weight: float = 0.35,
) -> list[list[int]]:
    if max_orders <= 0:
        return []

    lookup, depot_scores = _build_edge_lookup(example, edge_probabilities)
    customers = list(range(1, example.num_nodes))
    total_affinity = {
        customer: sum(
            score
            for (left, right), score in lookup.items()
            if customer in (left, right)
        )
        for customer in customers
    }
    farthest_customer = max(customers, key=lambda customer: float(example.depot_distances[customer]))

    ranked_depot = sorted(customers, key=lambda customer: depot_scores.get(customer, 0.0), reverse=True)
    ranked_affinity = sorted(customers, key=lambda customer: total_affinity[customer], reverse=True)

    seed_pool = [farthest_customer]
    seed_pool.extend(ranked_depot[: max_orders * 2])
    seed_pool.extend(ranked_affinity[: max_orders])

    unique_orders: list[list[int]] = []
    seen_seeds: set[int] = set()
    seen_orders: set[tuple[int, ...]] = set()
    for seed_customer in seed_pool:
        if seed_customer in seen_seeds:
            continue
        seen_seeds.add(seed_customer)
        order = greedy_chain_order(
            example,
            edge_probabilities,
            seed_customer=seed_customer,
            distance_weight=distance_weight,
        )
        order_key = tuple(order)
        if order_key not in seen_orders:
            unique_orders.append(order)
            seen_orders.add(order_key)
        reversed_order = list(reversed(order))
        reversed_key = tuple(reversed_order)
        if reversed_key not in seen_orders:
            unique_orders.append(reversed_order)
            seen_orders.add(reversed_key)
        if len(unique_orders) >= max_orders:
            break

    return unique_orders[:max_orders]
