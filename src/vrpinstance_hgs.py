from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import time
from typing import Iterable, Sequence

import numpy as np


Route = list
Routes = list
Pattern = tuple


class VRPInstance:
    numCustomers: int
    numVehicles: int
    vehicleCapacity: int
    demandOfCustomer: np.ndarray
    xCoordOfCustomer: np.ndarray
    yCoordOfCustomer: np.ndarray

    def __init__(self, filename: str):
        self.load_from_file(filename)
        self.solution = None
        self.objective_value = 0
        self._dist_matrix: np.ndarray | None = None

    def solve(self, time_budget_s: float | None = None, seed: int = 0):
        self._ensure_distance_matrix()
        routes, objective, _meta = solve_hgs_search(self, time_budget_s=time_budget_s, seed=seed)
        self.solution = routes
        self.objective_value = objective
        return self.solution, self.objective_value

    @property
    def customers(self) -> range:
        return range(1, self.numCustomers)

    @property
    def num_nodes(self) -> int:
        return self.numCustomers

    @property
    def num_vehicles(self) -> int:
        return self.numVehicles

    @property
    def vehicle_capacity(self) -> int:
        return int(self.vehicleCapacity)

    @property
    def demands(self) -> np.ndarray:
        return self.demandOfCustomer

    @property
    def total_demand(self) -> int:
        return int(self.demandOfCustomer[1:].sum())

    def _ensure_distance_matrix(self) -> None:
        if self._dist_matrix is not None:
            return
        n = self.numCustomers
        xs = self.xCoordOfCustomer
        ys = self.yCoordOfCustomer
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        self._dist_matrix = np.hypot(dx, dy)

    def distance(self, i: int, j: int) -> float:
        if self._dist_matrix is None:
            self._ensure_distance_matrix()
        return float(self._dist_matrix[i, j])

    def route_demand(self, route):
        if not route:
            return 0
        return int(self.demandOfCustomer[list(route)].sum())

    def route_distance(self, route):
        if not route:
            return 0.0
        if self._dist_matrix is None:
            self._ensure_distance_matrix()
        d = float(self._dist_matrix[0, route[0]]) + float(self._dist_matrix[route[-1], 0])
        for left, right in zip(route, route[1:]):
            d += float(self._dist_matrix[left, right])
        return d

    def routes_objective(self, routes):
        return sum(self.route_distance(route) for route in routes)

    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                content = f.read().split()
                iterator = iter(content)
                self.numCustomers = int(next(iterator))
                self.numVehicles = int(next(iterator))
                self.vehicleCapacity = int(next(iterator))
                self.demandOfCustomer = np.zeros(self.numCustomers, dtype=int)
                self.xCoordOfCustomer = np.zeros(self.numCustomers)
                self.yCoordOfCustomer = np.zeros(self.numCustomers)
                for i in range(self.numCustomers):
                    self.demandOfCustomer[i] = int(next(iterator))
                    self.xCoordOfCustomer[i] = float(next(iterator))
                    self.yCoordOfCustomer[i] = float(next(iterator))
        except Exception as e:
            print(f"Error reading instance file: {e}")
            exit(1)


def copy_routes(routes):
    return [list(route) for route in routes]


def flatten_routes(routes):
    flat = []
    for route in routes:
        flat.extend(route)
    return flat


def unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def split_giant_tour(instance, customer_order, max_routes=None):
    if max_routes is None:
        max_routes = instance.num_vehicles
    if not customer_order:
        return [[]], 0.0
    n = len(customer_order)
    prefix_demand = [0] * (n + 1)
    for index, customer in enumerate(customer_order, start=1):
        prefix_demand[index] = prefix_demand[index - 1] + int(instance.demands[customer])
    segment_cost = [[math.inf] * (n + 1) for _ in range(n)]
    for start in range(n):
        demand = 0
        route_cost = 0.0
        previous = 0
        for end in range(start, n):
            customer = customer_order[end]
            demand += int(instance.demands[customer])
            if demand > instance.vehicle_capacity:
                break
            route_cost += instance.distance(previous, customer)
            previous = customer
            segment_cost[start][end + 1] = route_cost + instance.distance(previous, 0)
    best = [[math.inf] * (max_routes + 1) for _ in range(n + 1)]
    parent = [[None] * (max_routes + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for end in range(1, n + 1):
        for routes_used in range(1, max_routes + 1):
            for start in range(end - 1, -1, -1):
                if prefix_demand[end] - prefix_demand[start] > instance.vehicle_capacity:
                    break
                current = segment_cost[start][end]
                if math.isinf(current) or math.isinf(best[start][routes_used - 1]):
                    continue
                candidate = best[start][routes_used - 1] + current
                if candidate < best[end][routes_used]:
                    best[end][routes_used] = candidate
                    parent[end][routes_used] = start
    route_count = min(range(max_routes + 1), key=lambda count: best[n][count])
    if math.isinf(best[n][route_count]):
        return None
    cuts = []
    end = n
    routes_used = route_count
    while end > 0:
        start = parent[end][routes_used]
        if start is None:
            return None
        cuts.append(start)
        end = start
        routes_used -= 1
    cuts.reverse()
    routes = []
    previous = 0
    for cut in cuts[1:]:
        routes.append(customer_order[previous:cut])
        previous = cut
    routes.append(customer_order[previous:n])
    return routes, best[n][route_count]


def repair_routes(instance, routes):
    if _is_clean_feasible_cover(instance, routes):
        return [route[:] for route in routes if route], instance.routes_objective(routes)
    valid_customers = [
        customer
        for customer in unique_preserve_order(customer for route in routes for customer in route)
        if 1 <= customer < instance.num_nodes
    ]
    missing = [customer for customer in instance.customers if customer not in set(valid_customers)]
    candidate_order = valid_customers + missing
    split = split_giant_tour(instance, candidate_order)
    if split is not None:
        return split
    packed = greedy_capacity_pack(
        instance,
        sorted(candidate_order, key=lambda customer: (-int(instance.demands[customer]), customer)),
    )
    return packed, instance.routes_objective(packed)


def greedy_capacity_pack(instance, customer_order):
    """First-fit-decreasing-style packing with cheapest-insertion within bins.

    Tolerates overflow: if first-fit can't seat a customer in any existing
    route AND we've already used ``num_vehicles`` routes, this opens an
    extra route rather than recursing into ``_pack_by_backtracking``. The
    caller (typically ``repair_routes``) feeds the result through
    ``split_giant_tour`` which will compact the cover back to
    ``num_vehicles`` routes when feasible. Backtracking is used only for
    very small instances where it's cheap.
    """
    if not customer_order:
        return []
    routes = []
    loads = []
    for customer in customer_order:
        placed = False
        for route_index in range(len(routes)):
            if loads[route_index] + int(instance.demands[customer]) > instance.vehicle_capacity:
                continue
            _, position = best_insertion_delta(instance, routes[route_index], customer)
            routes[route_index].insert(position, customer)
            loads[route_index] += int(instance.demands[customer])
            placed = True
            break
        if placed:
            continue
        # Open a new route. We allow exceeding instance.num_vehicles here; the
        # caller will compact via split_giant_tour. This avoids the
        # exponential-time _pack_by_backtracking on large instances.
        routes.append([customer])
        loads.append(int(instance.demands[customer]))

    # Only attempt exact backtracking on small instances where it's cheap.
    # If first-fit already uses <= num_vehicles routes, we're done.
    if len(routes) <= instance.num_vehicles:
        return routes
    if len(customer_order) <= 30:
        exact = _pack_by_backtracking(instance, customer_order, deadline=time.time() + 0.5)
        if exact is not None:
            return exact
    # Fall back to whatever first-fit produced; downstream split_giant_tour
    # will try to recompact into num_vehicles routes.
    return routes


def best_insertion_delta(instance, route, customer):
    if not route:
        return instance.distance(0, customer) + instance.distance(customer, 0), 0
    best_delta = math.inf
    best_position = 0
    full_route = [0] + list(route) + [0]
    for position in range(len(full_route) - 1):
        left = full_route[position]
        right = full_route[position + 1]
        delta = instance.distance(left, customer) + instance.distance(customer, right) - instance.distance(left, right)
        if delta < best_delta:
            best_delta = delta
            best_position = position
    return best_delta, best_position


def _is_clean_feasible_cover(instance, routes):
    if len(routes) > instance.num_vehicles:
        return False
    seen = set()
    for route in routes:
        if instance.route_demand(route) > instance.vehicle_capacity:
            return False
        for customer in route:
            if customer in seen or customer <= 0 or customer >= instance.num_nodes:
                return False
            seen.add(customer)
    return seen == set(instance.customers)


def _pack_by_backtracking(instance, customer_order, deadline=None):
    """Bounded-time backtracking packer. Returns None if it can't find a
    feasible packing within the deadline."""
    if deadline is None:
        deadline = time.time() + 0.5
    routes = [[] for _ in range(instance.num_vehicles)]
    loads = [0] * instance.num_vehicles
    ordered = sorted(customer_order, key=lambda customer: (-int(instance.demands[customer]), customer))
    aborted = [False]

    def search(index):
        if aborted[0] or time.time() >= deadline:
            aborted[0] = True
            return False
        if index == len(ordered):
            return True
        customer = ordered[index]
        demand = int(instance.demands[customer])
        used_loads = set()
        for route_index in range(instance.num_vehicles):
            if loads[route_index] in used_loads:
                continue
            if loads[route_index] + demand > instance.vehicle_capacity:
                continue
            used_loads.add(loads[route_index])
            _, position = best_insertion_delta(instance, routes[route_index], customer)
            routes[route_index].insert(position, customer)
            loads[route_index] += demand
            if search(index + 1):
                return True
            routes[route_index].pop(position)
            loads[route_index] -= demand
            if loads[route_index] == 0:
                break
        return False

    if not search(0):
        return None
    return [route for route in routes if route]


@dataclass
class _SolverRun:
    routes: Routes
    objective: float


def neighbor_count(instance):
    if instance.num_nodes <= 60:
        return min(20, instance.num_nodes - 1)
    if instance.num_nodes <= 180:
        return 30
    return 40


def nearest_neighbor_sets(instance, k):
    instance._ensure_distance_matrix()
    result = {}
    for customer in instance.customers:
        others = [c for c in instance.customers if c != customer]
        others.sort(key=lambda c: instance.distance(customer, c))
        result[customer] = set(others[:k])
    return result


def add_routes_to_pool(instance, route_pool, routes):
    for route in routes:
        if not route:
            continue
        key = frozenset(route)
        existing = route_pool.get(key)
        if existing is None or instance.route_distance(route) < instance.route_distance(existing):
            route_pool[key] = list(route)


def relaxed_nearest_tour_split(instance, rng, neighbors):
    customers = list(instance.customers)
    if not customers:
        return []
    visited = set()
    tour = []
    current = 0
    while len(tour) < len(customers):
        remaining = [c for c in customers if c not in visited]
        remaining.sort(key=lambda c: instance.distance(current, c))
        k = min(5, len(remaining))
        idx = int((rng.random() ** 2) * k)
        picked = remaining[idx]
        tour.append(picked)
        visited.add(picked)
        current = picked
    split = split_giant_tour(instance, tour)
    if split is None:
        return greedy_capacity_pack(instance, tour)
    return split[0]


def _customer_multiset(routes):
    out = []
    for r in routes:
        out.extend(r)
    out.sort()
    return tuple(out)


def local_search_hybrid(instance, routes, neighbors, deadline, tight_capacity):
    """First-improvement LS using or-opt (relocate 1..3) + 2-opt + swap.

    Two correctness/perf properties matter:
    - **Deadline is checked at every level** of the inner loops, not just
      between full passes. A single sweep on a 200+ customer instance can
      otherwise take many seconds and overrun the polish slice by an order
      of magnitude.
    - **Delta-based cost evaluation**: each candidate move is scored in O(1)
      from the local edges that change, rather than recomputing
      ``route_distance`` (which is O(L)) from scratch. This makes a single
      or_opt_step roughly len(route)x faster on long routes.
    """
    routes = [r[:] for r in routes if r]
    if not routes:
        return routes, 0.0
    cap = instance.vehicle_capacity
    invariant = _customer_multiset(routes)

    def or_opt_step():
        if time.time() >= deadline:
            return False
        for r1_idx in range(len(routes)):
            if time.time() >= deadline:
                return False
            r1 = routes[r1_idx]
            if not r1:
                continue
            for seg_len in (1, 2, 3):
                if seg_len > len(r1) or time.time() >= deadline:
                    if time.time() >= deadline:
                        return False
                    continue
                for i in range(len(r1) - seg_len + 1):
                    if time.time() >= deadline:
                        return False
                    segment = r1[i : i + seg_len]
                    seg_first = segment[0]
                    seg_last = segment[-1]
                    seg_demand = sum(int(instance.demands[c]) for c in segment)

                    # Cost saved by removing the segment from r1: the two edges
                    # touching the segment are replaced by a single edge.
                    a = r1[i - 1] if i > 0 else 0
                    d_after = r1[i + seg_len] if i + seg_len < len(r1) else 0
                    remove_gain = (
                        instance.distance(a, seg_first)
                        + instance.distance(seg_last, d_after)
                        - instance.distance(a, d_after)
                    )

                    best_delta = -1e-12
                    best_apply = None

                    for r2_idx in range(len(routes)):
                        if time.time() >= deadline:
                            return False
                        r2 = routes[r2_idx]
                        if r2_idx == r1_idx:
                            base_route = r1[:i] + r1[i + seg_len :]
                        else:
                            if instance.route_demand(r2) + seg_demand > cap:
                                continue
                            base_route = r2

                        for j in range(len(base_route) + 1):
                            left = base_route[j - 1] if j > 0 else 0
                            right = base_route[j] if j < len(base_route) else 0
                            base_edge = instance.distance(left, right)
                            for reverse in (False, True):
                                if reverse:
                                    sf, sl = seg_last, seg_first
                                else:
                                    sf, sl = seg_first, seg_last
                                insert_cost = (
                                    instance.distance(left, sf)
                                    + instance.distance(sl, right)
                                    - base_edge
                                )
                                delta = insert_cost - remove_gain
                                if delta < best_delta:
                                    if reverse:
                                        seg_used = list(reversed(segment))
                                    else:
                                        seg_used = list(segment)
                                    if r2_idx == r1_idx:
                                        new_r1 = base_route[:j] + seg_used + base_route[j:]
                                        new_r2 = None
                                    else:
                                        new_r1 = r1[:i] + r1[i + seg_len :]
                                        new_r2 = r2[:j] + seg_used + r2[j:]
                                    best_delta = delta
                                    best_apply = (r1_idx, r2_idx, new_r1, new_r2)

                    if best_apply is not None:
                        a_r1_idx, a_r2_idx, new_r1, new_r2 = best_apply
                        routes[a_r1_idx] = new_r1
                        if new_r2 is not None:
                            routes[a_r2_idx] = new_r2
                        return True
        return False

    def two_opt_step():
        if time.time() >= deadline:
            return False
        for r_idx in range(len(routes)):
            if time.time() >= deadline:
                return False
            r = routes[r_idx]
            if len(r) < 4:
                continue
            for i in range(len(r) - 1):
                if time.time() >= deadline:
                    return False
                for j in range(i + 2, len(r)):
                    a = r[i - 1] if i > 0 else 0
                    b = r[i]
                    c_node = r[j]
                    d = r[j + 1] if j + 1 < len(r) else 0
                    delta = (
                        instance.distance(a, c_node)
                        + instance.distance(b, d)
                        - instance.distance(a, b)
                        - instance.distance(c_node, d)
                    )
                    if delta < -1e-9:
                        r[i : j + 1] = r[i : j + 1][::-1]
                        return True
        return False

    def swap_step():
        if time.time() >= deadline:
            return False
        for r1_idx in range(len(routes)):
            if time.time() >= deadline:
                return False
            r1 = routes[r1_idx]
            if not r1:
                continue
            d1 = instance.route_demand(r1)
            for r2_idx in range(r1_idx + 1, len(routes)):
                if time.time() >= deadline:
                    return False
                r2 = routes[r2_idx]
                if not r2:
                    continue
                d2 = instance.route_demand(r2)
                for i in range(len(r1)):
                    cu = r1[i]
                    cu_d = int(instance.demands[cu])
                    a1 = r1[i - 1] if i > 0 else 0
                    b1 = r1[i + 1] if i + 1 < len(r1) else 0
                    for j in range(len(r2)):
                        cv = r2[j]
                        cv_d = int(instance.demands[cv])
                        if d1 - cu_d + cv_d > cap or d2 - cv_d + cu_d > cap:
                            continue
                        a2 = r2[j - 1] if j > 0 else 0
                        b2 = r2[j + 1] if j + 1 < len(r2) else 0
                        delta = (
                            instance.distance(a1, cv) + instance.distance(cv, b1)
                            + instance.distance(a2, cu) + instance.distance(cu, b2)
                            - instance.distance(a1, cu) - instance.distance(cu, b1)
                            - instance.distance(a2, cv) - instance.distance(cv, b2)
                        )
                        if delta < -1e-9:
                            r1[i] = cv
                            r2[j] = cu
                            return True
        return False

    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if iterations > 500:
            break
        if or_opt_step():
            assert _customer_multiset(routes) == invariant, "or_opt corrupted customer set"
            continue
        if two_opt_step():
            assert _customer_multiset(routes) == invariant, "two_opt corrupted customer set"
            continue
        if swap_step():
            assert _customer_multiset(routes) == invariant, "swap corrupted customer set"
            continue
        break

    routes = [r for r in routes if r]
    return routes, instance.routes_objective(routes)


def destroy_and_repair(instance, routes, rng, neighbors, deadline):
    routes = [r[:] for r in routes if r]
    all_customers = [c for r in routes for c in r]
    if not all_customers:
        return routes
    n = len(all_customers)
    target_remove = max(2, min(n - 1, n // 4))
    seed = rng.choice(all_customers)
    removed = {seed}
    while len(removed) < target_remove:
        if time.time() >= deadline:
            break
        ref = rng.choice(list(removed))
        candidates = [c for c in all_customers if c not in removed]
        if not candidates:
            break
        candidates.sort(key=lambda c: instance.distance(ref, c))
        idx = int((rng.random() ** 3) * min(len(candidates), 10))
        removed.add(candidates[idx])
    new_routes = [[c for c in r if c not in removed] for r in routes]
    new_routes = [r for r in new_routes if r]
    reinsert_order = sorted(removed, key=lambda c: -int(instance.demands[c]))
    for c in reinsert_order:
        best_target = -1
        best_pos = 0
        best_delta = math.inf
        for r_idx, r in enumerate(new_routes):
            if instance.route_demand(r) + int(instance.demands[c]) > instance.vehicle_capacity:
                continue
            d, pos = best_insertion_delta(instance, r, c)
            if d < best_delta:
                best_delta = d
                best_target = r_idx
                best_pos = pos
        if best_target < 0:
            if len(new_routes) < instance.num_vehicles:
                new_routes.append([c])
            else:
                new_routes.append([c])
        else:
            new_routes[best_target].insert(best_pos, c)
    repaired, _ = repair_routes(instance, new_routes)
    return repaired


def route_pool_recombine(instance, route_pool, deadline, best_routes=None):
    if not route_pool:
        return None
    candidates = []
    for key, route in route_pool.items():
        if not route:
            continue
        cost = instance.route_distance(route)
        candidates.append((cost / max(1, len(route)), cost, list(route)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    covered = set()
    selected = []
    for _, _, route in candidates:
        if time.time() >= deadline:
            break
        rs = set(route)
        if covered & rs:
            continue
        if len(selected) >= instance.num_vehicles:
            break
        selected.append(route[:])
        covered |= rs
    remaining = [c for c in instance.customers if c not in covered]
    for c in remaining:
        best_target = -1
        best_pos = 0
        best_delta = math.inf
        for r_idx, r in enumerate(selected):
            if instance.route_demand(r) + int(instance.demands[c]) > instance.vehicle_capacity:
                continue
            d, pos = best_insertion_delta(instance, r, c)
            if d < best_delta:
                best_delta = d
                best_target = r_idx
                best_pos = pos
        if best_target < 0:
            if len(selected) < instance.num_vehicles:
                selected.append([c])
            else:
                return None
        else:
            selected[best_target].insert(best_pos, c)
    repaired, _ = repair_routes(instance, selected)
    return repaired


def solve_hybrid(instance, time_budget_s, seed):
    rng = random.Random(seed)
    deadline = time.time() + max(0.05, time_budget_s)
    neighbors = nearest_neighbor_sets(instance, neighbor_count(instance))
    routes = relaxed_nearest_tour_split(instance, rng, neighbors)
    routes, _ = local_search_hybrid(instance, routes, neighbors, deadline, False)
    return _SolverRun(routes=routes, objective=instance.routes_objective(routes))


def solve_hgs_search(instance, time_budget_s, seed, incumbent=None):
    started = time.time()
    budget = time_budget_s if time_budget_s is not None else 300
    deadline = started + max(0.2, budget)
    rng = random.Random(seed)
    neighbors = nearest_neighbor_sets(instance, max(neighbor_count(instance), hgs_neighbor_count(instance)))
    tight_capacity = instance.total_demand / max(1, instance.num_vehicles * instance.vehicle_capacity) >= 0.86

    best_routes = None
    best_objective = math.inf
    best_origin = "none"
    population = []
    route_pool = {}
    crossovers = 0
    pattern_improvements = 0
    localized_improvements = 0
    pool_improvements = 0

    def consider(candidate, origin, intensify=True):
        nonlocal best_routes, best_objective, best_origin, pattern_improvements, localized_improvements
        if time.time() >= deadline:
            return False
        repaired, _ = repair_routes(instance, candidate)
        if intensify and time.time() < deadline:
            polish_deadline = min(deadline, time.time() + hgs_polish_slice(instance))
            repaired, _ = local_search_hybrid(instance, repaired, neighbors, polish_deadline, tight_capacity)
            try:
                from models.gradient.solver import gradient_route_descent
                repaired, _ = gradient_route_descent(instance, repaired, min(deadline, time.time() + hgs_polish_slice(instance)))
            except Exception:
                pass
        objective = instance.routes_objective(repaired)
        add_routes_to_pool(instance, route_pool, repaired)
        add_to_population(instance, population, repaired)
        if objective + 1e-9 < best_objective:
            best_routes = copy_routes(repaired)
            best_objective = objective
            best_origin = origin
            return True
        return False

    if incumbent is not None:
        consider(incumbent, "incumbent")

    seed_round = 0
    while len(population) < initial_population_size(instance) and time_remaining(deadline) > 0.08:
        seed_round += 1
        seed_budget = min(hybrid_seed_budget(instance), max(0.05, time_remaining(deadline) * 0.2))
        if seed_round % 3 == 0:
            candidate = relaxed_nearest_tour_split(instance, rng, neighbors)
        elif seed_round % 3 == 1:
            candidate = solve_hybrid(instance, time_budget_s=seed_budget, seed=rng.randrange(1 << 30)).routes
        else:
            order = sorted(instance.customers, key=lambda customer: (-int(instance.demands[customer]), rng.random()))
            candidate = greedy_capacity_pack(instance, order)
        consider(candidate, f"population_seed_{seed_round}")

    while time_remaining(deadline) > 0.05:
        if len(population) >= 2:
            parent_a, parent_b = select_parents(instance, population, rng)
            child_order = ordered_crossover(flatten_routes(parent_a), flatten_routes(parent_b), rng)
            child_split = split_giant_tour(instance, child_order)
            child = child_split[0] if child_split is not None else greedy_capacity_pack(instance, child_order)
            crossovers += 1
            consider(child, f"hgs_crossover_{crossovers}")

        patterns = mine_patterns(population, max_patterns=pattern_budget(instance))
        if best_routes is not None and patterns and time_remaining(deadline) > 0.04:
            before = best_objective
            injected = pattern_injection_search(instance, best_routes, patterns, deadline, rng)
            if injected is not None and consider(injected, "pils_pattern_injection"):
                if best_objective + 1e-9 < before:
                    pattern_improvements += 1

        if best_routes is not None and time_remaining(deadline) > 0.04:
            before = best_objective
            localized = localized_cache_search(instance, best_routes, rng, neighbors, deadline)
            if consider(localized, "filo_localized_shake"):
                if best_objective + 1e-9 < before:
                    localized_improvements += 1

        if route_pool and len(route_pool) >= instance.num_vehicles and crossovers % 3 == 0 and time_remaining(deadline) > 0.05:
            before = best_objective
            recombined = route_pool_recombine(instance, route_pool, deadline, best_routes=best_routes)
            if recombined is not None and consider(recombined, "hgs_route_pool"):
                if best_objective + 1e-9 < before:
                    pool_improvements += 1

        if len(population) < 2 and best_routes is not None:
            shaken = destroy_and_repair(instance, best_routes, rng, neighbors, deadline)
            consider(shaken, "hgs_lns_bootstrap")

    if best_routes is None:
        fallback = solve_hybrid(instance, time_budget_s=max(0.05, time_remaining(deadline)), seed=seed)
        best_routes = copy_routes(fallback.routes)
        best_objective = fallback.objective
        best_origin = "fallback_hybrid"

    return (best_routes, best_objective, {})


def add_to_population(instance, population, routes):
    cleaned = [route[:] for route in routes if route]
    signature = tuple(tuple(route) for route in cleaned)
    if any(tuple(tuple(route) for route in existing) == signature for existing in population):
        return
    population.append(cleaned)
    population.sort(key=instance.routes_objective)
    del population[max_population_size(instance):]


def select_parents(instance, population, rng):
    sample_size = min(len(population), 5)
    first = min(rng.sample(population, sample_size), key=instance.routes_objective)
    second = min(rng.sample(population, sample_size), key=lambda routes: parent_score(instance, first, routes))
    if first is second and len(population) > 1:
        second = population[1] if population[0] is first else population[0]
    return first, second


def parent_score(instance, first, second):
    objective = instance.routes_objective(second)
    first_edges = edge_set(first)
    second_edges = edge_set(second)
    shared = len(first_edges & second_edges)
    return objective + 0.05 * shared


def edge_set(routes):
    edges = set()
    for route in routes:
        path = [0] + route + [0]
        for left, right in zip(path, path[1:]):
            edges.add((left, right))
    return edges


def ordered_crossover(parent_a, parent_b, rng):
    if len(parent_a) < 3:
        return parent_a[:]
    start = rng.randrange(0, len(parent_a) - 1)
    end = rng.randrange(start + 1, len(parent_a) + 1)
    segment = parent_a[start:end]
    segment_set = set(segment)
    remainder = [customer for customer in parent_b if customer not in segment_set]
    return remainder[:start] + segment + remainder[start:]


def mine_patterns(population, max_patterns):
    counts = Counter()
    for routes in population[:12]:
        for route in routes:
            max_len = min(6, len(route))
            for length in range(2, max_len + 1):
                for start in range(0, len(route) - length + 1):
                    pattern = tuple(route[start : start + length])
                    counts[pattern] += 1
                    counts[tuple(reversed(pattern))] += 1
    ranked = sorted(counts, key=lambda pattern: (counts[pattern] * len(pattern), counts[pattern], len(pattern)), reverse=True)
    return ranked[:max_patterns]


def pattern_injection_search(instance, routes, patterns, deadline, rng):
    current_objective = instance.routes_objective(routes)
    current_order = flatten_routes(routes)
    best_routes = None
    best_objective = current_objective
    for pattern in patterns:
        if time.time() >= deadline:
            break
        pattern_set = set(pattern)
        if len(pattern_set) != len(pattern) or not pattern_set.issubset(current_order):
            continue
        remaining = [customer for customer in current_order if customer not in pattern_set]
        positions = candidate_pattern_positions(instance, remaining, pattern)
        if len(positions) > 32:
            positions = rng.sample(positions, 32)
        for position in positions:
            if time.time() >= deadline:
                break
            order = remaining[:position] + list(pattern) + remaining[position:]
            split = split_giant_tour(instance, order)
            if split is None:
                continue
            candidate, objective = split
            if objective + 1e-9 < best_objective:
                best_routes = candidate
                best_objective = objective
    return best_routes


def candidate_pattern_positions(instance, order, pattern):
    positions = {0, len(order)}
    first = pattern[0]
    last = pattern[-1]
    for index, customer in enumerate(order):
        if instance.distance(customer, first) <= instance.distance(0, first) or instance.distance(customer, last) <= instance.distance(0, last):
            positions.add(index)
            positions.add(index + 1)
    return sorted(position for position in positions if 0 <= position <= len(order))


def localized_cache_search(instance, routes, rng, neighbors, deadline):
    candidate = copy_routes(routes)
    non_empty = [route for route in candidate if route]
    if not non_empty or time.time() >= deadline:
        return candidate
    route = max(non_empty, key=lambda item: instance.route_distance(item) / max(1, len(item)))
    if len(route) <= 2:
        return destroy_and_repair(instance, candidate, rng, neighbors, deadline)
    start = rng.randrange(len(route))
    length = min(len(route) - start, rng.randint(2, min(8, max(2, len(route)))))
    seed_segment = route[start : start + length]
    remove_set = set(seed_segment)
    target_remove = max(3, min(24 if instance.num_nodes <= 180 else 42, (instance.num_nodes - 1) // 8))
    frontier = list(seed_segment)
    rng.shuffle(frontier)
    for customer in frontier:
        if len(remove_set) >= target_remove:
            break
        related = sorted(
            (other for other in flatten_routes(candidate) if other not in remove_set),
            key=lambda other: (instance.distance(customer, other), rng.random()),
        )
        for other in related[:3]:
            remove_set.add(other)
            if len(remove_set) >= target_remove:
                break
    partial = [[customer for customer in route_ if customer not in remove_set] for route_ in candidate]
    removed = sorted(remove_set, key=lambda customer: (-int(instance.demands[customer]), rng.random()))
    partial = [route_ for route_ in partial if route_]
    for customer in removed:
        best_route = None
        best_position = 0
        best_delta = math.inf
        loads = [instance.route_demand(route_) for route_ in partial]
        for route_index, route_ in enumerate(partial):
            if loads[route_index] + int(instance.demands[customer]) > instance.vehicle_capacity:
                continue
            for position in range(len(route_) + 1):
                left = 0 if position == 0 else route_[position - 1]
                right = 0 if position == len(route_) else route_[position]
                delta = instance.distance(left, customer) + instance.distance(customer, right) - instance.distance(left, right)
                if route_ and not any(neighbor in neighbors[customer] for neighbor in route_):
                    delta += 0.05 * instance.distance(0, customer)
                if delta < best_delta:
                    best_delta = delta
                    best_route = route_index
                    best_position = position
        if best_route is None:
            partial.append([customer])
        else:
            partial[best_route].insert(best_position, customer)
    repaired, _ = repair_routes(instance, partial)
    return repaired


def hybrid_seed_budget(instance):
    if instance.num_nodes <= 60:
        return 0.4
    if instance.num_nodes <= 180:
        return 0.8
    return 1.2


def hgs_polish_slice(instance):
    if instance.num_nodes <= 60:
        return 0.35
    if instance.num_nodes <= 180:
        return 0.55
    return 0.75


def hgs_neighbor_count(instance):
    if instance.num_nodes <= 80:
        return min(48, instance.num_nodes - 1)
    if instance.num_nodes <= 180:
        return 64
    return 80


def initial_population_size(instance):
    if instance.num_nodes <= 60:
        return 8
    if instance.num_nodes <= 180:
        return 10
    return 12


def max_population_size(instance):
    if instance.num_nodes <= 60:
        return 16
    if instance.num_nodes <= 180:
        return 20
    return 24


def pattern_budget(instance):
    if instance.num_nodes <= 60:
        return 20
    if instance.num_nodes <= 180:
        return 28
    return 36


def time_remaining(deadline):
    return deadline - time.time()