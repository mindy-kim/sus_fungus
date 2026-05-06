from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import time
from typing import Iterable, Sequence

import numpy as np


# =============================================================================
# Type aliases (mirroring cvrp.core)
# =============================================================================
Route = list  # list[int]
Routes = list  # list[list[int]]
Pattern = tuple  # tuple[int, ...]


# =============================================================================
# VRPInstance (the stencil) -- the HGS solver runs through .solve()
# =============================================================================
class VRPInstance:
    numCustomers: int  # number of nodes (depot + customers); index 0 is the depot
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

    # ---- public solve entry point ------------------------------------------
    def solve(self, time_budget_s: float | None = None, seed: int = 0):
        self._ensure_distance_matrix()
        routes, objective, _meta = solve_hgs_search(self, time_budget_s=time_budget_s, seed=seed)
        self.solution = routes
        self.objective_value = objective
        return self.solution, self.objective_value

    # ---- methods/properties expected by the HGS algorithm ------------------
    # These mirror the CVRPInstance API so the HGS code is unchanged.
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

    def route_demand(self, route: Sequence[int]) -> int:
        if not route:
            return 0
        return int(self.demandOfCustomer[list(route)].sum())

    def route_distance(self, route: Sequence[int]) -> float:
        if not route:
            return 0.0
        if self._dist_matrix is None:
            self._ensure_distance_matrix()
        d = float(self._dist_matrix[0, route[0]]) + float(self._dist_matrix[route[-1], 0])
        for left, right in zip(route, route[1:]):
            d += float(self._dist_matrix[left, right])
        return d

    def routes_objective(self, routes: Sequence[Sequence[int]]) -> float:
        return sum(self.route_distance(route) for route in routes)

    # ---- file I/O (preserved from stencil) ---------------------------------
    def load_from_file(self, filename: str):
        try:
            with open(filename, 'r') as f:
                content = f.read().split()
                iterator = iter(content)

                self.numCustomers = int(next(iterator))
                self.numVehicles = int(next(iterator))
                self.vehicleCapacity = int(next(iterator))

                print(f"Number of customers: {self.numCustomers}")
                print(f"Number of vehicles: {self.numVehicles}")
                print(f"Vehicle capacity: {self.vehicleCapacity}")

                self.demandOfCustomer = np.zeros(self.numCustomers, dtype=int)
                self.xCoordOfCustomer = np.zeros(self.numCustomers)
                self.yCoordOfCustomer = np.zeros(self.numCustomers)

                for i in range(self.numCustomers):
                    self.demandOfCustomer[i] = int(next(iterator))
                    self.xCoordOfCustomer[i] = float(next(iterator))
                    self.yCoordOfCustomer[i] = float(next(iterator))

                for i in range(self.numCustomers):
                    print(f"{self.demandOfCustomer[i]} {self.xCoordOfCustomer[i]} {self.yCoordOfCustomer[i]}")
        except Exception as e:
            print(f"Error reading instance file: {e}")
            exit(1)

    def __str__(self):
        out = f"Number of customers: {self.numCustomers}\n"
        out += f"Number of vehicles: {self.numVehicles}\n"
        out += f"Vehicle capacity: {self.vehicleCapacity}\n"
        for i in range(self.numCustomers):
            out += f"{self.demandOfCustomer[i]} {self.xCoordOfCustomer[i]} {self.yCoordOfCustomer[i]}\n"
        return out


# =============================================================================
# Core helpers (from cvrp.core)
# =============================================================================
def copy_routes(routes: Sequence[Sequence[int]]) -> Routes:
    return [list(route) for route in routes]


def flatten_routes(routes: Sequence[Sequence[int]]) -> list:
    flat: list = []
    for route in routes:
        flat.extend(route)
    return flat


def unique_preserve_order(items: Iterable[int]) -> list:
    seen: set = set()
    ordered: list = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


# =============================================================================
# Repair helpers (from cvrp.repair) -- preserved verbatim
# =============================================================================
def split_giant_tour(
    instance: VRPInstance,
    customer_order: list,
    max_routes: int | None = None,
):
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
    parent: list = [[None] * (max_routes + 1) for _ in range(n + 1)]
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

    cuts: list = []
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

    routes: Routes = []
    previous = 0
    for cut in cuts[1:]:
        routes.append(customer_order[previous:cut])
        previous = cut
    routes.append(customer_order[previous:n])
    return routes, best[n][route_count]


def repair_routes(instance: VRPInstance, routes: Routes):
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


def greedy_capacity_pack(instance: VRPInstance, customer_order: list) -> Routes:
    if not customer_order:
        return []

    routes: Routes = []
    loads: list = []
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
        if len(routes) >= instance.num_vehicles:
            break
        routes.append([customer])
        loads.append(int(instance.demands[customer]))

    if sum(len(route) for route in routes) != len(customer_order):
        exact = _pack_by_backtracking(instance, customer_order)
        if exact is None:
            raise ValueError("Unable to pack customer without violating capacity")
        routes = exact

    return routes


def best_insertion_delta(instance: VRPInstance, route: Route, customer: int):
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


def _is_clean_feasible_cover(instance: VRPInstance, routes: Routes) -> bool:
    if len(routes) > instance.num_vehicles:
        return False
    seen: set = set()
    for route in routes:
        if instance.route_demand(route) > instance.vehicle_capacity:
            return False
        for customer in route:
            if customer in seen or customer <= 0 or customer >= instance.num_nodes:
                return False
            seen.add(customer)
    return seen == set(instance.customers)


def _pack_by_backtracking(instance: VRPInstance, customer_order: list):
    routes: Routes = [[] for _ in range(instance.num_vehicles)]
    loads = [0] * instance.num_vehicles
    ordered = sorted(customer_order, key=lambda customer: (-int(instance.demands[customer]), customer))

    def search(index: int) -> bool:
        if index == len(ordered):
            return True
        customer = ordered[index]
        demand = int(instance.demands[customer])
        used_loads: set = set()
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


# =============================================================================
# Hybrid solver helpers (substitutes for models.hybrid.solver)
# =============================================================================
@dataclass
class _SolverRun:
    routes: Routes
    objective: float


def neighbor_count(instance: VRPInstance) -> int:
    if instance.num_nodes <= 60:
        return min(20, instance.num_nodes - 1)
    if instance.num_nodes <= 180:
        return 30
    return 40


def nearest_neighbor_sets(instance: VRPInstance, k: int) -> dict:
    instance._ensure_distance_matrix()
    result: dict = {}
    for customer in instance.customers:
        # Pull row of distances to all other customers (skip depot and self)
        others = [c for c in instance.customers if c != customer]
        others.sort(key=lambda c: instance.distance(customer, c))
        result[customer] = set(others[:k])
    return result


def add_routes_to_pool(instance: VRPInstance, route_pool: dict, routes: Routes) -> None:
    for route in routes:
        if not route:
            continue
        key = frozenset(route)
        existing = route_pool.get(key)
        if existing is None or instance.route_distance(route) < instance.route_distance(existing):
            route_pool[key] = list(route)


def relaxed_nearest_tour_split(instance: VRPInstance, rng: random.Random, neighbors: dict) -> Routes:
    customers = list(instance.customers)
    if not customers:
        return []
    visited: set = set()
    tour: list = []
    current = 0
    while len(tour) < len(customers):
        remaining = [c for c in customers if c not in visited]
        remaining.sort(key=lambda c: instance.distance(current, c))
        k = min(5, len(remaining))
        # Bias toward nearest with light randomness
        idx = int((rng.random() ** 2) * k)
        picked = remaining[idx]
        tour.append(picked)
        visited.add(picked)
        current = picked

    split = split_giant_tour(instance, tour)
    if split is None:
        return greedy_capacity_pack(instance, tour)
    return split[0]


def _customer_multiset(routes: Routes) -> tuple:
    out: list = []
    for r in routes:
        out.extend(r)
    out.sort()
    return tuple(out)


def local_search_hybrid(instance, routes, neighbors, deadline, tight_capacity):
    routes = [r[:] for r in routes if r]
    if not routes:
        return routes, 0.0
 
    cap = instance.vehicle_capacity
    d = instance.distance
    demands = instance.demands
    invariant = _customer_multiset(routes)
 
    # Incremental state: per-route load + (customer -> route_idx, position).
    loads = [instance.route_demand(r) for r in routes]
    cust_route: dict = {}
    cust_pos: dict = {}
    for ri, r in enumerate(routes):
        for pi, c in enumerate(r):
            cust_route[c] = ri
            cust_pos[c] = pi
 
    def _refresh_positions(r_idx: int) -> None:
        r = routes[r_idx]
        for pi, c in enumerate(r):
            cust_pos[c] = pi
            cust_route[c] = r_idx
 
    def or_opt_step() -> bool:
        # First-improvement: relocate a 1/2/3-customer segment, with optional
        # reversal, into the insertion slot adjacent to a neighbor of the
        # segment's endpoints (or to a depot end).
        for r1_idx in range(len(routes)):
            r1 = routes[r1_idx]
            if not r1 or time.time() >= deadline:
                continue
            len_r1 = len(r1)
            for seg_len in (1, 2, 3):
                if seg_len > len_r1:
                    continue
                for i in range(len_r1 - seg_len + 1):
                    seg = r1[i:i + seg_len]
                    seg_demand = sum(int(demands[c]) for c in seg)
 
                    # Edges around the removal site in r1.
                    a = r1[i - 1] if i > 0 else 0
                    b = seg[0]
                    e = seg[-1]
                    f = r1[i + seg_len] if i + seg_len < len_r1 else 0
                    delta_remove = d(a, f) - d(a, b) - d(e, f)
                    virt_len = len_r1 - seg_len  # length of r1 after removal
 
                    # Build candidate destinations (r2_idx, j). Allow
                    # duplicates -- a list is cheaper than a set of tuples.
                    cands: list = []
                    for n in neighbors.get(b, ()):
                        r2i = cust_route.get(n, -1)
                        if r2i < 0:
                            continue
                        pos = cust_pos[n]
                        cands.append((r2i, pos))
                        cands.append((r2i, pos + 1))
                    if e != b:
                        for n in neighbors.get(e, ()):
                            r2i = cust_route.get(n, -1)
                            if r2i < 0:
                                continue
                            pos = cust_pos[n]
                            cands.append((r2i, pos))
                            cands.append((r2i, pos + 1))
                    for r2_idx in range(len(routes)):
                        cands.append((r2_idx, 0))
                        cands.append((r2_idx, len(routes[r2_idx])))
 
                    for r2_idx, j in cands:
                        r2 = routes[r2_idx]
                        if r2_idx != r1_idx:
                            if loads[r2_idx] + seg_demand > cap:
                                continue
                            if j > len(r2):
                                continue
                            g = r2[j - 1] if j > 0 else 0
                            h = r2[j] if j < len(r2) else 0
                        else:
                            # Intra-route: j is a position in the virtual
                            # route obtained by removing seg from r1. Map
                            # back into r1's real indices for g and h.
                            if j > virt_len:
                                continue
                            if j == 0:
                                g = 0
                            elif j - 1 < i:
                                g = r1[j - 1]
                            else:
                                g = r1[j - 1 + seg_len]
                            if j == virt_len:
                                h = 0
                            elif j < i:
                                h = r1[j]
                            else:
                                h = r1[j + seg_len]
 
                        for reverse in (False, True):
                            # Skip identity move (intra-route, same slot, no
                            # reversal -- nothing changes).
                            if r2_idx == r1_idx and j == i and not reverse:
                                continue
                            seg_first = e if reverse else b
                            seg_last = b if reverse else e
                            delta = (
                                delta_remove
                                + d(g, seg_first)
                                + d(seg_last, h)
                                - d(g, h)
                            )
                            if delta < -1e-9:
                                seg_to_use = seg[::-1] if reverse else list(seg)
                                new_r1 = r1[:i] + r1[i + seg_len:]
                                if r2_idx == r1_idx:
                                    routes[r1_idx] = (
                                        new_r1[:j] + seg_to_use + new_r1[j:]
                                    )
                                    _refresh_positions(r1_idx)
                                else:
                                    routes[r1_idx] = new_r1
                                    routes[r2_idx] = (
                                        r2[:j] + seg_to_use + r2[j:]
                                    )
                                    loads[r1_idx] -= seg_demand
                                    loads[r2_idx] += seg_demand
                                    _refresh_positions(r1_idx)
                                    _refresh_positions(r2_idx)
                                return True
        return False
 
    def two_opt_step() -> bool:
        for r_idx in range(len(routes)):
            r = routes[r_idx]
            if len(r) < 4 or time.time() >= deadline:
                continue
            for i in range(len(r) - 1):
                for j in range(i + 2, len(r)):
                    a = r[i - 1] if i > 0 else 0
                    b = r[i]
                    c_node = r[j]
                    nxt = r[j + 1] if j + 1 < len(r) else 0
                    delta = (
                        d(a, c_node) + d(b, nxt) - d(a, b) - d(c_node, nxt)
                    )
                    if delta < -1e-9:
                        r[i:j + 1] = r[i:j + 1][::-1]
                        _refresh_positions(r_idx)
                        return True
        return False
 
    def swap_step() -> bool:
        for r1_idx in range(len(routes)):
            r1 = routes[r1_idx]
            if not r1 or time.time() >= deadline:
                continue
            d1 = loads[r1_idx]
            for i in range(len(r1)):
                cu = r1[i]
                cu_d = int(demands[cu])
                a1 = r1[i - 1] if i > 0 else 0
                b1 = r1[i + 1] if i + 1 < len(r1) else 0
                d_a1_cu = d(a1, cu)
                d_cu_b1 = d(cu, b1)
                # Neighbor-restricted swap partners.
                for cv in neighbors.get(cu, ()):
                    r2_idx = cust_route.get(cv)
                    if r2_idx is None or r2_idx == r1_idx:
                        continue
                    r2 = routes[r2_idx]
                    cv_d = int(demands[cv])
                    d2 = loads[r2_idx]
                    if d1 - cu_d + cv_d > cap or d2 - cv_d + cu_d > cap:
                        continue
                    j = cust_pos[cv]
                    a2 = r2[j - 1] if j > 0 else 0
                    b2 = r2[j + 1] if j + 1 < len(r2) else 0
                    old = d_a1_cu + d_cu_b1 + d(a2, cv) + d(cv, b2)
                    new = d(a1, cv) + d(cv, b1) + d(a2, cu) + d(cu, b2)
                    if new + 1e-9 < old:
                        r1[i] = cv
                        r2[j] = cu
                        loads[r1_idx] = d1 - cu_d + cv_d
                        loads[r2_idx] = d2 - cv_d + cu_d
                        cust_route[cv] = r1_idx
                        cust_route[cu] = r2_idx
                        cust_pos[cv] = i
                        cust_pos[cu] = j
                        return True
        return False
 
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if iterations > 5000:
            break
        if or_opt_step():
            assert _customer_multiset(routes) == invariant, "or_opt broke cover"
            continue
        if two_opt_step():
            assert _customer_multiset(routes) == invariant, "two_opt broke cover"
            continue
        if swap_step():
            assert _customer_multiset(routes) == invariant, "swap broke cover"
            continue
        break
 
    routes = [r for r in routes if r]
    return routes, instance.routes_objective(routes)


def destroy_and_repair(
    instance: VRPInstance,
    routes: Routes,
    rng: random.Random,
    neighbors: dict,
    deadline: float,
) -> Routes:
    routes = [r[:] for r in routes if r]
    all_customers = [c for r in routes for c in r]
    if not all_customers:
        return routes

    n = len(all_customers)
    target_remove = max(2, min(n - 1, n // 4))
    seed = rng.choice(all_customers)
    removed: set = {seed}

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
                # let repair sort it out
                new_routes.append([c])
        else:
            new_routes[best_target].insert(best_pos, c)

    repaired, _ = repair_routes(instance, new_routes)
    return repaired


def route_pool_recombine(
    instance: VRPInstance,
    route_pool: dict,
    deadline: float,
    best_routes: Routes | None = None,
) -> Routes | None:
    if not route_pool:
        return None

    candidates = []
    for key, route in route_pool.items():
        if not route:
            continue
        cost = instance.route_distance(route)
        candidates.append((cost / max(1, len(route)), cost, list(route)))
    candidates.sort(key=lambda item: (item[0], item[1]))

    covered: set = set()
    selected: Routes = []
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
                # Couldn't recombine into a feasible cover
                return None
        else:
            selected[best_target].insert(best_pos, c)

    repaired, _ = repair_routes(instance, selected)
    return repaired


def solve_hybrid(instance: VRPInstance, time_budget_s: float, seed: int) -> _SolverRun:
    rng = random.Random(seed)
    deadline = time.time() + max(0.05, time_budget_s)
    neighbors = nearest_neighbor_sets(instance, neighbor_count(instance))
    routes = relaxed_nearest_tour_split(instance, rng, neighbors)
    routes, _ = local_search_hybrid(instance, routes, neighbors, deadline, False)
    return _SolverRun(routes=routes, objective=instance.routes_objective(routes))


def sweep_clustering(instance: VRPInstance, angle_offset: float = 0.0) -> Routes:
    """Gillett-Miller sweep heuristic.

    1. Compute polar angle of every customer relative to the depot.
    2. Optionally rotate by ``angle_offset`` (in radians) so different offsets
       produce different clusterings -- useful for diversifying the initial
       population.
    3. Sweep angularly and pack customers into clusters greedily until the
       vehicle capacity is hit, then start a new cluster.
    4. Within each cluster, build a nearest-neighbor TSP tour from the depot
       and polish it with first-improvement 2-opt.

    Returns a list of routes covering every customer exactly once. The result
    may use more than ``instance.num_vehicles`` clusters when sweeping is
    not space-efficient; ``repair_routes`` (called by the consumer) will fix
    that via the giant-tour split.
    """
    customers = list(instance.customers)
    if not customers:
        return []

    depot_x = float(instance.xCoordOfCustomer[0])
    depot_y = float(instance.yCoordOfCustomer[0])

    def polar(c: int) -> float:
        dx = float(instance.xCoordOfCustomer[c]) - depot_x
        dy = float(instance.yCoordOfCustomer[c]) - depot_y
        # Customers coincident with the depot get a stable angle of 0
        if dx == 0.0 and dy == 0.0:
            return 0.0
        return math.atan2(dy, dx)

    two_pi = 2.0 * math.pi
    ordered = sorted(customers, key=lambda c: (polar(c) - angle_offset) % two_pi)

    # Sweep into capacity-feasible clusters
    cap = instance.vehicle_capacity
    clusters: list = []
    current: list = []
    current_load = 0
    for c in ordered:
        d = int(instance.demands[c])
        if current and current_load + d > cap:
            clusters.append(current)
            current = []
            current_load = 0
        current.append(c)
        current_load += d
    if current:
        clusters.append(current)

    # Order each cluster: nearest-neighbor from depot, then 2-opt
    routes: Routes = []
    for cluster in clusters:
        if not cluster:
            continue
        remaining = set(cluster)
        tour: list = []
        current_node = 0
        while remaining:
            next_c = min(remaining, key=lambda c: instance.distance(current_node, c))
            tour.append(next_c)
            remaining.remove(next_c)
            current_node = next_c

        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour)):
                    a = tour[i - 1] if i > 0 else 0
                    b = tour[i]
                    c_node = tour[j]
                    d = tour[j + 1] if j + 1 < len(tour) else 0
                    delta = (
                        instance.distance(a, c_node)
                        + instance.distance(b, d)
                        - instance.distance(a, b)
                        - instance.distance(c_node, d)
                    )
                    if delta < -1e-9:
                        tour[i : j + 1] = tour[i : j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        routes.append(tour)

    return routes


# =============================================================================
# HGS solver -- preserved as in the original (only imports redirected)
# =============================================================================
def solve_hgs_search(
    instance: VRPInstance,
    time_budget_s: float | None,
    seed: int,
    incumbent: Routes | None = None,
):
    started = time.time()
    budget = time_budget_s if time_budget_s is not None else 300
    deadline = started + max(0.2, budget)
    rng = random.Random(seed)
    neighbors = nearest_neighbor_sets(instance, max(neighbor_count(instance), hgs_neighbor_count(instance)))
    tight_capacity = instance.total_demand / max(1, instance.num_vehicles * instance.vehicle_capacity) >= 0.86

    best_routes: Routes | None = None
    best_objective = math.inf
    best_origin = "none"
    population: list = []
    route_pool: dict = {}
    crossovers = 0
    pattern_improvements = 0
    localized_improvements = 0
    pool_improvements = 0

    def consider(candidate: Routes, origin: str, intensify: bool = True) -> bool:
        nonlocal best_routes, best_objective, best_origin, pattern_improvements, localized_improvements
        if time.time() >= deadline:
            return False
        repaired, _ = repair_routes(instance, candidate)
        if intensify and time.time() < deadline:
            polish_deadline = min(deadline, time.time() + hgs_polish_slice(instance))
            repaired, _ = local_search_hybrid(instance, repaired, neighbors, polish_deadline, tight_capacity)
            try:
                from models.gradient.solver import gradient_route_descent  # type: ignore

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

    # Sweep clustering: cheap, deterministic starting point so HGS always has
    # a baseline ordering to compare against, even when the time budget is tight.
    if time_remaining(deadline) > 0.05:
        sweep_routes = sweep_clustering(instance, angle_offset=0.0)
        if sweep_routes:
            consider(sweep_routes, "sweep_clustering")

    seed_round = 0
    while len(population) < initial_population_size(instance) and time_remaining(deadline) > 0.08:
        seed_round += 1
        seed_budget = min(hybrid_seed_budget(instance), max(0.05, time_remaining(deadline) * 0.2))
        bucket = seed_round % 4
        if bucket == 0:
            candidate = relaxed_nearest_tour_split(instance, rng, neighbors)
        elif bucket == 1:
            candidate = solve_hybrid(instance, time_budget_s=seed_budget, seed=rng.randrange(1 << 30)).routes
        elif bucket == 2:
            order = sorted(instance.customers, key=lambda customer: (-int(instance.demands[customer]), rng.random()))
            candidate = greedy_capacity_pack(instance, order)
        else:
            # Rotated sweep -- different angle_offset gives a different
            # clustering, so this keeps the population diverse.
            offset = rng.uniform(0.0, 2.0 * math.pi)
            candidate = sweep_clustering(instance, angle_offset=offset)
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

    return (
        best_routes,
        best_objective,
        {
            "budget_s": round(max(0.2, budget), 6),
            "origin": best_origin,
            "population_size": len(population),
            "crossovers": crossovers,
            "route_pool_size": len(route_pool),
            "pattern_improvements": pattern_improvements,
            "localized_improvements": localized_improvements,
            "route_pool_improvements": pool_improvements,
            "runtime_budget_s": round(time.time() - started, 6),
        },
    )


def solve_pyvrp_hgs(*args, **kwargs):
    # Intentionally removed: pyvrp is a disallowed external solver for this
    # assignment. This stub is retained only so that any straggling reference
    # raises a clear error rather than silently importing pyvrp again.
    raise NotImplementedError("pyvrp is not used in this implementation")


def add_to_population(instance: VRPInstance, population: list, routes: Routes) -> None:
    cleaned = [route[:] for route in routes if route]
    signature = tuple(tuple(route) for route in cleaned)
    if any(tuple(tuple(route) for route in existing) == signature for existing in population):
        return
    population.append(cleaned)
    population.sort(key=instance.routes_objective)
    del population[max_population_size(instance):]


def select_parents(instance: VRPInstance, population: list, rng: random.Random):
    sample_size = min(len(population), 5)
    first = min(rng.sample(population, sample_size), key=instance.routes_objective)
    second = min(rng.sample(population, sample_size), key=lambda routes: parent_score(instance, first, routes))
    if first is second and len(population) > 1:
        second = population[1] if population[0] is first else population[0]
    return first, second


def parent_score(instance: VRPInstance, first: Routes, second: Routes) -> float:
    objective = instance.routes_objective(second)
    first_edges = edge_set(first)
    second_edges = edge_set(second)
    shared = len(first_edges & second_edges)
    return objective + 0.05 * shared


def edge_set(routes: Routes) -> set:
    edges: set = set()
    for route in routes:
        path = [0] + route + [0]
        for left, right in zip(path, path[1:]):
            edges.add((left, right))
    return edges


def ordered_crossover(parent_a: list, parent_b: list, rng: random.Random) -> list:
    if len(parent_a) < 3:
        return parent_a[:]
    start = rng.randrange(0, len(parent_a) - 1)
    end = rng.randrange(start + 1, len(parent_a) + 1)
    segment = parent_a[start:end]
    segment_set = set(segment)
    remainder = [customer for customer in parent_b if customer not in segment_set]
    return remainder[:start] + segment + remainder[start:]


def mine_patterns(population: list, max_patterns: int):
    counts: Counter = Counter()
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


def pattern_injection_search(
    instance: VRPInstance,
    routes: Routes,
    patterns: list,
    deadline: float,
    rng: random.Random,
) -> Routes | None:
    current_objective = instance.routes_objective(routes)
    current_order = flatten_routes(routes)
    best_routes: Routes | None = None
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


def candidate_pattern_positions(instance: VRPInstance, order: list, pattern) -> list:
    positions = {0, len(order)}
    first = pattern[0]
    last = pattern[-1]
    for index, customer in enumerate(order):
        if instance.distance(customer, first) <= instance.distance(0, first) or instance.distance(customer, last) <= instance.distance(0, last):
            positions.add(index)
            positions.add(index + 1)
    return sorted(position for position in positions if 0 <= position <= len(order))


def localized_cache_search(
    instance: VRPInstance,
    routes: Routes,
    rng: random.Random,
    neighbors: dict,
    deadline: float,
) -> Routes:
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


# =============================================================================
# HGS budget / size-tier helpers (preserved verbatim)
# =============================================================================
def default_hgs_budget(instance: VRPInstance) -> float:
    if instance.num_nodes <= 60:
        return 18.0
    if instance.num_nodes <= 180:
        return 36.0
    return 72.0


def hybrid_seed_budget(instance: VRPInstance) -> float:
    if instance.num_nodes <= 60:
        return 0.4
    if instance.num_nodes <= 180:
        return 0.8
    return 1.2


def hgs_polish_slice(instance: VRPInstance) -> float:
    if instance.num_nodes <= 60:
        return 0.35
    if instance.num_nodes <= 180:
        return 0.55
    return 0.75


def hgs_neighbor_count(instance: VRPInstance) -> int:
    if instance.num_nodes <= 80:
        return min(48, instance.num_nodes - 1)
    if instance.num_nodes <= 180:
        return 64
    return 80


def initial_population_size(instance: VRPInstance) -> int:
    if instance.num_nodes <= 60:
        return 8
    if instance.num_nodes <= 180:
        return 10
    return 12


def max_population_size(instance: VRPInstance) -> int:
    if instance.num_nodes <= 60:
        return 16
    if instance.num_nodes <= 180:
        return 20
    return 24


def pattern_budget(instance: VRPInstance) -> int:
    if instance.num_nodes <= 60:
        return 20
    if instance.num_nodes <= 180:
        return 28
    return 36


def time_remaining(deadline: float) -> float:
    return deadline - time.time()