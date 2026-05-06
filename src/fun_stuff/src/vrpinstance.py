import random
import sys
from math import atan2, hypot, pi
from pathlib import Path
from time import perf_counter


VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor"
if VENDOR_DIR.exists():
    vendor_path = str(VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

try:
    from ortools.linear_solver import pywraplp
except ImportError:
    pywraplp = None


def build_solver_config(**overrides):
    config = {
        "construction_candidate_limit": 3,
        "use_interroute_ls": True,
        "local_search_max_passes": 40,
        "use_destroy": False,
        "destroy_method": "spatial",
        "repair_method": "cheapest",
        "destroy_portfolio": None,
        "destroy_selection": "random",
        "random_seed": 0,
        "destroy_samples": 8,
        "destroy_fraction": 0.18,
        "destroy_min": 4,
        "destroy_max": 48,
        "lns_start_limit": 2,
        "lns_stall_limit": 4,
        "lns_restart_rounds": 1,
        "lns_time_budget_sec": None,
        "lns_improvement_reset": True,
        "post_repair_local_search": True,
        "use_oropt": False,
        "oropt_segment_lengths": (2, 3),
        "exact_repair_max_customers": 18,
        "exact_repair_max_routes": 3,
        "exact_repair_time_limit_sec": 0.25,
    }
    config.update(overrides)
    return config


BASELINE_SOLVER_CONFIG = build_solver_config(
    use_interroute_ls=False,
    use_destroy=False,
)

LOCAL_SEARCH_SOLVER_CONFIG = build_solver_config(
    use_interroute_ls=True,
    use_destroy=False,
)

TOP3_DESTROY_PORTFOLIO = [
    {"destroy_method": "related", "repair_method": "regret2"},
    {"destroy_method": "string", "repair_method": "cheapest"},
    {"destroy_method": "related", "repair_method": "local_split"},
]

DEFAULT_SOLVER_CONFIG = build_solver_config(
    use_interroute_ls=True,
    use_destroy=True,
    destroy_portfolio=TOP3_DESTROY_PORTFOLIO,
    destroy_selection="adaptive",
    random_seed=0,
    lns_restart_rounds=3,
    lns_time_budget_sec=10.0,
    destroy_samples=12,
)


class VRPInstance:
    def __init__(self, filename: str, config=None):
        self.solution = None
        self.objective_value = None
        self.config = DEFAULT_SOLVER_CONFIG.copy()
        if config is not None:
            self.config.update(config)
        self.load_from_file(filename)
        self._build_distance_matrix()

    def solve(self):
        if self.numCustomers == 0:
            return self._store_solution([[] for _ in range(self.numVehicles)], 0.0)

        candidates = self._construct_candidates()
        if not candidates:
            raise RuntimeError("Failed to build a feasible CVRP solution.")

        candidates = self._apply_local_search_to_candidates(candidates)
        return self._solve_candidate_pool(candidates)

    def solve_from_order(self, order):
        cleaned = [int(customer) for customer in order]
        expected = set(self.customers)
        actual = set(cleaned)
        if len(cleaned) != self.numCustomers or actual != expected:
            raise ValueError("Order must contain every customer exactly once.")

        routes, _ = self._split_order(cleaned)
        if routes is None:
            raise RuntimeError("Failed to split warm-start order into feasible routes.")
        return self.solve_from_routes(routes)

    def solve_from_routes(self, routes):
        candidate_routes = self._canonicalize_routes([route[:] for route in routes if route])
        self._validate_routes(candidate_routes)

        candidates = []
        self._store_candidate(
            candidates,
            candidate_routes,
            self.config["construction_candidate_limit"],
        )
        candidates = self._apply_local_search_to_candidates(candidates)
        return self._solve_candidate_pool(candidates)

    def _apply_local_search_to_candidates(self, candidates):
        if not self.config["use_interroute_ls"]:
            return candidates

        refined = []
        for _, routes in candidates:
            improved_routes = self._improve_with_local_search(routes)
            self._store_candidate(
                refined,
                improved_routes,
                self.config["construction_candidate_limit"],
            )
        return refined or candidates

    def _solve_candidate_pool(self, candidates):
        if self.config["use_destroy"]:
            refined = self._run_lns_rounds(candidates)
            if refined:
                candidates = refined

        best_cost, best_routes = candidates[0]
        return self._store_solution(best_routes, best_cost)

    def _store_solution(self, routes, objective_value):
        padded_routes = self._pad_routes(routes)
        self.solution = self._format_solution(padded_routes)
        self.objective_value = round(objective_value, 4)
        return self.solution, self.objective_value

    def _validate_routes(self, routes):
        if len(routes) > self.numVehicles:
            raise ValueError("Warm-start routes exceed the available number of vehicles.")

        seen = set()
        for route in routes:
            load = 0
            for customer in route:
                if customer < 1 or customer >= self.numNodes:
                    raise ValueError(f"Invalid customer index in warm-start routes: {customer}")
                if customer in seen:
                    raise ValueError(f"Customer {customer} appears more than once in warm-start routes.")
                seen.add(customer)
                load += self.demandOfCustomer[customer]
            if load > self.vehicleCapacity:
                raise ValueError("Warm-start route exceeds vehicle capacity.")

        expected = set(self.customers)
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise ValueError(
                f"Warm-start routes must cover every customer exactly once "
                f"(missing={missing[:8]}, extra={extra[:8]})."
            )

    def load_from_file(self, filename: str):
        try:
            with open(filename, "r", encoding="utf-8") as handle:
                content = handle.read().split()

            iterator = iter(content)
            self.numNodes = int(next(iterator))
            self.numVehicles = int(next(iterator))
            self.vehicleCapacity = int(next(iterator))
            self.numCustomers = self.numNodes - 1

            self.demandOfCustomer = [0] * self.numNodes
            self.xCoordOfCustomer = [0.0] * self.numNodes
            self.yCoordOfCustomer = [0.0] * self.numNodes

            for idx in range(self.numNodes):
                self.demandOfCustomer[idx] = int(next(iterator))
                self.xCoordOfCustomer[idx] = float(next(iterator))
                self.yCoordOfCustomer[idx] = float(next(iterator))
        except Exception as exc:
            raise RuntimeError(f"Error reading instance file: {exc}") from exc

    def _build_distance_matrix(self):
        self.distance = [[0.0] * self.numNodes for _ in range(self.numNodes)]
        for i in range(self.numNodes):
            xi = self.xCoordOfCustomer[i]
            yi = self.yCoordOfCustomer[i]
            for j in range(i + 1, self.numNodes):
                dist = hypot(
                    xi - self.xCoordOfCustomer[j],
                    yi - self.yCoordOfCustomer[j],
                )
                self.distance[i][j] = dist
                self.distance[j][i] = dist

        self.customers = list(range(1, self.numNodes))
        self.depotDistance = self.distance[0]
        self.angles = {}
        self.radii = {}
        for customer in self.customers:
            angle = atan2(
                self.yCoordOfCustomer[customer] - self.yCoordOfCustomer[0],
                self.xCoordOfCustomer[customer] - self.xCoordOfCustomer[0],
            )
            self.angles[customer] = angle
            self.radii[customer] = self.depotDistance[customer]

        self.minX = min(self.xCoordOfCustomer[1:], default=0.0)
        self.maxX = max(self.xCoordOfCustomer[1:], default=0.0)
        self.minY = min(self.yCoordOfCustomer[1:], default=0.0)
        self.maxY = max(self.yCoordOfCustomer[1:], default=0.0)

    def _construct_candidates(self):
        candidates = []
        for order in self._generate_orderings():
            routes, _ = self._split_order(order)
            if routes is None:
                continue

            improved_routes = [self._two_opt(route) for route in routes]
            self._store_candidate(
                candidates,
                improved_routes,
                self.config["construction_candidate_limit"],
            )
        return candidates

    def _store_candidate(self, candidates, routes, limit):
        canonical = self._canonicalize_routes(routes)
        signature = self._routes_signature(canonical)
        cost = self._total_cost(canonical)

        for idx, (_, existing_routes) in enumerate(candidates):
            if self._routes_signature(existing_routes) == signature:
                if cost + 1e-9 < candidates[idx][0]:
                    candidates[idx] = (cost, canonical)
                candidates.sort(key=lambda item: item[0])
                return

        candidates.append((cost, canonical))
        candidates.sort(key=lambda item: item[0])
        del candidates[limit:]

    def _run_lns_rounds(self, candidates):
        incumbent_pool = [(cost, routes[:]) for cost, routes in candidates]
        best_cost = incumbent_pool[0][0]
        lns_start = perf_counter()
        restart_rounds = max(1, self.config["lns_restart_rounds"])

        for round_idx in range(restart_rounds):
            remaining_budget = self._remaining_lns_budget(lns_start)
            if remaining_budget is not None and remaining_budget <= 0.0:
                break

            next_pool = []
            active = incumbent_pool[: self.config["lns_start_limit"]]
            if not active:
                break

            per_call_budget = None
            if remaining_budget is not None:
                per_call_budget = max(0.05, remaining_budget / max(1, len(active)))

            for candidate_idx, (_, routes) in enumerate(active):
                improved_routes = self._run_destroy_search(
                    routes,
                    seed_offset=round_idx * 4099 + candidate_idx * 257,
                    time_budget=per_call_budget,
                )
                self._store_candidate(
                    next_pool,
                    improved_routes,
                    self.config["construction_candidate_limit"],
                )

            for _, routes in incumbent_pool:
                self._store_candidate(
                    next_pool,
                    routes,
                    self.config["construction_candidate_limit"],
                )

            if not next_pool:
                break

            improved = next_pool[0][0] + 1e-9 < best_cost
            incumbent_pool = next_pool
            best_cost = incumbent_pool[0][0]

            if (
                improved
                and self.config["lns_improvement_reset"]
                and self.config["lns_time_budget_sec"] is not None
            ):
                lns_start = perf_counter()

        return incumbent_pool

    def _remaining_lns_budget(self, lns_start):
        time_budget = self.config["lns_time_budget_sec"]
        if time_budget is None:
            return None
        return time_budget - (perf_counter() - lns_start)

    def _generate_orderings(self):
        seen = set()
        orderings = []

        sweep_offsets = 8
        for step in range(sweep_offsets):
            offset = (2.0 * pi * step) / sweep_offsets
            angle_order = tuple(
                sorted(
                    self.customers,
                    key=lambda customer: (
                        (self.angles[customer] - offset) % (2.0 * pi),
                        self.radii[customer],
                        customer,
                    ),
                )
            )
            self._append_order(orderings, seen, angle_order)
            self._append_order(orderings, seen, tuple(reversed(angle_order)))

        self._append_order(
            orderings,
            seen,
            tuple(
                sorted(
                    self.customers,
                    key=lambda customer: (
                        self.xCoordOfCustomer[customer],
                        self.yCoordOfCustomer[customer],
                        customer,
                    ),
                )
            ),
        )
        self._append_order(
            orderings,
            seen,
            tuple(
                sorted(
                    self.customers,
                    key=lambda customer: (
                        self.yCoordOfCustomer[customer],
                        self.xCoordOfCustomer[customer],
                        customer,
                    ),
                )
            ),
        )
        self._append_order(
            orderings,
            seen,
            tuple(
                sorted(
                    self.customers,
                    key=lambda customer: (
                        self.xCoordOfCustomer[customer] + self.yCoordOfCustomer[customer],
                        self.xCoordOfCustomer[customer] - self.yCoordOfCustomer[customer],
                        customer,
                    ),
                )
            ),
        )
        morton_order = tuple(
            sorted(
                self.customers,
                key=lambda customer: (
                    self._morton_code(
                        self.xCoordOfCustomer[customer],
                        self.yCoordOfCustomer[customer],
                    ),
                    customer,
                ),
            )
        )
        self._append_order(orderings, seen, morton_order)
        self._append_order(orderings, seen, tuple(reversed(morton_order)))
        self._append_order(
            orderings,
            seen,
            tuple(
                self._packed_route_order(
                    key=lambda customer: (
                        self.demandOfCustomer[customer],
                        self.radii[customer],
                        customer,
                    )
                )
            ),
        )
        self._append_order(
            orderings,
            seen,
            tuple(
                self._packed_route_order(
                    key=lambda customer: (
                        self.radii[customer],
                        self.demandOfCustomer[customer],
                        customer,
                    )
                )
            ),
        )

        for seed in self._nearest_neighbor_seeds():
            self._append_order(orderings, seen, tuple(self._nearest_neighbor_order(seed)))

        return orderings

    def _append_order(self, orderings, seen, order):
        if order and order not in seen:
            orderings.append(order)
            seen.add(order)

    def _morton_code(self, x, y):
        if self.maxX == self.minX:
            norm_x = 0
        else:
            norm_x = int(((x - self.minX) / (self.maxX - self.minX)) * 65535)

        if self.maxY == self.minY:
            norm_y = 0
        else:
            norm_y = int(((y - self.minY) / (self.maxY - self.minY)) * 65535)

        code = 0
        for bit in range(16):
            code |= ((norm_x >> bit) & 1) << (2 * bit)
            code |= ((norm_y >> bit) & 1) << (2 * bit + 1)
        return code

    def _nearest_neighbor_seeds(self):
        seeds = []
        by_radius = sorted(self.customers, key=lambda customer: self.radii[customer], reverse=True)
        for customer in by_radius[:4]:
            if customer not in seeds:
                seeds.append(customer)

        angle_sorted = sorted(self.customers, key=lambda customer: self.angles[customer])
        for customer in (angle_sorted[0], angle_sorted[len(angle_sorted) // 2], angle_sorted[-1]):
            if customer not in seeds:
                seeds.append(customer)
        return seeds

    def _nearest_neighbor_order(self, seed):
        remaining = set(self.customers)
        order = [seed]
        remaining.remove(seed)
        current = seed

        while remaining:
            next_customer = min(
                remaining,
                key=lambda customer: (self.distance[current][customer], self.radii[customer]),
            )
            order.append(next_customer)
            remaining.remove(next_customer)
            current = next_customer

        return order

    def _packed_route_order(self, key):
        routes = [[] for _ in range(self.numVehicles)]
        remaining_capacity = [self.vehicleCapacity] * self.numVehicles
        route_centers = [[0.0, 0.0, 0] for _ in range(self.numVehicles)]

        for customer in sorted(self.customers, key=key, reverse=True):
            best_route = None
            best_score = None
            demand = self.demandOfCustomer[customer]

            for route_idx in range(self.numVehicles):
                if remaining_capacity[route_idx] < demand:
                    continue

                slack = remaining_capacity[route_idx] - demand
                if route_centers[route_idx][2] == 0:
                    proximity = self.radii[customer]
                else:
                    center_x = route_centers[route_idx][0] / route_centers[route_idx][2]
                    center_y = route_centers[route_idx][1] / route_centers[route_idx][2]
                    proximity = hypot(
                        self.xCoordOfCustomer[customer] - center_x,
                        self.yCoordOfCustomer[customer] - center_y,
                    )

                score = (slack, proximity, route_idx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_route = route_idx

            if best_route is None:
                return []

            routes[best_route].append(customer)
            remaining_capacity[best_route] -= demand
            route_centers[best_route][0] += self.xCoordOfCustomer[customer]
            route_centers[best_route][1] += self.yCoordOfCustomer[customer]
            route_centers[best_route][2] += 1

        non_empty_routes = [route for route in routes if route]
        non_empty_routes.sort(key=self._route_angle)

        flattened = []
        for route in non_empty_routes:
            flattened.extend(self._order_route_customers(route))
        return flattened

    def _order_route_customers(self, route):
        if len(route) <= 2:
            return route[:]

        start = max(route, key=lambda customer: self.radii[customer])
        remaining = set(route)
        ordered = [start]
        remaining.remove(start)
        current = start

        while remaining:
            next_customer = min(
                remaining,
                key=lambda customer: (self.distance[current][customer], self.distance[0][customer]),
            )
            ordered.append(next_customer)
            remaining.remove(next_customer)
            current = next_customer

        return ordered

    def _split_order(self, order, max_routes=None):
        customer_count = len(order)
        route_limit = min(max_routes or self.numVehicles, customer_count)
        segments = [[] for _ in range(customer_count)]

        for start in range(customer_count):
            load = 0
            route_cost = 0.0
            for end in range(start, customer_count):
                customer = order[end]
                load += self.demandOfCustomer[customer]
                if load > self.vehicleCapacity:
                    break

                if end == start:
                    route_cost = 2.0 * self.depotDistance[customer]
                else:
                    previous = order[end - 1]
                    route_cost += (
                        self.distance[previous][customer]
                        + self.depotDistance[customer]
                        - self.depotDistance[previous]
                    )

                segments[start].append((end + 1, route_cost))

        inf = float("inf")
        predecessors = [[-1] * (customer_count + 1) for _ in range(route_limit + 1)]
        dp_prev = [inf] * (customer_count + 1)
        dp_prev[0] = 0.0

        best_cost = inf
        best_route_count = -1

        for route_count in range(1, route_limit + 1):
            dp_cur = [inf] * (customer_count + 1)
            for start in range(customer_count):
                if dp_prev[start] == inf:
                    continue
                base_cost = dp_prev[start]
                for end, route_cost in segments[start]:
                    total_cost = base_cost + route_cost
                    if total_cost + 1e-9 < dp_cur[end]:
                        dp_cur[end] = total_cost
                        predecessors[route_count][end] = start

            if dp_cur[customer_count] + 1e-9 < best_cost:
                best_cost = dp_cur[customer_count]
                best_route_count = route_count
            dp_prev = dp_cur

        if best_route_count < 0:
            return None, None

        routes = []
        end = customer_count
        route_count = best_route_count
        while route_count > 0:
            start = predecessors[route_count][end]
            if start < 0:
                return None, None
            routes.append(list(order[start:end]))
            end = start
            route_count -= 1

        routes.reverse()
        return routes, best_cost

    def _improve_with_local_search(self, routes):
        routes = self._canonicalize_routes(routes)
        route_loads = [self._route_load(route) for route in routes]

        passes = 0
        improved = True
        while improved and passes < self.config["local_search_max_passes"]:
            improved = False
            operators = [self._try_relocate_move]
            if self.config["use_oropt"]:
                operators.append(self._try_oropt_move)
            operators.extend((self._try_swap_move, self._try_two_opt_star_move))

            for operator in operators:
                changed, routes, route_loads = operator(routes, route_loads)
                if changed:
                    improved = True
                    passes += 1
                    break

        final_routes = [self._two_opt(route) for route in routes if route]
        return self._canonicalize_routes(final_routes)

    def _try_relocate_move(self, routes, route_loads):
        route_count = len(routes)
        allow_new_route = route_count < self.numVehicles

        for source_idx, source_route in enumerate(routes):
            source_cost = self._route_cost(source_route)
            for source_pos, customer in enumerate(source_route):
                demand = self.demandOfCustomer[customer]
                source_removed = source_route[:source_pos] + source_route[source_pos + 1 :]
                source_removed_cost = self._route_cost(source_removed)

                target_limit = route_count + (1 if allow_new_route else 0)
                for target_idx in range(target_limit):
                    if target_idx == route_count:
                        if not source_removed:
                            continue
                        new_target = [customer]
                        delta = source_removed_cost + self._route_cost(new_target) - source_cost
                        if delta < -1e-9:
                            new_routes = [route[:] for route in routes]
                            new_loads = route_loads[:]
                            new_routes[source_idx] = self._two_opt(source_removed)
                            new_loads[source_idx] -= demand
                            new_routes.append(new_target)
                            new_loads.append(demand)
                            return self._compress_routes(new_routes, new_loads)
                        continue

                    target_route = routes[target_idx]
                    target_cost = self._route_cost(target_route)

                    if source_idx == target_idx:
                        for target_pos in range(len(target_route) + 1):
                            if target_pos == source_pos or target_pos == source_pos + 1:
                                continue
                            insert_pos = target_pos
                            if target_pos > source_pos:
                                insert_pos -= 1
                            new_route = source_removed[:]
                            new_route.insert(insert_pos, customer)
                            new_cost = self._route_cost(new_route)
                            if new_cost + 1e-9 < source_cost:
                                new_routes = [route[:] for route in routes]
                                new_loads = route_loads[:]
                                new_routes[source_idx] = self._two_opt(new_route)
                                return True, new_routes, new_loads
                        continue

                    if route_loads[target_idx] + demand > self.vehicleCapacity:
                        continue

                    for target_pos in range(len(target_route) + 1):
                        new_target = target_route[:]
                        new_target.insert(target_pos, customer)
                        delta = (
                            source_removed_cost
                            + self._route_cost(new_target)
                            - source_cost
                            - target_cost
                        )
                        if delta < -1e-9:
                            new_routes = [route[:] for route in routes]
                            new_loads = route_loads[:]
                            new_routes[source_idx] = self._two_opt(source_removed)
                            new_routes[target_idx] = self._two_opt(new_target)
                            new_loads[source_idx] -= demand
                            new_loads[target_idx] += demand
                            return self._compress_routes(new_routes, new_loads)

        return False, routes, route_loads

    def _try_oropt_move(self, routes, route_loads):
        route_count = len(routes)
        allow_new_route = route_count < self.numVehicles

        for segment_length in self.config["oropt_segment_lengths"]:
            for source_idx, source_route in enumerate(routes):
                if len(source_route) <= segment_length:
                    continue
                source_cost = self._route_cost(source_route)

                for start_pos in range(len(source_route) - segment_length + 1):
                    segment = source_route[start_pos : start_pos + segment_length]
                    segment_load = sum(self.demandOfCustomer[customer] for customer in segment)
                    source_removed = (
                        source_route[:start_pos] + source_route[start_pos + segment_length :]
                    )
                    source_removed_cost = self._route_cost(source_removed)

                    target_limit = route_count + (1 if allow_new_route else 0)
                    for target_idx in range(target_limit):
                        if target_idx == route_count:
                            if not source_removed:
                                continue
                            new_target = segment[:]
                            delta = source_removed_cost + self._route_cost(new_target) - source_cost
                            if delta < -1e-9:
                                new_routes = [route[:] for route in routes]
                                new_loads = route_loads[:]
                                new_routes[source_idx] = self._two_opt(source_removed)
                                new_loads[source_idx] -= segment_load
                                new_routes.append(new_target)
                                new_loads.append(segment_load)
                                return self._compress_routes(new_routes, new_loads)
                            continue

                        target_route = routes[target_idx]
                        target_cost = self._route_cost(target_route)

                        if source_idx == target_idx:
                            for target_pos in range(len(target_route) + 1):
                                if start_pos <= target_pos <= start_pos + segment_length:
                                    continue
                                insert_pos = target_pos
                                if target_pos > start_pos:
                                    insert_pos -= segment_length
                                new_route = source_removed[:]
                                for offset, customer in enumerate(segment):
                                    new_route.insert(insert_pos + offset, customer)
                                new_cost = self._route_cost(new_route)
                                if new_cost + 1e-9 < source_cost:
                                    new_routes = [route[:] for route in routes]
                                    new_loads = route_loads[:]
                                    new_routes[source_idx] = self._two_opt(new_route)
                                    return True, new_routes, new_loads
                            continue

                        if route_loads[target_idx] + segment_load > self.vehicleCapacity:
                            continue

                        for target_pos in range(len(target_route) + 1):
                            new_target = target_route[:]
                            for offset, customer in enumerate(segment):
                                new_target.insert(target_pos + offset, customer)
                            delta = (
                                source_removed_cost
                                + self._route_cost(new_target)
                                - source_cost
                                - target_cost
                            )
                            if delta < -1e-9:
                                new_routes = [route[:] for route in routes]
                                new_loads = route_loads[:]
                                new_routes[source_idx] = self._two_opt(source_removed)
                                new_routes[target_idx] = self._two_opt(new_target)
                                new_loads[source_idx] -= segment_load
                                new_loads[target_idx] += segment_load
                                return self._compress_routes(new_routes, new_loads)

        return False, routes, route_loads

    def _try_swap_move(self, routes, route_loads):
        for first_idx in range(len(routes)):
            first_route = routes[first_idx]
            first_cost = self._route_cost(first_route)
            for second_idx in range(first_idx + 1, len(routes)):
                second_route = routes[second_idx]
                second_cost = self._route_cost(second_route)
                for first_pos, first_customer in enumerate(first_route):
                    first_demand = self.demandOfCustomer[first_customer]
                    for second_pos, second_customer in enumerate(second_route):
                        second_demand = self.demandOfCustomer[second_customer]
                        new_first_load = route_loads[first_idx] - first_demand + second_demand
                        new_second_load = route_loads[second_idx] - second_demand + first_demand
                        if (
                            new_first_load > self.vehicleCapacity
                            or new_second_load > self.vehicleCapacity
                        ):
                            continue

                        new_first = first_route[:]
                        new_second = second_route[:]
                        new_first[first_pos] = second_customer
                        new_second[second_pos] = first_customer
                        delta = (
                            self._route_cost(new_first)
                            + self._route_cost(new_second)
                            - first_cost
                            - second_cost
                        )
                        if delta < -1e-9:
                            new_routes = [route[:] for route in routes]
                            new_loads = route_loads[:]
                            new_routes[first_idx] = self._two_opt(new_first)
                            new_routes[second_idx] = self._two_opt(new_second)
                            new_loads[first_idx] = new_first_load
                            new_loads[second_idx] = new_second_load
                            return True, new_routes, new_loads

        return False, routes, route_loads

    def _try_two_opt_star_move(self, routes, route_loads):
        for first_idx in range(len(routes)):
            first_route = routes[first_idx]
            first_prefix_loads = self._prefix_loads(first_route)
            first_cost = self._route_cost(first_route)
            for second_idx in range(first_idx + 1, len(routes)):
                second_route = routes[second_idx]
                second_prefix_loads = self._prefix_loads(second_route)
                second_cost = self._route_cost(second_route)

                for first_cut in range(len(first_route) + 1):
                    first_prefix = first_route[:first_cut]
                    first_tail = first_route[first_cut:]
                    first_prefix_load = first_prefix_loads[first_cut]
                    first_tail_load = route_loads[first_idx] - first_prefix_load

                    for second_cut in range(len(second_route) + 1):
                        if first_cut == len(first_route) and second_cut == len(second_route):
                            continue
                        if first_cut == 0 and second_cut == 0:
                            continue

                        second_prefix = second_route[:second_cut]
                        second_tail = second_route[second_cut:]
                        second_prefix_load = second_prefix_loads[second_cut]
                        second_tail_load = route_loads[second_idx] - second_prefix_load

                        new_first_load = first_prefix_load + second_tail_load
                        new_second_load = second_prefix_load + first_tail_load
                        if (
                            new_first_load > self.vehicleCapacity
                            or new_second_load > self.vehicleCapacity
                        ):
                            continue

                        new_first = first_prefix + second_tail
                        new_second = second_prefix + first_tail
                        delta = (
                            self._route_cost(new_first)
                            + self._route_cost(new_second)
                            - first_cost
                            - second_cost
                        )
                        if delta < -1e-9:
                            new_routes = [route[:] for route in routes]
                            new_loads = route_loads[:]
                            new_routes[first_idx] = self._two_opt(new_first)
                            new_routes[second_idx] = self._two_opt(new_second)
                            new_loads[first_idx] = new_first_load
                            new_loads[second_idx] = new_second_load
                            return self._compress_routes(new_routes, new_loads)

        return False, routes, route_loads

    def _prefix_loads(self, route):
        prefix = [0]
        running = 0
        for customer in route:
            running += self.demandOfCustomer[customer]
            prefix.append(running)
        return prefix

    def _compress_routes(self, routes, route_loads):
        compressed_routes = []
        compressed_loads = []
        for route, load in zip(routes, route_loads):
            if route:
                compressed_routes.append(route)
                compressed_loads.append(load)
        return True, compressed_routes, compressed_loads

    def _run_destroy_search(self, routes, seed_offset=0, time_budget=None):
        current_routes = self._canonicalize_routes(routes)
        current_cost = self._total_cost(current_routes)
        stall = 0
        rng = self._build_rng(seed_offset)
        pair_state = self._initialize_pair_state()
        search_start = perf_counter()

        sample_idx = 0
        while sample_idx < self.config["destroy_samples"]:
            if time_budget is not None and perf_counter() - search_start >= time_budget:
                break

            destroy_method, repair_method, pair_idx = self._select_destroy_repair_pair(
                sample_idx,
                rng,
                pair_state,
            )
            remove_count = self._destroy_size(current_routes)
            removed = self._select_destroy_set(
                current_routes,
                destroy_method,
                sample_idx,
                remove_count,
            )
            if not removed or len(removed) >= self.numCustomers:
                continue

            state = self._remove_customers(current_routes, removed)
            repaired = self._repair_routes(state, repair_method)
            if repaired is None:
                self._update_pair_state(pair_state, pair_idx, improved=False, relative_gain=0.0)
                sample_idx += 1
                continue

            if self.config["post_repair_local_search"]:
                repaired = self._improve_with_local_search(repaired)
            else:
                repaired = self._canonicalize_routes([self._two_opt(route) for route in repaired])

            repaired_cost = self._total_cost(repaired)
            if repaired_cost + 1e-9 < current_cost:
                relative_gain = (current_cost - repaired_cost) / current_cost if current_cost > 1e-9 else 0.0
                self._update_pair_state(pair_state, pair_idx, improved=True, relative_gain=relative_gain)
                current_routes = repaired
                current_cost = repaired_cost
                stall = 0
            else:
                self._update_pair_state(pair_state, pair_idx, improved=False, relative_gain=0.0)
                stall += 1
                if stall >= self.config["lns_stall_limit"]:
                    break

            sample_idx += 1

        return current_routes

    def _build_rng(self, seed_offset=0):
        signature = (
            self.config["random_seed"]
            + 701 * seed_offset
            + 97 * self.numNodes
            + 193 * self.numVehicles
            + 389 * self.vehicleCapacity
            + 53 * sum(self.demandOfCustomer)
        )
        return random.Random(signature)

    def _initialize_pair_state(self):
        portfolio = self.config.get("destroy_portfolio")
        if not portfolio:
            return None
        return {
            "scores": [1.0 for _ in portfolio],
            "tries": [0 for _ in portfolio],
        }

    def _select_destroy_repair_pair(self, sample_idx, rng, pair_state):
        portfolio = self.config.get("destroy_portfolio")
        if portfolio:
            selection = self.config["destroy_selection"]
            if selection == "schedule":
                offset = self.config["random_seed"] % len(portfolio)
                pair_idx = (sample_idx + offset) % len(portfolio)
            elif selection == "adaptive":
                pair_idx = self._select_adaptive_pair_index(rng, pair_state)
            else:
                pair_idx = rng.randrange(len(portfolio))
            pair = portfolio[pair_idx]
            return pair["destroy_method"], pair["repair_method"], pair_idx
        return self.config["destroy_method"], self.config["repair_method"], None

    def _select_adaptive_pair_index(self, rng, pair_state):
        scores = pair_state["scores"]
        total = sum(scores)
        if total <= 1e-9:
            return rng.randrange(len(scores))
        threshold = rng.random() * total
        cumulative = 0.0
        for idx, score in enumerate(scores):
            cumulative += score
            if cumulative >= threshold:
                return idx
        return len(scores) - 1

    def _update_pair_state(self, pair_state, pair_idx, improved, relative_gain):
        if pair_state is None or pair_idx is None:
            return
        pair_state["tries"][pair_idx] += 1
        if improved:
            reward = 1.0 + min(5.0, 200.0 * relative_gain)
            pair_state["scores"][pair_idx] = 0.65 * pair_state["scores"][pair_idx] + 0.35 * reward
        else:
            pair_state["scores"][pair_idx] = max(0.5, pair_state["scores"][pair_idx] * 0.92)

    def _destroy_size(self, routes):
        suggested = int(round(self.numCustomers * self.config["destroy_fraction"]))
        if routes:
            suggested = max(suggested, max(len(route) for route in routes))
        suggested = max(self.config["destroy_min"], suggested)
        suggested = min(self.config["destroy_max"], suggested)
        return min(self.numCustomers - 1, suggested)

    def _select_destroy_set(self, routes, method, sample_idx, remove_count):
        route_map = self._route_membership(routes)
        customers = [customer for route in routes for customer in route]
        seed = customers[(sample_idx * 37) % len(customers)]

        if method == "spatial":
            ordered = sorted(
                customers,
                key=lambda customer: (
                    self.distance[seed][customer],
                    abs(self.demandOfCustomer[seed] - self.demandOfCustomer[customer]),
                    customer,
                ),
            )
            return set(ordered[:remove_count])

        if method == "sector":
            seed_angle = self.angles[seed]
            ordered = sorted(
                customers,
                key=lambda customer: (
                    self._angle_gap(seed_angle, self.angles[customer]),
                    abs(self.radii[seed] - self.radii[customer]),
                    customer,
                ),
            )
            return set(ordered[:remove_count])

        if method == "corridor":
            partner = self._corridor_partner(seed, customers, route_map)
            ordered = sorted(
                customers,
                key=lambda customer: (
                    self._point_to_segment_distance(customer, seed, partner),
                    min(self.distance[seed][customer], self.distance[partner][customer]),
                    customer,
                ),
            )
            return set(ordered[:remove_count])

        if method == "related":
            removed = [seed]
            removed_set = {seed}
            while len(removed) < remove_count:
                best_customer = None
                best_score = None
                for customer in customers:
                    if customer in removed_set:
                        continue
                    score = min(
                        self._related_score(customer, anchor, route_map)
                        for anchor in removed
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_customer = customer
                if best_customer is None:
                    break
                removed.append(best_customer)
                removed_set.add(best_customer)
            return removed_set

        if method == "string":
            return self._string_destroy(routes, route_map, seed, sample_idx, remove_count)

        raise ValueError(f"Unknown destroy method: {method}")

    def _route_membership(self, routes):
        membership = {}
        for route_idx, route in enumerate(routes):
            for pos, customer in enumerate(route):
                membership[customer] = (route_idx, pos)
        return membership

    def _angle_gap(self, first, second):
        diff = abs(first - second)
        return min(diff, 2.0 * pi - diff)

    def _corridor_partner(self, seed, customers, route_map):
        seed_route = route_map[seed][0]
        partner = None
        partner_score = None

        for customer in customers:
            if customer == seed:
                continue
            cross_route = 0 if route_map[customer][0] != seed_route else 1
            score = (cross_route, self.distance[seed][customer], customer)
            if partner_score is None or score < partner_score:
                partner_score = score
                partner = customer

        return partner if partner is not None else seed

    def _point_to_segment_distance(self, customer, first, second):
        px = self.xCoordOfCustomer[customer]
        py = self.yCoordOfCustomer[customer]
        ax = self.xCoordOfCustomer[first]
        ay = self.yCoordOfCustomer[first]
        bx = self.xCoordOfCustomer[second]
        by = self.yCoordOfCustomer[second]

        dx = bx - ax
        dy = by - ay
        if dx == 0.0 and dy == 0.0:
            return hypot(px - ax, py - ay)

        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return hypot(px - proj_x, py - proj_y)

    def _related_score(self, customer, anchor, route_map):
        route_penalty = 0.0 if route_map[customer][0] == route_map[anchor][0] else 25.0
        demand_penalty = abs(self.demandOfCustomer[customer] - self.demandOfCustomer[anchor])
        angle_penalty = 15.0 * self._angle_gap(self.angles[customer], self.angles[anchor])
        return self.distance[customer][anchor] + 0.35 * demand_penalty + angle_penalty + route_penalty

    def _string_destroy(self, routes, route_map, seed, sample_idx, remove_count):
        seed_route_idx, seed_pos = route_map[seed]
        seed_route = routes[seed_route_idx]
        remaining = remove_count
        removed = set()

        ordered_routes = sorted(
            range(len(routes)),
            key=lambda route_idx: (
                self._route_centroid_distance(routes[seed_route_idx], routes[route_idx]),
                route_idx,
            ),
        )

        for route_rank, route_idx in enumerate(ordered_routes):
            if remaining <= 0:
                break
            route = routes[route_idx]
            if not route:
                continue

            routes_left = len(ordered_routes) - route_rank
            frag_length = max(1, remaining // max(1, routes_left))
            frag_length = min(frag_length, len(route), max(2, remove_count // 3))

            if route_idx == seed_route_idx:
                anchor = seed_pos
            else:
                anchor = (sample_idx * 3 + len(route) // 2) % len(route)

            start = max(0, min(anchor - frag_length // 2, len(route) - frag_length))
            fragment = route[start : start + frag_length]
            removed.update(fragment)
            remaining = remove_count - len(removed)

        if len(removed) < remove_count:
            customers = [customer for route in routes for customer in route if customer not in removed]
            extras = sorted(
                customers,
                key=lambda customer: (
                    self.distance[seed][customer],
                    customer,
                ),
            )
            for customer in extras:
                if len(removed) >= remove_count:
                    break
                removed.add(customer)

        return removed

    def _route_centroid_distance(self, first_route, second_route):
        first_x, first_y = self._route_centroid(first_route)
        second_x, second_y = self._route_centroid(second_route)
        return hypot(first_x - second_x, first_y - second_y)

    def _remove_customers(self, routes, removed):
        routes_after = []
        untouched_routes = []
        touched_survivors = []
        route_budget = 0

        for route in routes:
            survivor = [customer for customer in route if customer not in removed]
            if len(survivor) != len(route):
                route_budget += 1
                touched_survivors.append(survivor)
            else:
                untouched_routes.append(route[:])
            if survivor:
                routes_after.append(survivor)

        return {
            "removed_customers": list(removed),
            "routes_after_removal": routes_after,
            "loads_after_removal": [self._route_load(route) for route in routes_after],
            "untouched_routes": untouched_routes,
            "touched_survivors": touched_survivors,
            "route_budget": max(1, route_budget),
        }

    def _repair_routes(self, state, method):
        if method == "cheapest":
            return self._repair_by_insertion(state, strategy="cheapest")
        if method == "regret2":
            return self._repair_by_insertion(state, strategy="regret2")
        if method == "local_split":
            return self._repair_by_local_split(state)
        if method == "exact":
            return self._repair_by_exact_mip(state)
        raise ValueError(f"Unknown repair method: {method}")

    def _repair_by_insertion(self, state, strategy):
        routes = [route[:] for route in state["routes_after_removal"]]
        loads = state["loads_after_removal"][:]
        remaining = sorted(
            state["removed_customers"],
            key=lambda customer: (
                self.demandOfCustomer[customer],
                self.radii[customer],
                customer,
            ),
            reverse=True,
        )

        if strategy == "cheapest":
            for customer in remaining:
                options = self._insertion_options(customer, routes, loads, 1)
                if not options:
                    return None
                self._apply_insertion(routes, loads, customer, options[0])
        else:
            regret_k = 2
            while remaining:
                best_customer = None
                best_options = None
                best_key = None
                for customer in remaining:
                    options = self._insertion_options(customer, routes, loads, regret_k + 1)
                    if not options:
                        return None
                    best_delta = options[0][0]
                    compare_idx = min(regret_k, len(options) - 1)
                    regret = options[compare_idx][0] - best_delta
                    key = (-regret, best_delta, -self.demandOfCustomer[customer], customer)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_customer = customer
                        best_options = options
                self._apply_insertion(routes, loads, best_customer, best_options[0])
                remaining.remove(best_customer)

        return self._canonicalize_routes([self._two_opt(route) for route in routes if route])

    def _repair_subset_data(self, state):
        subset_customers = []
        for route in state["touched_survivors"]:
            subset_customers.extend(route)
        subset_customers.extend(state["removed_customers"])
        route_limit = min(state["route_budget"], self.numVehicles)
        return subset_customers, route_limit

    def _repair_by_local_split(self, state):
        subset_customers, route_limit = self._repair_subset_data(state)

        if not subset_customers:
            return self._canonicalize_routes(state["untouched_routes"])

        subset_orders = self._generate_subset_orderings(subset_customers)
        best_routes = None
        best_cost = float("inf")

        for order in subset_orders:
            routes, _ = self._split_order(order, max_routes=route_limit)
            if routes is None:
                continue
            routes = [self._two_opt(route) for route in routes]
            cost = self._total_cost(routes)
            if cost + 1e-9 < best_cost:
                best_cost = cost
                best_routes = routes

        if best_routes is None:
            return None

        merged = [route[:] for route in state["untouched_routes"]]
        merged.extend(best_routes)
        return self._canonicalize_routes(merged)

    def _repair_by_exact_mip(self, state):
        subset_customers, route_limit = self._repair_subset_data(state)

        if not subset_customers:
            return self._canonicalize_routes(state["untouched_routes"])

        if (
            pywraplp is None
            or len(subset_customers) > self.config["exact_repair_max_customers"]
            or route_limit > self.config["exact_repair_max_routes"]
            or self.config["exact_repair_time_limit_sec"] <= 0.0
        ):
            return self._repair_by_local_split(state)

        exact_routes = self._solve_subset_exact_mip(subset_customers, route_limit)
        if exact_routes is None:
            return self._repair_by_local_split(state)

        merged = [route[:] for route in state["untouched_routes"]]
        merged.extend(exact_routes)
        return self._canonicalize_routes(merged)

    def _solve_subset_exact_mip(self, subset_customers, route_limit):
        solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            return None

        solver.SetTimeLimit(max(1, int(round(1000.0 * self.config["exact_repair_time_limit_sec"]))))

        local_nodes = [0] + list(subset_customers)
        customer_indices = range(1, len(local_nodes))
        node_indices = range(len(local_nodes))
        capacity = self.vehicleCapacity

        arcs = {}
        for vehicle in range(route_limit):
            for start in node_indices:
                for end in node_indices:
                    if start == end:
                        continue
                    arcs[(vehicle, start, end)] = solver.BoolVar(
                        f"x_{vehicle}_{start}_{end}"
                    )

        loads = {}
        for vehicle in range(route_limit):
            for node_idx in customer_indices:
                loads[(vehicle, node_idx)] = solver.IntVar(
                    0,
                    capacity,
                    f"u_{vehicle}_{node_idx}",
                )

        for node_idx in customer_indices:
            incoming = solver.Sum(
                arcs[(vehicle, start, node_idx)]
                for vehicle in range(route_limit)
                for start in node_indices
                if start != node_idx
            )
            outgoing = solver.Sum(
                arcs[(vehicle, node_idx, end)]
                for vehicle in range(route_limit)
                for end in node_indices
                if end != node_idx
            )
            solver.Add(incoming == 1)
            solver.Add(outgoing == 1)

        for vehicle in range(route_limit):
            depot_out = solver.Sum(arcs[(vehicle, 0, end)] for end in customer_indices)
            depot_in = solver.Sum(arcs[(vehicle, start, 0)] for start in customer_indices)
            solver.Add(depot_out == depot_in)
            solver.Add(depot_out <= 1)

            for node_idx in customer_indices:
                incoming = solver.Sum(
                    arcs[(vehicle, start, node_idx)]
                    for start in node_indices
                    if start != node_idx
                )
                outgoing = solver.Sum(
                    arcs[(vehicle, node_idx, end)]
                    for end in node_indices
                    if end != node_idx
                )
                demand = self.demandOfCustomer[local_nodes[node_idx]]
                solver.Add(incoming == outgoing)
                solver.Add(loads[(vehicle, node_idx)] >= demand * incoming)
                solver.Add(loads[(vehicle, node_idx)] <= capacity * incoming)

            for start in customer_indices:
                for end in customer_indices:
                    if start == end:
                        continue
                    demand_end = self.demandOfCustomer[local_nodes[end]]
                    solver.Add(
                        loads[(vehicle, end)]
                        >= loads[(vehicle, start)]
                        + demand_end
                        - capacity * (1 - arcs[(vehicle, start, end)])
                    )

        objective = solver.Objective()
        for (vehicle, start, end), variable in arcs.items():
            objective.SetCoefficient(
                variable,
                self.distance[local_nodes[start]][local_nodes[end]],
            )
        objective.SetMinimization()

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            return None

        routes = []
        visited = []
        for vehicle in range(route_limit):
            successor = {}
            for start in node_indices:
                for end in node_indices:
                    if start == end:
                        continue
                    if arcs[(vehicle, start, end)].solution_value() > 0.5:
                        successor[start] = end

            if 0 not in successor:
                continue

            route = []
            seen = {0}
            current = successor[0]
            while current != 0:
                if current in seen or current not in successor:
                    return None
                seen.add(current)
                route.append(local_nodes[current])
                visited.append(local_nodes[current])
                current = successor[current]
            if route:
                routes.append(route)

        if len(visited) != len(subset_customers) or len(set(visited)) != len(subset_customers):
            return None

        return routes

    def _generate_subset_orderings(self, customers):
        seen = set()
        orderings = []
        subset = tuple(customers)

        for step in range(4):
            offset = (2.0 * pi * step) / 4
            order = tuple(
                sorted(
                    subset,
                    key=lambda customer: (
                        (self.angles[customer] - offset) % (2.0 * pi),
                        self.radii[customer],
                        customer,
                    ),
                )
            )
            self._append_order(orderings, seen, order)
            self._append_order(orderings, seen, tuple(reversed(order)))

        self._append_order(
            orderings,
            seen,
            tuple(
                sorted(
                    subset,
                    key=lambda customer: (
                        self.xCoordOfCustomer[customer],
                        self.yCoordOfCustomer[customer],
                        customer,
                    ),
                )
            ),
        )
        self._append_order(
            orderings,
            seen,
            tuple(
                sorted(
                    subset,
                    key=lambda customer: (
                        self.yCoordOfCustomer[customer],
                        self.xCoordOfCustomer[customer],
                        customer,
                    ),
                )
            ),
        )

        start = max(subset, key=lambda customer: self.radii[customer])
        self._append_order(orderings, seen, tuple(self._nearest_neighbor_subset(start, subset)))
        return orderings

    def _nearest_neighbor_subset(self, seed, customers):
        remaining = set(customers)
        order = [seed]
        remaining.remove(seed)
        current = seed

        while remaining:
            next_customer = min(
                remaining,
                key=lambda customer: (self.distance[current][customer], customer),
            )
            order.append(next_customer)
            remaining.remove(next_customer)
            current = next_customer
        return order

    def _insertion_options(self, customer, routes, loads, limit):
        demand = self.demandOfCustomer[customer]
        options = []

        for route_idx, route in enumerate(routes):
            if loads[route_idx] + demand > self.vehicleCapacity:
                continue
            old_cost = self._route_cost(route)
            for pos in range(len(route) + 1):
                new_route = route[:]
                new_route.insert(pos, customer)
                delta = self._route_cost(new_route) - old_cost
                options.append((delta, route_idx, pos))

        if len(routes) < self.numVehicles:
            options.append((2.0 * self.depotDistance[customer], len(routes), 0))

        options.sort(key=lambda item: (item[0], item[1], item[2]))
        return options[:limit]

    def _apply_insertion(self, routes, loads, customer, option):
        _, route_idx, pos = option
        demand = self.demandOfCustomer[customer]
        if route_idx == len(routes):
            routes.append([customer])
            loads.append(demand)
            return

        routes[route_idx].insert(pos, customer)
        loads[route_idx] += demand

    def _two_opt(self, route):
        if len(route) < 4:
            return route[:]

        best_route = route[:]
        improved = True

        while improved:
            improved = False
            for start in range(len(best_route) - 1):
                first = 0 if start == 0 else best_route[start - 1]
                second = best_route[start]
                for end in range(start + 1, len(best_route)):
                    third = best_route[end]
                    fourth = 0 if end == len(best_route) - 1 else best_route[end + 1]
                    delta = (
                        self.distance[first][third]
                        + self.distance[second][fourth]
                        - self.distance[first][second]
                        - self.distance[third][fourth]
                    )
                    if delta < -1e-9:
                        best_route[start : end + 1] = reversed(best_route[start : end + 1])
                        improved = True
                        break
                if improved:
                    break

        return best_route

    def _route_load(self, route):
        return sum(self.demandOfCustomer[customer] for customer in route)

    def _route_cost(self, route):
        if not route:
            return 0.0

        cost = self.depotDistance[route[0]] + self.depotDistance[route[-1]]
        for idx in range(len(route) - 1):
            cost += self.distance[route[idx]][route[idx + 1]]
        return cost

    def _total_cost(self, routes):
        return sum(self._route_cost(route) for route in routes)

    def _route_centroid(self, route):
        if not route:
            return self.xCoordOfCustomer[0], self.yCoordOfCustomer[0]
        return (
            sum(self.xCoordOfCustomer[customer] for customer in route) / len(route),
            sum(self.yCoordOfCustomer[customer] for customer in route) / len(route),
        )

    def _route_angle(self, route):
        center_x, center_y = self._route_centroid(route)
        return atan2(center_y - self.yCoordOfCustomer[0], center_x - self.xCoordOfCustomer[0])

    def _canonicalize_routes(self, routes):
        normalized = [route[:] for route in routes if route]
        normalized.sort(
            key=lambda route: (
                self._route_angle(route),
                self.radii[route[0]],
                route[0],
                len(route),
            )
        )
        return normalized

    def _routes_signature(self, routes):
        return tuple(tuple(route) for route in routes)

    def _pad_routes(self, routes):
        padded = [route[:] for route in routes]
        while len(padded) < self.numVehicles:
            padded.append([])
        return padded

    def _format_solution(self, routes):
        tokens = []
        for route in routes:
            if route:
                tokens.append("0")
                tokens.extend(str(customer) for customer in route)
                tokens.append("0")
            else:
                tokens.extend(("0", "0"))
        return " ".join(tokens)

    def __str__(self):
        out = [
            f"Number of locations: {self.numNodes}",
            f"Number of customers: {self.numCustomers}",
            f"Number of vehicles: {self.numVehicles}",
            f"Vehicle capacity: {self.vehicleCapacity}",
        ]
        for idx in range(self.numNodes):
            out.append(
                f"{self.demandOfCustomer[idx]} "
                f"{self.xCoordOfCustomer[idx]} "
                f"{self.yCoordOfCustomer[idx]}"
            )
        return "\n".join(out)
