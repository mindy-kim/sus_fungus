"""
CVRP solver for CSCI 2951-O Project 5 (Transportation & Logistics).

Approach:
  - Parse the .vrp instance into a numpy-backed object.
  - Pre-compute an integer-scaled Euclidean distance matrix (OR-Tools uses
    integer arc costs, so we scale by DISTANCE_SCALE for precision).
  - Build an OR-Tools RoutingModel with a vehicle-capacity dimension.
  - First solution: PATH_CHEAPEST_ARC (cheap nearest-neighbour insertion).
  - Local search:   GUIDED_LOCAL_SEARCH (best general-purpose CVRP
                    metaheuristic in OR-Tools; penalises features of bad
                    arcs to escape local optima).
  - Time-bounded; on timeout, return the incumbent.
  - Final objective is recomputed from the raw float Euclidean distances
    so that scaling does not affect the reported value.
  - If OR-Tools fails to find any feasible solution (rare — would only
    happen on a malformed instance), fall back to a capacity-aware
    nearest-neighbour heuristic.
"""

import math
import os
import sys
import numpy as np

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


# Scale factor for converting float Euclidean distances to integers for
# OR-Tools.  1000 gives ~3 decimal places of precision, which is more
# than enough given that the final reported objective is recomputed from
# the raw float distances.
DISTANCE_SCALE = 1000

# Default solver time limit in seconds.  The handout allows ~5 minutes
# (300s) per instance; we leave a small buffer for I/O and shutdown.
# Override with the VRP_TIME_LIMIT environment variable (useful for
# unit-testing or when running with a non-default outer timeout).
DEFAULT_TIME_LIMIT_S = int(os.environ.get('VRP_TIME_LIMIT', 290))


class VRPInstance:
    numCustomers: int        # number of locations including the depot (index 0)
    numVehicles: int         # size of the homogeneous fleet
    vehicleCapacity: int     # capacity of every vehicle
    demandOfCustomer: np.ndarray
    xCoordOfCustomer: np.ndarray
    yCoordOfCustomer: np.ndarray

    def __init__(self, filename: str):
        self.load_from_file(filename)
        self.solution = None
        self.objective_value = 0
        self._raw_dist = None    # float Euclidean matrix (n, n)
        self._int_dist = None    # scaled int Euclidean matrix (n, n)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def load_from_file(self, filename: str):
        try:
            with open(filename, 'r') as f:
                tokens = iter(f.read().split())

                self.numCustomers = int(next(tokens))
                self.numVehicles = int(next(tokens))
                self.vehicleCapacity = int(next(tokens))

                self.demandOfCustomer = np.zeros(self.numCustomers, dtype=np.int64)
                self.xCoordOfCustomer = np.zeros(self.numCustomers, dtype=np.float64)
                self.yCoordOfCustomer = np.zeros(self.numCustomers, dtype=np.float64)

                for i in range(self.numCustomers):
                    self.demandOfCustomer[i] = int(next(tokens))
                    self.xCoordOfCustomer[i] = float(next(tokens))
                    self.yCoordOfCustomer[i] = float(next(tokens))
        except Exception as e:
            # Send debug info to stderr so it is not parsed as the result.
            print(f"Error reading instance file: {e}", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Distance matrices
    # ------------------------------------------------------------------
    def _build_distance_matrices(self):
        coords = np.column_stack((self.xCoordOfCustomer, self.yCoordOfCustomer))
        # Vectorised pairwise Euclidean distances.
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        self._raw_dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        # OR-Tools requires integer arc costs.
        self._int_dist = np.rint(self._raw_dist * DISTANCE_SCALE).astype(np.int64)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def solve(self, time_limit_seconds: int = DEFAULT_TIME_LIMIT_S):
        self._build_distance_matrices()

        ok = self._solve_with_ortools(time_limit_seconds)
        if not ok:
            # Defensive fallback; rarely triggered.
            self._solve_greedy()

        return self.solution, self.objective_value

    # ------------------------------------------------------------------
    # OR-Tools solve
    # ------------------------------------------------------------------
    def _solve_with_ortools(self, time_limit_seconds: int) -> bool:
        n = self.numCustomers
        v = self.numVehicles
        depot = 0

        manager = pywrapcp.RoutingIndexManager(n, v, depot)
        routing = pywrapcp.RoutingModel(manager)

        int_dist = self._int_dist  # local alias for the closure

        def distance_callback(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return int(int_dist[i, j])

        transit_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Capacity dimension.  Each vehicle accumulates the demand of the
        # nodes it visits and may not exceed `vehicleCapacity`.
        demand = self.demandOfCustomer

        def demand_callback(from_index):
            return int(demand[manager.IndexToNode(from_index)])

        demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx,
            0,                                          # null capacity slack
            [int(self.vehicleCapacity)] * v,            # per-vehicle capacities
            True,                                       # cumul starts at zero
            'Capacity',
        )

        # Search parameters.
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.time_limit.seconds = int(time_limit_seconds)
        # Keep solver quiet; main.py owns stdout for the final JSON line.
        params.log_search = False

        assignment = routing.SolveWithParameters(params)
        if assignment is None:
            return False

        # Extract per-vehicle routes and the true Euclidean objective.
        raw = self._raw_dist
        routes, total = [], 0.0
        for vehicle_id in range(v):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                next_index = assignment.Value(routing.NextVar(index))
                total += raw[node, manager.IndexToNode(next_index)]
                index = next_index
            route.append(manager.IndexToNode(index))   # final depot
            routes.append(route)

        self.solution = self._format_solution(routes)
        self.objective_value = round(float(total), 4)
        return True

    # ------------------------------------------------------------------
    # Greedy fallback (capacity-aware nearest neighbour)
    # ------------------------------------------------------------------
    def _solve_greedy(self):
        n = self.numCustomers
        v = self.numVehicles
        cap = self.vehicleCapacity
        raw = self._raw_dist

        unvisited = set(range(1, n))
        routes = []

        for _ in range(v):
            route = [0]
            current = 0
            remaining = cap
            while unvisited:
                best, best_d = None, math.inf
                for c in unvisited:
                    if self.demandOfCustomer[c] <= remaining:
                        d = raw[current, c]
                        if d < best_d:
                            best_d, best = d, c
                if best is None:
                    break
                route.append(best)
                remaining -= int(self.demandOfCustomer[best])
                unvisited.remove(best)
                current = best
            route.append(0)
            routes.append(route)

        if unvisited:
            # Could not assign every customer — instance is infeasible
            # for this fleet/capacity.  Report failure cleanly.
            self.solution = None
            self.objective_value = 0
            return

        total = 0.0
        for r in routes:
            for a, b in zip(r[:-1], r[1:]):
                total += raw[a, b]

        self.solution = self._format_solution(routes)
        self.objective_value = round(float(total), 4)

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_solution(routes):
        # Each route already starts and ends with the depot (0).
        # The expected solution string is every route concatenated with
        # single spaces, e.g.  "0 1 2 3 0 0 4 0 0 0 0 0".
        return " ".join(str(node) for route in routes for node in route)

    # ------------------------------------------------------------------
    def __str__(self):
        out = [f"Number of customers: {self.numCustomers}",
               f"Number of vehicles: {self.numVehicles}",
               f"Vehicle capacity: {self.vehicleCapacity}"]
        for i in range(self.numCustomers):
            out.append(f"{self.demandOfCustomer[i]} "
                       f"{self.xCoordOfCustomer[i]} "
                       f"{self.yCoordOfCustomer[i]}")
        return "\n".join(out)