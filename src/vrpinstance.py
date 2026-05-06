from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
import time
from typing import Sequence

import numpy as np

Routes = list

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
        self._dist_rows: list | None = None

    def solve(self):

        try:
            time_budget = float(os.environ.get("VRP_TIME_BUDGET", "290"))
        except ValueError:
            time_budget = 290.0
        try:
            seed = int(os.environ.get("VRP_SEED", "42"))
        except ValueError:
            seed = 42

        self._ensure_distance_matrix()
        routes, objective, _meta = solve_hgs_search(
            self, time_budget_s=time_budget, seed=seed
        )

        self.solution = _format_solution(routes, self.numVehicles, optimality_flag=0)
        self.objective_value = float(objective)
        return self.solution, self.objective_value

    @property
    def customers(self) -> range: return range(1, self.numCustomers)
    @property
    def num_nodes(self) -> int: return self.numCustomers
    @property
    def num_vehicles(self) -> int: return self.numVehicles
    @property
    def vehicle_capacity(self) -> int: return int(self.vehicleCapacity)
    @property
    def demands(self) -> np.ndarray: return self.demandOfCustomer
    @property
    def total_demand(self) -> int: return int(self.demandOfCustomer[1:].sum())

    def _ensure_distance_matrix(self) -> None:
        if self._dist_matrix is not None:
            return
        xs = self.xCoordOfCustomer
        ys = self.yCoordOfCustomer
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        self._dist_matrix = np.hypot(dx, dy)
        self._dist_rows = [row.tolist() for row in self._dist_matrix]

    def distance(self, i: int, j: int) -> float:
        if self._dist_matrix is None:
            self._ensure_distance_matrix()
        return self._dist_rows[i][j]

    def route_demand(self, route: Sequence[int]) -> int:
        if not route: return 0
        return int(self.demandOfCustomer[list(route)].sum())

    def route_distance(self, route: Sequence[int]) -> float:
        if not route: return 0.0
        if self._dist_matrix is None:
            self._ensure_distance_matrix()
        Drow = self._dist_rows
        d = Drow[0][route[0]] + Drow[route[-1]][0]
        for left, right in zip(route, route[1:]):
            d += Drow[left][right]
        return d

    def routes_objective(self, routes: Sequence[Sequence[int]]) -> float:
        return sum(self.route_distance(r) for r in routes)

    def load_from_file(self, filename: str):
        try:
            with open(filename, 'r') as f:
                content = f.read().split()
                it = iter(content)
                self.numCustomers = int(next(it))
                self.numVehicles = int(next(it))
                self.vehicleCapacity = int(next(it))
                self.demandOfCustomer = np.zeros(self.numCustomers, dtype=int)
                self.xCoordOfCustomer = np.zeros(self.numCustomers)
                self.yCoordOfCustomer = np.zeros(self.numCustomers)
                for i in range(self.numCustomers):
                    self.demandOfCustomer[i] = int(next(it))
                    self.xCoordOfCustomer[i] = float(next(it))
                    self.yCoordOfCustomer[i] = float(next(it))
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

def _format_solution(routes, num_vehicles, optimality_flag=0):
    used = [list(r) for r in routes if r]
    while len(used) < num_vehicles:
        used.append([])
    used = used[:num_vehicles]
    parts = [str(optimality_flag)]
    for r in used:
        parts.append("0")
        for c in r:
            parts.append(str(int(c)))
        parts.append("0")
    return " ".join(parts)

def copy_routes(routes): return [list(r) for r in routes]

def flatten_routes(routes):
    flat = []
    for r in routes: flat.extend(r)
    return flat

def unique_preserve_order(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def split_giant_tour(instance, customer_order, max_routes=None):
    if max_routes is None:
        max_routes = instance.num_vehicles
    if not customer_order:
        return [[]], 0.0
    n = len(customer_order)
    Drow = instance._dist_rows
    cap = instance.vehicle_capacity
    demands = instance.demandOfCustomer

    prefix_demand = [0] * (n + 1)
    for idx, c in enumerate(customer_order, start=1):
        prefix_demand[idx] = prefix_demand[idx - 1] + int(demands[c])

    segment_cost = [[math.inf] * (n + 1) for _ in range(n)]
    for start in range(n):
        demand = 0
        cost = 0.0
        prev = 0
        for end in range(start, n):
            c = customer_order[end]
            demand += int(demands[c])
            if demand > cap:
                break
            cost += Drow[prev][c]
            prev = c
            segment_cost[start][end + 1] = cost + Drow[prev][0]

    best = [[math.inf] * (max_routes + 1) for _ in range(n + 1)]
    parent = [[None] * (max_routes + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for end in range(1, n + 1):
        for ru in range(1, max_routes + 1):
            for start in range(end - 1, -1, -1):
                if prefix_demand[end] - prefix_demand[start] > cap:
                    break
                cur = segment_cost[start][end]
                if math.isinf(cur) or math.isinf(best[start][ru - 1]):
                    continue
                cand = best[start][ru - 1] + cur
                if cand < best[end][ru]:
                    best[end][ru] = cand
                    parent[end][ru] = start

    rc = min(range(max_routes + 1), key=lambda c: best[n][c])
    if math.isinf(best[n][rc]):
        return None
    cuts = []
    end = n; ru = rc
    while end > 0:
        st = parent[end][ru]
        if st is None: return None
        cuts.append(st); end = st; ru -= 1
    cuts.reverse()
    routes = []
    prev = 0
    for cut in cuts[1:]:
        routes.append(customer_order[prev:cut]); prev = cut
    routes.append(customer_order[prev:n])
    return routes, best[n][rc]

def repair_routes(instance, routes):
    if _is_clean_feasible_cover(instance, routes):
        return [r[:] for r in routes if r], instance.routes_objective(routes)
    valid = [c for c in unique_preserve_order(c for r in routes for c in r)
             if 1 <= c < instance.num_nodes]
    missing = [c for c in instance.customers if c not in set(valid)]
    candidate = valid + missing
    sp = split_giant_tour(instance, candidate)
    if sp is not None: return sp
    packed = greedy_capacity_pack(instance, sorted(candidate, key=lambda c: (-int(instance.demands[c]), c)))
    return packed, instance.routes_objective(packed)

def greedy_capacity_pack(instance, customer_order):
    if not customer_order: return []
    routes, loads = [], []
    cap = instance.vehicle_capacity
    demands = instance.demandOfCustomer
    for c in customer_order:
        placed = False
        for ri in range(len(routes)):
            if loads[ri] + int(demands[c]) > cap: continue
            _, pos = best_insertion_delta(instance, routes[ri], c)
            routes[ri].insert(pos, c)
            loads[ri] += int(demands[c])
            placed = True
            break
        if placed: continue
        if len(routes) >= instance.num_vehicles: break
        routes.append([c]); loads.append(int(demands[c]))
    if sum(len(r) for r in routes) != len(customer_order):
        exact = _pack_by_backtracking(instance, customer_order)
        if exact is None:
            raise ValueError("Unable to pack customer without violating capacity")
        routes = exact
    return routes

def best_insertion_delta(instance, route, customer):
    Drow = instance._dist_rows
    if not route:
        return Drow[0][customer] + Drow[customer][0], 0
    best_delta = math.inf; best_pos = 0
    Dc = Drow[customer]
    full = [0] + list(route) + [0]
    for p in range(len(full) - 1):
        l = full[p]; r = full[p + 1]
        delta = Drow[l][customer] + Dc[r] - Drow[l][r]
        if delta < best_delta:
            best_delta = delta; best_pos = p
    return best_delta, best_pos

def _is_clean_feasible_cover(instance, routes):
    if len(routes) > instance.num_vehicles: return False
    seen = set()
    for r in routes:
        if instance.route_demand(r) > instance.vehicle_capacity: return False
        for c in r:
            if c in seen or c <= 0 or c >= instance.num_nodes: return False
            seen.add(c)
    return seen == set(instance.customers)

def _pack_by_backtracking(instance, customer_order):
    routes = [[] for _ in range(instance.num_vehicles)]
    loads = [0] * instance.num_vehicles
    ordered = sorted(customer_order, key=lambda c: (-int(instance.demands[c]), c))
    cap = instance.vehicle_capacity
    demands = instance.demandOfCustomer
    def search(i):
        if i == len(ordered): return True
        c = ordered[i]; d = int(demands[c])
        used = set()
        for ri in range(instance.num_vehicles):
            if loads[ri] in used: continue
            if loads[ri] + d > cap: continue
            used.add(loads[ri])
            _, pos = best_insertion_delta(instance, routes[ri], c)
            routes[ri].insert(pos, c); loads[ri] += d
            if search(i + 1): return True
            routes[ri].pop(pos); loads[ri] -= d
            if loads[ri] == 0: break
        return False
    if not search(0): return None
    return [r for r in routes if r]

@dataclass
class _SolverRun:
    routes: Routes
    objective: float

def neighbor_count(instance):
    if instance.num_nodes <= 60: return min(20, instance.num_nodes - 1)
    if instance.num_nodes <= 180: return 30
    return 40

def nearest_neighbor_sets(instance, k):
    instance._ensure_distance_matrix()
    Drow = instance._dist_rows
    custs = list(instance.customers)
    result = {}
    for c in custs:
        Dc = Drow[c]
        others = [(Dc[o], o) for o in custs if o != c]
        others.sort()
        result[c] = [o for _, o in others[:k]]
    return result

def add_routes_to_pool(instance, pool, routes):
    for r in routes:
        if not r: continue
        k = frozenset(r)
        ex = pool.get(k)
        if ex is None or instance.route_distance(r) < instance.route_distance(ex):
            pool[k] = list(r)

def relaxed_nearest_tour_split(instance, rng, neighbors):
    customers = list(instance.customers)
    if not customers: return []
    Drow = instance._dist_rows
    visited = set(); tour = []; cur = 0
    while len(tour) < len(customers):
        rem = [c for c in customers if c not in visited]
        rem.sort(key=lambda c: Drow[cur][c])
        k = min(5, len(rem))
        idx = int((rng.random() ** 2) * k)
        picked = rem[idx]
        tour.append(picked); visited.add(picked); cur = picked
    sp = split_giant_tour(instance, tour)
    if sp is None: return greedy_capacity_pack(instance, tour)
    return sp[0]

def local_search_v3(instance, routes, neighbors, deadline):
    routes = [r[:] for r in routes if r]
    if not routes:
        return routes, 0.0

    cap = instance.vehicle_capacity
    instance._ensure_distance_matrix()
    Drow = instance._dist_rows
    demands_arr = instance.demandOfCustomer
    n = instance.num_nodes

    dem = [int(demands_arr[i]) for i in range(n)]

    loads = [sum(dem[c] for c in r) if r else 0 for r in routes]

    def build_pdem(r):
        pd = [0] * (len(r) + 1)
        s = 0
        for p, c in enumerate(r):
            s += dem[c]
            pd[p + 1] = s
        return pd
    route_pdem = [build_pdem(r) for r in routes]

    cust_route = [-1] * n
    cust_pos = [-1] * n
    for ri, r in enumerate(routes):
        for pi, c in enumerate(r):
            cust_route[c] = ri
            cust_pos[c] = pi

    dlb = [True] * n
    for c in range(1, n):
        dlb[c] = False

    def refresh(ri):
        for pi, c in enumerate(routes[ri]):
            cust_route[c] = ri
            cust_pos[c] = pi
        route_pdem[ri] = build_pdem(routes[ri])

    def wake(c):
        if 0 < c < n: dlb[c] = False

    or_opt_seg_lens = (1, 2, 3) if n > 250 else (1, 2, 3, 4)

    def or_opt_from(c1):
        ri = cust_route[c1]
        if ri < 0: return False
        r1 = routes[ri]
        i = cust_pos[c1]
        len1 = len(r1)
        pdem1 = route_pdem[ri]

        for seg_len in or_opt_seg_lens:
            if i + seg_len > len1: break
            seg_demand = pdem1[i + seg_len] - pdem1[i]
            b = r1[i]; e = r1[i + seg_len - 1]
            a = r1[i - 1] if i > 0 else 0
            f = r1[i + seg_len] if i + seg_len < len1 else 0
            Da = Drow[a]; De = Drow[e]; Db = Drow[b]
            delta_remove = Da[f] - Da[b] - De[f]

            cands = []
            for nbr in neighbors.get(b, ()):
                r2i = cust_route[nbr]
                if r2i < 0: continue
                pos = cust_pos[nbr]
                cands.append((r2i, pos))
                cands.append((r2i, pos + 1))
            if e != b:
                for nbr in neighbors.get(e, ()):
                    r2i = cust_route[nbr]
                    if r2i < 0: continue
                    pos = cust_pos[nbr]
                    cands.append((r2i, pos))
                    cands.append((r2i, pos + 1))

            virt_len = len1 - seg_len
            for r2_idx, j in cands:
                r2 = routes[r2_idx]
                if r2_idx != ri:
                    if loads[r2_idx] + seg_demand > cap: continue
                    if j > len(r2): continue
                    g = r2[j - 1] if j > 0 else 0
                    h = r2[j] if j < len(r2) else 0
                else:
                    if j > virt_len: continue
                    if j == 0: g = 0
                    elif j - 1 < i: g = r1[j - 1]
                    else: g = r1[j - 1 + seg_len]
                    if j == virt_len: h = 0
                    elif j < i: h = r1[j]
                    else: h = r1[j + seg_len]

                Dg = Drow[g]
                base = delta_remove - Dg[h]
                d_fwd = base + Dg[b] + De[h]
                d_rev = base + Dg[e] + Db[h] if seg_len > 1 else math.inf

                if d_fwd < -1e-9 and d_fwd <= d_rev:
                    seg_use = r1[i:i + seg_len]
                    if r2_idx == ri:
                        nr = r1[:i] + r1[i + seg_len:]
                        routes[ri] = nr[:j] + seg_use + nr[j:]
                        refresh(ri)
                    else:
                        routes[ri] = r1[:i] + r1[i + seg_len:]
                        routes[r2_idx] = r2[:j] + seg_use + r2[j:]
                        loads[ri] -= seg_demand; loads[r2_idx] += seg_demand
                        refresh(ri); refresh(r2_idx)
                    for x in seg_use: wake(x)
                    wake(a); wake(f); wake(g); wake(h)
                    return True
                if d_rev < -1e-9:
                    seg_use = r1[i:i + seg_len][::-1]
                    if r2_idx == ri:
                        nr = r1[:i] + r1[i + seg_len:]
                        routes[ri] = nr[:j] + seg_use + nr[j:]
                        refresh(ri)
                    else:
                        routes[ri] = r1[:i] + r1[i + seg_len:]
                        routes[r2_idx] = r2[:j] + seg_use + r2[j:]
                        loads[ri] -= seg_demand; loads[r2_idx] += seg_demand
                        refresh(ri); refresh(r2_idx)
                    for x in seg_use: wake(x)
                    wake(a); wake(f); wake(g); wake(h)
                    return True
        return False

    def two_opt_intra(ri):
        r = routes[ri]
        L = len(r)
        if L < 4: return False
        i = 0
        while i < L - 1:
            a = r[i - 1] if i > 0 else 0
            b = r[i]
            Da = Drow[a]; Db = Drow[b]
            base = Da[b]
            j = i + 2
            while j < L:
                c = r[j]
                nx = r[j + 1] if j + 1 < L else 0
                delta = Da[c] + Db[nx] - base - Drow[c][nx]
                if delta < -1e-9:
                    r[i:j + 1] = r[i:j + 1][::-1]
                    refresh(ri)
                    for x in r: wake(x)
                    return True
                j += 1
            i += 1
        return False

    def two_opt_star_from(c1):
        r1_idx = cust_route[c1]
        if r1_idx < 0: return False
        r1 = routes[r1_idx]
        i = cust_pos[c1]
        L1 = len(r1)
        pdem1 = route_pdem[r1_idx]

        x = r1[i + 1] if i + 1 < L1 else 0
        suffix1_dem_A = pdem1[L1] - pdem1[i + 1]
        prefix1_dem_A = pdem1[i + 1]
        Dc1 = Drow[c1]; d_c1_x = Dc1[x]

        for v in neighbors.get(c1, ()):
            r2_idx = cust_route[v]
            if r2_idx < 0 or r2_idx == r1_idx: continue
            r2 = routes[r2_idx]
            j = cust_pos[v]
            L2 = len(r2)
            pdem2 = route_pdem[r2_idx]
            y = r2[j + 1] if j + 1 < L2 else 0

            suffix2_dem = pdem2[L2] - pdem2[j + 1]
            prefix2_dem = pdem2[j + 1]
            new_load1 = prefix1_dem_A + suffix2_dem
            new_load2 = prefix2_dem + suffix1_dem_A
            if new_load1 > cap or new_load2 > cap: continue

            delta = Dc1[y] + Drow[v][x] - d_c1_x - Drow[v][y]
            if delta < -1e-9:
                new_r1 = r1[:i + 1] + r2[j + 1:]
                new_r2 = r2[:j + 1] + r1[i + 1:]
                routes[r1_idx] = new_r1
                routes[r2_idx] = new_r2
                loads[r1_idx] = new_load1
                loads[r2_idx] = new_load2
                refresh(r1_idx); refresh(r2_idx)
                for x_ in new_r1: wake(x_)
                for x_ in new_r2: wake(x_)
                return True

        a = r1[i - 1] if i > 0 else 0
        suffix1_dem_B = pdem1[L1] - pdem1[i]
        prefix1_dem_B = pdem1[i]
        d_a_c1 = Drow[a][c1]

        for v in neighbors.get(c1, ()):
            r2_idx = cust_route[v]
            if r2_idx < 0 or r2_idx == r1_idx: continue
            r2 = routes[r2_idx]
            j = cust_pos[v]
            L2 = len(r2)
            pdem2 = route_pdem[r2_idx]

            b = r2[j - 1] if j > 0 else 0
            suffix2_dem_B = pdem2[L2] - pdem2[j]
            prefix2_dem_B = pdem2[j]
            new_load1 = prefix1_dem_B + suffix2_dem_B
            new_load2 = prefix2_dem_B + suffix1_dem_B
            if new_load1 > cap or new_load2 > cap: continue

            delta = Drow[a][v] + Drow[b][c1] - d_a_c1 - Drow[b][v]
            if delta < -1e-9:
                new_r1 = r1[:i] + r2[j:]
                new_r2 = r2[:j] + r1[i:]
                routes[r1_idx] = new_r1
                routes[r2_idx] = new_r2
                loads[r1_idx] = new_load1
                loads[r2_idx] = new_load2
                refresh(r1_idx); refresh(r2_idx)
                for x_ in new_r1: wake(x_)
                for x_ in new_r2: wake(x_)
                return True
        return False

    def swap_star_from(u):
        r1_idx = cust_route[u]
        if r1_idx < 0: return False
        r1 = routes[r1_idx]
        iu = cust_pos[u]
        L1 = len(r1)
        au = r1[iu - 1] if iu > 0 else 0
        bu = r1[iu + 1] if iu + 1 < L1 else 0
        u_d = dem[u]
        Du = Drow[u]
        Dau = Drow[au]
        rem_u_cost = Dau[u] + Du[bu] - Dau[bu]
        load1 = loads[r1_idx]

        for v in neighbors.get(u, ()):
            r2_idx = cust_route[v]
            if r2_idx < 0 or r2_idx == r1_idx: continue
            r2 = routes[r2_idx]
            iv = cust_pos[v]
            L2 = len(r2)
            v_d = dem[v]
            if load1 - u_d + v_d > cap: continue
            if loads[r2_idx] - v_d + u_d > cap: continue

            av = r2[iv - 1] if iv > 0 else 0
            bv = r2[iv + 1] if iv + 1 < L2 else 0
            Dv = Drow[v]
            Dav = Drow[av]
            rem_v_cost = Dav[v] + Dv[bv] - Dav[bv]

            best_ins_u = math.inf; best_ins_u_pos = 0
            prev_node = 0
            edge_idx = 0
            for k in range(L2 + 1):
                if k == iv:
                    continue
                next_node = r2[k] if k < L2 else 0
                d = Drow[prev_node][u] + Du[next_node] - Drow[prev_node][next_node]
                if d < best_ins_u:
                    best_ins_u = d
                    best_ins_u_pos = edge_idx
                prev_node = next_node
                edge_idx += 1

            best_ins_v = math.inf; best_ins_v_pos = 0
            prev_node = 0
            edge_idx = 0
            for k in range(L1 + 1):
                if k == iu:
                    continue
                next_node = r1[k] if k < L1 else 0
                d = Drow[prev_node][v] + Dv[next_node] - Drow[prev_node][next_node]
                if d < best_ins_v:
                    best_ins_v = d
                    best_ins_v_pos = edge_idx
                prev_node = next_node
                edge_idx += 1

            delta = -rem_u_cost - rem_v_cost + best_ins_u + best_ins_v
            if delta < -1e-9:
                r1_no_u = r1[:iu] + r1[iu + 1:]
                new_r1 = r1_no_u[:best_ins_v_pos] + [v] + r1_no_u[best_ins_v_pos:]
                r2_no_v = r2[:iv] + r2[iv + 1:]
                new_r2 = r2_no_v[:best_ins_u_pos] + [u] + r2_no_v[best_ins_u_pos:]
                routes[r1_idx] = new_r1
                routes[r2_idx] = new_r2
                loads[r1_idx] = load1 - u_d + v_d
                loads[r2_idx] = loads[r2_idx] - v_d + u_d
                refresh(r1_idx); refresh(r2_idx)
                for x in new_r1: wake(x)
                for x in new_r2: wake(x)
                return True
        return False

    def cross_exchange_from(c1):
        r1_idx = cust_route[c1]
        if r1_idx < 0: return False
        r1 = routes[r1_idx]
        i = cust_pos[c1]
        L1 = len(r1)
        pdem1 = route_pdem[r1_idx]

        for L1seg in (1, 2):
            if i + L1seg > L1: break
            seg1_dem = pdem1[i + L1seg] - pdem1[i]
            s1a = r1[i]; s1b = r1[i + L1seg - 1]
            a = r1[i - 1] if i > 0 else 0
            f = r1[i + L1seg] if i + L1seg < L1 else 0
            Da = Drow[a]; Df = Drow[f]
            old_r1 = Da[s1a] + Drow[s1b][f]
            load1_after_remove = loads[r1_idx] - seg1_dem

            for nbr in neighbors.get(c1, ()):
                r2_idx = cust_route[nbr]
                if r2_idx < 0 or r2_idx == r1_idx: continue
                r2 = routes[r2_idx]
                j = cust_pos[nbr]
                L2 = len(r2)
                pdem2 = route_pdem[r2_idx]
                load2_full = loads[r2_idx]

                for L2seg in (1, 2):
                    if j + L2seg > L2: break
                    seg2_dem = pdem2[j + L2seg] - pdem2[j]
                    new_load1 = load1_after_remove + seg2_dem
                    new_load2 = load2_full - seg2_dem + seg1_dem
                    if new_load1 > cap or new_load2 > cap: continue

                    s2a = r2[j]; s2b = r2[j + L2seg - 1]
                    g = r2[j - 1] if j > 0 else 0
                    h = r2[j + L2seg] if j + L2seg < L2 else 0
                    Dg = Drow[g]; Dh_node = Drow[h]
                    old_r2 = Dg[s2a] + Drow[s2b][h]
                    old_total = old_r1 + old_r2

                    options = []

                    options.append((Da[s2a] + Drow[s2b][f] + Dg[s1a] + Drow[s1b][h],
                                    False, False))
                    if L2seg > 1:
                        options.append((Da[s2b] + Drow[s2a][f] + Dg[s1a] + Drow[s1b][h],
                                        True, False))
                    if L1seg > 1:
                        options.append((Da[s2a] + Drow[s2b][f] + Dg[s1b] + Drow[s1a][h],
                                        False, True))
                    if L1seg > 1 and L2seg > 1:
                        options.append((Da[s2b] + Drow[s2a][f] + Dg[s1b] + Drow[s1a][h],
                                        True, True))

                    best_new, rev2, rev1 = min(options, key=lambda t: t[0])
                    delta = best_new - old_total
                    if delta < -1e-9:
                        seg2_use = r2[j:j + L2seg]
                        seg1_use = r1[i:i + L1seg]
                        if rev2: seg2_use = seg2_use[::-1]
                        if rev1: seg1_use = seg1_use[::-1]
                        new_r1 = r1[:i] + seg2_use + r1[i + L1seg:]
                        new_r2 = r2[:j] + seg1_use + r2[j + L2seg:]
                        routes[r1_idx] = new_r1
                        routes[r2_idx] = new_r2
                        loads[r1_idx] = new_load1
                        loads[r2_idx] = new_load2
                        refresh(r1_idx); refresh(r2_idx)
                        for x in new_r1: wake(x)
                        for x in new_r2: wake(x)
                        return True
        return False

    custs = list(range(1, n))
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if iterations > 200_000: break
        any_improvement = False

        for c in custs:
            if dlb[c]: continue
            if cust_route[c] < 0:
                dlb[c] = True; continue
            if time.time() >= deadline: break

            if (or_opt_from(c)
                or two_opt_star_from(c)
                or swap_star_from(c)
                or cross_exchange_from(c)):
                any_improvement = True
                continue
            dlb[c] = True

        if time.time() >= deadline: break

        for ri in range(len(routes)):
            if not routes[ri]: continue
            if time.time() >= deadline: break
            if two_opt_intra(ri):
                any_improvement = True

        if not any_improvement:
            break

    routes = [r for r in routes if r]
    return routes, instance.routes_objective(routes)

def _related_removal(instance, routes, rng, target):
    Drow = instance._dist_rows
    all_c = [c for r in routes for c in r]
    if not all_c: return set()
    seed = rng.choice(all_c)
    removed = {seed}
    while len(removed) < target:
        ref = rng.choice(list(removed))
        cands = [c for c in all_c if c not in removed]
        if not cands: break
        cands.sort(key=lambda c: Drow[ref][c])
        idx = int((rng.random() ** 3) * min(len(cands), 10))
        removed.add(cands[idx])
    return removed

def _string_removal(instance, routes, rng, target):
    Drow = instance._dist_rows
    all_c = [c for r in routes for c in r]
    if not all_c: return set()
    avg_route_len = max(2, sum(len(r) for r in routes) // max(1, len(routes)))
    L_max = min(10, avg_route_len)

    seed = rng.choice(all_c)

    cust_route = {}
    for ri, r in enumerate(routes):
        for c in r:
            cust_route[c] = ri

    removed = set()
    ruined_routes = set()
    Dseed = Drow[seed]
    nearest = sorted(all_c, key=lambda c: Dseed[c])
    for c in nearest:
        if len(removed) >= target: break
        ri = cust_route.get(c)
        if ri is None or ri in ruined_routes: continue
        r = routes[ri]
        if not r: continue
        L = max(1, min(L_max, len(r)))
        slen = rng.randint(1, L)

        cidx = r.index(c)
        lo = max(0, cidx - rng.randint(0, slen - 1))
        hi = min(len(r), lo + slen)
        for x in r[lo:hi]:
            removed.add(x)
            if len(removed) >= target: break
        ruined_routes.add(ri)
    return removed

def _worst_removal(instance, routes, rng, target):
    Drow = instance._dist_rows
    all_c = []
    for r in routes:
        for p, c in enumerate(r):
            a = r[p - 1] if p > 0 else 0
            b = r[p + 1] if p + 1 < len(r) else 0
            saving = Drow[a][c] + Drow[c][b] - Drow[a][b]
            all_c.append((saving, c))
    all_c.sort(reverse=True)
    removed = set()
    while len(removed) < target and all_c:

        idx = int((rng.random() ** 2) * min(len(all_c), 12))
        removed.add(all_c.pop(idx)[1])
    return removed

def _reinsert_greedy(instance, routes, removed, rng):
    new_routes = [[c for c in r if c not in removed] for r in routes]
    new_routes = [r for r in new_routes if r]
    cap = instance.vehicle_capacity
    demands = instance.demandOfCustomer

    pending = list(removed)
    if len(pending) <= 3:

        pending.sort(key=lambda c: -int(demands[c]))
        for c in pending:
            bt, bp, bd = -1, 0, math.inf
            d_c = int(demands[c])
            for ri, r in enumerate(new_routes):
                if instance.route_demand(r) + d_c > cap:
                    continue
                d, p = best_insertion_delta(instance, r, c)
                if d < bd: bd = d; bt = ri; bp = p
            if bt < 0:
                new_routes.append([c])
            else:
                new_routes[bt].insert(bp, c)
        return new_routes

    while pending:

        route_dem_cache = [instance.route_demand(r) for r in new_routes]

        best_choice = None                                                       
        for c in pending:
            d_c = int(demands[c])
            best1_cost = math.inf; best1_ri = -1; best1_pos = 0
            best2_cost = math.inf
            for ri, r in enumerate(new_routes):
                if route_dem_cache[ri] + d_c > cap:
                    continue
                d, p = best_insertion_delta(instance, r, c)
                if d < best1_cost:
                    best2_cost = best1_cost
                    best1_cost = d; best1_ri = ri; best1_pos = p
                elif d < best2_cost:
                    best2_cost = d

            if best1_ri < 0:
                regret = math.inf                   

                Drow = instance._dist_rows
                best1_cost = Drow[0][c] + Drow[c][0]

            elif math.isinf(best2_cost):
                regret = math.inf                           
            else:
                regret = best2_cost - best1_cost

            score = (regret, -best1_cost)
            if best_choice is None or score > best_choice[0]:
                best_choice = (score, best1_cost, c, best1_ri, best1_pos)

        _, _, c, ri, pos = best_choice
        if ri < 0:
            new_routes.append([c])
        else:
            new_routes[ri].insert(pos, c)
        pending.remove(c)

    return new_routes

_LNS_OPS = ['related', 'string', 'worst']

def destroy_and_repair(instance, routes, rng, neighbors, deadline,
                       removal_frac=0.25, op_weights=None):
    routes = [r[:] for r in routes if r]
    all_c = [c for r in routes for c in r]
    if not all_c: return routes, 'related'
    n = len(all_c)
    target = max(2, min(n - 1, int(n * removal_frac)))

    if op_weights is None:
        op_weights = {k: 1.0 for k in _LNS_OPS}
    total = sum(op_weights.values())
    pick = rng.random() * total
    acc = 0.0
    chosen = _LNS_OPS[0]
    for k in _LNS_OPS:
        acc += op_weights[k]
        if pick <= acc: chosen = k; break

    if chosen == 'related':
        removed = _related_removal(instance, routes, rng, target)
    elif chosen == 'string':
        removed = _string_removal(instance, routes, rng, target)
    else:
        removed = _worst_removal(instance, routes, rng, target)

    new_routes = _reinsert_greedy(instance, routes, removed, rng)
    repaired, _ = repair_routes(instance, new_routes)
    return repaired, chosen

def route_pool_recombine(instance, pool, deadline, best_routes=None):
    if not pool: return None
    cands = []
    for k, r in pool.items():
        if not r: continue
        cost = instance.route_distance(r)
        cands.append((cost / max(1, len(r)), cost, list(r)))
    cands.sort(key=lambda x: (x[0], x[1]))
    covered, sel = set(), []
    for _, _, r in cands:
        if time.time() >= deadline: break
        rs = set(r)
        if covered & rs: continue
        if len(sel) >= instance.num_vehicles: break
        sel.append(r[:]); covered |= rs
    rem = [c for c in instance.customers if c not in covered]
    for c in rem:
        bt, bp, bd = -1, 0, math.inf
        for ri, r in enumerate(sel):
            if instance.route_demand(r) + int(instance.demands[c]) > instance.vehicle_capacity:
                continue
            d, p = best_insertion_delta(instance, r, c)
            if d < bd: bd = d; bt = ri; bp = p
        if bt < 0:
            if len(sel) < instance.num_vehicles: sel.append([c])
            else: return None
        else:
            sel[bt].insert(bp, c)
    repaired, _ = repair_routes(instance, sel)
    return repaired

def sweep_clustering(instance, angle_offset=0.0):
    customers = list(instance.customers)
    if not customers: return []
    Drow = instance._dist_rows
    dx0 = float(instance.xCoordOfCustomer[0])
    dy0 = float(instance.yCoordOfCustomer[0])

    def polar(c):
        dx = float(instance.xCoordOfCustomer[c]) - dx0
        dy = float(instance.yCoordOfCustomer[c]) - dy0
        if dx == 0.0 and dy == 0.0: return 0.0
        return math.atan2(dy, dx)

    tp = 2.0 * math.pi
    ordered = sorted(customers, key=lambda c: (polar(c) - angle_offset) % tp)
    cap = instance.vehicle_capacity
    clusters = []; cur = []; cur_load = 0
    for c in ordered:
        d = int(instance.demands[c])
        if cur and cur_load + d > cap:
            clusters.append(cur); cur = []; cur_load = 0
        cur.append(c); cur_load += d
    if cur: clusters.append(cur)

    routes = []
    for cl in clusters:
        if not cl: continue
        rem = set(cl); tour = []; cn = 0
        while rem:
            nx = min(rem, key=lambda c: Drow[cn][c])
            tour.append(nx); rem.remove(nx); cn = nx
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour)):
                    a = tour[i - 1] if i > 0 else 0
                    b = tour[i]; c = tour[j]
                    dn = tour[j + 1] if j + 1 < len(tour) else 0
                    delta = Drow[a][c] + Drow[b][dn] - Drow[a][b] - Drow[c][dn]
                    if delta < -1e-9:
                        tour[i:j + 1] = tour[i:j + 1][::-1]
                        improved = True; break
                if improved: break
        routes.append(tour)
    return routes

def solve_hybrid(instance, time_budget_s, seed):
    rng = random.Random(seed)
    deadline = time.time() + max(0.05, time_budget_s)
    neighbors = nearest_neighbor_sets(instance, neighbor_count(instance))
    routes = relaxed_nearest_tour_split(instance, rng, neighbors)
    routes, _ = local_search_v3(instance, routes, neighbors, deadline)
    return _SolverRun(routes=routes, objective=instance.routes_objective(routes))

def solve_hgs_search(instance, time_budget_s, seed, incumbent=None):
    started = time.time()
    budget = time_budget_s if time_budget_s is not None else 100
    deadline = started + max(0.2, budget)
    rng = random.Random(seed)
    neighbors = nearest_neighbor_sets(
        instance, max(neighbor_count(instance), hgs_neighbor_count(instance))
    )
    tight = instance.total_demand / max(1, instance.num_vehicles * instance.vehicle_capacity) >= 0.86

    best_routes = None; best_obj = math.inf; best_origin = "none"
    population = []; pool = {}
    crossovers = 0
    no_improve_rounds = 0

    op_weights = {'related': 1.0, 'string': 1.0, 'worst': 1.0}
    op_attempts = {'related': 0, 'string': 0, 'worst': 0}
    op_successes = {'related': 0, 'string': 0, 'worst': 0}

    def update_op_weights():

        for k in _LNS_OPS:
            sr = op_successes[k] / max(1, op_attempts[k])
            op_weights[k] = 0.1 + sr

    def consider(cand, origin, polish=True):
        nonlocal best_routes, best_obj, best_origin
        if time.time() >= deadline: return False
        rep, _ = repair_routes(instance, cand)

        if polish and time.time() < deadline:
            raw_obj = instance.routes_objective(rep)
            slice_s = hgs_polish_slice(instance)
            if best_obj < math.inf and raw_obj > best_obj * 1.05:
                slice_s *= 0.4                                                
            polish_dl = min(deadline, time.time() + slice_s)
            rep, _ = local_search_v3(instance, rep, neighbors, polish_dl)
        obj = instance.routes_objective(rep)
        add_routes_to_pool(instance, pool, rep)
        add_to_population(instance, population, rep)
        if obj + 1e-9 < best_obj:
            best_routes = copy_routes(rep); best_obj = obj; best_origin = origin
            return True
        return False

    if incumbent is not None: consider(incumbent, "incumbent")
    if time_remaining(deadline) > 0.05:
        sw = sweep_clustering(instance, 0.0)
        if sw: consider(sw, "sweep_0")

    sr = 0
    while len(population) < initial_population_size(instance) and time_remaining(deadline) > 0.08:
        sr += 1
        sb = min(hybrid_seed_budget(instance), max(0.05, time_remaining(deadline) * 0.2))
        bk = sr % 4
        if bk == 0: cand = relaxed_nearest_tour_split(instance, rng, neighbors)
        elif bk == 1: cand = solve_hybrid(instance, sb, rng.randrange(1 << 30)).routes
        elif bk == 2:
            order = sorted(instance.customers, key=lambda c: (-int(instance.demands[c]), rng.random()))
            cand = greedy_capacity_pack(instance, order)
        else:
            off = rng.uniform(0, 2 * math.pi)
            cand = sweep_clustering(instance, off)
        consider(cand, f"seed_{sr}")

    final_polish_budget = min(max(2.0, budget * 0.08), 8.0)
    last_improve_time = time.time()

    while time_remaining(deadline) > 0.05 + final_polish_budget:
        improved_this_round = False

        elapsed_since_improve = time.time() - last_improve_time
        if (elapsed_since_improve > 0.25 * budget and
                len(population) >= 4 and
                time_remaining(deadline) > 5.0 + final_polish_budget):

            keep = len(population) // 2
            population.sort(key=instance.routes_objective)
            del population[keep:]
            for _ in range(keep):
                if time_remaining(deadline) <= 1.0 + final_polish_budget: break
                bk = rng.randrange(4)
                if bk == 0: cand = relaxed_nearest_tour_split(instance, rng, neighbors)
                elif bk == 1:
                    sb = min(hybrid_seed_budget(instance), max(0.05, time_remaining(deadline) * 0.05))
                    cand = solve_hybrid(instance, sb, rng.randrange(1 << 30)).routes
                elif bk == 2:
                    order = sorted(instance.customers, key=lambda c: (-int(instance.demands[c]), rng.random()))
                    cand = greedy_capacity_pack(instance, order)
                else:
                    off = rng.uniform(0, 2 * math.pi)
                    cand = sweep_clustering(instance, off)
                consider(cand, "diversify")
            last_improve_time = time.time()                                 
            continue

        if len(population) >= 2:
            pa, pb = select_parents(instance, population, rng)
            order = ordered_crossover(flatten_routes(pa), flatten_routes(pb), rng)
            sp = split_giant_tour(instance, order)
            child = sp[0] if sp else greedy_capacity_pack(instance, order)
            crossovers += 1
            if consider(child, f"xo_{crossovers}"):
                improved_this_round = True

        if (pool and len(pool) >= instance.num_vehicles
                and crossovers % 3 == 0
                and time_remaining(deadline) > 0.05 + final_polish_budget):
            rec = route_pool_recombine(instance, pool, deadline, best_routes=best_routes)
            if rec is not None and consider(rec, "pool"):
                improved_this_round = True

        if best_routes is not None and time_remaining(deadline) > 0.05 + final_polish_budget:
            base = 0.20 if not tight else 0.25
            rf = min(0.55, base + 0.05 * no_improve_rounds)
            update_op_weights()
            shaken, op_used = destroy_and_repair(
                instance, best_routes, rng, neighbors, deadline,
                removal_frac=rf, op_weights=op_weights,
            )
            op_attempts[op_used] += 1
            if consider(shaken, f"lns_{op_used}_{rf:.2f}"):
                op_successes[op_used] += 1
                improved_this_round = True

        if not improved_this_round and time_remaining(deadline) > 0.10 + final_polish_budget:
            off = rng.uniform(0, 2 * math.pi)
            cand = sweep_clustering(instance, off)
            consider(cand, "kick_sweep")

        if no_improve_rounds >= 4 and len(population) > 0 and time_remaining(deadline) > 0.15 + final_polish_budget:
            victim = rng.choice(population)
            update_op_weights()
            big, op_used = destroy_and_repair(
                instance, victim, rng, neighbors, deadline,
                removal_frac=min(0.65, 0.40 + 0.05 * no_improve_rounds),
                op_weights=op_weights,
            )
            op_attempts[op_used] += 1
            if consider(big, f"big_kick_{op_used}"):
                op_successes[op_used] += 1
                improved_this_round = True
                no_improve_rounds = 0

        no_improve_rounds = 0 if improved_this_round else (no_improve_rounds + 1)
        if improved_this_round:
            last_improve_time = time.time()

    if best_routes is not None and time_remaining(deadline) > 0.1:
        final_dl = deadline
        polished, polished_obj = local_search_v3(instance, best_routes, neighbors, final_dl)
        if polished_obj + 1e-9 < best_obj:
            best_routes = copy_routes(polished); best_obj = polished_obj
            best_origin = "final_polish"

    if best_routes is None:
        fb = solve_hybrid(instance, max(0.05, time_remaining(deadline)), seed)
        best_routes = copy_routes(fb.routes); best_obj = fb.objective
        best_origin = "fallback_hybrid"

    return (
        best_routes,
        best_obj,
        {
            "budget_s": round(max(0.2, budget), 6),
            "origin": best_origin,
            "population_size": len(population),
            "crossovers": crossovers,
            "route_pool_size": len(pool),
            "lns_op_successes": dict(op_successes),
            "lns_op_attempts": dict(op_attempts),
            "runtime_s": round(time.time() - started, 6),
        },
    )

def add_to_population(instance, pop, routes):
    cleaned = [r[:] for r in routes if r]
    sig = tuple(tuple(r) for r in cleaned)
    if any(tuple(tuple(r) for r in e) == sig for e in pop): return
    pop.append(cleaned)
    pop.sort(key=instance.routes_objective)
    del pop[max_population_size(instance):]

def _edge_set(routes):
    edges = set()
    for r in routes:
        path = [0] + list(r) + [0]
        for x, y in zip(path, path[1:]):
            edges.add((x, y))
    return edges

def select_parents(instance, pop, rng):
    s = min(len(pop), 5)
    a = min(rng.sample(pop, s), key=instance.routes_objective)
    a_edges = _edge_set(a)
    def score_b(r):
        return instance.routes_objective(r) + 0.05 * len(_edge_set(r) & a_edges)
    b = min(rng.sample(pop, s), key=score_b)
    if a is b and len(pop) > 1:
        b = pop[1] if pop[0] is a else pop[0]
    return a, b

def ordered_crossover(pa, pb, rng):
    if len(pa) < 3: return pa[:]
    s = rng.randrange(0, len(pa) - 1)
    e = rng.randrange(s + 1, len(pa) + 1)
    seg = pa[s:e]; ss = set(seg)
    rem = [c for c in pb if c not in ss]
    return rem[:s] + seg + rem[s:]

def hybrid_seed_budget(instance):
    if instance.num_nodes <= 60: return 0.4
    if instance.num_nodes <= 180: return 0.8
    return 1.2
def hgs_polish_slice(instance):

    if instance.num_nodes <= 60: return 0.40
    if instance.num_nodes <= 180: return 0.70
    return 1.50
def hgs_neighbor_count(instance):
    if instance.num_nodes <= 80: return min(48, instance.num_nodes - 1)
    if instance.num_nodes <= 180: return 64

    return 50
def initial_population_size(instance):
    if instance.num_nodes <= 60: return 8
    if instance.num_nodes <= 180: return 10
    return 12
def max_population_size(instance):
    if instance.num_nodes <= 60: return 16
    if instance.num_nodes <= 180: return 20
    return 24
def time_remaining(deadline): return deadline - time.time()
