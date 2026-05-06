from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from cvrp.core import CVRPInstance, Routes
from models.classical.solver import HeuristicSolution


@dataclass(frozen=True)
class FungusConfig:
    grid_size: int = 160
    max_steps: int = 0
    frame_stride: int = 3
    branch_limit: int = 12000
    max_returned_routes: int = 0
    num_headings: int = 32
    turn_window: int = 3
    initial_tip_count: int = 6
    service_radius: int = 1
    tip_wobble: float = 0.18
    node_burst_count: int = 12
    node_burst_energy_retention: float = 0.78
    node_burst_jitter: float = 0.18
    branch_probability: float = 0.24
    secondary_branch_probability: float = 0.07
    crowding_weight: float = 1.15
    structure_weight: float = 0.7
    nutrient_weight: float = 1.0
    persistence_weight: float = 1.1
    turn_penalty: float = 0.18
    randomness: float = 0.28
    anastomosis_threshold: float = 4.2
    energy_loss_per_step: float = 0.06
    min_energy: float = 0.8
    initial_energy_scale: float = 3.2
    service_cost: float = 0.25
    service_energy_retention: float = 0.9
    density_deposit: float = 1.0
    density_decay: float = 0.03
    density_diffusion: float = 0.2
    halo_feed_rate: float = 0.05
    body_deposit: float = 0.7
    body_decay: float = 0.008
    body_diffusion: float = 0.28
    body_feed_rate: float = 0.11
    body_growth_rate: float = 0.06
    body_crowding_weight: float = 0.18


@dataclass(frozen=True)
class FungusGeometry:
    grid_size: int
    node_cells: list[list[int]]
    min_x: float
    min_y: float
    scale: float


@dataclass
class FungusFrame:
    step: int
    active_count: int
    returned_count: int
    total_hits: int
    total_branches_created: int
    frame_grid: int
    density: list[float]
    vein_density: list[float]
    active_density: list[float]
    service_counts: list[int]


def solve_fungus(
    instance: CVRPInstance,
    time_budget_s: float | None = None,
    seed: int = 0,
    config: FungusConfig | None = None,
    collect_frames: bool = False,
) -> HeuristicSolution:
    config = config or FungusConfig()
    rng = np.random.default_rng(seed)
    geometry = make_geometry(instance, config.grid_size)
    grid = geometry.grid_size
    node_cells = geometry.node_cells
    customer_fields, customer_field = build_customer_fields(instance, geometry)

    density = np.zeros((grid, grid), dtype=np.float32)
    structure = np.zeros((grid, grid), dtype=np.float32)
    body = np.zeros((grid, grid), dtype=np.float32)
    service_counts = np.zeros(instance.num_nodes, dtype=np.int32)
    served = np.zeros(instance.num_nodes, dtype=bool)
    served[0] = True

    heading_dr, heading_dc = heading_table(config.num_headings)
    rows, cols, headings, energies, capacities, lineages, ages = initialize_tips(instance, geometry, config, rng)
    depot_row = np.array([geometry.node_cells[0][0]], dtype=np.int32)
    depot_col = np.array([geometry.node_cells[0][1]], dtype=np.int32)
    add_kernel(density, depot_row, depot_col, center=2.4, cardinal=0.9, diagonal=0.55)
    add_kernel(structure, depot_row, depot_col, center=1.2, cardinal=0.3, diagonal=0.0)
    add_kernel(body, depot_row, depot_col, center=2.1, cardinal=1.05, diagonal=0.6)

    lineage_parent: list[int] = [-1]
    lineage_node: list[int] = [0]
    route_cache: dict[int, tuple[int, ...]] = {0: ()}

    returned_routes: Routes = []
    route_signatures: set[tuple[int, ...]] = set()
    returned_route_overflow = 0
    total_hits = 0
    total_growth_branches = 0
    total_node_burst_branches = 0
    total_branches_created = 0
    max_active_sources = int(rows.size)
    frames: list[FungusFrame] = []
    deadline = time.time() + time_budget_s if time_budget_s is not None else None
    termination_reason = "extinct"

    if collect_frames:
        frames.append(
            make_frame(
                step=0,
                density=density,
                structure=structure,
                body=body,
                rows=rows,
                cols=cols,
                service_counts=service_counts,
                returned_count=len(returned_routes),
                total_hits=total_hits,
                total_branches_created=total_branches_created,
            )
        )

    steps = 0
    while rows.size > 0:
        if config.max_steps > 0 and steps >= config.max_steps:
            termination_reason = "max_steps"
            break
        if deadline is not None and time.time() >= deadline:
            termination_reason = "time_budget"
            break

        steps += 1
        density *= np.float32(max(0.0, 1.0 - config.density_decay))
        density = diffuse_field(density, config.density_diffusion)
        density += diffuse_field(structure, 0.55) * np.float32(config.halo_feed_rate)
        structure *= np.float32(max(0.0, 1.0 - min(0.012, config.density_decay * 0.18)))
        body *= np.float32(max(0.0, 1.0 - config.body_decay))
        body = diffuse_field(body, config.body_diffusion)
        mature_structure = np.clip(structure - np.float32(0.16), 0.0, None)
        body += diffuse_field(mature_structure, 0.72) * np.float32(config.body_feed_rate)
        body += diffuse_field(diffuse_field(mature_structure, 0.72), 0.72) * np.float32(config.body_feed_rate * 0.55)
        body += diffuse_field(density, 0.52) * np.float32(config.body_growth_rate)
        crowding = crowding_field(density, structure, body, config.body_crowding_weight)

        prev_rows = rows.copy()
        prev_cols = cols.copy()
        rows, cols, headings, energies, capacities, lineages, ages = advance_tips(
            rows=rows,
            cols=cols,
            headings=headings,
            energies=energies,
            capacities=capacities,
            lineages=lineages,
            ages=ages,
            heading_dr=heading_dr,
            heading_dc=heading_dc,
            crowding=crowding,
            structure=structure,
            customer_field=customer_field,
            config=config,
            rng=rng,
        )

        cell_rows, cell_cols = discrete_cells(rows, cols, grid)
        prev_cell_rows, prev_cell_cols = discrete_cells(prev_rows, prev_cols, grid)
        visible_mask = in_bounds_mask(rows, cols, grid)
        prev_visible_mask = in_bounds_mask(prev_rows, prev_cols, grid)

        local_structure = np.zeros(rows.size, dtype=np.float32)
        local_nutrient = np.zeros(rows.size, dtype=np.float32)
        if visible_mask.any():
            local_structure[visible_mask] = structure[cell_rows[visible_mask], cell_cols[visible_mask]]
            local_nutrient[visible_mask] = customer_field[cell_rows[visible_mask], cell_cols[visible_mask]]
        fusion_mask = visible_mask & (local_structure >= config.anastomosis_threshold) & (ages >= 12) & (local_nutrient < 0.08)
        energies = energies - fusion_mask.astype(np.float32) * np.float32(0.08)
        alive_mask = (~fusion_mask) & (energies > config.min_energy)
        rows, cols, headings, energies, capacities, lineages, ages, cell_rows, cell_cols = filter_tip_state(
            alive_mask, rows, cols, headings, energies, capacities, lineages, ages, cell_rows, cell_cols
        )
        prev_cell_rows = prev_cell_rows[alive_mask]
        prev_cell_cols = prev_cell_cols[alive_mask]
        visible_mask = visible_mask[alive_mask]
        prev_visible_mask = prev_visible_mask[alive_mask]

        if rows.size == 0:
            break

        deposit_mask = visible_mask | prev_visible_mask
        deposit_density(
            density=density,
            structure=structure,
            body=body,
            cell_rows=cell_rows[deposit_mask],
            cell_cols=cell_cols[deposit_mask],
            prev_rows=prev_cell_rows[deposit_mask],
            prev_cols=prev_cell_cols[deposit_mask],
            deposit=config.density_deposit,
            body_deposit=config.body_deposit,
        )
        (
            rows,
            cols,
            headings,
            energies,
            capacities,
            lineages,
            ages,
            cell_rows,
            cell_cols,
            service_hits,
            emitted,
            node_burst_branches,
            returned_route_overflow,
        ) = service_customer_hits(
            instance=instance,
            node_cells=node_cells,
            cell_rows=cell_rows,
            cell_cols=cell_cols,
            rows=rows,
            cols=cols,
            headings=headings,
            energies=energies,
            capacities=capacities,
            lineages=lineages,
            ages=ages,
            served=served,
            service_counts=service_counts,
            customer_fields=customer_fields,
            customer_field=customer_field,
            lineage_parent=lineage_parent,
            lineage_node=lineage_node,
            route_cache=route_cache,
            route_signatures=route_signatures,
            returned_routes=returned_routes,
            returned_route_overflow=returned_route_overflow,
            config=config,
            rng=rng,
        )
        total_hits += service_hits
        total_node_burst_branches += node_burst_branches
        total_branches_created = total_growth_branches + total_node_burst_branches

        if rows.size == 0:
            break

        crowding = crowding_field(density, structure, body, config.body_crowding_weight)
        rows, cols, headings, energies, capacities, lineages, ages, created = spawn_branches(
            rows=rows,
            cols=cols,
            headings=headings,
            energies=energies,
            capacities=capacities,
            lineages=lineages,
            ages=ages,
            customer_field=customer_field,
            crowding=crowding,
            config=config,
            rng=rng,
        )
        total_growth_branches += created
        total_branches_created = total_growth_branches + total_node_burst_branches

        rows, cols, headings, energies, capacities, lineages, ages = dedupe_and_trim_tips(
            rows=rows,
            cols=cols,
            headings=headings,
            energies=energies,
            capacities=capacities,
            lineages=lineages,
            ages=ages,
            grid=grid,
            branch_limit=config.branch_limit,
            customer_field=customer_field,
            crowding=crowding,
        )
        max_active_sources = max(max_active_sources, int(rows.size))

        if collect_frames and (steps % config.frame_stride == 0 or rows.size == 0 or emitted > 0):
            frames.append(
                make_frame(
                    step=steps,
                    density=density,
                    structure=structure,
                    body=body,
                    rows=rows,
                    cols=cols,
                    service_counts=service_counts,
                    returned_count=len(returned_routes),
                    total_hits=total_hits,
                    total_branches_created=total_branches_created,
                )
            )

    if rows.size == 0 and steps > 0:
        termination_reason = "extinct"

    final_routes = sorted(returned_routes, key=lambda route: (len(route), tuple(route)))
    metadata: dict[str, object] = {
        "solver": "fungus",
        "seed": seed,
        "steps": steps,
        "active_tip_count": int(rows.size),
        "max_active_sources": max_active_sources,
        "returned_route_count": len(final_routes),
        "returned_route_overflow": returned_route_overflow,
        "total_hits": total_hits,
        "total_branches_created": total_branches_created,
        "total_growth_branches": total_growth_branches,
        "total_node_burst_branches": total_node_burst_branches,
        "termination_reason": termination_reason,
        "served_customers": int(served[1:].sum()),
        "node_cells": [list(cell) for cell in node_cells],
        "geometry": asdict(geometry),
        "returned_routes": [list(route) for route in final_routes],
    }
    if collect_frames:
        metadata["frames"] = [asdict(frame) for frame in frames]

    return HeuristicSolution(
        routes=final_routes,
        objective=instance.routes_objective(final_routes),
        metadata=metadata,
    )


def make_geometry(instance: CVRPInstance, grid_size: int) -> FungusGeometry:
    xs = np.asarray(instance.xs, dtype=np.float32)
    ys = np.asarray(instance.ys, dtype=np.float32)
    min_x = float(xs.min(initial=0.0))
    max_x = float(xs.max(initial=1.0))
    min_y = float(ys.min(initial=0.0))
    max_y = float(ys.max(initial=1.0))
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    usable = max(4, grid_size - 5)
    scale = usable / max(span_x, span_y)
    pad = 2.0

    cols = np.floor((xs - min_x) * scale + pad + 0.5).astype(np.int32)
    rows = np.floor((ys - min_y) * scale + pad + 0.5).astype(np.int32)
    cols = np.clip(cols, 0, grid_size - 1)
    rows = np.clip(rows, 0, grid_size - 1)
    node_cells = [[int(row), int(col)] for row, col in zip(rows.tolist(), cols.tolist())]
    return FungusGeometry(grid_size=grid_size, node_cells=node_cells, min_x=min_x, min_y=min_y, scale=float(scale))


def build_cell_node_lookup(node_cells: list[list[int]]) -> dict[tuple[int, int], list[int]]:
    lookup: dict[tuple[int, int], list[int]] = {}
    for node, (row, col) in enumerate(node_cells):
        lookup.setdefault((row, col), []).append(node)
    return lookup


def build_customer_fields(
    instance: CVRPInstance,
    geometry: FungusGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    grid = geometry.grid_size
    rr = np.arange(grid, dtype=np.float32)[:, None]
    cc = np.arange(grid, dtype=np.float32)[None, :]
    fields = np.zeros((instance.num_nodes, grid, grid), dtype=np.float32)
    sigma = max(2.0, grid / 20.0)
    denom = np.float32(2.0 * sigma * sigma)

    for node in instance.customers:
        row = np.float32(geometry.node_cells[node][0])
        col = np.float32(geometry.node_cells[node][1])
        demand_scale = 1.0 + 0.18 * float(instance.demands[node]) / max(1, instance.vehicle_capacity)
        sq = (rr - row) * (rr - row) + (cc - col) * (cc - col)
        fields[node] = np.exp(-sq / denom, dtype=np.float32) * np.float32(demand_scale)

    return fields, fields.sum(axis=0, dtype=np.float32)


def heading_table(num_headings: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, num_headings, endpoint=False, dtype=np.float32)
    return -np.sin(angles, dtype=np.float32), np.cos(angles, dtype=np.float32)


def initialize_tips(
    instance: CVRPInstance,
    geometry: FungusGeometry,
    config: FungusConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depot_row = np.float32(geometry.node_cells[0][0])
    depot_col = np.float32(geometry.node_cells[0][1])
    initial_count = max(3, min(config.initial_tip_count, config.num_headings))
    if initial_count >= config.num_headings:
        headings = np.arange(config.num_headings, dtype=np.int16)
    else:
        base = np.floor(np.linspace(0, config.num_headings, num=initial_count, endpoint=False, dtype=np.float32)).astype(np.int16)
        rotation = np.int16(rng.integers(0, config.num_headings))
        jitter = rng.integers(-1, 2, size=initial_count, endpoint=False, dtype=np.int16)
        headings = (base + rotation + jitter) % config.num_headings
    rows = depot_row + rng.normal(0.0, 0.22, size=initial_count).astype(np.float32)
    cols = depot_col + rng.normal(0.0, 0.22, size=initial_count).astype(np.float32)
    max_radius = 0.0
    for row, col in geometry.node_cells[1:]:
        max_radius = max(max_radius, math.hypot(row - depot_row, col - depot_col))
    initial_energy = max(
        config.min_energy + 3.0,
        config.initial_energy_scale * (max_radius + 4.0) * max(config.energy_loss_per_step, 0.04),
    )
    energies = np.full(initial_count, np.float32(initial_energy), dtype=np.float32)
    capacities = np.full(initial_count, instance.vehicle_capacity, dtype=np.int32)
    lineages = np.zeros(initial_count, dtype=np.int32)
    ages = np.zeros(initial_count, dtype=np.int16)
    return rows, cols, headings, energies, capacities, lineages, ages


def advance_tips(
    rows: np.ndarray,
    cols: np.ndarray,
    headings: np.ndarray,
    energies: np.ndarray,
    capacities: np.ndarray,
    lineages: np.ndarray,
    ages: np.ndarray,
    heading_dr: np.ndarray,
    heading_dc: np.ndarray,
    crowding: np.ndarray,
    structure: np.ndarray,
    customer_field: np.ndarray,
    config: FungusConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(-config.turn_window, config.turn_window + 1, dtype=np.int16)
    candidate_headings = (headings[:, None] + offsets[None, :]) % config.num_headings
    candidate_rows = rows[:, None] + heading_dr[candidate_headings]
    candidate_cols = cols[:, None] + heading_dc[candidate_headings]

    grid = crowding.shape[0]
    candidate_cell_rows, candidate_cell_cols = discrete_cells(candidate_rows, candidate_cols, grid)
    current_visible = in_bounds_mask(rows, cols, grid)
    valid = (
        (candidate_rows >= 0.0)
        & (candidate_rows <= grid - 1.0)
        & (candidate_cols >= 0.0)
        & (candidate_cols <= grid - 1.0)
    )

    nutrient = customer_field[candidate_cell_rows, candidate_cell_cols] * valid.astype(np.float32)
    local_crowding = crowding[candidate_cell_rows, candidate_cell_cols] * valid.astype(np.float32)
    local_structure = structure[candidate_cell_rows, candidate_cell_cols] * valid.astype(np.float32)
    persistence = (offsets == 0).astype(np.float32)[None, :] * np.float32(config.persistence_weight)
    turn_penalty = np.abs(offsets, dtype=np.float32)[None, :] * np.float32(config.turn_penalty)
    noise = rng.normal(0.0, config.randomness, size=candidate_headings.shape).astype(np.float32)
    edge_bonus = (~valid).astype(np.float32) * np.float32(0.08)

    scores = (
        persistence
        + nutrient * np.float32(config.nutrient_weight)
        - local_crowding * np.float32(config.crowding_weight)
        - local_structure * np.float32(config.structure_weight)
        - turn_penalty
        + edge_bonus
        + noise
    )
    offscreen_mask = ~current_visible
    if offscreen_mask.any():
        scores[offscreen_mask, :] = np.float32(-1.0e9)
        scores[offscreen_mask, config.turn_window] = np.float32(1.0)
    best = scores.argmax(axis=1)
    selector = np.arange(rows.size, dtype=np.int32)

    best_offsets = offsets[best].astype(np.float32)
    headings = candidate_headings[selector, best].astype(np.int16)
    step_rows = heading_dr[headings].astype(np.float32)
    step_cols = heading_dc[headings].astype(np.float32)
    wobble = rng.normal(0.0, config.tip_wobble, size=rows.size).astype(np.float32)
    rows = rows + step_rows - step_cols * wobble
    cols = cols + step_cols + step_rows * wobble
    energies = energies - np.float32(config.energy_loss_per_step) - np.abs(best_offsets) * np.float32(0.02)
    ages = np.minimum(ages.astype(np.int32) + 1, 32767).astype(np.int16)
    return rows, cols, headings, energies, capacities, lineages, ages


def discrete_cells(rows: np.ndarray, cols: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    row_cells = np.clip(np.floor(rows + 0.5).astype(np.int32), 0, grid - 1)
    col_cells = np.clip(np.floor(cols + 0.5).astype(np.int32), 0, grid - 1)
    return row_cells, col_cells


def in_bounds_mask(rows: np.ndarray, cols: np.ndarray, grid: int) -> np.ndarray:
    return (rows >= 0.0) & (rows <= grid - 1.0) & (cols >= 0.0) & (cols <= grid - 1.0)


def filter_tip_state(
    mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    headings: np.ndarray,
    energies: np.ndarray,
    capacities: np.ndarray,
    lineages: np.ndarray,
    ages: np.ndarray,
    cell_rows: np.ndarray,
    cell_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        rows[mask],
        cols[mask],
        headings[mask],
        energies[mask],
        capacities[mask],
        lineages[mask],
        ages[mask],
        cell_rows[mask],
        cell_cols[mask],
    )


def deposit_density(
    density: np.ndarray,
    structure: np.ndarray,
    body: np.ndarray,
    cell_rows: np.ndarray,
    cell_cols: np.ndarray,
    prev_rows: np.ndarray,
    prev_cols: np.ndarray,
    deposit: float,
    body_deposit: float,
) -> None:
    if cell_rows.size == 0:
        return
    mid_rows = ((cell_rows + prev_rows) // 2).astype(np.int32)
    mid_cols = ((cell_cols + prev_cols) // 2).astype(np.int32)
    add_kernel(density, cell_rows, cell_cols, center=deposit * 0.52, cardinal=deposit * 0.11, diagonal=deposit * 0.06)
    add_kernel(density, mid_rows, mid_cols, center=deposit * 0.22, cardinal=deposit * 0.05, diagonal=deposit * 0.025)
    add_kernel(density, prev_rows, prev_cols, center=deposit * 0.16, cardinal=deposit * 0.03, diagonal=deposit * 0.015)
    add_kernel(structure, cell_rows, cell_cols, center=deposit * 0.58, cardinal=deposit * 0.06, diagonal=0.0)
    add_kernel(structure, mid_rows, mid_cols, center=deposit * 0.14, cardinal=deposit * 0.02, diagonal=0.0)
    add_kernel(structure, prev_rows, prev_cols, center=deposit * 0.1, cardinal=0.0, diagonal=0.0)
    add_kernel(body, cell_rows, cell_cols, center=body_deposit * 0.34, cardinal=body_deposit * 0.2, diagonal=body_deposit * 0.11)
    add_kernel(body, mid_rows, mid_cols, center=body_deposit * 0.2, cardinal=body_deposit * 0.12, diagonal=body_deposit * 0.08)
    add_kernel(body, prev_rows, prev_cols, center=body_deposit * 0.14, cardinal=body_deposit * 0.08, diagonal=body_deposit * 0.05)
    add_wide_kernel(
        body,
        cell_rows,
        cell_cols,
        cardinal2=body_deposit * 0.09,
        knight=body_deposit * 0.055,
        diagonal2=body_deposit * 0.03,
    )
    add_wide_kernel(
        body,
        mid_rows,
        mid_cols,
        cardinal2=body_deposit * 0.045,
        knight=body_deposit * 0.03,
        diagonal2=body_deposit * 0.018,
    )


def add_kernel(
    field: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    center: float,
    cardinal: float,
    diagonal: float,
) -> None:
    if rows.size == 0:
        return
    grid = field.shape[0]
    if center != 0.0:
        np.add.at(field, (rows, cols), np.float32(center))
    if cardinal != 0.0:
        for row_offset, col_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = np.clip(rows + row_offset, 0, grid - 1)
            cc = np.clip(cols + col_offset, 0, grid - 1)
            np.add.at(field, (rr, cc), np.float32(cardinal))
    if diagonal != 0.0:
        for row_offset, col_offset in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = np.clip(rows + row_offset, 0, grid - 1)
            cc = np.clip(cols + col_offset, 0, grid - 1)
            np.add.at(field, (rr, cc), np.float32(diagonal))


def add_wide_kernel(
    field: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    cardinal2: float,
    knight: float,
    diagonal2: float,
) -> None:
    if rows.size == 0:
        return
    grid = field.shape[0]
    if cardinal2 != 0.0:
        for row_offset, col_offset in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            rr = np.clip(rows + row_offset, 0, grid - 1)
            cc = np.clip(cols + col_offset, 0, grid - 1)
            np.add.at(field, (rr, cc), np.float32(cardinal2))
    if knight != 0.0:
        for row_offset, col_offset in (
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ):
            rr = np.clip(rows + row_offset, 0, grid - 1)
            cc = np.clip(cols + col_offset, 0, grid - 1)
            np.add.at(field, (rr, cc), np.float32(knight))
    if diagonal2 != 0.0:
        for row_offset, col_offset in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
            rr = np.clip(rows + row_offset, 0, grid - 1)
            cc = np.clip(cols + col_offset, 0, grid - 1)
            np.add.at(field, (rr, cc), np.float32(diagonal2))


def node_burst_headings(num_headings: int, burst_count: int, base_heading: int) -> np.ndarray:
    if burst_count <= 0 or num_headings <= 0:
        return np.empty(0, dtype=np.int16)
    count = min(burst_count, num_headings)
    offsets = np.floor(np.arange(count, dtype=np.float32) * (num_headings / count)).astype(np.int16)
    return ((base_heading + offsets) % num_headings).astype(np.int16, copy=False)


def service_customer_hits(
    instance: CVRPInstance,
    node_cells: list[list[int]],
    cell_rows: np.ndarray,
    cell_cols: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    headings: np.ndarray,
    energies: np.ndarray,
    capacities: np.ndarray,
    lineages: np.ndarray,
    ages: np.ndarray,
    served: np.ndarray,
    service_counts: np.ndarray,
    customer_fields: np.ndarray,
    customer_field: np.ndarray,
    lineage_parent: list[int],
    lineage_node: list[int],
    route_cache: dict[int, tuple[int, ...]],
    route_signatures: set[tuple[int, ...]],
    returned_routes: Routes,
    returned_route_overflow: int,
    config: FungusConfig,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
]:
    if rows.size == 0:
        return (
            rows,
            cols,
            headings,
            energies,
            capacities,
            lineages,
            ages,
            cell_rows,
            cell_cols,
            0,
            0,
            0,
            returned_route_overflow,
        )

    dead_mask = np.zeros(rows.size, dtype=bool)
    busy_mask = np.zeros(rows.size, dtype=bool)
    total_hits = 0
    emitted = 0
    node_burst_branches = 0
    burst_rows: list[np.ndarray] = []
    burst_cols: list[np.ndarray] = []
    burst_headings: list[np.ndarray] = []
    burst_energies: list[np.ndarray] = []
    burst_capacities: list[np.ndarray] = []
    burst_lineages: list[np.ndarray] = []
    burst_ages: list[np.ndarray] = []
    visible_mask = in_bounds_mask(rows, cols, customer_field.shape[0])

    for node in instance.customers:
        if served[node]:
            continue
        node_row = int(node_cells[node][0])
        node_col = int(node_cells[node][1])
        near_mask = (
            visible_mask
            & (~dead_mask)
            & (~busy_mask)
            & (np.abs(cell_rows - node_row) <= config.service_radius)
            & (np.abs(cell_cols - node_col) <= config.service_radius)
        )
        tip_indices = np.flatnonzero(near_mask)
        if tip_indices.size == 0:
            continue

        demand = instance.demands[node]
        feasible = tip_indices[capacities[tip_indices] >= demand]
        infeasible = tip_indices[capacities[tip_indices] < demand]
        if infeasible.size > 0:
            dead_mask[infeasible] = True
        if feasible.size == 0:
            continue

        distances = np.maximum(
            np.abs(cell_rows[feasible] - node_row),
            np.abs(cell_cols[feasible] - node_col),
        ).astype(np.float32)
        scores = energies[feasible] - distances * np.float32(0.18) - ages[feasible].astype(np.float32) * np.float32(0.01)
        winner_local = feasible[np.argmax(scores)]
        busy_mask[winner_local] = True
        served[node] = True
        service_counts[node] += 1
        customer_field -= customer_fields[node]
        total_hits += 1

        new_energy = max(
            config.min_energy + 0.05,
            float((energies[winner_local] - np.float32(config.service_cost)) * np.float32(config.service_energy_retention)),
        )
        remaining_capacity = int(capacities[winner_local] - demand)
        parent = int(lineages[winner_local])
        route = route_cache[parent] + (int(node),)
        new_lineage = len(lineage_parent)
        lineage_parent.append(parent)
        lineage_node.append(int(node))
        route_cache[new_lineage] = route

        lineages[winner_local] = np.int32(new_lineage)
        capacities[winner_local] = np.int32(remaining_capacity)
        burst_energy = max(
            config.min_energy + 0.05,
            float(new_energy * np.float32(config.node_burst_energy_retention)),
        )
        headings_to_emit = node_burst_headings(
            config.num_headings,
            config.node_burst_count,
            int(headings[winner_local]),
        )
        if headings_to_emit.size > 0:
            rows[winner_local] = np.float32(node_row) + np.float32(rng.normal(0.0, config.node_burst_jitter))
            cols[winner_local] = np.float32(node_col) + np.float32(rng.normal(0.0, config.node_burst_jitter))
            headings[winner_local] = np.int16(headings_to_emit[0])
        energies[winner_local] = np.float32(burst_energy)
        ages[winner_local] = np.int16(0)

        available_children = max(0, config.branch_limit - (rows.size + node_burst_branches))
        extra_count = min(max(0, headings_to_emit.size - 1), available_children)
        if extra_count > 0:
            burst_rows.append(
                np.full(extra_count, np.float32(node_row), dtype=np.float32)
                + rng.normal(0.0, config.node_burst_jitter, size=extra_count).astype(np.float32)
            )
            burst_cols.append(
                np.full(extra_count, np.float32(node_col), dtype=np.float32)
                + rng.normal(0.0, config.node_burst_jitter, size=extra_count).astype(np.float32)
            )
            burst_headings.append(headings_to_emit[1 : 1 + extra_count].astype(np.int16, copy=False))
            burst_energies.append(np.full(extra_count, np.float32(burst_energy), dtype=np.float32))
            burst_capacities.append(np.full(extra_count, np.int32(remaining_capacity), dtype=np.int32))
            burst_lineages.append(np.full(extra_count, np.int32(new_lineage), dtype=np.int32))
            burst_ages.append(np.zeros(extra_count, dtype=np.int16))
            node_burst_branches += extra_count

        if route not in route_signatures:
            if config.max_returned_routes > 0 and len(returned_routes) >= config.max_returned_routes:
                returned_route_overflow += 1
            else:
                returned_routes.append(list(route))
                route_signatures.add(route)
                emitted += 1

    if dead_mask.any():
        alive_mask = ~dead_mask & (energies > config.min_energy)
        rows, cols, headings, energies, capacities, lineages, ages, cell_rows, cell_cols = filter_tip_state(
            alive_mask, rows, cols, headings, energies, capacities, lineages, ages, cell_rows, cell_cols
        )

    if burst_rows:
        rows = np.concatenate((rows, *burst_rows), dtype=np.float32)
        cols = np.concatenate((cols, *burst_cols), dtype=np.float32)
        headings = np.concatenate((headings, *burst_headings), dtype=np.int16)
        energies = np.concatenate((energies, *burst_energies), dtype=np.float32)
        capacities = np.concatenate((capacities, *burst_capacities), dtype=np.int32)
        lineages = np.concatenate((lineages, *burst_lineages), dtype=np.int32)
        ages = np.concatenate((ages, *burst_ages), dtype=np.int16)
        cell_rows, cell_cols = discrete_cells(rows, cols, customer_field.shape[0])

    return (
        rows,
        cols,
        headings,
        energies,
        capacities,
        lineages,
        ages,
        cell_rows,
        cell_cols,
        total_hits,
        emitted,
        node_burst_branches,
        returned_route_overflow,
    )


def spawn_branches(
    rows: np.ndarray,
    cols: np.ndarray,
    headings: np.ndarray,
    energies: np.ndarray,
    capacities: np.ndarray,
    lineages: np.ndarray,
    ages: np.ndarray,
    customer_field: np.ndarray,
    crowding: np.ndarray,
    config: FungusConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    active_count = rows.size
    if active_count == 0 or active_count >= config.branch_limit:
        return rows, cols, headings, energies, capacities, lineages, ages, 0

    grid = customer_field.shape[0]
    visible_mask = in_bounds_mask(rows, cols, grid)
    if not visible_mask.any():
        return rows, cols, headings, energies, capacities, lineages, ages, 0

    cell_rows, cell_cols = discrete_cells(rows, cols, grid)
    nutrient = customer_field[cell_rows, cell_cols]
    local_crowding = crowding[cell_rows, cell_cols]
    energy_span = max(float(energies.max(initial=0.0) - config.min_energy), 1.0)
    energy_factor = np.clip((energies - np.float32(config.min_energy)) / np.float32(energy_span), 0.0, 1.0)
    nutrient_scale = np.clip(nutrient / np.float32(max(0.2, float(customer_field.max(initial=0.0)))), 0.0, 1.0)
    openness = np.clip(1.15 - local_crowding, 0.12, 1.0)
    base_probability = (
        np.float32(config.branch_probability)
        * (np.float32(0.38) + np.float32(0.62) * energy_factor)
        * (np.float32(0.3) + np.float32(0.7) * np.maximum(nutrient_scale, np.float32(0.2)))
        * openness
    )
    branch_mask = rng.random(active_count, dtype=np.float32) < base_probability
    branch_mask &= visible_mask
    candidate_indices = np.flatnonzero(branch_mask)
    if candidate_indices.size == 0:
        return rows, cols, headings, energies, capacities, lineages, ages, 0

    remaining = max(0, config.branch_limit - active_count)
    if candidate_indices.size > remaining:
        scores = energies[candidate_indices] + nutrient[candidate_indices] * np.float32(0.5) - local_crowding[candidate_indices]
        keep = np.argpartition(scores, -remaining)[-remaining:] if remaining > 0 else np.empty(0, dtype=np.int32)
        candidate_indices = candidate_indices[keep]
    if candidate_indices.size == 0:
        return rows, cols, headings, energies, capacities, lineages, ages, 0

    branch_span = max(2, config.num_headings // 10)
    offsets = rng.integers(branch_span, branch_span + 3, size=candidate_indices.size, endpoint=False, dtype=np.int16)
    signs = np.where(rng.random(candidate_indices.size) < 0.5, np.int16(-1), np.int16(1))
    daughter_headings = (headings[candidate_indices] + offsets * signs) % config.num_headings
    daughter_rows = rows[candidate_indices].copy()
    daughter_cols = cols[candidate_indices].copy()
    daughter_energies = energies[candidate_indices] * np.float32(0.78)
    daughter_capacities = capacities[candidate_indices].copy()
    daughter_lineages = lineages[candidate_indices].copy()
    daughter_ages = np.zeros(candidate_indices.size, dtype=np.int16)

    energies[candidate_indices] *= np.float32(0.94)
    rows = np.concatenate((rows, daughter_rows), dtype=np.float32)
    cols = np.concatenate((cols, daughter_cols), dtype=np.float32)
    headings = np.concatenate((headings, daughter_headings.astype(np.int16)), dtype=np.int16)
    energies = np.concatenate((energies, daughter_energies.astype(np.float32)), dtype=np.float32)
    capacities = np.concatenate((capacities, daughter_capacities.astype(np.int32)), dtype=np.int32)
    lineages = np.concatenate((lineages, daughter_lineages.astype(np.int32)), dtype=np.int32)
    ages = np.concatenate((ages, daughter_ages), dtype=np.int16)

    if config.secondary_branch_probability > 0.0:
        secondary_mask = rng.random(candidate_indices.size, dtype=np.float32) < np.float32(config.secondary_branch_probability)
        secondary_parents = candidate_indices[secondary_mask]
        remaining = max(0, config.branch_limit - rows.size)
        if secondary_parents.size > remaining:
            secondary_parents = secondary_parents[:remaining]
        if secondary_parents.size > 0:
            secondary_offsets = rng.integers(branch_span + 1, branch_span + 4, size=secondary_parents.size, endpoint=False, dtype=np.int16)
            secondary_signs = np.where(rng.random(secondary_parents.size) < 0.5, np.int16(-1), np.int16(1))
            secondary_headings = (headings[secondary_parents] + secondary_offsets * secondary_signs) % config.num_headings
            rows = np.concatenate((rows, rows[secondary_parents].copy()), dtype=np.float32)
            cols = np.concatenate((cols, cols[secondary_parents].copy()), dtype=np.float32)
            headings = np.concatenate((headings, secondary_headings.astype(np.int16)), dtype=np.int16)
            energies = np.concatenate((energies, (energies[secondary_parents] * np.float32(0.72)).astype(np.float32)), dtype=np.float32)
            capacities = np.concatenate((capacities, capacities[secondary_parents].copy()), dtype=np.int32)
            lineages = np.concatenate((lineages, lineages[secondary_parents].copy()), dtype=np.int32)
            ages = np.concatenate((ages, np.zeros(secondary_parents.size, dtype=np.int16)), dtype=np.int16)

    return rows, cols, headings, energies, capacities, lineages, ages, int(candidate_indices.size)


def dedupe_and_trim_tips(
    rows: np.ndarray,
    cols: np.ndarray,
    headings: np.ndarray,
    energies: np.ndarray,
    capacities: np.ndarray,
    lineages: np.ndarray,
    ages: np.ndarray,
    grid: int,
    branch_limit: int,
    customer_field: np.ndarray,
    crowding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rows.size == 0:
        return rows, cols, headings, energies, capacities, lineages, ages

    visible_mask = in_bounds_mask(rows, cols, grid)
    if visible_mask.any():
        visible_indices = np.flatnonzero(visible_mask)
        offscreen_indices = np.flatnonzero(~visible_mask)
        cell_rows, cell_cols = discrete_cells(rows[visible_indices], cols[visible_indices], grid)
        order_local = np.lexsort((ages[visible_indices], -energies[visible_indices], headings[visible_indices], lineages[visible_indices], cell_cols, cell_rows))
        order = visible_indices[order_local]
        sorted_rows = cell_rows[order_local]
        sorted_cols = cell_cols[order_local]
        sorted_lineages = lineages[order]
        sorted_headings = headings[order]
        unique_mask = np.ones(order.size, dtype=bool)
        unique_mask[1:] = (
            (sorted_rows[1:] != sorted_rows[:-1])
            | (sorted_cols[1:] != sorted_cols[:-1])
            | (sorted_lineages[1:] != sorted_lineages[:-1])
            | (sorted_headings[1:] != sorted_headings[:-1])
        )
        keep = np.concatenate((order[unique_mask], offscreen_indices), dtype=np.int32)

        rows = rows[keep]
        cols = cols[keep]
        headings = headings[keep]
        energies = energies[keep]
        capacities = capacities[keep]
        lineages = lineages[keep]
        ages = ages[keep]

    if rows.size <= branch_limit:
        return rows, cols, headings, energies, capacities, lineages, ages

    cell_rows, cell_cols = discrete_cells(rows, cols, grid)
    scores = energies + customer_field[cell_rows, cell_cols] * np.float32(0.35) - crowding[cell_rows, cell_cols] * np.float32(0.1)
    chosen = np.argpartition(scores, -branch_limit)[-branch_limit:]
    return (
        rows[chosen],
        cols[chosen],
        headings[chosen],
        energies[chosen],
        capacities[chosen],
        lineages[chosen],
        ages[chosen],
    )


def crowding_field(
    density: np.ndarray,
    structure: np.ndarray,
    body: np.ndarray,
    body_weight: float,
) -> np.ndarray:
    base = density * np.float32(0.5) + structure + body * np.float32(body_weight)
    padded = np.pad(base, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    cardinal = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) * np.float32(0.16)
    diagonal = (
        padded[:-2, :-2]
        + padded[:-2, 2:]
        + padded[2:, :-2]
        + padded[2:, 2:]
    ) * np.float32(0.08)
    return center * np.float32(0.42) + cardinal + diagonal


def diffuse_field(field: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0 or field.size == 0:
        return field
    padded = np.pad(field, 1, mode="constant")
    blurred = (
        padded[1:-1, 1:-1] * np.float32(0.28)
        + (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        * np.float32(0.12)
        + (
            padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        )
        * np.float32(0.06)
    )
    blend = np.float32(np.clip(amount, 0.0, 1.0))
    return field * (np.float32(1.0) - blend) + blurred * blend


def make_frame(
    step: int,
    density: np.ndarray,
    structure: np.ndarray,
    body: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    service_counts: np.ndarray,
    returned_count: int,
    total_hits: int,
    total_branches_created: int,
) -> FungusFrame:
    active_density = diffuse_field(tip_density_grid(rows, cols, density.shape[0]), 0.2) * np.float32(0.55)
    halo_density = diffuse_field(density, 0.42)
    body_glow = diffuse_field(body, 0.35)
    mature_glow = diffuse_field(diffuse_field(structure, 0.78), 0.62)
    visible_density = np.clip(
        body_glow * np.float32(1.05)
        + mature_glow * np.float32(0.46)
        + halo_density * np.float32(0.16),
        0.0,
        None,
    )
    visible_density = diffuse_field(visible_density, 0.16)
    visible_veins = np.clip(
        structure * np.float32(1.12)
        + diffuse_field(structure, 0.35) * np.float32(0.42)
        + diffuse_field(density, 0.22) * np.float32(0.08),
        0.0,
        None,
    )
    return FungusFrame(
        step=step,
        active_count=int(rows.size),
        returned_count=returned_count,
        total_hits=total_hits,
        total_branches_created=total_branches_created,
        frame_grid=int(density.shape[0]),
        density=visible_density.astype(np.float32, copy=False).ravel().tolist(),
        vein_density=visible_veins.astype(np.float32, copy=False).ravel().tolist(),
        active_density=active_density.astype(np.float32, copy=False).ravel().tolist(),
        service_counts=service_counts.astype(np.int32, copy=False).tolist(),
    )


def tip_density_grid(rows: np.ndarray, cols: np.ndarray, grid: int) -> np.ndarray:
    active = np.zeros((grid, grid), dtype=np.float32)
    if rows.size == 0:
        return active
    cell_rows, cell_cols = discrete_cells(rows, cols, grid)
    np.add.at(active, (cell_rows, cell_cols), np.float32(1.0))
    padded = np.pad(active, 1, mode="constant")
    return (
        padded[1:-1, 1:-1]
        + (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        * np.float32(0.45)
        + (
            padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        )
        * np.float32(0.2)
    )
