from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from cvrp.core import CVRPInstance


BACKGROUND = np.array([0, 0, 0], dtype=np.float32)
FUNGUS_BODY_LOW = np.array([28, 20, 0], dtype=np.float32)
FUNGUS_BODY_HIGH = np.array([232, 198, 26], dtype=np.float32)
FUNGUS_VEIN_LOW = np.array([92, 70, 0], dtype=np.float32)
FUNGUS_VEIN_HIGH = np.array([255, 234, 72], dtype=np.float32)
ACTIVE_GLOW = np.array([255, 226, 88], dtype=np.float32)
DEPOT_COLOR = np.array([255, 236, 110], dtype=np.float32)
CUSTOMER_COLOR = np.array([48, 34, 8], dtype=np.float32)
CUSTOMER_HIT_COLOR = np.array([156, 118, 10], dtype=np.float32)
DEPOT_RADIUS = 2
CUSTOMER_RADIUS = 1


def write_fungus_evolution_gif(
    instance: CVRPInstance,
    frames: Sequence[object],
    output_path: str | Path,
    node_cells: Sequence[Sequence[int]],
    size: int = 720,
    duration_ms: int = 45,
) -> None:
    normalized = [normalize_frame(frame) for frame in frames]
    if not normalized:
        raise ValueError("Cannot render fungus GIF without frames")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    images = [draw_frame(instance, frame, node_cells, size) for frame in normalized]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def normalize_frame(frame: object) -> dict[str, Any]:
    if is_dataclass(frame):
        return asdict(frame)
    if isinstance(frame, dict):
        return dict(frame)
    raise TypeError(f"Unsupported frame object: {type(frame)!r}")


def draw_frame(
    instance: CVRPInstance,
    frame: dict[str, Any],
    node_cells: Sequence[Sequence[int]],
    size: int,
) -> Image.Image:
    grid = int(frame["frame_grid"])
    body = np.asarray(frame.get("density", []), dtype=np.float32).reshape(grid, grid)
    veins = np.asarray(frame.get("vein_density", []), dtype=np.float32).reshape(grid, grid)
    active = np.asarray(frame.get("active_density", []), dtype=np.float32).reshape(grid, grid)
    body = soften_field(soften_field(body, center=0.48, cardinal=0.14, diagonal=0.04), center=0.42, cardinal=0.11, diagonal=0.03)
    veins = soften_field(veins, center=0.24, cardinal=0.08, diagonal=0.02)
    active = soften_field(active, center=0.18, cardinal=0.05, diagonal=0.0)
    service_counts = np.asarray(frame.get("service_counts", []), dtype=np.int32)

    rgb = np.broadcast_to(BACKGROUND, (grid, grid, 3)).copy()
    paint_density(rgb, body, FUNGUS_BODY_LOW, FUNGUS_BODY_HIGH, gamma=0.68, alpha=0.96)
    paint_density(rgb, veins, FUNGUS_VEIN_LOW, FUNGUS_VEIN_HIGH, gamma=0.62, alpha=0.58)
    paint_density(rgb, active, FUNGUS_VEIN_HIGH, ACTIVE_GLOW, gamma=0.82, alpha=0.34)
    paint_nodes(rgb, node_cells, service_counts)

    image = Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8), mode="RGB")
    return image.resize((size, size), resample=Image.NEAREST)


def paint_nodes(
    rgb: np.ndarray,
    node_cells: Sequence[Sequence[int]],
    service_counts: np.ndarray,
) -> None:
    if not node_cells:
        return
    depot_row = int(node_cells[0][0])
    depot_col = int(node_cells[0][1])
    paint_node_blob(rgb, depot_row, depot_col, DEPOT_COLOR, DEPOT_RADIUS)
    for node in range(1, len(node_cells)):
        row = int(node_cells[node][0])
        col = int(node_cells[node][1])
        color = CUSTOMER_HIT_COLOR if node < service_counts.size and service_counts[node] > 0 else CUSTOMER_COLOR
        paint_node_blob(rgb, row, col, color, CUSTOMER_RADIUS)


def paint_node_blob(
    rgb: np.ndarray,
    row: int,
    col: int,
    color: np.ndarray,
    radius: int,
) -> None:
    grid = rgb.shape[0]
    for row_offset in range(-radius, radius + 1):
        rr = row + row_offset
        if rr < 0 or rr >= grid:
            continue
        for col_offset in range(-radius, radius + 1):
            cc = col + col_offset
            if cc < 0 or cc >= grid:
                continue
            distance = abs(row_offset) + abs(col_offset)
            if distance > radius + (radius > 1):
                continue
            if distance == 0:
                weight = 1.0
            elif distance <= radius:
                weight = 0.78
            else:
                weight = 0.52
            target = color * np.float32(weight)
            rgb[rr, cc, :] = np.maximum(rgb[rr, cc, :], target)


def paint_density(
    rgb: np.ndarray,
    field: np.ndarray,
    low_color: np.ndarray,
    high_color: np.ndarray,
    gamma: float,
    alpha: float,
) -> None:
    if field.size == 0:
        return
    max_value = float(field.max(initial=0.0))
    if max_value <= 0.0:
        return
    norm = np.clip(field / max_value, 0.0, 1.0)
    intensity = np.power(norm, gamma)[..., None]
    target = low_color[None, None, :] * (1.0 - intensity) + high_color[None, None, :] * intensity
    blend = np.clip(intensity * alpha, 0.0, 1.0)
    rgb[:] = rgb * (1.0 - blend) + target * blend


def soften_field(
    field: np.ndarray,
    center: float,
    cardinal: float,
    diagonal: float,
) -> np.ndarray:
    padded = np.pad(field, 1, mode="constant")
    return (
        padded[1:-1, 1:-1] * np.float32(center)
        + (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        * np.float32(cardinal)
        + (
            padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        )
        * np.float32(diagonal)
    )
