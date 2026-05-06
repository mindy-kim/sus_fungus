import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import torch

from fm_warmstart import (
    build_sampling_generator,
    choose_torch_device,
    load_fm_example_batch,
    load_fm_model,
)
from learning.vrp_flow_matching import sample_edge_probabilities
ROUTE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]
DEPOT_COLOR = "#222222"
NODE_COLOR  = "#aaaaaa"
EDGE_CMAP   = plt.get_cmap("YlOrRd")


def find_label_record(instance_path: Path) -> dict | None:
    stem = instance_path.stem
    for candidate_dir in [instance_path.parent, instance_path.parent.parent]:
        for labels_file in candidate_dir.rglob("labels.jsonl"):
            with labels_file.open() as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("instance_id") == stem:
                        return rec
    return None


def _draw_nodes(ax, coords, demands, capacity):
    xs = coords[:, 0].numpy()
    ys = coords[:, 1].numpy()

    cust_x, cust_y = xs[1:], ys[1:]
    cust_d = demands[1:].numpy() / float(capacity)
    sizes = 30 + cust_d * 120
    ax.scatter(cust_x, cust_y, s=sizes, c=NODE_COLOR,
               zorder=3, linewidths=0.4, edgecolors="#555555")

    ax.scatter(xs[0], ys[0], s=200, marker="*",
               c=DEPOT_COLOR, zorder=5, linewidths=0)


def _axis_style(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")


def draw_optimal(ax, example):
    coords = example.coords.numpy()
    routes = example.target_routes

    if routes is None:
        ax.set_title("optimal\n(no label)", fontsize=9, pad=4)
        _draw_nodes(ax, example.coords, example.demands, example.vehicle_capacity)
        _axis_style(ax)
        return

    for r_idx, route in enumerate(routes):
        color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
        full = [0] + route + [0]
        for a, b in zip(full, full[1:]):
            ax.plot(
                [coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                color=color, linewidth=1.0, alpha=0.8, zorder=2,
            )

    _draw_nodes(ax, example.coords, example.demands, example.vehicle_capacity)
    _axis_style(ax)

    n_routes = len([r for r in routes if r])
    ax.set_title(f"optimal  ({n_routes} routes)", fontsize=9, pad=4)


def draw_sample(ax, example, edge_probs: torch.Tensor, sample_idx: int):
    coords = example.coords.numpy()
    edges = example.candidate_edges
    probs = edge_probs.numpy()

    order = probs.argsort()

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    for i in order:
        u, v = edges[i]
        p = float(probs[i])
        if p < 0.05:
            continue
        color = EDGE_CMAP(norm(p))
        ax.plot(
            [coords[u, 0], coords[v, 0]],
            [coords[u, 1], coords[v, 1]],
            color=color,
            linewidth=0.6 + p * 1.4,
            alpha=0.2 + p * 0.7,
            zorder=2,
        )

    _draw_nodes(ax, example.coords, example.demands, example.vehicle_capacity)
    _axis_style(ax)
    ax.set_title(f"sample {sample_idx + 1}", fontsize=9, pad=4)


def parse_args():
    p = ArgumentParser(description="Visualize FM model samples vs. optimal route.")
    p.add_argument("--checkpoint",         type=str, required=True)
    p.add_argument("--instance-path",      type=str, required=True)
    p.add_argument("--label-record",       type=str, default=None,
                   help="JSON string for the label record (optional; auto-detected otherwise)")
    p.add_argument("--num-samples",        type=int, default=4)
    p.add_argument("--integration-steps",  type=int, default=32)
    p.add_argument("--device",             type=str, default=None)
    p.add_argument("--seed",               type=int, default=0)
    p.add_argument("--output",             type=str, default="samples.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = choose_torch_device(args.device)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    model, config = load_fm_model(checkpoint_path, device)

    instance_path = Path(args.instance_path)
    instance_id = instance_path.stem

    record = None
    if args.label_record:
        record = json.loads(args.label_record)
    else:
        record = find_label_record(instance_path)
    example, batch = load_fm_example_batch(
        instance_path,
        config,
        device,
        record=record,
        instance_id=instance_id,
    )
    generator = build_sampling_generator(device, args.seed)

    samples: list[torch.Tensor] = []
    for _ in range(args.num_samples):
        probs = sample_edge_probabilities(
            model, batch,
            steps=args.integration_steps,
            noise_scale=config["noise_scale"],
            generator=generator if device.type != "mps" else None,
        ).cpu()
        samples.append(probs)

    total_panels = 1 + args.num_samples
    ncols = min(total_panels, 4)
    nrows = (total_panels + ncols - 1) // ncols

    fig_w = ncols * 3.5
    fig_h = nrows * 3.5 + 0.6

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.06, "hspace": 0.18})

    all_axes = axes.flatten()
    draw_optimal(all_axes[0], example)

    for i, probs in enumerate(samples):
        draw_sample(all_axes[i + 1], example, probs, i)

    for ax in all_axes[total_panels:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=EDGE_CMAP,
                               norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=all_axes[1:total_panels],
                        orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("edge probability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    legend_handles = [
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=DEPOT_COLOR, markersize=10, label="depot"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=NODE_COLOR, markersize=7,
               markeredgecolor="#555555", markeredgewidth=0.4, label="customer"),
    ]
    all_axes[0].legend(handles=legend_handles, loc="lower right",
                       fontsize=7, framealpha=0.8, edgecolor="#cccccc")

    obj_str = f"  -  optimal obj = {record['objective']:.2f}" if record and "objective" in record else ""
    fig.suptitle(
        f"{instance_id}{obj_str}",
        fontsize=10, y=1.01, fontweight="bold",
    )

    out_path = Path(args.output)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
