from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import time

from cvrp.core import CVRPInstance
from fungus.solver import FungusConfig, solve_fungus
from fungus.viz import write_fungus_evolution_gif


def main() -> None:
    defaults = FungusConfig()
    parser = ArgumentParser(description="Run the experimental hyphal-tip fungus automata and write an evolution GIF.")
    parser.add_argument("instance_path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-budget", type=float, default=None)
    parser.add_argument("--grid-size", type=int, default=defaults.grid_size)
    parser.add_argument("--steps", type=int, default=0, help="0 means run until all sources die")
    parser.add_argument("--frame-stride", type=int, default=defaults.frame_stride)
    parser.add_argument("--initial-tip-count", type=int, default=defaults.initial_tip_count)
    parser.add_argument("--branch-limit", type=int, default=defaults.branch_limit)
    parser.add_argument("--max-returned-routes", type=int, default=0)
    parser.add_argument("--energy-loss-per-step", type=float, default=defaults.energy_loss_per_step)
    parser.add_argument("--min-energy", type=float, default=defaults.min_energy)
    parser.add_argument("--initial-energy-scale", type=float, default=defaults.initial_energy_scale)
    parser.add_argument("--service-cost", type=float, default=defaults.service_cost)
    parser.add_argument("--service-energy-retention", type=float, default=defaults.service_energy_retention)
    parser.add_argument("--density-deposit", type=float, default=defaults.density_deposit)
    parser.add_argument("--density-decay", type=float, default=defaults.density_decay)
    parser.add_argument("--density-diffusion", type=float, default=defaults.density_diffusion)
    parser.add_argument("--body-deposit", type=float, default=defaults.body_deposit)
    parser.add_argument("--body-decay", type=float, default=defaults.body_decay)
    parser.add_argument("--body-diffusion", type=float, default=defaults.body_diffusion)
    parser.add_argument("--body-feed-rate", type=float, default=defaults.body_feed_rate)
    parser.add_argument("--body-growth-rate", type=float, default=defaults.body_growth_rate)
    parser.add_argument("--tip-wobble", type=float, default=defaults.tip_wobble)
    parser.add_argument("--num-headings", type=int, default=defaults.num_headings)
    parser.add_argument("--branch-probability", type=float, default=defaults.branch_probability)
    parser.add_argument("--secondary-branch-probability", type=float, default=defaults.secondary_branch_probability)
    parser.add_argument("--node-burst-count", type=int, default=defaults.node_burst_count)
    parser.add_argument("--node-burst-energy-retention", type=float, default=defaults.node_burst_energy_retention)
    args = parser.parse_args()

    instance_path = Path(args.instance_path)
    instance = CVRPInstance.from_file(instance_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("artifacts") / f"fungus_evolution_{instance_path.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FungusConfig(
        grid_size=args.grid_size,
        max_steps=args.steps,
        frame_stride=args.frame_stride,
        initial_tip_count=args.initial_tip_count,
        branch_limit=args.branch_limit,
        max_returned_routes=args.max_returned_routes,
        num_headings=args.num_headings,
        branch_probability=args.branch_probability,
        secondary_branch_probability=args.secondary_branch_probability,
        node_burst_count=args.node_burst_count,
        node_burst_energy_retention=args.node_burst_energy_retention,
        energy_loss_per_step=args.energy_loss_per_step,
        min_energy=args.min_energy,
        initial_energy_scale=args.initial_energy_scale,
        service_cost=args.service_cost,
        service_energy_retention=args.service_energy_retention,
        tip_wobble=args.tip_wobble,
        density_deposit=args.density_deposit,
        density_decay=args.density_decay,
        density_diffusion=args.density_diffusion,
        body_deposit=args.body_deposit,
        body_decay=args.body_decay,
        body_diffusion=args.body_diffusion,
        body_feed_rate=args.body_feed_rate,
        body_growth_rate=args.body_growth_rate,
    )

    started = time.time()
    solution = solve_fungus(
        instance,
        time_budget_s=args.time_budget,
        seed=args.seed,
        config=config,
        collect_frames=True,
    )
    runtime_s = time.time() - started

    frames = solution.metadata.get("frames", [])
    node_cells = solution.metadata.get("node_cells", [])
    write_fungus_evolution_gif(
        instance,
        frames if isinstance(frames, list) else [],
        node_cells=node_cells if isinstance(node_cells, list) else [],
        output_path=output_dir / "evolution.gif",
    )

    routes_text = "\n".join(
        f"route {index + 1}: 0 {' '.join(str(node) for node in route)} 0"
        for index, route in enumerate(solution.routes)
    )
    (output_dir / "routes.txt").write_text((routes_text + "\n") if routes_text else "", encoding="utf-8")
    (output_dir / "routes.json").write_text(json.dumps(solution.routes, indent=2) + "\n", encoding="utf-8")

    record = {
        "instance_id": instance.name,
        "source": str(instance_path),
        "model": "fungus",
        "seed": args.seed,
        "runtime_s": runtime_s,
        "returned_route_count": len(solution.routes),
        **safe_summary(solution.metadata),
    }
    (output_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(record, indent=2))
    if solution.routes:
        print(routes_text)
    else:
        print("no returned routes")
    print(f"wrote {output_dir / 'evolution.gif'}")
    print(f"wrote {output_dir / 'routes.txt'}")


def safe_summary(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"frames", "node_cells", "geometry", "returned_routes"}
        and (isinstance(value, (str, int, float, bool)) or value is None)
    }


if __name__ == "__main__":
    main()
