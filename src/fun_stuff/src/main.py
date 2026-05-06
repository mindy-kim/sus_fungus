import json
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

from vrpinstance import VRPInstance


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a trained FM model checkpoint. When provided, uses "
            "flow-matching warm starts instead of classical construction heuristics."
        ),
    )
    parser.add_argument("--fm-samples", type=int, default=4)
    parser.add_argument("--fm-orders", type=int, default=6)
    parser.add_argument("--fm-steps", type=int, default=32)
    parser.add_argument("--fm-seed", type=int, default=0)
    parser.add_argument(
        "--no-fungus",
        action="store_true",
        help="Disable the fungus automaton warm start.",
    )
    parser.add_argument("--fungus-time-budget", type=float, default=2.0)
    return parser.parse_args()


def solve_instance(args) -> tuple[str | None, float | None]:
    if args.checkpoint:
        from fm_warmstart import solve_with_fm_checkpoint

        solution, objective, *_ = solve_with_fm_checkpoint(
            instance_path=args.input_file,
            checkpoint_path=args.checkpoint,
            num_samples=args.fm_samples,
            max_orders=args.fm_orders,
            integration_steps=args.fm_steps,
            seed=args.fm_seed,
            use_fungus=not args.no_fungus,
            fungus_time_budget=args.fungus_time_budget,
        )
        return solution, objective

    instance = VRPInstance(args.input_file)
    return instance.solve()


def main():
    args = parse_args()
    instance_path = Path(args.input_file)
    started = perf_counter()
    solution, objective = solve_instance(args)
    elapsed = perf_counter() - started

    output = {
        "Instance": instance_path.name,
        "Time": f"{elapsed:.2f}",
        "Result": objective if solution is not None else "--",
        "Solution": solution if solution is not None else "--",
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
