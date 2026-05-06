import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from fm_warmstart import (
    build_sampling_generator,
    choose_torch_device,
    load_fm_example_batch,
    load_fm_model,
)
from learning.vrp_flow_matching import sample_edge_probabilities
from learning.vrp_warm_start import generate_candidate_orders
from vrpinstance import VRPInstance


def parse_args():
    parser = ArgumentParser(description="Sample VRP warm starts from a trained edge flow model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--instance-path", type=str, required=True)
    parser.add_argument("--instance-id", type=str, default=None)
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-orders", type=int, default=6)
    parser.add_argument("--distance-weight", type=float, default=0.35)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lns-time-budget-sec", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_torch_device(args.device)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    model, config = load_fm_model(checkpoint_path, device)

    instance_path = Path(args.instance_path)
    example, batch = load_fm_example_batch(
        instance_path,
        config,
        device,
        instance_id=args.instance_id,
    )

    best_result = None
    all_candidates = []
    generator = build_sampling_generator(device, args.seed)
    solver = VRPInstance(
        str(instance_path),
        config={"lns_time_budget_sec": args.lns_time_budget_sec},
    )

    for sample_idx in range(args.num_samples):
        edge_probabilities = sample_edge_probabilities(
            model,
            batch,
            steps=args.integration_steps,
            noise_scale=config["noise_scale"],
            generator=generator if device.type != "mps" else None,
        ).cpu()

        orders = generate_candidate_orders(
            example,
            edge_probabilities=edge_probabilities,
            max_orders=args.max_orders,
            distance_weight=args.distance_weight,
        )

        for order_idx, order in enumerate(orders):
            try:
                solution, objective = solver.solve_from_order(order)
            except RuntimeError:
                continue
            candidate = {
                "sample": sample_idx,
                "order_index": order_idx,
                "objective": objective,
                "solution": solution,
                "order_prefix": order[:12],
            }
            all_candidates.append(candidate)
            if best_result is None or objective < best_result["objective"]:
                best_result = candidate

    if best_result is None:
        raise RuntimeError("No warm-start candidate orders were produced.")

    print(
        json.dumps(
            {
                "instance": str(instance_path),
                "checkpoint": str(checkpoint_path),
                "best": best_result,
                "candidates_evaluated": len(all_candidates),
            }
        )
    )


if __name__ == "__main__":
    main()
