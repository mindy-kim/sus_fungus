import json
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from learning.vrp_flow_data import VRPGraphDataset, collate_graph_batch
from learning.vrp_flow_matching import compute_batch_losses, summarize_metrics
from learning.vrp_flow_model import EdgeFlowMatchingModel


def parse_args():
    parser = ArgumentParser(description="Train a graph-based edge flow matching model for VRP.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/vrp_edge_flow")
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: EdgeFlowMatchingModel,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    noise_scale: float,
    classification_weight: float,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    rows = []

    for batch in loader:
        batch = batch.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_batch_losses(
            model,
            batch,
            noise_scale=noise_scale,
            classification_weight=classification_weight,
        )
        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        rows.append(metrics)

    return summarize_metrics(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    train_dataset = VRPGraphDataset(
        data_dir=args.data_dir,
        split=args.train_split,
        knn_k=args.knn_k,
        limit=args.train_limit,
    )
    val_dataset = VRPGraphDataset(
        data_dir=args.data_dir,
        split=args.val_split,
        knn_k=args.knn_k,
        limit=args.val_limit,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
    )

    sample_batch = collate_graph_batch([train_dataset[0]])
    model = EdgeFlowMatchingModel(
        node_feature_dim=sample_batch.node_features.size(1),
        edge_feature_dim=sample_batch.edge_features.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        time_dim=args.time_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_path = checkpoint_dir / "best.pt"
    history = []

    config = {
        "data_dir": args.data_dir,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "knn_k": args.knn_k,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "time_dim": args.time_dim,
        "dropout": args.dropout,
        "noise_scale": args.noise_scale,
        "classification_weight": args.classification_weight,
        "node_feature_dim": sample_batch.node_features.size(1),
        "edge_feature_dim": sample_batch.edge_features.size(1),
    }

    for epoch_idx in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            noise_scale=args.noise_scale,
            classification_weight=args.classification_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                noise_scale=args.noise_scale,
                classification_weight=args.classification_weight,
            )

        row = {
            "epoch": epoch_idx,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            json.dumps(
                {
                    "epoch": epoch_idx,
                    "train_loss": round(train_metrics.get("loss", 0.0), 6),
                    "train_f1": round(train_metrics.get("f1", 0.0), 6),
                    "val_loss": round(val_metrics.get("loss", 0.0), 6),
                    "val_f1": round(val_metrics.get("f1", 0.0), 6),
                }
            )
        )

        if val_metrics.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "history": history,
                },
                best_path,
            )

    history_path = checkpoint_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(
        json.dumps(
            {
                "best_checkpoint": str(best_path),
                "history": str(history_path),
                "best_val_loss": round(best_val_loss, 6),
                "device": str(device),
            }
        )
    )


if __name__ == "__main__":
    main()
