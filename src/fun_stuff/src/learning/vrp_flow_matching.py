from collections.abc import Iterable

import torch
import torch.nn.functional as F

from .vrp_flow_data import GraphBatch
from .vrp_flow_model import EdgeFlowMatchingModel


def _binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= 0.5
    truth = labels >= 0.5

    tp = (predictions & truth).sum().item()
    fp = (predictions & ~truth).sum().item()
    fn = (~predictions & truth).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)
    accuracy = (predictions == truth).to(torch.float32).mean().item()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_batch_losses(
    model: EdgeFlowMatchingModel,
    batch: GraphBatch,
    noise_scale: float = 1.0,
    classification_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    if batch.edge_labels is None:
        raise ValueError("Flow matching requires labeled edge targets.")

    device = batch.node_features.device
    timesteps = torch.rand(batch.num_graphs, device=device)
    edge_timesteps = timesteps[batch.edge_batch]

    base_state = torch.randn_like(batch.edge_labels, device=device) * noise_scale
    target_edges = batch.edge_labels
    intermediate_state = ((1.0 - edge_timesteps) * base_state) + (edge_timesteps * target_edges)
    target_velocity = target_edges - base_state

    predicted_velocity, edge_logits = model(batch, intermediate_state, timesteps)

    flow_loss = F.mse_loss(predicted_velocity, target_velocity)
    positive_count = torch.clamp(target_edges.sum(), min=1.0)
    negative_count = torch.clamp(target_edges.numel() - target_edges.sum(), min=1.0)
    pos_weight = torch.clamp(negative_count / positive_count, min=1.0)
    classification_loss = F.binary_cross_entropy_with_logits(
        edge_logits,
        target_edges,
        pos_weight=pos_weight,
    )

    loss = flow_loss + (classification_weight * classification_loss)
    metrics = {
        "loss": float(loss.detach().item()),
        "flow_loss": float(flow_loss.detach().item()),
        "classification_loss": float(classification_loss.detach().item()),
    }
    metrics.update(_binary_metrics(edge_logits.detach(), target_edges.detach()))
    return loss, metrics


@torch.no_grad()
def sample_edge_probabilities(
    model: EdgeFlowMatchingModel,
    batch: GraphBatch,
    steps: int = 32,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive.")

    device = batch.node_features.device
    edge_state = torch.randn(
        batch.edge_features.size(0),
        device=device,
        generator=generator,
    ) * noise_scale

    dt = 1.0 / steps
    for step_idx in range(steps):
        midpoint = (step_idx + 0.5) * dt
        timesteps = torch.full((batch.num_graphs,), midpoint, device=device)
        predicted_velocity, _ = model(batch, edge_state, timesteps)
        edge_state = edge_state + (dt * predicted_velocity)

    final_timesteps = torch.ones(batch.num_graphs, device=device)
    _, final_logits = model(batch, edge_state, final_timesteps)
    return torch.sigmoid(final_logits)


def summarize_metrics(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in keys
    }
