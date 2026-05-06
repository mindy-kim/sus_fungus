import math

import torch
from torch import nn

from .vrp_flow_data import GraphBatch


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [input_dim]
        for _ in range(max(0, depth - 1)):
            dims.append(hidden_dim)
        dims.append(output_dim)

        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proj = MLP(embedding_dim, embedding_dim, embedding_dim, depth=2)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            raise ValueError("Time embedding dimension must be at least 2.")

        step = max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10000.0) / step)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0) * (2.0 * math.pi)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        if embedding.size(1) < self.embedding_dim:
            padding = self.embedding_dim - embedding.size(1)
            embedding = torch.nn.functional.pad(embedding, (0, padding))
        return self.proj(embedding)


class ResidualEdgeGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        edge_input_dim = (hidden_dim * 4) + time_dim
        node_input_dim = (hidden_dim * 2) + time_dim
        edge_update_dim = (hidden_dim * 4) + time_dim

        self.edge_message = MLP(edge_input_dim, hidden_dim, hidden_dim, depth=2, dropout=dropout)
        self.node_update = MLP(node_input_dim, hidden_dim, hidden_dim, depth=2, dropout=dropout)
        self.edge_update = MLP(edge_update_dim, hidden_dim, hidden_dim, depth=2, dropout=dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        edge_index: torch.Tensor,
        node_time: torch.Tensor,
        edge_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        edge_message_input = torch.cat(
            (
                edge_state,
                node_state[source],
                node_state[target],
                node_state[source] * node_state[target],
                edge_time,
            ),
            dim=1,
        )
        messages = self.edge_message(edge_message_input)

        aggregated = node_state.new_zeros(node_state.size(0), messages.size(1))
        aggregated.index_add_(0, source, messages)
        aggregated.index_add_(0, target, messages)

        node_update = self.node_update(torch.cat((node_state, aggregated, node_time), dim=1))
        node_state = self.node_norm(node_state + node_update)

        edge_update_input = torch.cat(
            (
                edge_state,
                node_state[source],
                node_state[target],
                messages,
                edge_time,
            ),
            dim=1,
        )
        edge_update = self.edge_update(edge_update_input)
        edge_state = self.edge_norm(edge_state + edge_update)
        return node_state, edge_state


class EdgeFlowMatchingModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.node_encoder = MLP(node_feature_dim, hidden_dim, hidden_dim, depth=2, dropout=dropout)
        self.edge_encoder = MLP(
            edge_feature_dim + 1 + time_dim,
            hidden_dim,
            hidden_dim,
            depth=2,
            dropout=dropout,
        )
        self.time_encoder = SinusoidalTimeEmbedding(time_dim)
        self.blocks = nn.ModuleList(
            ResidualEdgeGraphBlock(hidden_dim, time_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.flow_head = MLP(
            (hidden_dim * 3) + time_dim,
            hidden_dim,
            1,
            depth=2,
            dropout=dropout,
        )
        self.classifier_head = MLP(hidden_dim * 3, hidden_dim, 1, depth=2, dropout=dropout)

    def forward(
        self,
        batch: GraphBatch,
        edge_state: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timesteps.ndim != 1 or timesteps.size(0) != batch.num_graphs:
            raise ValueError("timesteps must have shape [num_graphs].")
        if edge_state.ndim != 1 or edge_state.size(0) != batch.edge_features.size(0):
            raise ValueError("edge_state must have shape [num_edges].")

        node_state = self.node_encoder(batch.node_features)
        time_state = self.time_encoder(timesteps)
        node_time = time_state[batch.node_batch]
        edge_time = time_state[batch.edge_batch]

        edge_state_hidden = self.edge_encoder(
            torch.cat((batch.edge_features, edge_state.unsqueeze(1), edge_time), dim=1)
        )

        for block in self.blocks:
            node_state, edge_state_hidden = block(
                node_state=node_state,
                edge_state=edge_state_hidden,
                edge_index=batch.edge_index,
                node_time=node_time,
                edge_time=edge_time,
            )

        source, target = batch.edge_index
        flow_input = torch.cat(
            (
                edge_state_hidden,
                node_state[source],
                node_state[target],
                edge_time,
            ),
            dim=1,
        )
        classifier_input = torch.cat(
            (
                edge_state_hidden,
                node_state[source],
                node_state[target],
            ),
            dim=1,
        )
        predicted_flow = self.flow_head(flow_input).squeeze(1)
        edge_logits = self.classifier_head(classifier_input).squeeze(1)
        return predicted_flow, edge_logits
