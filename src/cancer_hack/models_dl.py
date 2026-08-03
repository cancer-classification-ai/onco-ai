"""Neural baselines for cancer-subtype classification.

The set models are permutation invariant at both hierarchy levels:
mutation tokens are pooled within a gene, then mutated genes are pooled within
a sample.  No padding or arbitrary gene ordering is used.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .dataset_dl import GENE_RULE_FEATURE_GROUPS, GENE_RULE_FEATURE_NAMES


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend(
            [
                nn.Linear(width, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def segment_mean(
    values: torch.Tensor, indices: torch.Tensor, count: int
) -> torch.Tensor:
    """Mean rows in ``values`` by integer segment index."""
    output = values.new_zeros((count, values.shape[-1]))
    if values.numel() == 0:
        return output
    output.index_add_(0, indices, values)
    sizes = values.new_zeros(count)
    sizes.index_add_(0, indices, values.new_ones(len(indices)))
    return output / sizes.clamp_min(1).unsqueeze(1)


def segment_max(
    values: torch.Tensor, indices: torch.Tensor, count: int
) -> torch.Tensor:
    """Max rows by segment; empty segments become zeros."""
    output = values.new_full((count, values.shape[-1]), float("-inf"))
    if values.numel():
        expanded = indices.unsqueeze(1).expand_as(values)
        output.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return output.masked_fill(torch.isinf(output), 0.0)


def segment_sum_sqrt(
    values: torch.Tensor, indices: torch.Tensor, count: int
) -> torch.Tensor:
    """Sum rows by segment and divide by ``sqrt(segment size)``."""
    output = values.new_zeros((count, values.shape[-1]))
    if values.numel() == 0:
        return output
    output.index_add_(0, indices, values)
    sizes = values.new_zeros(count)
    sizes.index_add_(0, indices, values.new_ones(len(indices)))
    return output / sizes.clamp_min(1).sqrt().unsqueeze(1)


class TabularMLP(nn.Module):
    """Dense-feature MLP baseline."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = _mlp(
            input_dim, hidden_dims, num_classes, dropout=dropout
        )

    def forward(self, batch: dict[str, object] | torch.Tensor) -> torch.Tensor:
        dense = batch["dense"] if isinstance(batch, dict) else batch
        if dense is None:
            raise ValueError("TabularMLP requires dense features")
        return self.network(dense)


class MutationTokenEncoder(nn.Module):
    """Embed deterministic mutation-token fields and combine them with an MLP."""

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        *,
        embedding_dim: int = 24,
        token_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gene = nn.Embedding(vocab_sizes["gene"], embedding_dim, padding_idx=0)
        self.kind = nn.Embedding(vocab_sizes["kind"], embedding_dim, padding_idx=0)
        self.ref = nn.Embedding(
            vocab_sizes["amino_acid"], embedding_dim, padding_idx=0
        )
        self.alt = nn.Embedding(
            vocab_sizes["amino_acid"], embedding_dim, padding_idx=0
        )
        self.position = nn.Embedding(
            vocab_sizes["position"], embedding_dim, padding_idx=0
        )
        self.signature = nn.Embedding(
            vocab_sizes["signature"], embedding_dim, padding_idx=0
        )
        self.network = _mlp(
            6 * embedding_dim, (token_dim,), token_dim, dropout=dropout
        )

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        parts = (
            self.gene(batch["gene_id"]),
            self.kind(batch["kind_id"]),
            self.ref(batch["ref_id"]),
            self.alt(batch["alt_id"]),
            self.position(batch["position_id"]),
            self.signature(batch["signature_id"]),
        )
        return self.network(torch.cat(parts, dim=-1))


class HierarchicalGeneSetEncoder(nn.Module):
    """Deep Sets encoder over mutation tokens, then over genes."""

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        num_classes: int,
        *,
        embedding_dim: int = 24,
        token_dim: int = 96,
        gene_dim: int = 128,
        sample_dim: int = 256,
        classifier_hidden: Sequence[int] = (256,),
        dropout: float = 0.15,
        gene_rule_features: dict[str, object] | None = None,
        gene_rule_projection_dim: int = 16,
        use_sum_sqrt_pool: bool = False,
        explicit_gene_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.token_encoder = MutationTokenEncoder(
            vocab_sizes,
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            dropout=dropout,
        )
        rule_config = dict(gene_rule_features or {})
        self.rule_enabled = bool(rule_config.get("enabled", False))
        self.log_transform_counts = bool(
            rule_config.get("log_transform_counts", True)
        )
        self.use_sum_sqrt_pool = bool(use_sum_sqrt_pool)
        self.explicit_gene_embedding = bool(explicit_gene_embedding)
        rule_mask = torch.zeros(len(GENE_RULE_FEATURE_NAMES), dtype=torch.float32)
        switches = {
            "use_counts": "counts",
            "use_type_ratios": "type_ratios",
            "use_position_stats": "position_stats",
            "use_duplicate_stats": "duplicate_stats",
            "use_state_flags": "state_flags",
        }
        for switch, group in switches.items():
            if bool(rule_config.get(switch, False)):
                rule_mask[list(GENE_RULE_FEATURE_GROUPS[group])] = 1.0
        if self.rule_enabled and not bool(rule_mask.any()):
            raise ValueError("gene_rule_features is enabled but no feature group is active")
        self.register_buffer("gene_rule_mask", rule_mask, persistent=False)
        self.rule_projection = (
            _mlp(
                len(GENE_RULE_FEATURE_NAMES),
                (gene_rule_projection_dim,),
                gene_rule_projection_dim,
                dropout=dropout,
            )
            if self.rule_enabled
            else None
        )
        pooling_width = token_dim * (3 if self.use_sum_sqrt_pool else 2)
        gene_input_width = pooling_width
        if self.explicit_gene_embedding:
            gene_input_width += embedding_dim
        if self.rule_enabled:
            gene_input_width += gene_rule_projection_dim
        self.gene_encoder = _mlp(
            gene_input_width, (gene_dim,), gene_dim, dropout=dropout
        )
        self.sample_encoder = _mlp(
            gene_dim * 2, (sample_dim,), sample_dim, dropout=dropout
        )
        self.classifier = _mlp(
            sample_dim,
            classifier_hidden,
            num_classes,
            dropout=dropout,
        )
        self.output_dim = sample_dim

    def encode(self, batch: dict[str, object]) -> torch.Tensor:
        token_values = self.token_encoder(batch)
        gene_count = int(batch["gene_to_sample"].numel())
        token_to_gene = batch["token_to_gene"]
        gene_mean = segment_mean(token_values, token_to_gene, gene_count)
        gene_max = segment_max(token_values, token_to_gene, gene_count)
        gene_parts = [gene_max, gene_mean]
        if self.use_sum_sqrt_pool:
            gene_parts.append(
                segment_sum_sqrt(token_values, token_to_gene, gene_count)
            )
        if self.explicit_gene_embedding:
            gene_parts.append(self.token_encoder.gene(batch["gene_ids"]))
        if self.rule_enabled:
            rule_stats = batch["gene_rule_stats"]
            if self.log_transform_counts:
                rule_stats = torch.cat(
                    [torch.log1p(rule_stats[:, :4]), rule_stats[:, 4:]],
                    dim=1,
                )
            masked_stats = rule_stats * self.gene_rule_mask
            gene_parts.append(self.rule_projection(masked_stats))
        gene_values = self.gene_encoder(torch.cat(gene_parts, dim=-1))

        sample_count = int(batch["sample_count"])
        gene_to_sample = batch["gene_to_sample"]
        sample_mean = segment_mean(gene_values, gene_to_sample, sample_count)
        sample_max = segment_max(gene_values, gene_to_sample, sample_count)
        return self.sample_encoder(torch.cat([sample_mean, sample_max], dim=-1))

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        return self.classifier(self.encode(batch))


class HybridSetMLP(nn.Module):
    """Fuse hierarchical mutation-set representation with engineered features."""

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        dense_dim: int,
        num_classes: int,
        *,
        dense_hidden: int = 256,
        fusion_hidden: Sequence[int] = (384, 192),
        dropout: float = 0.15,
        **set_encoder_kwargs,
    ) -> None:
        super().__init__()
        self.set_encoder = HierarchicalGeneSetEncoder(
            vocab_sizes,
            num_classes,
            dropout=dropout,
            **set_encoder_kwargs,
        )
        self.dense_encoder = _mlp(
            dense_dim, (dense_hidden,), dense_hidden, dropout=dropout
        )
        self.classifier = _mlp(
            self.set_encoder.output_dim + dense_hidden,
            fusion_hidden,
            num_classes,
            dropout=dropout,
        )

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        if batch["dense"] is None:
            raise ValueError("HybridSetMLP requires dense features")
        set_values = self.set_encoder.encode(batch)
        dense_values = self.dense_encoder(batch["dense"])
        return self.classifier(torch.cat([set_values, dense_values], dim=-1))


def create_dl_model(
    name: str,
    *,
    num_classes: int,
    dense_dim: int,
    vocab_sizes: dict[str, int] | None,
    model_params: dict[str, object] | None = None,
) -> nn.Module:
    """Model factory shared by the training and smoke-test scripts."""
    params = dict(model_params or {})
    if name == "mlp":
        return TabularMLP(dense_dim, num_classes, **params)
    if vocab_sizes is None:
        raise ValueError(f"{name} requires tokenizer vocabulary sizes")
    if name == "set_encoder":
        return HierarchicalGeneSetEncoder(vocab_sizes, num_classes, **params)
    if name == "hybrid":
        return HybridSetMLP(
            vocab_sizes, dense_dim, num_classes, **params
        )
    raise ValueError(f"unknown DL model: {name}")
