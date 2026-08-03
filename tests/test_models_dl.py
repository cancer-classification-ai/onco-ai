from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from cancer_hack.dataset_dl import (
    MutationSampleDataset,
    MutationTokenizer,
    collate_mutation_samples,
    tokenize_frame,
)
from cancer_hack.models_dl import create_dl_model


def _batch():
    frame = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "TP53": ["R273H Q369*", "WT"],
            "KRAS": ["G12D", "K16fs"],
        }
    )
    tokenizer = MutationTokenizer(["TP53", "KRAS"], signature_buckets=32)
    samples = tokenize_frame(frame, tokenizer)
    dataset = MutationSampleDataset(
        samples,
        frame["ID"],
        labels=[0, 1],
        dense=np.ones((2, 5), dtype=np.float32),
    )
    return tokenizer, samples, collate_mutation_samples([dataset[0], dataset[1]])


def test_all_dl_models_return_class_logits() -> None:
    tokenizer, _, batch = _batch()
    for name in ("mlp", "set_encoder", "hybrid"):
        params = {"hidden_dims": [8], "dropout": 0.0} if name == "mlp" else {
            "embedding_dim": 4,
            "token_dim": 8,
            "gene_dim": 8,
            "sample_dim": 8,
            "classifier_hidden": [8],
            "dropout": 0.0,
        }
        if name == "hybrid":
            params.update(dense_hidden=8, fusion_hidden=[8])
        model = create_dl_model(
            name,
            num_classes=3,
            dense_dim=5,
            vocab_sizes=tokenizer.vocab_sizes,
            model_params=params,
        )
        assert model(batch).shape == (2, 3)


def test_set_encoder_is_permutation_invariant() -> None:
    tokenizer, samples, _ = _batch()
    permuted = copy.deepcopy(samples)
    for sample in permuted:
        sample.reverse()
        for gene_tokens in sample:
            gene_tokens.reverse()
    original_dataset = MutationSampleDataset(samples, ["a", "b"])
    permuted_dataset = MutationSampleDataset(permuted, ["a", "b"])
    original = collate_mutation_samples([original_dataset[0], original_dataset[1]])
    changed = collate_mutation_samples([permuted_dataset[0], permuted_dataset[1]])
    model = create_dl_model(
        "set_encoder",
        num_classes=3,
        dense_dim=0,
        vocab_sizes=tokenizer.vocab_sizes,
        model_params={
            "embedding_dim": 4,
            "token_dim": 8,
            "gene_dim": 8,
            "sample_dim": 8,
            "classifier_hidden": [8],
            "dropout": 0.0,
        },
    ).eval()

    with torch.no_grad():
        assert torch.allclose(model(original), model(changed), atol=1e-6)
