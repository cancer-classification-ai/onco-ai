#!/usr/bin/env python
"""Small synthetic forward/backward smoke test for every DL architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.dataset_dl import (  # noqa: E402
    MutationSampleDataset,
    MutationTokenizer,
    collate_mutation_samples,
    tokenize_frame,
)
from cancer_hack.models_dl import create_dl_model  # noqa: E402


def main() -> None:
    frame = pd.DataFrame(
        {
            "ID": ["s1", "s2", "s3", "s4"],
            "SUBCLASS": ["A", "B", "C", "A"],
            "TP53": ["R273H", "Q369*", "WT", "R175H R175H"],
            "KRAS": ["WT", "G12D", "K16fs", "WT"],
            "EGFR": ["E746_A750del", "WT", "WT", "L858R"],
        }
    )
    tokenizer = MutationTokenizer(["TP53", "KRAS", "EGFR"], signature_buckets=64)
    samples = tokenize_frame(frame, tokenizer)
    dense = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
    dataset = MutationSampleDataset(
        samples,
        frame["ID"],
        labels=[0, 1, 2, 0],
        dense=dense,
    )
    batch = collate_mutation_samples([dataset[index] for index in range(len(dataset))])
    for name in ("mlp", "set_encoder", "hybrid"):
        model = create_dl_model(
            name,
            num_classes=3,
            dense_dim=dense.shape[1],
            vocab_sizes=tokenizer.vocab_sizes,
            model_params={
                "hidden_dims": [16],
                "dropout": 0.0,
            }
            if name == "mlp"
            else (
                {
                    "embedding_dim": 4,
                    "token_dim": 8,
                    "gene_dim": 8,
                    "sample_dim": 12,
                    "classifier_hidden": [8],
                    "dropout": 0.0,
                }
                if name == "set_encoder"
                else {
                    "embedding_dim": 4,
                    "token_dim": 8,
                    "gene_dim": 8,
                    "sample_dim": 12,
                    "classifier_hidden": [8],
                    "dense_hidden": 8,
                    "fusion_hidden": [12],
                    "dropout": 0.0,
                }
            ),
        )
        logits = model(batch)
        if logits.shape != (4, 3) or not torch.isfinite(logits).all():
            raise RuntimeError(f"{name}: invalid logits {logits.shape}")
        torch.nn.functional.cross_entropy(logits, batch["labels"]).backward()
        print(f"{name:12s} OK  logits={tuple(logits.shape)}")
    print("DL smoke test passed")


if __name__ == "__main__":
    main()
