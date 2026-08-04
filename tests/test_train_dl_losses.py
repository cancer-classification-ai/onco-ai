from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F


def _load_train_dl_module():
    path = Path(__file__).resolve().parents[1] / "scripts/train_dl.py"
    spec = importlib.util.spec_from_file_location("train_dl_losses", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_class_weight_resolves_to_weighted_ce() -> None:
    train_dl = _load_train_dl_module()
    assert train_dl.resolve_loss_config({"class_weight": True}) == {
        "name": "cross_entropy",
        "class_weight": "balanced",
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
    }


def test_sqrt_balanced_weights_are_square_root_of_balanced() -> None:
    train_dl = _load_train_dl_module()
    labels = np.asarray([0, 0, 0, 1])
    balanced = train_dl.class_weights(labels, 2, mode="balanced")
    square_root = train_dl.class_weights(labels, 2, mode="sqrt_balanced")
    assert balanced is not None and square_root is not None
    assert torch.allclose(square_root, balanced.sqrt())


def test_focal_gamma_zero_matches_unweighted_cross_entropy() -> None:
    train_dl = _load_train_dl_module()
    logits = torch.tensor([[2.0, 0.0], [0.2, 0.8]], dtype=torch.float32)
    targets = torch.tensor([0, 1])
    focal = train_dl.FocalCrossEntropy(
        gamma=0.0,
        weight=None,
        label_smoothing=0.0,
    )
    assert focal(logits, targets) == pytest.approx(
        float(F.cross_entropy(logits, targets))
    )


@pytest.mark.parametrize(
    ("name", "weight", "smoothing", "gamma"),
    [
        ("cross_entropy", "balanced", 0.02, 0.0),
        ("focal", "none", 0.0, 1.0),
        ("focal", "sqrt_balanced", 0.02, 2.0),
    ],
)
def test_supported_loss_configs_resolve(name, weight, smoothing, gamma) -> None:
    train_dl = _load_train_dl_module()
    resolved = train_dl.resolve_loss_config(
        {
            "loss": {
                "name": name,
                "class_weight": weight,
                "label_smoothing": smoothing,
                "focal_gamma": gamma,
            }
        }
    )
    assert resolved == {
        "name": name,
        "class_weight": weight,
        "label_smoothing": smoothing,
        "focal_gamma": gamma,
    }
