from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_train_dl_module():
    path = Path(__file__).resolve().parents[1] / "scripts/train_dl.py"
    spec = importlib.util.spec_from_file_location("train_dl_full_features", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_feature_configs_enable_frequency_and_both_latent_methods() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "mlp_full_features.yaml",
        "hybrid_set_mlp_v2_full_features.yaml",
    ):
        with (root / "configs" / name).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        dense = config["dense_features"]
        assert dense["domain_feature_set"] == "all"
        assert dense["frequency_blocks"] == ["freq21", "aatrans9"]
        assert dense["latent"]["enabled"] is True
        assert dense["latent"]["methods"] == ["svd", "nmf"]


def test_fold_engineered_features_append_frequency_and_latent() -> None:
    train_dl = _load_train_dl_module()
    raw_train = pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "SUBCLASS": ["X", "X", "Y", "Y"],
            "G1": ["R1H", "R1H", "WT", "WT"],
            "G2": ["WT", "WT", "Q2*", "Q2*"],
            "G3": ["A3V", "WT", "A3V", "WT"],
        }
    )
    raw_test = pd.DataFrame(
        {
            "ID": ["e"],
            "G1": ["R1H"],
            "G2": ["Q2*"],
            "G3": ["WT"],
        }
    )
    latent_train = np.asarray(
        [[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.float32
    )
    latent_test = np.asarray([[1, 1, 0]], dtype=np.float32)
    train, test, names = train_dl.append_fold_engineered_features(
        np.zeros((4, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.asarray([0, 1, 2, 3]),
        labels=np.asarray(["X", "X", "Y", "Y"]),
        frequency_blocks=["freq21", "aatrans9"],
        raw_train=raw_train,
        raw_test=raw_test,
        raw_gene_columns=["G1", "G2", "G3"],
        latent_source=(["G1", "G2", "G3"], latent_train, latent_test),
        latent_config={
            "methods": ["svd", "nmf"],
            "n_components": 2,
            "row_norm": "l2",
            "value": "proj",
            "mode": "mutated",
            "gene_weight": "none",
            "min_gene_support": 1,
            "random_state": 0,
        },
    )

    expected_added = (
        len(train_dl.FREQUENCY_RARITY_FEATURE_COLUMNS)
        + len(train_dl.AA_TRANSITION_FEATURE_COLUMNS)
        + 4
    )
    assert train.shape == (4, 2 + expected_added)
    assert test.shape == (1, 2 + expected_added)
    assert len(names) == expected_added
    assert any(name.startswith("freq21__") for name in names)
    assert any(name.startswith("aatrans9__") for name in names)
    assert any(name.startswith("lat__svd__") for name in names)
    assert any(name.startswith("lat__nmf__") for name in names)
    assert np.isfinite(train).all()
    assert np.isfinite(test).all()
