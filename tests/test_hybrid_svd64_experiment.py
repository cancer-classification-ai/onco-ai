from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from cancer_hack.features_latent import (
    _binarize,
    build_fold_latent_block,
    fit_latent_basis,
    transform_latent_basis,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "configs/hybrid_set_mlp.yaml"
CANDIDATE_CONFIG = ROOT / "configs/hybrid_set_mlp_svd64.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_train_dl():
    path = ROOT / "scripts/train_dl.py"
    spec = importlib.util.spec_from_file_location("train_dl_svd64_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_compare_dl():
    path = ROOT / "scripts/compare_dl_runs.py"
    spec = importlib.util.spec_from_file_location("compare_dl_svd64_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix() -> np.ndarray:
    rng = np.random.default_rng(12)
    matrix = (rng.random((90, 70)) < 0.22).astype(np.float32)
    matrix[::3, ::7] = 2.0
    return matrix


def _kwargs() -> dict:
    return {
        "gene_names": [f"G{i}" for i in range(70)],
        "method": "svd",
        "n_components": 64,
        "mode": "mutated",
        "row_norm": "l2",
        "gene_weight": "none",
        "min_gene_support": 5,
        "random_state": 0,
    }


def test_mutated_binary_conversion_maps_enc3_presence() -> None:
    encoded = np.asarray([[0, 1, 2]], dtype=np.float32)
    assert _binarize(encoded, "mutated").toarray().tolist() == [[0.0, 1.0, 1.0]]
    assert _binarize(encoded, "functional").toarray().tolist() == [[0.0, 0.0, 1.0]]


def test_fold_isolation_preserves_basis_support_and_fit_transform() -> None:
    matrix = _matrix()
    train_index = np.arange(72)
    changed = matrix.copy()
    changed[72:] = 2.0 - changed[72:]

    basis_a = fit_latent_basis(matrix, train_index, **_kwargs())
    basis_b = fit_latent_basis(changed, train_index, **_kwargs())

    assert np.array_equal(basis_a.gene_index, basis_b.gene_index)
    assert np.array_equal(basis_a.components, basis_b.components)
    assert np.array_equal(
        transform_latent_basis(matrix[train_index], basis_a),
        transform_latent_basis(changed[train_index], basis_b),
    )


def test_test_isolation_changes_only_test_transform() -> None:
    matrix = _matrix()
    train_index = np.arange(72)
    labels = np.asarray(["A", "B"] * 36)
    test_a = matrix[:8].copy()
    test_b = np.zeros_like(test_a)

    _, train_a, transformed_a, _, basis_a = build_fold_latent_block(
        matrix, test_a, train_index, labels, **_kwargs()
    )
    _, train_b, transformed_b, _, basis_b = build_fold_latent_block(
        matrix, test_b, train_index, labels, **_kwargs()
    )

    assert np.array_equal(basis_a.gene_index, basis_b.gene_index)
    assert np.array_equal(basis_a.components, basis_b.components)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(train_a[72:], train_b[72:])
    assert not np.array_equal(transformed_a, transformed_b)


def test_svd64_is_deterministic_and_has_expected_shapes() -> None:
    matrix = _matrix()
    train_index = np.arange(72)
    basis_a = fit_latent_basis(matrix, train_index, **_kwargs())
    basis_b = fit_latent_basis(matrix, train_index, **_kwargs())
    train_a = transform_latent_basis(matrix, basis_a)
    train_b = transform_latent_basis(matrix, basis_b)
    test = transform_latent_basis(matrix[:11], basis_a)

    assert basis_a.n_components == 64
    assert np.allclose(basis_a.components, basis_b.components, atol=1e-7)
    assert np.allclose(train_a, train_b, atol=1e-7)
    assert train_a.shape == (90, 64)
    assert test.shape == (11, 64)
    assert 591 + train_a.shape[1] == 655


def test_candidate_config_diff_is_latent_svd64_only() -> None:
    baseline = _load_yaml(BASELINE_CONFIG)
    candidate = _load_yaml(CANDIDATE_CONFIG)
    latent = candidate["dense_features"]["latent"]

    candidate_without_dense = copy.deepcopy(candidate)
    candidate_without_dense.pop("dense_features")
    assert candidate_without_dense == baseline
    assert latent == {
        "enabled": True,
        "methods": ["svd"],
        "n_components": 64,
        "mode": "mutated",
        "row_norm": "l2",
        "gene_weight": "none",
        "min_gene_support": 5,
        "random_state": 0,
        "value": "proj",
    }
    assert "nmf" not in latent["methods"]


def test_train_dl_records_fold_local_latent_metadata() -> None:
    train_dl = _load_train_dl()
    matrix = _matrix()
    metadata: list[dict[str, object]] = []
    baseline_train = np.zeros((90, 591), dtype=np.float32)
    baseline_test = np.zeros((11, 591), dtype=np.float32)

    train, test, names = train_dl.append_fold_engineered_features(
        baseline_train,
        baseline_test,
        np.arange(72),
        labels=np.asarray(["A", "B"] * 45),
        frequency_blocks=[],
        raw_train=None,
        raw_test=None,
        raw_gene_columns=None,
        latent_source=(_kwargs()["gene_names"], matrix, matrix[:11]),
        latent_config={
            key: value for key, value in _kwargs().items() if key != "gene_names"
        },
        latent_metadata=metadata,
    )

    assert train.shape == (90, 655)
    assert test.shape == (11, 655)
    assert len(names) == 64
    assert metadata == [
        {
            "method": "svd",
            "dimension": 64,
            "fit_row_count": 72,
            "support_gene_count": 70,
        }
    ]


def test_compare_dl_runs_applies_adoption_rule_and_per_class_delta() -> None:
    compare_dl = _load_compare_dl()
    baseline = compare_dl.baseline_fallback()
    baseline["per_class_f1"] = {"A": 0.2, "B": 0.4}
    candidate = {
        "oof_macro_f1": compare_dl.BASELINE_OOF_MACRO_F1 + 0.006,
        "oof_macro_f1_singleton": compare_dl.BASELINE_SINGLETON_F1 - 0.002,
        "oof_accuracy": compare_dl.BASELINE_ACCURACY + 0.001,
        "fold_macro_f1": [
            score + delta
            for score, delta in zip(
                compare_dl.BASELINE_FOLDS, [0.01, 0.01, 0.01, -0.01, -0.01]
            )
        ],
        "per_class_f1": {"A": 0.3, "B": 0.35},
    }

    result = compare_dl.compare(baseline, candidate)

    assert result["verdict"] == "ADOPT_CANDIDATE"
    assert result["improved_folds"] == 3
    assert result["worsened_folds"] == 2
    assert result["per_class_f1_delta"]["A"] == pytest.approx(0.1)
