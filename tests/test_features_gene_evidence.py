from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.features_gene_evidence import (  # noqa: E402
    BLOCK_TO_METHOD,
    build_fold_gene_evidence_block,
    build_gene_evidence_fold,
)


def _load_train_gbdt():
    path = PROJECT_ROOT / "scripts/train_gbdt.py"
    spec = importlib.util.spec_from_file_location("train_gbdt_evidence_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy():
    X = sparse.csr_matrix(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    y = np.array(["A", "B"] * 5)
    return X, y


def test_model_agnostic_fold_api_returns_cross_fitted_arrays() -> None:
    X, y = _toy()
    for method in BLOCK_TO_METHOD.values():
        block = build_gene_evidence_fold(
            X_train=X,
            y_train=y,
            X_valid=X[:3],
            X_test=X[3:6],
            method=method,
            classes=["A", "B"],
            inner_n_splits=5,
        )
        assert block.train.shape == (10, 2)
        assert block.valid.shape == (3, 2)
        assert block.test is not None and block.test.shape == (3, 2)
        assert len(block.feature_names) == 2
        assert block.diagnostics["train_encoding"] == "inner_cross_fitted"
        assert np.isfinite(block.train).all()
        assert block.train.dtype == np.float32


def test_gbdt_adapter_places_cross_fit_values_only_on_outer_train_rows() -> None:
    X, y = _toy()
    train_index = np.arange(8)
    names, train_out, test_out, diagnostics = build_fold_gene_evidence_block(
        X,
        X[8:],
        train_index,
        y[train_index],
        block="ebovr",
        classes=["A", "B"],
        inner_n_splits=4,
    )
    assert len(names) == 2
    assert train_out.shape == (10, 2)
    assert test_out.shape == (2, 2)
    assert diagnostics["outer_train_rows"] == 8
    assert diagnostics["train_encoding"] == "inner_cross_fitted"
    assert diagnostics["valid_test_encoding"] == "outer_train_fit_transform_only"


def test_train_gbdt_registers_evidence_as_explicit_fold_only_blocks() -> None:
    train_gbdt = _load_train_gbdt()
    assert train_gbdt.EVIDENCE_BLOCKS == ("ebovr", "ebbnb")
    assert set(train_gbdt.EVIDENCE_BLOCKS) <= set(train_gbdt.FOLD_MATRIX_BLOCKS)
    assert train_gbdt.BLOCK_SOURCES["ebovr"].endswith(
        "gene_mutated_matrix.parquet"
    )
    assert train_gbdt.CONFIGS["r16_base"]["blocks"] == ("domain", "rollup16")
    assert train_gbdt.CONFIGS["r16_ebovr"]["blocks"] == (
        "domain",
        "rollup16",
        "ebovr",
    )
    assert train_gbdt.CONFIGS["r16_ebbnb"]["blocks"] == (
        "domain",
        "rollup16",
        "ebbnb",
    )
    assert train_gbdt.CONFIGS["r16_ebboth"]["blocks"] == (
        "domain",
        "rollup16",
        "ebovr",
        "ebbnb",
    )
    assert train_gbdt.CONFIGS["f4r_ebovr"]["blocks"][-1] == "ebovr"
    assert train_gbdt.CONFIGS["f4r_ebbnb"]["blocks"][-1] == "ebbnb"
    assert train_gbdt.CONFIGS["f4r_ebboth"]["blocks"][-2:] == (
        "ebovr",
        "ebbnb",
    )
    assert not (
        set(train_gbdt.CONFIGS["full_all"]["blocks"])
        & set(train_gbdt.EVIDENCE_BLOCKS)
    )
    assert train_gbdt.CONFIGS["full_all"]["blocks"] == train_gbdt.FULL_ALL_BLOCKS
    assert train_gbdt.CONFIGS["full_all_ebbnb"]["blocks"] == (
        *train_gbdt.FULL_ALL_BLOCKS,
        "ebbnb",
    )
    assert train_gbdt.CONFIGS["full_all_ebovr"]["blocks"] == (
        *train_gbdt.FULL_ALL_BLOCKS,
        "ebovr",
    )


def test_candidate_combinations_are_explicit_only_and_keep_f4r_fixed() -> None:
    train_gbdt = _load_train_gbdt()
    candidates = set(train_gbdt.EXPERIMENTAL_COMBINATION_CONFIGS)
    assert candidates == {
        "f4r_gtype_csig",
        "f4r_gtype_ebbnb",
        "f4r_gtype_ebovr",
        "f4r_compact",
        "full_all_ebbnb",
        "full_all_ebovr",
        "f4r_gtype_csig_ebbnb",
        "f4r_compact_ebbnb",
    }
    common = {"domain", "rollup16", "enc3"}
    assert all(common <= set(train_gbdt.CONFIGS[name]["blocks"]) for name in candidates)


def test_evidence_slug_changes_with_every_training_parameter() -> None:
    train_gbdt = _load_train_gbdt()
    base = {"strength": 20.0, "prior_weight": 0.0, "inner_n_splits": 5}
    slugs = {
        train_gbdt._evidence_slug(base),
        train_gbdt._evidence_slug({**base, "strength": 10.0}),
        train_gbdt._evidence_slug({**base, "prior_weight": 0.25}),
        train_gbdt._evidence_slug({**base, "inner_n_splits": 4}),
    }
    assert len(slugs) == 4
