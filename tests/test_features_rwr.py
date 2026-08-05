"""Leakage and shape contracts for fold-local unsupervised RWR256."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np

from cancer_hack.features_rwr import (
    build_fold_rwr_block,
    fit_rwr_basis,
    fit_rwr_graph,
    propagate_rwr,
    transform_rwr_basis,
)

GENES = ["GA", "GB", "GC", "GD", "GE", "GF", "GG", "GH"]
MATRIX = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        # valid-only GG/GH co-mutation must not enter graph or SVD fit.
        [1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
    ],
    dtype=np.float32,
)
TRAIN_INDEX = np.arange(8)
KWARGS = dict(
    gene_names=GENES,
    n_components=3,
    min_gene_support=1,
    min_edge_support=1,
    topk_neighbors=2,
    random_state=0,
)


def test_rwr_fit_signatures_accept_neither_test_nor_labels():
    for function in (fit_rwr_graph, fit_rwr_basis):
        parameters = inspect.signature(function).parameters
        assert not [name for name in parameters if "test" in name.lower()]
        assert not [
            name
            for name in parameters
            if name in {"y", "labels", "y_train", "y_train_fold"}
        ]


def test_rwr_basis_ignores_outer_valid_row_content():
    disturbed = MATRIX.copy()
    disturbed[8:] = 1.0
    first = fit_rwr_basis(MATRIX, TRAIN_INDEX, **KWARGS)
    second = fit_rwr_basis(disturbed, TRAIN_INDEX, **KWARGS)

    assert np.array_equal(
        first.graph.transition.toarray(), second.graph.transition.toarray()
    )
    assert np.allclose(first.components, second.components)


def test_valid_only_comutation_is_absent_from_graph():
    graph = fit_rwr_graph(
        MATRIX,
        TRAIN_INDEX,
        gene_names=GENES,
        min_gene_support=1,
        min_edge_support=1,
        topk_neighbors=2,
    )
    gg, gh = GENES.index("GG"), GENES.index("GH")
    assert graph.transition[gg, gh] == 0
    assert graph.transition[gh, gg] == 0


def test_transition_rows_preserve_probability_mass():
    graph = fit_rwr_graph(
        MATRIX,
        TRAIN_INDEX,
        gene_names=GENES,
        min_gene_support=1,
        min_edge_support=1,
        topk_neighbors=2,
    )
    assert np.allclose(np.asarray(graph.transition.sum(axis=1)).ravel(), 1.0)

    propagated = propagate_rwr(MATRIX, graph)
    nonempty = MATRIX.sum(axis=1) > 0
    assert np.allclose(propagated[nonempty].sum(axis=1), 1.0, atol=1e-5)


def test_rwr_transform_is_row_independent_and_burden_normalized():
    basis = fit_rwr_basis(MATRIX, TRAIN_INDEX, **KWARGS)
    full = transform_rwr_basis(MATRIX, basis)
    for row in (0, 5, 10):
        single = transform_rwr_basis(MATRIX[[row]], basis)
        assert np.allclose(single[0], full[row], atol=1e-6)
    norms = np.linalg.norm(full, axis=1)
    # A validation row made only of fold-train-unseen genes may project to zero.
    # Every representable row is L2 normalized and no row may exceed unit norm.
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)
    assert (norms <= 1.0 + 1e-5).all()


def test_zero_mutation_row_stays_zero_and_finite():
    basis = fit_rwr_basis(MATRIX, TRAIN_INDEX, **KWARGS)
    zero = np.zeros((1, MATRIX.shape[1]), dtype=np.float32)
    output = transform_rwr_basis(zero, basis)
    assert np.isfinite(output).all()
    assert np.array_equal(output, np.zeros_like(output))


def test_fold_builder_returns_all_rows_and_requested_width():
    names, train_out, test_out, diagnostics, _ = build_fold_rwr_block(
        MATRIX,
        MATRIX[:3],
        TRAIN_INDEX,
        **KWARGS,
    )
    assert len(names) == 3
    assert train_out.shape == (len(MATRIX), 3)
    assert test_out.shape == (3, 3)
    assert diagnostics["n_genes"] == len(GENES)
    assert diagnostics["n_edges"] > 0


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "train_gbdt_rwr_test", PROJECT_ROOT / "scripts/train_gbdt.py"
)
TRAIN_GBDT = importlib.util.module_from_spec(_SPEC)
sys.modules["train_gbdt_rwr_test"] = TRAIN_GBDT
_SPEC.loader.exec_module(TRAIN_GBDT)


def test_dense_configs_differ_only_by_rwr256_and_are_model_agnostic():
    base = TRAIN_GBDT.CONFIGS["dense_base"]
    rwr = TRAIN_GBDT.CONFIGS["dense_rwr256"]
    assert base == {
        "blocks": ("domain", "rollup16", "gecr"),
        "weight": "balanced",
        "desc": "RWR dense 대조군",
    }
    assert rwr["blocks"] == (*base["blocks"], "rwr256")
    assert rwr["weight"] == base["weight"] == "balanced"
    assert all("cat" not in name and "xgb" not in name for name in TRAIN_GBDT.RWR_DENSE_CONFIGS)


def test_rwr_block_is_fold_only_and_uses_enc3_source():
    assert "rwr256" in TRAIN_GBDT.RWR_BLOCKS
    assert "rwr256" in TRAIN_GBDT.FOLD_MATRIX_BLOCKS
    assert TRAIN_GBDT.BLOCK_SOURCES["rwr256"] == TRAIN_GBDT.BLOCK_SOURCES["enc3"]
    assert TRAIN_GBDT.block_cache_key("rwr256") == TRAIN_GBDT.block_cache_key("enc3")


def test_catboost_baseline_params_are_untuned_defaults():
    assert TRAIN_GBDT.MODEL_PARAMS["catboost"] == {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
    }
    assert "cbopt10" not in TRAIN_GBDT.CONFIGS


def test_rwr_slug_changes_with_every_implementation_axis():
    base = {
        "n_components": 256,
        "restart": 0.5,
        "max_iter": 20,
        "tol": 1e-6,
        "min_gene_support": 5,
        "min_edge_support": 2,
        "topk_neighbors": 32,
        "random_state": 0,
    }
    seen = {TRAIN_GBDT._rwr_slug(base)}
    for key, value in (
        ("n_components", 128),
        ("restart", 0.3),
        ("max_iter", 30),
        ("tol", 1e-5),
        ("min_gene_support", 10),
        ("min_edge_support", 3),
        ("topk_neighbors", 16),
        ("random_state", 1),
    ):
        slug = TRAIN_GBDT._rwr_slug({**base, key: value})
        assert slug not in seen
        seen.add(slug)
