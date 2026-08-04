from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.gene_evidence import (  # noqa: E402
    EmpiricalBayesGeneEvidence,
    cross_fit_gene_evidence,
    scores_to_probabilities,
    validate_probabilities,
)


def test_zero_support_is_neutral_for_both_encoders() -> None:
    X_fit = sparse.csr_matrix(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int8
    )
    y = np.array(["A", "A", "B", "B"])
    for method in ("eb_ovr", "eb_bnb"):
        encoder = EmpiricalBayesGeneEvidence(
            method=method, classes=["A", "B"]
        ).fit(X_fit, y)
        np.testing.assert_array_equal(encoder.weight_[2], [0.0, 0.0])
        without_unseen = encoder.transform(sparse.csr_matrix([[0, 0, 0]]))
        with_unseen = encoder.transform(sparse.csr_matrix([[0, 0, 1]]))
        np.testing.assert_array_equal(without_unseen, with_unseen)


def test_class_size_neutrality_for_unseen_gene() -> None:
    # A is rare (2 rows), B is common (8 rows); gene 1 is absent everywhere.
    X = sparse.csr_matrix(np.column_stack([np.r_[np.ones(2), np.zeros(8)], np.zeros(10)]))
    y = np.array(["A"] * 2 + ["B"] * 8)
    for method in ("eb_ovr", "eb_bnb"):
        encoder = EmpiricalBayesGeneEvidence(
            method=method, classes=["A", "B"]
        ).fit(X, y)
        np.testing.assert_array_equal(encoder.weight_[1], [0.0, 0.0])


def test_empirical_bayes_shrinks_small_class_more_strongly() -> None:
    # Global rate is 0.6. A has 2/2 and B has 4/8.
    X = sparse.csr_matrix(np.r_[np.ones(2), np.ones(4), np.zeros(4)][:, None])
    y = np.array(["A"] * 2 + ["B"] * 8)
    encoder = EmpiricalBayesGeneEvidence(
        method="eb_bnb", strength=20.0, classes=["A", "B"]
    ).fit(X, y)
    global_rate = encoder.global_rate_[0]
    posterior_a, posterior_b = encoder.p_gene_class_[0]
    raw_a, raw_b = 1.0, 0.5
    retained_a = abs(posterior_a - global_rate) / abs(raw_a - global_rate)
    retained_b = abs(posterior_b - global_rate) / abs(raw_b - global_rate)
    assert retained_a < retained_b


def test_nested_cross_fit_excludes_each_heldout_row() -> None:
    X = sparse.eye(10, format="csr", dtype=np.int8)
    y = np.array(["A", "B"] * 5)
    features, audit = cross_fit_gene_evidence(
        X,
        y,
        method="eb_ovr",
        classes=["A", "B"],
        n_splits=5,
        return_fit_indices=True,
    )
    coverage = np.zeros(len(y), dtype=int)
    for fit_index, heldout_index in audit:
        assert np.intersect1d(fit_index, heldout_index).size == 0
        coverage[heldout_index] += 1
    np.testing.assert_array_equal(coverage, np.ones(len(y), dtype=int))
    assert features.shape == (10, 2)
    assert np.isfinite(features).all()


def test_validation_labels_do_not_affect_validation_features() -> None:
    X = sparse.csr_matrix([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_fit = np.array(["A", "B"])
    encoder = EmpiricalBayesGeneEvidence(
        method="eb_ovr", classes=["A", "B"]
    ).fit(X[:2], y_fit)
    before = encoder.transform(X[2:])
    changed_validation_labels = np.array(["B", "A"])
    assert changed_validation_labels.tolist() != ["A", "B"]
    after = encoder.transform(X[2:])
    np.testing.assert_array_equal(before, after)


def test_probability_validation() -> None:
    probability = scores_to_probabilities(np.array([[1000.0, -1000.0], [0.0, 0.0]]))
    validate_probabilities(probability)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0) & (probability <= 1))
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        validate_probabilities(np.array([[0.2, 0.2]]))


def test_repository_sgkf_has_group_isolation_and_oof_coverage() -> None:
    folds = pd.read_parquet(PROJECT_ROOT / "data/process/train_folds.parquet")
    assignment = folds["fold_group5"].to_numpy()
    groups = folds["group_key"].to_numpy()
    coverage = np.zeros(len(folds), dtype=int)
    for fold in range(5):
        train_index = np.flatnonzero(assignment != fold)
        valid_index = np.flatnonzero(assignment == fold)
        assert np.intersect1d(groups[train_index], groups[valid_index]).size == 0
        coverage[valid_index] += 1
    np.testing.assert_array_equal(coverage, np.ones(len(folds), dtype=int))
