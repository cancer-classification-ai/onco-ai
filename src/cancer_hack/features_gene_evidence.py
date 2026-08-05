"""Model-agnostic supervised gene evidence feature blocks.

`ebovr` and `ebbnb` both map a binary sample-by-gene matrix to one score per
canonical class.  Unlike ordinary fold-local transforms, the training rows
must be produced by inner cross-fitting because the encoder reads labels.

The public :func:`build_gene_evidence_fold` API is independent of XGBoost,
LightGBM, neural networks, and the repository's GBDT driver.  The
:func:`build_fold_gene_evidence_block` adapter matches the full-train/full-test
array contract used by ``scripts/train_gbdt.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .gene_evidence import (
    EVIDENCE_METHODS,
    EmpiricalBayesGeneEvidence,
    cross_fit_gene_evidence,
)

BLOCK_TO_METHOD = {
    "ebovr": "eb_ovr",
    "ebbnb": "eb_bnb",
}


@dataclass(frozen=True)
class GeneEvidenceFold:
    """Leakage-safe feature arrays for one caller-defined outer fold."""

    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray | None
    feature_names: tuple[str, ...]
    diagnostics: dict


def evidence_feature_names(method: str, classes: Sequence[str]) -> tuple[str, ...]:
    if method not in EVIDENCE_METHODS:
        raise ValueError(f"unknown evidence method: {method}")
    slug = "ebovr" if method == "eb_ovr" else "ebbnb"
    return tuple(f"evidence__{slug}__{label}" for label in classes)


def build_gene_evidence_fold(
    *,
    X_train: sparse.spmatrix | np.ndarray,
    y_train: Sequence[str],
    X_valid: sparse.spmatrix | np.ndarray,
    X_test: sparse.spmatrix | np.ndarray | None = None,
    method: str,
    classes: Sequence[str],
    strength: float = 20.0,
    prior_weight: float = 0.0,
    inner_n_splits: int = 5,
    random_state: int = 42,
) -> GeneEvidenceFold:
    """Build model-ready train/valid/test evidence for one outer fold.

    ``train`` is inner cross-fitted: every row is transformed by an encoder
    whose fit subset excludes that row.  ``valid`` and ``test`` are transformed
    by one encoder fitted on all of ``X_train``.
    """
    train_matrix = sparse.csr_matrix(X_train)
    valid_matrix = sparse.csr_matrix(X_valid)
    test_matrix = None if X_test is None else sparse.csr_matrix(X_test)
    labels = np.asarray(y_train, dtype=str)
    class_array = np.asarray(classes, dtype=str)
    if train_matrix.shape[0] != labels.size:
        raise ValueError("X_train rows and y_train length differ")
    if valid_matrix.shape[1] != train_matrix.shape[1]:
        raise ValueError("X_valid gene dimension differs from X_train")
    if test_matrix is not None and test_matrix.shape[1] != train_matrix.shape[1]:
        raise ValueError("X_test gene dimension differs from X_train")

    train_features = cross_fit_gene_evidence(
        train_matrix,
        labels,
        method=method,
        classes=class_array,
        strength=strength,
        prior_weight=prior_weight,
        n_splits=inner_n_splits,
        random_state=random_state,
    ).astype(np.float32, copy=False)

    outer_encoder = EmpiricalBayesGeneEvidence(
        method=method,
        strength=strength,
        prior_weight=prior_weight,
        classes=class_array,
    ).fit(train_matrix, labels)
    valid_features = outer_encoder.transform(valid_matrix).astype(
        np.float32, copy=False
    )
    test_features = (
        None
        if test_matrix is None
        else outer_encoder.transform(test_matrix).astype(np.float32, copy=False)
    )
    names = evidence_feature_names(method, class_array)
    expected_width = class_array.size
    for split, values in (
        ("train", train_features),
        ("valid", valid_features),
        ("test", test_features),
    ):
        if values is not None and values.shape[1] != expected_width:
            raise AssertionError(f"{split} evidence width is not {expected_width}")
        if values is not None and not np.isfinite(values).all():
            raise FloatingPointError(f"{split} evidence contains non-finite values")

    return GeneEvidenceFold(
        train=train_features,
        valid=valid_features,
        test=test_features,
        feature_names=names,
        diagnostics={
            "method": method,
            "strength": float(strength),
            "prior_weight": float(prior_weight),
            "inner_n_splits": int(inner_n_splits),
            "inner_random_state": int(random_state),
            "n_input_genes": int(train_matrix.shape[1]),
            "n_output_features": int(expected_width),
            "n_zero_support_genes": int(outer_encoder.zero_support_.sum()),
            "train_encoding": "inner_cross_fitted",
            "valid_test_encoding": "outer_train_fit_transform_only",
        },
    )


def build_fold_gene_evidence_block(
    train_matrix: sparse.spmatrix | np.ndarray,
    test_matrix: sparse.spmatrix | np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence[str],
    *,
    block: str,
    classes: Sequence[str],
    strength: float = 20.0,
    prior_weight: float = 0.0,
    inner_n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Adapt :func:`build_gene_evidence_fold` to the GBDT block contract.

    The returned train matrix covers every repository train row.  Outer-train
    positions contain inner cross-fitted values; all other positions contain
    outer-train-fitted transform-only values and are therefore suitable for the
    current outer validation fold.
    """
    if block not in BLOCK_TO_METHOD:
        raise ValueError(f"unknown gene evidence block: {block}")
    fit_rows = np.asarray(train_index, dtype=np.int64)
    full_train = sparse.csr_matrix(train_matrix)
    labels = np.asarray(y_train_fold, dtype=str)
    if labels.size != fit_rows.size:
        raise ValueError("y_train_fold length differs from train_index")

    fold = build_gene_evidence_fold(
        X_train=full_train[fit_rows],
        y_train=labels,
        X_valid=full_train,
        X_test=test_matrix,
        method=BLOCK_TO_METHOD[block],
        classes=classes,
        strength=strength,
        prior_weight=prior_weight,
        inner_n_splits=inner_n_splits,
        random_state=random_state,
    )
    train_out = fold.valid.copy()
    train_out[fit_rows] = fold.train
    diagnostics = {
        **fold.diagnostics,
        "block": block,
        "outer_train_rows": int(fit_rows.size),
        "full_train_rows": int(full_train.shape[0]),
    }
    if fold.test is None:
        raise AssertionError("GBDT block adapter requires test features")
    return list(fold.feature_names), train_out, fold.test, diagnostics
