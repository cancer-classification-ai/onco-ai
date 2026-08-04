"""Empirical-Bayes gene evidence encoders and leakage-safe cross-fitting.

The input is a binary sample-by-gene matrix.  Both encoders use a gene's
fit-partition prevalence as the shrinkage prior, so an unobserved gene cannot
become positive evidence merely because a class is small.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold


EVIDENCE_METHODS = ("eb_ovr", "eb_bnb")


class EmpiricalBayesGeneEvidence:
    """Fit EB-OVR or EB-Bernoulli gene-by-class evidence."""

    def __init__(
        self,
        *,
        method: str,
        strength: float = 20.0,
        prior_weight: float = 0.0,
        classes: Sequence[str] | None = None,
        eps: float = 1e-7,
    ) -> None:
        if method not in EVIDENCE_METHODS:
            raise ValueError(f"method must be one of {EVIDENCE_METHODS}")
        if strength <= 0:
            raise ValueError("strength must be positive")
        if not 0 < eps < 0.5:
            raise ValueError("eps must be between 0 and 0.5")
        self.method = method
        self.strength = float(strength)
        self.prior_weight = float(prior_weight)
        self.requested_classes = (
            None if classes is None else np.asarray(classes, dtype=str)
        )
        self.eps = float(eps)

    @staticmethod
    def _binary_matrix(X: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
        matrix = sparse.csr_matrix(X, dtype=np.float64)
        if matrix.data.size and (
            np.any(matrix.data < 0)
            or np.any(matrix.data > 1)
            or np.any(matrix.data != 1)
        ):
            raise ValueError("X must be a binary 0/1 matrix")
        return matrix

    def fit(
        self, X: sparse.spmatrix | np.ndarray, y: Sequence[str]
    ) -> "EmpiricalBayesGeneEvidence":
        matrix = self._binary_matrix(X)
        labels = np.asarray(y, dtype=str)
        if matrix.shape[0] != labels.size:
            raise ValueError(f"X rows ({matrix.shape[0]}) != y rows ({labels.size})")
        classes = (
            np.unique(labels)
            if self.requested_classes is None
            else self.requested_classes.copy()
        )
        unknown = sorted(set(labels) - set(classes))
        if unknown:
            raise ValueError(f"labels absent from requested classes: {unknown}")

        indicator = np.column_stack([labels == label for label in classes]).astype(
            np.float64
        )
        class_counts = indicator.sum(axis=0)
        if np.any(class_counts == 0):
            raise ValueError(
                f"fit partition has no samples for classes: "
                f"{classes[class_counts == 0].tolist()}"
            )

        global_count = np.asarray(matrix.sum(axis=0)).ravel()
        global_rate = global_count / labels.size
        zero_support = global_count == 0
        class_mutated = np.asarray(matrix.T @ indicator, dtype=np.float64)
        p_class = (
            class_mutated + self.strength * global_rate[:, None]
        ) / (class_counts[None, :] + self.strength)
        p_class = np.clip(p_class, self.eps, 1.0 - self.eps)

        if self.method == "eb_ovr":
            rest_mutated = global_count[:, None] - class_mutated
            rest_counts = labels.size - class_counts
            p_rest = (
                rest_mutated + self.strength * global_rate[:, None]
            ) / (rest_counts[None, :] + self.strength)
            p_rest = np.clip(p_rest, self.eps, 1.0 - self.eps)
            weights = (
                np.log(p_class) - np.log1p(-p_class)
                - np.log(p_rest) + np.log1p(-p_rest)
            )
            intercept = np.zeros(classes.size, dtype=np.float64)
            self.p_rest_ = p_rest
        else:
            weights = np.log(p_class) - np.log1p(-p_class)
            # Unsupported genes are excluded, rather than clipped to eps and
            # multiplied by thousands inside the absence term.
            intercept = np.log1p(-p_class[~zero_support]).sum(axis=0)
            if self.prior_weight:
                class_prior = class_counts / class_counts.sum()
                intercept += self.prior_weight * np.log(class_prior)

        # This is an explicit scientific contract, not merely a numerical
        # consequence of the prevalence prior.
        weights[zero_support, :] = 0.0
        self.classes_ = classes
        self.class_counts_ = class_counts.astype(np.int64)
        self.class_mutated_counts_ = class_mutated.astype(np.int64)
        self.global_count_ = global_count.astype(np.int64)
        self.global_rate_ = global_rate
        self.zero_support_ = zero_support
        self.p_gene_class_ = p_class
        self.weight_ = weights
        self.intercept_ = intercept
        self.n_features_in_ = matrix.shape[1]
        if not np.isfinite(weights).all() or not np.isfinite(intercept).all():
            raise FloatingPointError("non-finite evidence parameters")
        return self

    def transform(self, X: sparse.spmatrix | np.ndarray) -> np.ndarray:
        self._check_fitted()
        matrix = self._binary_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} genes; expected {self.n_features_in_}"
            )
        scores = np.asarray(matrix @ self.weight_, dtype=np.float64)
        scores += self.intercept_[None, :]
        if not np.isfinite(scores).all():
            raise FloatingPointError("non-finite evidence score")
        return scores

    def predict_proba(self, X: sparse.spmatrix | np.ndarray) -> np.ndarray:
        return scores_to_probabilities(self.transform(X))

    def _check_fitted(self) -> None:
        if not hasattr(self, "weight_"):
            raise RuntimeError("encoder is not fitted")


def cross_fit_gene_evidence(
    X: sparse.spmatrix | np.ndarray,
    y: Sequence[str],
    *,
    method: str,
    classes: Sequence[str],
    strength: float = 20.0,
    prior_weight: float = 0.0,
    n_splits: int = 5,
    random_state: int = 42,
    return_fit_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Create training evidence with every row excluded from its encoder fit."""
    matrix = sparse.csr_matrix(X)
    labels = np.asarray(y, dtype=str)
    classes_array = np.asarray(classes, dtype=str)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    output = np.full(
        (matrix.shape[0], classes_array.size), np.nan, dtype=np.float64
    )
    coverage = np.zeros(matrix.shape[0], dtype=np.int8)
    audit: list[tuple[np.ndarray, np.ndarray]] = []
    for fit_index, heldout_index in splitter.split(np.zeros(labels.size), labels):
        if np.intersect1d(fit_index, heldout_index).size:
            raise AssertionError("inner fit/held-out overlap")
        encoder = EmpiricalBayesGeneEvidence(
            method=method,
            strength=strength,
            prior_weight=prior_weight,
            classes=classes_array,
        ).fit(matrix[fit_index], labels[fit_index])
        output[heldout_index] = encoder.transform(matrix[heldout_index])
        coverage[heldout_index] += 1
        audit.append((fit_index.copy(), heldout_index.copy()))
    if not np.all(coverage == 1) or not np.isfinite(output).all():
        raise AssertionError("invalid inner cross-fit coverage or values")
    return (output, audit) if return_fit_indices else output


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite 2D array")
    probabilities = softmax(values, axis=1)
    validate_probabilities(probabilities)
    return probabilities


def validate_probabilities(probabilities: np.ndarray, *, atol: float = 1e-7) -> None:
    values = np.asarray(probabilities)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite 2D array")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("probability outside [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=atol):
        raise ValueError("probability rows do not sum to one")
