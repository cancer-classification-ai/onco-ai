"""Fold-safe linear classifiers.

The project feature matrices mix binary indicators, counts, TF-IDF values, and
latent components.  A linear model therefore needs scaling, but the scaler must
be fit on the fold's training rows only.  ``ScaledLogisticRegression`` owns the
scaler so callers cannot accidentally fit it on the full OOF matrix.

Sparse CSR is the internal representation.  ``StandardScaler(with_mean=False)``
preserves sparsity while putting columns on comparable variance scales.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

SEED = 42
SCALERS = ("standard", "maxabs", "none")


def _as_csr(values) -> sparse.csr_matrix:
    """Return a finite float64 CSR matrix accepted by scikit-learn."""
    matrix = sparse.csr_matrix(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"feature matrix must be 2D, got {matrix.shape}")
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise ValueError("feature matrix contains NaN or Inf")
    return matrix


def _make_scaler(name: str):
    if name == "standard":
        return StandardScaler(with_mean=False)
    if name == "maxabs":
        return MaxAbsScaler()
    if name == "none":
        return None
    raise ValueError(f"unknown scaler {name!r}; choose one of {SCALERS}")


class ScaledLogisticRegression:
    """Sparse multinomial Logistic Regression with an owned fold-fit scaler."""

    name = "logistic"

    def __init__(
        self,
        *,
        C: float = 0.1,
        solver: str = "lbfgs",
        scaler: str = "standard",
        max_iter: int = 2_000,
        tol: float = 1e-4,
        random_state: int = SEED,
        verbose: int = 0,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if solver not in {"saga", "lbfgs", "newton-cg", "newton-cholesky", "sag"}:
            raise ValueError(
                "solver must support multinomial L2 Logistic Regression; "
                f"got {solver!r}"
            )

        self.C = float(C)
        self.solver = solver
        self.scaler_name = scaler
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.verbose = int(verbose)

        self.scaler_: StandardScaler | MaxAbsScaler | None = None
        self.estimator_: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None
        self.n_iter_: np.ndarray | None = None
        self.converged_: bool | None = None
        self.convergence_warnings_: list[str] = []
        self.calibration_: dict[str, Any] | None = None

    def fit(
        self,
        X,
        y: Sequence,
        sample_weight: np.ndarray | None = None,
    ) -> "ScaledLogisticRegression":
        matrix = _as_csr(X)
        labels = np.asarray(y)
        if matrix.shape[0] != len(labels):
            raise ValueError(f"X rows {matrix.shape[0]} != y rows {len(labels)}")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (len(labels),):
                raise ValueError(
                    f"sample_weight shape {sample_weight.shape} != {(len(labels),)}"
                )
            if not np.isfinite(sample_weight).all() or np.any(sample_weight <= 0):
                raise ValueError("sample_weight must contain finite positive values")

        self.scaler_ = _make_scaler(self.scaler_name)
        transformed = (
            matrix if self.scaler_ is None else self.scaler_.fit_transform(matrix)
        )

        # ``l1_ratio=0`` means L2 in current scikit-learn.  Omitting the
        # deprecated ``multi_class`` and ``penalty`` parameters also keeps this
        # compatible with both pre-1.8 and 1.8+ releases.
        self.estimator_ = LogisticRegression(
            C=self.C,
            l1_ratio=0.0,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.estimator_.fit(
                transformed,
                labels,
                sample_weight=sample_weight,
            )
        self.convergence_warnings_ = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        self.classes_ = self.estimator_.classes_.copy()
        self.n_iter_ = np.asarray(self.estimator_.n_iter_, dtype=np.int64)
        self.converged_ = not self.convergence_warnings_ and bool(
            np.all(self.n_iter_ < self.max_iter)
        )
        return self

    def _transform(self, X) -> sparse.csr_matrix:
        if self.estimator_ is None:
            raise RuntimeError("model is not fitted")
        matrix = _as_csr(X)
        if matrix.shape[1] != self.estimator_.n_features_in_:
            raise ValueError(
                f"feature width {matrix.shape[1]} != fitted width "
                f"{self.estimator_.n_features_in_}"
            )
        return matrix if self.scaler_ is None else self.scaler_.transform(matrix)

    def predict_proba(self, X) -> np.ndarray:
        transformed = self._transform(X)
        probability = np.asarray(
            self.estimator_.predict_proba(transformed),
            dtype=np.float64,
        )
        if probability.ndim != 2 or probability.shape[1] != len(self.classes_):
            raise RuntimeError(
                f"bad probability shape {probability.shape} for {len(self.classes_)} classes"
            )
        return probability

    def predict(self, X) -> np.ndarray:
        transformed = self._transform(X)
        return np.asarray(self.estimator_.predict(transformed))

    def describe(self) -> dict[str, Any]:
        if self.estimator_ is None:
            raise RuntimeError("model is not fitted")
        coefficient = np.asarray(self.estimator_.coef_)
        detail = {
            "name": self.name,
            "device": "cpu",
            "params": {
                "C": self.C,
                "solver": self.solver,
                "scaler": self.scaler_name,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state,
            },
            "n_iter": self.n_iter_.tolist(),
            "converged": bool(self.converged_),
            "convergence_warnings": self.convergence_warnings_,
            "coefficient_density": float(np.count_nonzero(coefficient) / coefficient.size),
            "coefficient_l2_norm": float(np.linalg.norm(coefficient)),
        }
        if self.calibration_ is not None:
            detail["calibration"] = self.calibration_
        return detail


def create_logistic_model(**params) -> ScaledLogisticRegression:
    """Factory kept small so scripts and tests share one construction path."""
    return ScaledLogisticRegression(**params)
