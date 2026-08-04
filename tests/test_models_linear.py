from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from cancer_hack.models_linear import ScaledLogisticRegression, create_logistic_model


def multiclass_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(120, 8))
    score = np.column_stack(
        [
            1.5 * matrix[:, 0] - matrix[:, 1],
            -matrix[:, 0] + 1.2 * matrix[:, 2],
            matrix[:, 1] - matrix[:, 2] + matrix[:, 3],
        ]
    )
    labels = np.asarray(["ACC", "BRCA", "DLBC"])[score.argmax(axis=1)]
    return matrix, labels


@pytest.mark.parametrize("sparse_input", [False, True])
def test_logistic_probability_contract(sparse_input):
    matrix, labels = multiclass_data()
    values = sparse.csr_matrix(matrix) if sparse_input else matrix
    model = create_logistic_model(C=1.0, max_iter=1_000, tol=1e-5)
    model.fit(values, labels)

    probability = model.predict_proba(values)
    assert probability.shape == (len(labels), 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert list(model.classes_) == ["ACC", "BRCA", "DLBC"]
    assert model.describe()["device"] == "cpu"


def test_scaler_is_fit_only_on_rows_passed_to_fit():
    train = np.asarray([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0], [6.0, 12.0]])
    labels = np.asarray(["a", "a", "b", "b"])
    held_out = np.asarray([[10_000.0, 20_000.0]])
    model = ScaledLogisticRegression(C=1.0, max_iter=500)
    model.fit(train, labels)

    assert np.allclose(model.scaler_.mean_, train.mean(axis=0))
    assert not np.allclose(model.scaler_.mean_, np.vstack([train, held_out]).mean(axis=0))
    assert model.predict_proba(held_out).shape == (1, 2)


def test_invalid_sample_weight_is_rejected():
    matrix, labels = multiclass_data()
    model = create_logistic_model()
    with pytest.raises(ValueError, match="sample_weight"):
        model.fit(matrix, labels, sample_weight=np.ones(len(labels) - 1))


def test_unknown_scaler_is_rejected_when_fit_starts():
    matrix, labels = multiclass_data()
    model = create_logistic_model(scaler="mystery")
    with pytest.raises(ValueError, match="unknown scaler"):
        model.fit(matrix, labels)


def test_predict_before_fit_is_rejected():
    with pytest.raises(RuntimeError, match="not fitted"):
        create_logistic_model().predict_proba([[0.0, 1.0]])
