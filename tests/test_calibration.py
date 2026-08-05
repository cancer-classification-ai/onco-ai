"""클래스별 로짓 결정 보정 테스트."""

import numpy as np
import pytest

from cancer_hack.calibration import MacroF1LogitBias, apply_logit_bias
from cancer_hack.metrics import macro_f1


def test_zero_bias_preserves_probabilities() -> None:
    proba = np.array([[0.2, 0.8], [0.7, 0.3]])
    adjusted = apply_logit_bias(proba, [0.0, 0.0])
    assert np.allclose(adjusted, proba)


def test_logit_bias_returns_valid_probabilities() -> None:
    adjusted = apply_logit_bias(
        np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]),
        [-0.2, 0.1, 0.4],
    )
    assert np.all(adjusted >= 0)
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_logit_bias_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="bias 형상"):
        apply_logit_bias(np.array([[0.2, 0.8]]), [0.0])


def test_macro_f1_bias_can_recover_underpredicted_classes() -> None:
    classes = np.array(["A", "B", "C"])
    y_true = np.array(["A"] * 30 + ["B"] * 10 + ["C"] * 10)
    proba = np.zeros((50, 3))
    proba[:30] = [0.70, 0.20, 0.10]
    proba[30:40] = [0.45, 0.40, 0.15]
    proba[40:] = [0.45, 0.15, 0.40]
    before = macro_f1(y_true, classes[proba.argmax(axis=1)])

    fitted = MacroF1LogitBias().fit(proba, y_true, classes)
    after = macro_f1(y_true, fitted.predict(proba))

    assert after > before
    assert after == pytest.approx(1.0)
    assert fitted.bias_.mean() == pytest.approx(0.0)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        MacroF1LogitBias().predict_proba(np.array([[0.2, 0.8]]))
