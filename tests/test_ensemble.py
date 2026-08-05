"""OOF 확률 앙상블과 분포 경보 테스트."""

import numpy as np
import pytest

from cancer_hack.ensemble import MacroF1Blender, weighted_average
from cancer_hack.metrics import (
    prediction_distribution_report,
    total_variation_distance,
)


def test_weighted_average_matches_manual_result() -> None:
    first = np.array([[0.8, 0.2], [0.3, 0.7]])
    second = np.array([[0.4, 0.6], [0.5, 0.5]])
    observed = weighted_average([first, second], [0.75, 0.25])
    assert np.allclose(observed, first * 0.75 + second * 0.25)
    assert np.allclose(observed.sum(axis=1), 1.0)


def test_weighted_average_rejects_negative_weights() -> None:
    proba = np.array([[0.8, 0.2]])
    with pytest.raises(ValueError, match="weights"):
        weighted_average([proba, proba], [1.1, -0.1])


def test_blender_selects_the_better_model() -> None:
    classes = ["A", "B"]
    y_true = np.array(["A", "B", "A", "B"])
    perfect = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    reversed_model = perfect[:, ::-1]

    fitted = MacroF1Blender().fit([perfect, reversed_model], y_true, classes)

    assert fitted.weights_[0] == pytest.approx(1.0)
    assert fitted.weights_[1] == pytest.approx(0.0)
    assert fitted.train_macro_f1_ == pytest.approx(1.0)


def test_tvd_known_value() -> None:
    reference = {"A": 0.5, "B": 0.5}
    observed = {"A": 0.8, "B": 0.2}
    assert total_variation_distance(reference, observed) == pytest.approx(0.3)


def test_prediction_distribution_report_is_guardrail() -> None:
    report = prediction_distribution_report(
        ["A", "A", "B", "B"],
        ["A", "A", "A", "A"],
        ["A", "B"],
        warning_threshold=0.2,
    )
    assert report["tvd"] == pytest.approx(0.5)
    assert report["warning"] is True
    assert report["interpretation"].startswith("guardrail_only")
