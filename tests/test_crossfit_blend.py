"""`crossfit_calibrated_blend` 의 계약.

이 함수를 `scripts/calibrate_ensemble.py` 와 제출 노트북이 **같이** 부른다. 로직이 두
벌이면 노트북이 제출본을 재현하지 못하는 순간이 오고, 그때는 대회 심사에서 코드와
제출물이 어긋난 것으로 보인다.
"""

from __future__ import annotations

import numpy as np
import pytest

from cancer_hack.calibration import MacroF1LogitBias
from cancer_hack.ensemble import crossfit_calibrated_blend, weighted_average

CLASSES = ["ACC", "BRCA", "LGG"]


@pytest.fixture
def toy():
    """3클래스 · 60행 · 소스 2개. fold 는 3개로 균등하게 나눈다."""
    rng = np.random.default_rng(0)
    y = np.asarray([CLASSES[i % 3] for i in range(60)])
    folds = np.asarray([i % 3 for i in range(60)])
    sources = []
    for shift in (0.0, 0.3):
        logits = rng.normal(size=(60, 3))
        for i, label in enumerate(y):
            logits[i, CLASSES.index(label)] += 1.0 + shift
        proba = np.exp(logits)
        sources.append(proba / proba.sum(axis=1, keepdims=True))
    return sources, y, folds


def test_raw_blend_equals_plain_weighted_average(toy):
    """고정 가중이면 fold 밖에서 학습할 게 없다 — raw 는 그냥 가중평균이어야 한다."""
    sources, y, folds = toy
    weights = [0.7, 0.3]
    raw, _, _ = crossfit_calibrated_blend(sources, y, CLASSES, folds, weights)
    np.testing.assert_allclose(raw, weighted_average(sources, weights))


def test_every_row_is_filled_exactly_once(toy):
    """fold 하나가 빠지면 그 행의 확률이 0으로 남는데, argmax 는 조용히 첫 클래스를 낸다."""
    sources, y, folds = toy
    _, calibrated, _ = crossfit_calibrated_blend(sources, y, CLASSES, folds, [0.5, 0.5])
    assert (calibrated.sum(axis=1) > 0).all()
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)


def test_fold_results_cover_all_folds(toy):
    sources, y, folds = toy
    _, _, fold_results = crossfit_calibrated_blend(sources, y, CLASSES, folds, [0.5, 0.5])
    assert [r["fold"] for r in fold_results] == [0, 1, 2]
    assert sum(r["n_valid"] for r in fold_results) == len(y)
    for r in fold_results:
        assert r["n_train"] + r["n_valid"] == len(y)
        assert set(r["bias"]) == set(CLASSES)


def test_bias_is_fitted_without_the_validation_fold(toy):
    """교차적합이 실제로 되고 있는지 — fold 0 의 보정이 fold 0 을 안 보고 나와야 한다.

    같은 입력으로 fold 0 을 뺀 나머지에 직접 fit 한 바이어스와 일치하는지 본다.
    전체 OOF 에 한 번에 맞추면 이 값이 달라진다.
    """
    sources, y, folds = toy
    weights = [0.6, 0.4]
    _, _, fold_results = crossfit_calibrated_blend(sources, y, CLASSES, folds, weights)

    train_mask = folds != 0
    expected = MacroF1LogitBias().fit(
        weighted_average([s[train_mask] for s in sources], weights), y[train_mask], CLASSES
    )
    assert fold_results[0]["bias"] == expected.bias_by_class()


def test_calibration_can_lose_out_of_fold(toy):
    """보정이 held-out fold 에서 **나빠질 수 있다**는 게 교차적합을 쓰는 이유다.

    전체 OOF 에 한 번에 맞추면 보정은 정의상 절대 안 나빠진다(fit 과 evaluate 가 같은
    행이라서다). 그 값으로 구성을 고르면 실측으로 0.5210 을 0.5438 로 읽게 된다.
    여기서는 두 점수가 fold 마다 따로 계산돼 서로 다를 수 있다는 것만 확인한다.
    """
    sources, y, folds = toy
    _, _, fold_results = crossfit_calibrated_blend(sources, y, CLASSES, folds, [0.5, 0.5])
    assert any(r["calibrated_macro_f1"] != r["raw_blend_macro_f1"] for r in fold_results)
    assert all(0.0 <= r["calibrated_macro_f1"] <= 1.0 for r in fold_results)
