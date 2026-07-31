"""fold 학습 부분에서만 fit 하는지 검증한다.

대회 규정이 콕 집어 금지한 것 — 인코더·스케일러·집계 통계를 test 로 fit 하는 것.
`hypermutated_flag` 와 `burden_quantile_bin` 은 분위수 경계를 잡아야 해서 이 위험이
있는 유일한 피처다. 나머지는 행마다 독립 계산이라 애초에 샐 곳이 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancer_hack.features_basic import (
    BurdenBinner,
    make_sample_mutation_features,
)


def _burden(values: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"mutation_event_count": values})


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        BurdenBinner().transform(_burden([1, 2, 3]))


def test_threshold_depends_only_on_fit_data():
    train = _burden(list(range(100)))
    other = _burden([10_000] * 100)

    binner = BurdenBinner(hypermutated_quantile=0.95).fit(train)
    threshold = binner.threshold_
    edges = binner.edges_.copy()

    binner.transform(other)  # transform 은 상태를 건드리면 안 된다
    assert binner.threshold_ == threshold
    assert np.array_equal(binner.edges_, edges)


def test_fitting_on_more_data_changes_threshold():
    """경계가 fit 데이터에 실제로 반응해야 의미 있는 검증이다."""
    small = BurdenBinner().fit(_burden(list(range(100))))
    large = BurdenBinner().fit(_burden(list(range(100)) + [10_000] * 100))
    assert small.threshold_ != large.threshold_


def test_valid_rows_can_exceed_train_range():
    """학습 fold 밖 값이 들어와도 터지지 않고 마지막 구간으로 간다."""
    binner = BurdenBinner(n_bins=4).fit(_burden([0, 1, 2, 3, 4, 5, 6, 7]))
    out = binner.transform(_burden([-5, 3, 999]))
    assert int(out["hypermutated_flag"].iloc[-1]) == 1
    assert int(out["burden_quantile_bin"].iloc[0]) == 0
    assert int(out["burden_quantile_bin"].iloc[-1]) == len(binner.edges_)


def test_transform_preserves_input_columns():
    features = _burden([1, 2, 3])
    features["other"] = [9, 9, 9]
    out = BurdenBinner(n_bins=2).fit(features).transform(features)
    assert list(out.columns)[:2] == ["mutation_event_count", "other"]
    assert "hypermutated_flag" in out.columns
    assert "burden_quantile_bin" in out.columns


def test_sample_features_are_row_independent(toy_frame):
    """행을 섞거나 잘라도 같은 샘플의 피처 값이 바뀌지 않아야 한다.

    이게 성립하면 train/test 를 따로 돌려도 누수가 없다.
    """
    genes = ["TP53", "KRAS", "EGFR"]
    full = make_sample_mutation_features(toy_frame, gene_columns=genes)
    subset = make_sample_mutation_features(
        toy_frame.iloc[[2]].reset_index(drop=True), gene_columns=genes
    )
    pd.testing.assert_frame_equal(
        full.iloc[[2]].reset_index(drop=True), subset, check_dtype=True
    )
