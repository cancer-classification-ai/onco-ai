"""비율·변환 변수 10종 모델 입력 interface 통합 테스트.

`RatioTransformFeatures`는 새 계산식을 추가하지 않는다 — `make_sample_mutation_features()`가
낸 8개 stateless 컬럼과 `BurdenBinner`가 만드는 2개 stateful 컬럼을 정본 순서로 잇기만
한다. 여기서는 그 연결 계약(정확히 10열, 순서, index 보존, 입력 불변, fold-safe fit)만
검증하고 원래 계산식은 `test_mutation_parser.py`/`test_feature_alignment.py`가 이미 본다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancer_hack.features_basic import (
    BurdenBinner,
    RATIO_TRANSFORM_FEATURE_COLUMNS,
    RatioTransformFeatures,
    make_sample_mutation_features,
)


def _sample_features(mutation_event_count: list[int]) -> pd.DataFrame:
    """`make_sample_mutation_features()` 출력 형태를 흉내 낸 최소 synthetic 프레임."""
    n = len(mutation_event_count)
    rng = np.arange(n, dtype="float32")
    return pd.DataFrame(
        {
            "log1p_mutated_gene_count": rng,
            "log1p_mutation_event_count": rng,
            "functional_ratio": (rng % 2).astype("float32"),
            "synonymous_ratio": np.zeros(n, dtype="float32"),
            "missense_ratio": (rng % 2).astype("float32"),
            "nonsense_ratio": np.zeros(n, dtype="float32"),
            "frameshift_ratio": np.zeros(n, dtype="float32"),
            "multihit_gene_ratio": np.zeros(n, dtype="float32"),
            "mutation_event_count": mutation_event_count,
        },
        index=[f"s{i}" for i in range(n)],
    )


def test_ratio_transform_columns_are_exact_and_ordered():
    assert RATIO_TRANSFORM_FEATURE_COLUMNS == (
        "log1p_mutated_gene_count",
        "log1p_mutation_event_count",
        "functional_ratio",
        "synonymous_ratio",
        "missense_ratio",
        "nonsense_ratio",
        "frameshift_ratio",
        "multihit_gene_ratio",
        "hypermutated_flag",
        "burden_quantile_bin",
    )


def test_transform_returns_exactly_ten_columns_in_order():
    train = _sample_features(list(range(20)))
    out = RatioTransformFeatures().fit(train).transform(train)
    assert list(out.columns) == list(RATIO_TRANSFORM_FEATURE_COLUMNS)
    assert out.shape == (20, 10)


def test_index_and_row_order_are_preserved():
    train = _sample_features(list(range(10)))
    shuffled = train.iloc[[3, 0, 7, 1]]
    out = RatioTransformFeatures().fit(train).transform(shuffled)
    assert list(out.index) == list(shuffled.index)


def test_input_frame_is_not_mutated():
    train = _sample_features(list(range(10)))
    before = train.copy(deep=True)
    RatioTransformFeatures().fit(train).transform(train)
    pd.testing.assert_frame_equal(train, before)


def test_first_eight_values_match_input_exactly():
    train = _sample_features(list(range(10)))
    out = RatioTransformFeatures().fit(train).transform(train)
    for col in RATIO_TRANSFORM_FEATURE_COLUMNS[:8]:
        pd.testing.assert_series_equal(out[col], train[col], check_names=True)


def test_last_two_values_match_burden_binner_directly():
    train = _sample_features([0, 1, 2, 3, 100] * 4)
    rt_out = RatioTransformFeatures().fit(train).transform(train)
    binner_out = BurdenBinner().fit(train).transform(train)
    pd.testing.assert_series_equal(
        rt_out["hypermutated_flag"], binner_out["hypermutated_flag"]
    )
    pd.testing.assert_series_equal(
        rt_out["burden_quantile_bin"], binner_out["burden_quantile_bin"]
    )


def test_transform_before_fit_raises():
    train = _sample_features(list(range(5)))
    with pytest.raises(RuntimeError):
        RatioTransformFeatures().transform(train)


def test_missing_required_column_raises_with_column_name():
    train = _sample_features(list(range(5))).drop(columns=["missense_ratio"])
    with pytest.raises(ValueError, match="missense_ratio"):
        RatioTransformFeatures().fit(_sample_features(list(range(5)))).transform(train)

    no_burden = _sample_features(list(range(5))).drop(columns=["mutation_event_count"])
    with pytest.raises(ValueError, match="mutation_event_count"):
        RatioTransformFeatures().fit(no_burden)


def test_validation_extreme_values_do_not_change_train_boundaries():
    train = _sample_features(list(range(100)))
    rt = RatioTransformFeatures().fit(train)
    threshold_before = rt._binner.threshold_
    edges_before = rt._binner.edges_.copy()

    extreme = _sample_features([-10, 10_000, 0])
    rt.transform(extreme)

    assert rt._binner.threshold_ == threshold_before
    assert np.array_equal(rt._binner.edges_, edges_before)


def test_repeated_fit_on_same_train_is_deterministic():
    train = _sample_features(list(range(50)))
    out_a = RatioTransformFeatures().fit(train).transform(train)
    out_b = RatioTransformFeatures().fit(train).transform(train)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_output_has_no_nan_or_inf():
    train = _sample_features(list(range(30)))
    out = RatioTransformFeatures().fit(train).transform(train)
    values = out.to_numpy(dtype="float64")
    assert not np.isnan(values).any()
    assert not np.isinf(values).any()


def test_missense_nonsense_ratio_correct_for_stop_gain_notations():
    """parser.py 이름 충돌 수정이 실제로 nonsense/missense 비율에 반영되는지 확인한다.

    Q369*(train 표기)·Q369X(test 표기) 모두 nonsense로 잡혀야 하고, 그 값이
    RatioTransformFeatures 출력까지 그대로 흘러가야 한다.
    """
    df = pd.DataFrame(
        {
            "TP53": ["Q369*", "Q369X", "G827R"],
            "KRAS": ["WT", "WT", "WT"],
        }
    )
    sample_feats = make_sample_mutation_features(df, gene_columns=["TP53", "KRAS"])

    assert sample_feats["nonsense_ratio"].tolist() == [1.0, 1.0, 0.0]
    assert sample_feats["missense_ratio"].tolist() == [0.0, 0.0, 1.0]

    out = RatioTransformFeatures().fit(sample_feats).transform(sample_feats)
    assert out["nonsense_ratio"].tolist() == [1.0, 1.0, 0.0]
    assert out["missense_ratio"].tolist() == [0.0, 0.0, 1.0]
