"""중복 그룹 가중치 실험(A~E) — same-label 확장 계약.

`balanced_sample_weight`/`group_size_inverse_weight`/`resolve_sample_weight`
계약은 `tests/test_models_gbdt.py`가 이미 지킨다. 여기서는 그걸 다시 테스트하지
않고, `sample_weights.py`가 새로 추가한 것만 검증한다 — `(SUBCLASS, profile_hash)`
same-label 그룹핑(D)과 그 단순곱의 클래스별 총weight 불균형을 고치는 교정식(E).

원본 데이터·개인 경로 없이 합성 배열만으로 전부 돈다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancer_hack.sample_weights import (
    EXPERIMENT_SCHEMES,
    effective_sample_size,
    normalize_to_unit_mean,
    resolve_experiment_weight,
    same_label_duplicate_weight,
    same_label_rebalanced_weight,
)

# 프로필 p1이 라벨 A,A,B로 갈라지는 경우를 반드시 포함한다 — profile-only 그룹핑과
# same-label 그룹핑이 실제로 다른 값을 내는 핵심 케이스다.
TOY_Y = np.array(["A", "A", "A", "B", "B", "C", "C", "C", "C"])
TOY_PH = np.array(["p1", "p1", "p2", "p1", "p3", "p4", "p4", "p4", "p5"])


# ---------------------------------------------------------------------------
# same_label_duplicate_weight (D 성분)
# ---------------------------------------------------------------------------


def test_same_label_duplicate_weight_splits_by_label():
    """profile-only라면 p1 세 행(0,1,3)이 한 그룹이지만, same-label은 (A,p1)=2행과
    (B,p1)=1행으로 갈라진다."""
    w = same_label_duplicate_weight(TOY_Y, TOY_PH)
    assert w[0] == pytest.approx(w[1])  # (A,p1) 그룹 안에서는 동일
    assert w[3] != pytest.approx(w[0])  # (B,p1)은 별개 그룹이라 값이 다름
    assert w[3] > w[0]  # (B,p1)은 단독이라 더 큰 가중치


def test_same_label_duplicate_weight_mean_is_one():
    w = same_label_duplicate_weight(TOY_Y, TOY_PH)
    assert float(w.mean()) == pytest.approx(1.0)


def test_same_label_duplicate_weight_matches_hand_calc():
    w = same_label_duplicate_weight(["A", "A", "B"], ["p1", "p1", "p1"])
    assert float(w[0]) == pytest.approx(0.75)
    assert float(w[2]) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# same_label_rebalanced_weight (E)
# ---------------------------------------------------------------------------


def test_e_scheme_per_class_total_weight_equals_n_over_k():
    w = same_label_rebalanced_weight(TOY_Y, TOY_PH)
    n, k = len(TOY_Y), len(np.unique(TOY_Y))
    expected = n / k
    totals = pd.Series(w).groupby(TOY_Y).sum()
    for cls, total in totals.items():
        assert total == pytest.approx(expected, abs=1e-9), f"클래스 {cls} 총weight={total}, 기대={expected}"


def test_e_scheme_each_unique_group_in_same_class_has_equal_total_weight():
    w = same_label_rebalanced_weight(TOY_Y, TOY_PH)
    frame = pd.DataFrame({"y": TOY_Y, "ph": TOY_PH, "w": w})
    frame["group"] = frame["y"].astype(str) + "\x1f" + frame["ph"].astype(str)
    per_group_total = frame.groupby(["y", "group"])["w"].sum().reset_index()
    for cls, sub in per_group_total.groupby("y"):
        assert sub["w"].std() < 1e-9, f"클래스 {cls} 안의 그룹별 총weight가 다르다: {sub['w'].tolist()}"


def test_e_scheme_weight_within_group_inversely_proportional_to_size():
    w = same_label_rebalanced_weight(TOY_Y, TOY_PH)
    # (C,p4) 그룹 크기 3(인덱스 5,6,7) — 그룹 안에서는 상수
    assert w[5] == pytest.approx(w[6]) == pytest.approx(w[7])
    # (A,p1) 그룹 크기 2 vs (A,p2) 그룹 크기 1 -> 큰 그룹 쪽이 더 작아야 한다
    assert w[0] < w[2]


def test_e_scheme_sum_equals_n_and_mean_is_one():
    w = same_label_rebalanced_weight(TOY_Y, TOY_PH)
    assert float(w.sum()) == pytest.approx(len(TOY_Y), abs=1e-9)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-9)


def test_e_scheme_matches_doctest_hand_calc():
    w = same_label_rebalanced_weight(["A", "A", "A", "B", "B"], ["p1", "p1", "p2", "p1", "p3"])
    assert float(w.sum()) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# D의 알려진 결함 — 클래스 불균형을 숨기지 않는다
# ---------------------------------------------------------------------------


def test_d_naive_product_breaks_class_balance_while_e_does_not():
    from cancer_hack.sample_weights import resolve_experiment_weight

    d = resolve_experiment_weight("D_balanced_same_label_naive", TOY_Y, TOY_PH)
    e = resolve_experiment_weight("E_same_label_rebalanced", TOY_Y, TOY_PH)

    d_totals = pd.Series(d).groupby(TOY_Y).sum()
    e_totals = pd.Series(e).groupby(TOY_Y).sum()

    assert e_totals.std() < 1e-9, "E는 클래스별 총weight가 균등해야 한다"
    assert d_totals.std() > 1e-6, "D는 이 합성 예제에서 클래스별 총weight가 불균등해야 한다(구조적 성질)"


# ---------------------------------------------------------------------------
# normalize_to_unit_mean / effective_sample_size
# ---------------------------------------------------------------------------


def test_normalize_to_unit_mean_preserves_ratio():
    raw = np.array([1.0, 2.0, 4.0])
    normalized = normalize_to_unit_mean(raw)
    assert float(normalized.mean()) == pytest.approx(1.0)
    assert float(normalized[1] / normalized[0]) == pytest.approx(raw[1] / raw[0])


def test_normalize_to_unit_mean_rejects_nonpositive_sum():
    with pytest.raises(ValueError):
        normalize_to_unit_mean(np.array([0.0, 0.0]))


def test_normalize_to_unit_mean_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        normalize_to_unit_mean(np.array([1.0, np.nan, 2.0]))


def test_normalize_to_unit_mean_rejects_inf():
    with pytest.raises(ValueError, match="NaN 또는 inf"):
        normalize_to_unit_mean(np.array([1.0, np.inf, 2.0]))


def test_normalize_to_unit_mean_rejects_negative_inf():
    with pytest.raises(ValueError, match="NaN 또는 inf"):
        normalize_to_unit_mean(np.array([1.0, -np.inf, 2.0]))


def test_normalize_to_unit_mean_rejects_negative_weight_even_if_sum_positive():
    """합만 보면 5로 양수라 통과할 수 있지만, 음수 원소 자체가 잘못된 입력이다."""
    with pytest.raises(ValueError, match="음수"):
        normalize_to_unit_mean(np.array([-5.0, 10.0]))


def test_normalize_to_unit_mean_rejects_empty_array():
    with pytest.raises(ValueError, match="빈"):
        normalize_to_unit_mean(np.array([]))


def test_effective_sample_size_no_weight_equals_n():
    assert effective_sample_size(np.ones(12)) == pytest.approx(12.0)


def test_effective_sample_size_decreases_with_skew():
    uniform = np.ones(10)
    skewed = np.array([9.0] + [0.11111111] * 9)
    assert effective_sample_size(skewed) < effective_sample_size(uniform)


def test_effective_sample_size_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        effective_sample_size(np.array([1.0, np.nan]))


def test_effective_sample_size_rejects_inf():
    with pytest.raises(ValueError, match="NaN 또는 inf"):
        effective_sample_size(np.array([1.0, np.inf]))


def test_effective_sample_size_rejects_negative_weight():
    with pytest.raises(ValueError, match="음수"):
        effective_sample_size(np.array([-1.0, 2.0]))


def test_effective_sample_size_rejects_empty_array():
    with pytest.raises(ValueError, match="빈"):
        effective_sample_size(np.array([]))


@pytest.mark.parametrize("scheme", ["B_balanced", "C_balanced_profile", "D_balanced_same_label_naive", "E_same_label_rebalanced"])
def test_resolved_weight_never_contains_nan_inf_or_negative(scheme):
    """실제 스킴 출력 자체도 방어검사를 통과해야 한다 — 회귀 방지용 통합 점검."""
    w = resolve_experiment_weight(scheme, TOY_Y, TOY_PH)
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)


# ---------------------------------------------------------------------------
# resolve_experiment_weight — scheme resolver
# ---------------------------------------------------------------------------


def test_resolve_experiment_weight_a_returns_none():
    assert resolve_experiment_weight("A_none", TOY_Y, TOY_PH) is None


@pytest.mark.parametrize("scheme", ["B_balanced", "C_balanced_profile", "D_balanced_same_label_naive", "E_same_label_rebalanced"])
def test_resolve_experiment_weight_all_weighted_schemes_are_finite_positive_and_normalized(scheme):
    w = resolve_experiment_weight(scheme, TOY_Y, TOY_PH)
    assert np.isfinite(w).all()
    assert (w > 0).all()
    assert len(w) == len(TOY_Y)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-9)


def test_resolve_experiment_weight_unknown_scheme_raises():
    with pytest.raises(ValueError, match="모르는 scheme"):
        resolve_experiment_weight("Z_unknown", TOY_Y, TOY_PH)


def test_all_declared_schemes_are_resolvable():
    for scheme in EXPERIMENT_SCHEMES:
        resolve_experiment_weight(scheme, TOY_Y, TOY_PH)  # 예외 없이 끝나면 통과


# ---------------------------------------------------------------------------
# 방어적 오류 처리 — 길이 불일치·빈 입력·결측
# ---------------------------------------------------------------------------


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="길이가 다르다"):
        same_label_duplicate_weight(["A", "B"], ["p1"])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="빈 입력"):
        same_label_duplicate_weight([], [])


def test_missing_value_in_profile_hash_raises():
    with pytest.raises(ValueError, match="결측"):
        same_label_duplicate_weight(["A", "B"], ["p1", None])


def test_missing_value_in_label_raises():
    with pytest.raises(ValueError, match="결측"):
        same_label_duplicate_weight(["A", None], ["p1", "p2"])


# ---------------------------------------------------------------------------
# 결정론성 · validation 정보 없이 계산 가능
# ---------------------------------------------------------------------------


def test_computation_is_deterministic():
    for scheme in EXPERIMENT_SCHEMES:
        first = resolve_experiment_weight(scheme, TOY_Y, TOY_PH)
        second = resolve_experiment_weight(scheme, TOY_Y, TOY_PH)
        if first is None:
            assert second is None
        else:
            assert np.array_equal(first, second)


def test_weight_depends_only_on_the_rows_it_is_given():
    """validation 정보를 전달하지 않아도(=넘기지 않은 행은 계산에 전혀 관여하지
    않아도) train 부분만으로 계산이 완결된다는 것을 보여준다. 뒤에 행을 추가하면
    앞 행의 가중치가 달라진다는 사실 자체가 이 함수에 "train 전체" 개념이 없고
    호출자가 넘긴 배열만 본다는 것을 증명한다 — 그래서 validation 행을 실수로
    섞으면 안 된다."""
    only_first_seven = same_label_rebalanced_weight(TOY_Y[:7], TOY_PH[:7])
    with_all_rows = same_label_rebalanced_weight(TOY_Y, TOY_PH)
    assert not np.allclose(only_first_seven, with_all_rows[:7])
