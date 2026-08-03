"""샘플 가중치 계약 — 전부 평균 1 이어야 한다.

`balanced_sample_weight` 는 sklearn `compute_sample_weight("balanced")` 라 평균이 1 이다.
새로 붙는 가중치가 다른 스케일을 쓰면 둘을 곱했을 때 실효 학습률이 조용히 달라지고,
지금 튜닝된 `learning_rate`·`reg_lambda` 가 의미를 잃는다. 그래서 정규화를 테스트로
못 박아 둔다.

그룹 가중치는 **fold 의 train 부분에서만** 계산한다. skf5 에서 그룹 364 개가 fold 를
가로지르므로 전역 카운트를 쓰면 학습에 한 번만 들어간 행까지 깎인다.
"""

from __future__ import annotations

import numpy as np
import pytest

from cancer_hack.models_gbdt import (
    SAMPLE_WEIGHT_BUILDERS,
    balanced_sample_weight,
    group_size_inverse_weight,
    resolve_sample_weight,
)

# 실측 train 그룹 크기 분포 {1: 5185, 2: 445, 3: 2, 4: 2, 18: 1, 94: 1} 의 축소판.
TOY_GROUPS = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
# 라벨을 일부러 불균형하게, 그리고 그룹 크기와 상관되게 둔다. 5:5 로 두면 balanced
# 가중치가 전부 1 이 되어 곱셈 관련 테스트가 아무것도 검증하지 못한다.
TOY_LABELS = list("aabbbbbbbb")


def test_balanced_weight_mean_is_one():
    """기존 계약을 못 박는다 — 지금 이걸 지키는 테스트가 없었다."""
    weight = balanced_sample_weight(list("aaabbc"))
    assert float(weight.mean()) == pytest.approx(1.0)


def test_group_weight_mean_is_one():
    weight = group_size_inverse_weight(TOY_GROUPS)
    assert float(weight.mean()) == pytest.approx(1.0)


def test_group_weight_is_inverse_of_group_size():
    """크기 2 그룹의 행은 단독 행의 정확히 절반을 받아야 한다."""
    weight = group_size_inverse_weight([0, 1, 1])
    assert float(weight[1] / weight[0]) == pytest.approx(0.5)
    assert float(weight[1]) == pytest.approx(float(weight[2]))


def test_group_weight_rows_in_one_group_are_equal():
    weight = group_size_inverse_weight(TOY_GROUPS)
    for group in set(TOY_GROUPS):
        members = weight[np.asarray(TOY_GROUPS) == group]
        assert np.allclose(members, members[0])


def test_group_weight_power_softens_large_groups():
    """`power=0.5` 는 94 행짜리 무변이 그룹이 통째로 사라지는 걸 막는다."""
    full = group_size_inverse_weight(TOY_GROUPS, power=1.0)
    soft = group_size_inverse_weight(TOY_GROUPS, power=0.5)
    assert soft.min() > full.min()
    assert float(soft.mean()) == pytest.approx(1.0)


def test_group_weight_rejects_nonpositive_power():
    with pytest.raises(ValueError, match="power"):
        group_size_inverse_weight(TOY_GROUPS, power=0.0)


def test_group_weight_accepts_string_group_keys():
    """`group_key` 가 정수든 문자열이든 받아야 한다."""
    numeric = group_size_inverse_weight([0, 0, 1])
    textual = group_size_inverse_weight(["x", "x", "y"])
    assert np.allclose(numeric, textual)


def test_group_weight_uses_only_the_rows_it_is_given():
    """fold 의 train 슬라이스만 넘어오면 그 안의 크기로만 세야 한다.

    skf5 에서 그룹 364 개가 fold 를 가로지른다. 쌍둥이 중 한쪽만 학습에 들어갔으면
    그 행은 한 번만 기여하므로 단독 행과 같은 가중치를 받아야 한다.
    """
    groups = np.array([0, 0, 1, 2])
    train_index = np.array([0, 2, 3])  # 그룹 0 의 한쪽만 들어온다
    weight = group_size_inverse_weight(groups[train_index])
    assert np.allclose(weight, weight[0])


@pytest.mark.parametrize(
    "spec", ["balanced", "group", "group_sqrt", "balanced+group", "balanced+group_sqrt"]
)
def test_resolved_weight_mean_is_one(spec):
    weight = resolve_sample_weight(spec, TOY_LABELS, TOY_GROUPS)
    assert float(weight.mean()) == pytest.approx(1.0)
    assert len(weight) == len(TOY_LABELS)


@pytest.mark.parametrize("spec", [None, "", "none"])
def test_no_weight_specs_return_none(spec):
    assert resolve_sample_weight(spec, TOY_LABELS, TOY_GROUPS) is None


def test_unknown_weight_name_raises():
    with pytest.raises(ValueError, match="모르는 가중치"):
        resolve_sample_weight("balanced+nope", TOY_LABELS, TOY_GROUPS)


def test_composed_weight_needs_the_renormalization():
    """개별 가중치가 평균 1 이어도 곱의 평균은 1 이 아니다. 그래서 다시 맞춘다."""
    raw = balanced_sample_weight(TOY_LABELS) * group_size_inverse_weight(TOY_GROUPS)
    assert float(raw.mean()) != pytest.approx(1.0)
    assert float(
        resolve_sample_weight("balanced+group", TOY_LABELS, TOY_GROUPS).mean()
    ) == pytest.approx(1.0)


def test_every_builder_name_is_resolvable():
    """리졸버와 빌더 사전이 어긋나면 config 문자열이 조용히 죽는다."""
    for name in SAMPLE_WEIGHT_BUILDERS:
        assert resolve_sample_weight(name, TOY_LABELS, TOY_GROUPS) is not None
