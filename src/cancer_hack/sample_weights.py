"""중복 그룹 가중치 실험(A~E) 전용 확장 — canonical 구현을 재사용한다.

`balanced_sample_weight` / `group_size_inverse_weight` / `resolve_sample_weight`
는 이미 `cancer_hack.models_gbdt`에 있다(PR #17). 이 모듈은 그걸 다시 만들지
않고, 거기 없는 두 가지만 추가한다 — `(SUBCLASS, profile_hash)` 조인으로 그룹을
정의하는 same-label 중복 가중치(D)와, 그 단순곱의 클래스별 총weight 불균형을
고치는 교정식(E). A(무가중)/B(클래스균형)/C(profile-only 역수)는 새로 만들지
않고 `models_gbdt`의 `resolve_sample_weight`를 그대로 호출한다.

모든 함수는 **fold의 train 부분만** 받는 순수 함수다. 전역 상태가 없고,
validation/test 인덱스를 넘기면 안 된다 — 함수 자체는 넘어온 배열 전체를
"train"으로 취급해 그 안에서 그룹 크기를 센다.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .models_gbdt import balanced_sample_weight, resolve_sample_weight

__all__ = [
    "EXPERIMENT_SCHEMES",
    "same_label_duplicate_weight",
    "same_label_rebalanced_weight",
    "normalize_to_unit_mean",
    "effective_sample_size",
    "resolve_experiment_weight",
]

#: 08_duplicate_group_weight_experiment 노트북/문서와 이름을 맞춘다.
EXPERIMENT_SCHEMES: tuple[str, ...] = (
    "A_none",
    "B_balanced",
    "C_balanced_profile",
    "D_balanced_same_label_naive",
    "E_same_label_rebalanced",
)


def _check_inputs(y: Sequence, profile_hash: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    profile_hash = np.asarray(profile_hash)
    if len(y) != len(profile_hash):
        raise ValueError(f"y와 profile_hash 길이가 다르다: {len(y)} vs {len(profile_hash)}")
    if len(y) == 0:
        raise ValueError("빈 입력이다 — 최소 1행이 필요하다")
    if pd.isna(y).any() or pd.isna(profile_hash).any():
        raise ValueError("y 또는 profile_hash에 결측값이 있다")
    return y, profile_hash


def _check_weight_values(weight: np.ndarray) -> None:
    """원소 단위 방어검사 — 집계값(합 등)만 보면 개별 NaN/inf/음수를 놓친다.

    예: weight=[-5, 10]은 합이 5로 양수라 합계만 보는 검사는 통과하지만, 음수
    가중치 자체가 이미 잘못된 입력이다. 여기서 원소 단위로 먼저 막는다.
    """
    if weight.size == 0:
        raise ValueError("빈 weight 배열이다")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weight에 NaN 또는 inf가 있다")
    if np.any(weight < 0):
        raise ValueError("weight에 음수가 있다")


def normalize_to_unit_mean(weight: np.ndarray) -> np.ndarray:
    """평균이 1이 되도록 전역 정규화한다. 상대 비율은 그대로 유지된다."""
    weight = np.asarray(weight, dtype=np.float64)
    _check_weight_values(weight)
    total = weight.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"weight 합이 정규화할 수 없는 값이다: {total}")
    return weight * (len(weight) / total)


def effective_sample_size(weight: np.ndarray) -> float:
    """Kish's ESS = (sum w)^2 / sum(w^2). 가중치로 인한 정보 손실을 정량화한다.

    >>> effective_sample_size(np.ones(10))
    10.0
    """
    weight = np.asarray(weight, dtype=np.float64)
    _check_weight_values(weight)
    denom = float(np.sum(weight ** 2))
    if denom <= 0:
        raise ValueError("weight 제곱합이 0 이하라 ESS를 계산할 수 없다")
    return float((weight.sum() ** 2) / denom)


def _same_label_group_codes(y: np.ndarray, profile_hash: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joint = pd.Series(y).astype(str) + "\x1f" + pd.Series(profile_hash).astype(str)
    codes, _ = pd.factorize(joint.to_numpy())
    sizes = np.bincount(codes).astype(np.float64)
    return codes, sizes


def same_label_duplicate_weight(y: Sequence, profile_hash: Sequence) -> np.ndarray:
    """**폴드 내부의 train 부분에서만 호출한다.**

    `(SUBCLASS, profile_hash)` 조인으로 그룹을 만들어 그 크기의 역수를 준다 —
    `group_size_inverse_weight`(profile_hash 단독, `models_gbdt`)와 달리 라벨이
    다르면 별개 그룹으로 센다. 평균 1로 정규화된다.

    >>> w = same_label_duplicate_weight(["A", "A", "B"], ["p1", "p1", "p1"])
    >>> round(float(w[0]), 4), round(float(w[2]), 4)
    (0.75, 1.5)
    """
    y_arr, ph_arr = _check_inputs(y, profile_hash)
    codes, sizes = _same_label_group_codes(y_arr, ph_arr)
    return normalize_to_unit_mean(1.0 / sizes[codes])


def same_label_rebalanced_weight(y: Sequence, profile_hash: Sequence) -> np.ndarray:
    """**폴드 내부의 train 부분에서만 호출한다.**

    `balanced_sample_weight() * same_label_duplicate_weight()` 단순곱(D)은
    클래스별 총weight 균형이 깨진다 — 이 함수는 그 결함을 고치는 교정식이다.

        weight_i = (N / K) / (G_c * n_cg)

    N = 전체 행 수, K = 클래스 수, G_c = 클래스 c 안의 고유
    `(SUBCLASS, profile_hash)` 그룹 수, n_cg = 그 행이 속한 same-label 그룹 크기.

    이미 평균 1·클래스별 총weight = N/K로 설계돼 정규화가 필요 없다.

    >>> w = same_label_rebalanced_weight(["A","A","A","B","B"], ["p1","p1","p2","p1","p3"])
    >>> round(float(w.sum()), 6)
    5.0
    """
    y_arr, ph_arr = _check_inputs(y, profile_hash)
    n = len(y_arr)
    classes, y_codes = np.unique(y_arr, return_inverse=True)
    k = len(classes)

    group_codes, group_sizes = _same_label_group_codes(y_arr, ph_arr)
    n_cg = group_sizes[group_codes]

    frame = pd.DataFrame({"y_code": y_codes, "group_code": group_codes})
    groups_per_class = frame.groupby("y_code")["group_code"].nunique()
    g_c = groups_per_class.loc[y_codes].to_numpy().astype(np.float64)

    return (n / k) / (g_c * n_cg)


def resolve_experiment_weight(
    scheme: str, y: Sequence, profile_hash: Sequence
) -> np.ndarray | None:
    """A~E 다섯 실험 scheme 이름으로 sample_weight를 만든다. fold의 train 부분만 받는다.

    A/B/C는 새로 만들지 않고 `cancer_hack.models_gbdt.resolve_sample_weight`를
    그대로 재사용한다 — B = ``resolve_sample_weight("balanced", ...)``,
    C = ``resolve_sample_weight("balanced+group", ...)``(PR #17 `f4rw`와 같은 정의).
    D/E만 이 모듈에서 새로 정의한다.

    >>> resolve_experiment_weight("A_none", ["a", "b"], ["p1", "p2"]) is None
    True
    """
    if scheme == "A_none":
        return None
    if scheme == "B_balanced":
        return resolve_sample_weight("balanced", y, profile_hash)
    if scheme == "C_balanced_profile":
        return resolve_sample_weight("balanced+group", y, profile_hash)
    if scheme == "D_balanced_same_label_naive":
        y_arr, ph_arr = _check_inputs(y, profile_hash)
        raw = balanced_sample_weight(y_arr) * same_label_duplicate_weight(y_arr, ph_arr)
        return normalize_to_unit_mean(raw)
    if scheme == "E_same_label_rebalanced":
        return same_label_rebalanced_weight(y, profile_hash)
    raise ValueError(f"모르는 scheme: {scheme!r} (가능: {EXPERIMENT_SCHEMES})")
