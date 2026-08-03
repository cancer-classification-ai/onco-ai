"""RF 산출물(OOF/test 확률/submission) 스키마·계약 검증 — spec §9.2.

`Ticket 1a`/`1b`/`3`/`4`/`5` 가 전부 이 모듈을 그대로 호출한다(재구현 금지).
문제를 찾으면 `ValueError` 를 낸다 — 통과하면 반환값 없음(`None`).

`cancer_hack.metrics.macro_f1` 을 그대로 쓰지 않는 이유: 그 함수는 `labels=`
인자가 없어 `sklearn.metrics.f1_score` 의 기본 라벨 추론(관측된 라벨 집합)에
기댄다. RF 작업 지시(§spec, Ticket 1a/1b)는 Macro F1 계산에
`labels=canonical_class_order` 를 항상 명시하라고 못 박아 뒀다 — 특정 fold의
validation 에 없는 클래스도 0점으로 정확히 반영되게 하기 위해서다. 공용
`metrics.py` 를 이 요구에 맞춰 바꾸면 GBDT 경로까지 건드리게 되므로, RF 전용
이 작은 헬퍼(`macro_f1_with_labels`)를 여기 따로 둔다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from cancer_hack.metrics import probability_columns

__all__ = [
    "macro_f1_with_labels",
    "validate_oof_frame",
    "validate_test_probability_frame",
    "validate_submission_frame",
]


def macro_f1_with_labels(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> float:
    """`labels=` 를 항상 명시하는 Macro F1. RF 산출물 평가는 전부 이걸 쓴다."""
    return float(
        f1_score(y_true, y_pred, average="macro", labels=list(labels), zero_division=0)
    )


# ---------------------------------------------------------------- 공통 헬퍼
def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, label: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} 에 없는 컬럼: {missing}")


def _reject_extra_columns(frame: pd.DataFrame, allowed: Sequence[str], *, label: str) -> None:
    extra = [c for c in frame.columns if c not in allowed]
    if extra:
        raise ValueError(f"{label} 에 예상 밖 컬럼이 있다(index/Unnamed 열 포함 가능): {extra}")


def _check_ids_no_dup_no_missing(ids: pd.Series, expected: Sequence[str], *, label: str) -> pd.Series:
    ids = ids.astype(str)
    dup = ids[ids.duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"{label} ID 중복 {len(dup)}개 (예: {dup[:5]})")
    if len(ids) != len(expected):
        raise ValueError(f"{label} 행 수 {len(ids)}, 기대 {len(expected)}")
    expected_set = set(map(str, expected))
    got_set = set(ids.tolist())
    missing = expected_set - got_set
    extra = got_set - expected_set
    if missing or extra:
        raise ValueError(
            f"{label} ID 집합 불일치 — 누락 {len(missing)}개(예: {sorted(missing)[:5]}), "
            f"예상외 {len(extra)}개(예: {sorted(extra)[:5]})"
        )
    return ids


def _argmax_mismatch(proba: np.ndarray, chosen_index: np.ndarray, *, atol: float) -> np.ndarray:
    """`chosen_index` 가 각 행의 최댓값과 `atol` 이내인지(=동점 포함 유효한 argmax인지).

    CSV round-trip 이 마지막 몇 ULP 를 깎아 실제로는 동점이던 두 확률이 원본에서는
    미세하게 갈렸을 수 있다(실측: `pandas.to_csv` 기본 포맷팅이 완전한 왕복을
    보장하지 않는다). 정확히 `argmax` 인덱스 하나만 정답으로 인정하면 이런 근접
    동점에서 재현 불가능한 오탐이 난다 — 그래서 최댓값과 `atol` 이내인 클래스는
    전부 유효한 argmax 로 받아들인다.
    """
    row_max = proba.max(axis=1)
    chosen = proba[np.arange(len(proba)), chosen_index]
    return (row_max - chosen) > atol


def _check_probabilities(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> np.ndarray:
    proba = frame[list(columns)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(proba)):
        raise ValueError(f"{label} 확률에 NaN/Inf 가 있다")
    if (proba < 0).any() or (proba > 1).any():
        raise ValueError(f"{label} 확률이 [0,1] 범위를 벗어난다")
    sums = proba.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        bad = int((~np.isclose(sums, 1.0, atol=1e-6)).sum())
        raise ValueError(f"{label} 확률 행 합이 1 이 아닌 행 {bad}개")
    return proba


# ---------------------------------------------------------------- OOF
def validate_oof_frame(
    frame: pd.DataFrame,
    *,
    class_order: Sequence[str],
    train_ids: Sequence[str],
    n_splits: int,
    group_key_by_id: Mapping[str, object] | None = None,
    expected_macro_f1: float | None = None,
    atol: float = 1e-6,
) -> None:
    """spec §9.2 OOF 계약(`ID, fold, y_true, y_pred, p_{class}*len(class_order)`).

    `group_key_by_id` 를 주면 동일 그룹의 fold 교차를 함께 검사한다(선택).
    `expected_macro_f1` 을 주면 저장된 값과 독립 재계산 결과를 `atol` 이내로
    비교한다(§9.2 "저장된 Macro F1이 재계산한 값과 일치").
    """
    label = "OOF"
    proba_columns = probability_columns(class_order)
    required = ["ID", "fold", "y_true", "y_pred", *proba_columns]
    _require_columns(frame, required, label=label)
    _reject_extra_columns(frame, required, label=label)

    _check_ids_no_dup_no_missing(frame["ID"], train_ids, label=label)

    fold_raw = frame["fold"].to_numpy()
    fold_values = fold_raw.astype(np.int64)
    if not np.array_equal(fold_values, fold_raw.astype(np.float64)):
        raise ValueError(f"{label} fold 값이 정수가 아니다")
    if fold_values.min() < 0 or fold_values.max() > n_splits - 1:
        raise ValueError(
            f"{label} fold 값 범위 이탈: [{int(fold_values.min())}, {int(fold_values.max())}], "
            f"기대 [0, {n_splits - 1}]"
        )

    canonical = set(map(str, class_order))
    for column in ("y_true", "y_pred"):
        values = set(frame[column].astype(str).tolist())
        unexpected = values - canonical
        if unexpected:
            raise ValueError(f"{label} {column} 에 canonical 클래스 밖 값: {sorted(unexpected)}")

    proba = _check_probabilities(frame, proba_columns, label=label)

    class_index = {str(c): i for i, c in enumerate(class_order)}
    chosen = np.array([class_index[v] for v in frame["y_pred"].astype(str)])
    mismatched = _argmax_mismatch(proba, chosen, atol=atol)
    if mismatched.any():
        raise ValueError(f"{label} y_pred 가 argmax 와 다른 행 {int(mismatched.sum())}개")

    if group_key_by_id is not None:
        ids = frame["ID"].astype(str)
        groups = ids.map(group_key_by_id)
        if groups.isna().any():
            unknown = ids[groups.isna()].tolist()
            raise ValueError(f"{label} group_key_by_id 에 없는 ID 가 있다: {unknown[:5]}")
        check = pd.DataFrame({"group": groups.to_numpy(), "fold": fold_values})
        crossing = check.groupby("group")["fold"].nunique()
        bad_groups = crossing[crossing > 1]
        if len(bad_groups):
            raise ValueError(
                f"{label} 동일 group 이 fold 를 넘는 사례 {len(bad_groups)}개 "
                f"(예: {bad_groups.index[:5].tolist()})"
            )

    if expected_macro_f1 is not None:
        recomputed = macro_f1_with_labels(
            frame["y_true"].astype(str), frame["y_pred"].astype(str), class_order
        )
        if abs(recomputed - expected_macro_f1) > atol:
            raise ValueError(
                f"{label} 저장된 Macro F1({expected_macro_f1:.6f})이 재계산 값"
                f"({recomputed:.6f})과 atol={atol} 이내로 일치하지 않는다"
            )


# ---------------------------------------------------------------- test 확률
def validate_test_probability_frame(
    frame: pd.DataFrame,
    *,
    class_order: Sequence[str],
    sample_submission_ids: Sequence[str],
) -> None:
    """spec §9.2 test 확률 계약(`ID, p_{class}*len(class_order)`, sample_submission 순서).

    `build_prediction_frame` 을 그대로 재사용하면(spec §8) `y_pred` 열이 함께
    따라온다 — 기존 `train_gbdt.py` 의 test_predictions 산출물도 이미 그렇다.
    그래서 필수 열의 **존재**만 확인하고 컬럼 집합을 정확히 맞추라고 요구하지
    않는다("정확히 ID,SUBCLASS" 제약은 submission 에만 있다, spec §9.2).
    """
    label = "test probability"
    proba_columns = probability_columns(class_order)
    required = ["ID", *proba_columns]
    _require_columns(frame, required, label=label)

    expected = [str(i) for i in sample_submission_ids]
    ids = _check_ids_no_dup_no_missing(frame["ID"], expected, label=label)
    if ids.tolist() != expected:
        raise ValueError(f"{label} ID 순서가 sample_submission 과 다르다")

    _check_probabilities(frame, proba_columns, label=label)


# ---------------------------------------------------------------- submission
def validate_submission_frame(
    frame: pd.DataFrame,
    *,
    class_order: Sequence[str],
    sample_submission_ids: Sequence[str],
    test_proba_frame: pd.DataFrame | None = None,
    atol: float = 1e-6,
) -> None:
    """spec §9.2 submission 계약(`ID,SUBCLASS` 정확히, canonical 클래스, argmax 일치)."""
    label = "submission"
    if list(frame.columns) != ["ID", "SUBCLASS"]:
        raise ValueError(
            f"{label} 컬럼이 정확히 ID,SUBCLASS 가 아니다(index/Unnamed 열 포함 가능): "
            f"{list(frame.columns)}"
        )

    expected = [str(i) for i in sample_submission_ids]
    ids = _check_ids_no_dup_no_missing(frame["ID"], expected, label=label)
    if ids.tolist() != expected:
        raise ValueError(f"{label} ID 순서가 sample_submission 과 다르다")

    subclass = frame["SUBCLASS"]
    if subclass.isna().any():
        raise ValueError(f"{label} SUBCLASS 에 결측이 있다")
    subclass = subclass.astype(str)
    canonical = set(map(str, class_order))
    unexpected = set(subclass.tolist()) - canonical
    if unexpected:
        raise ValueError(f"{label} SUBCLASS 에 canonical 클래스 밖 값: {sorted(unexpected)}")

    if test_proba_frame is not None:
        proba_columns = probability_columns(class_order)
        proba_ids = test_proba_frame["ID"].astype(str).tolist()
        proba = test_proba_frame[proba_columns].to_numpy(dtype=np.float64)
        proba_by_id = dict(zip(proba_ids, proba))
        class_index = {str(c): i for i, c in enumerate(class_order)}

        mismatched = []
        for id_, label_value in zip(ids.tolist(), subclass.tolist()):
            row = proba_by_id.get(id_)
            if row is None:
                mismatched.append(id_)
                continue
            if _argmax_mismatch(row[None, :], np.array([class_index[label_value]]), atol=atol)[0]:
                mismatched.append(id_)
        if mismatched:
            raise ValueError(
                f"{label} SUBCLASS 가 test probability argmax 와 다른 ID {len(mismatched)}개 "
                f"(예: {mismatched[:5]})"
            )
