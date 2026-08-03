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


def _canonical_argmax_labels(proba: np.ndarray, class_order: Sequence[str]) -> np.ndarray:
    """행별 canonical `np.argmax` 라벨. 정확한 동점은 `class_order` 에서 먼저
    나오는 클래스를 고른다(`np.argmax` 자체의 규칙 — 최댓값의 첫 위치를 돌려준다).

    y_pred/SUBCLASS 판정은 **항상 이 값과 정확히 일치**해야 한다. 근접 동점을
    허용하는 오차범위(atol)는 여기 없다 — 오차범위를 두면 "최댓값에 가깝지만
    아닌" 클래스가 조용히 통과한다. 대신 CSV round-trip 으로 인한 부동소수점
    오차 문제(`pandas.to_csv` 기본 포맷팅이 float64 완전 왕복을 보장하지
    않음 — 실측됨)는 **저장·재읽은 값을 기준으로 y_pred/SUBCLASS 를 계산**하는
    방식으로 `scripts/train_rf.py` 쪽에서 해결한다. 즉, 이 함수가 보는 `proba`
    는 이미 최종적으로 저장될(또는 저장된) 값이어야 한다.
    """
    classes = np.asarray([str(c) for c in class_order])
    return classes[proba.argmax(axis=1)]


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
    비교한다(§9.2 "저장된 Macro F1이 재계산한 값과 일치") — `atol` 은 **이
    Macro F1 비교에만** 쓰인다. `y_pred` 가 `p_{class}` 의 canonical
    `np.argmax` 와 일치하는지는 오차범위 없이 정확히 비교한다(근접 동점을
    허용하면 "최댓값에 가깝지만 아닌" 클래스가 조용히 통과한다).
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

    argmax_pred = _canonical_argmax_labels(proba, class_order)
    mismatched = frame["y_pred"].astype(str).to_numpy() != argmax_pred
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
    """spec §9.2 test 확률 계약 — 컬럼이 정확히 `ID` + `p_{class}*len(class_order)`.

    `build_prediction_frame` 을 그대로 쓰면 `y_pred` 열이 따라오는데, 저장용
    test 확률 파일에는 spec §8.2 스키마대로 `y_pred` 를 넣지 않는다
    (`scripts/train_rf.py` 가 저장 전 `["ID", *p_columns]` 로 잘라낸다).
    그래서 이 validator 는 `y_pred`/index/`Unnamed` 등 예상 밖 컬럼이 하나라도
    섞이면 실패한다 — 있으면 저장 경로가 스키마를 어긴 것이다.
    """
    label = "test probability"
    proba_columns = probability_columns(class_order)
    required = ["ID", *proba_columns]
    _require_columns(frame, required, label=label)
    _reject_extra_columns(frame, required, label=label)

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
) -> None:
    """spec §9.2 submission 계약(`ID,SUBCLASS` 정확히, canonical 클래스, argmax 일치).

    `test_proba_frame` 을 주면 SUBCLASS 가 그 확률의 canonical `np.argmax` 와
    **정확히** 일치하는지 검사한다(오차범위 없음, `_canonical_argmax_labels`
    참고). `test_proba_frame` 은 실제로 저장·재읽은 test 확률이어야 한다 —
    저장 전 메모리 값과 비교하면 CSV round-trip 부동소수점 오차로 오탐이 난다.
    """
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
        argmax_by_id = dict(zip(proba_ids, _canonical_argmax_labels(proba, class_order)))

        mismatched = [
            id_
            for id_, label_value in zip(ids.tolist(), subclass.tolist())
            if argmax_by_id.get(id_) != label_value
        ]
        if mismatched:
            raise ValueError(
                f"{label} SUBCLASS 가 test probability argmax 와 다른 ID {len(mismatched)}개 "
                f"(예: {mismatched[:5]})"
            )
