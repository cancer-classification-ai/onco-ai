"""평가 지표와 예측 결과 프레임.

대회 평가 산식은 **Macro F1** 하나다. 정확도는 참고로만 본다 — 클래스가 26개인데
BRCA 786명 대 DLBC 38명이라 정확도는 다수 클래스에 끌려간다.

아티팩트 I/O 는 여기 넣지 않는다. 파일을 쓰는 건 `io.py` 와 scripts 의 몫이다.

확률 컬럼 이름(`p_{class}`) 규약을 이 모듈이 단독으로 정의한다. OOF 와
test_predictions 두 곳에서 쓰이는데 정의가 두 벌이 되면 조용히 어긋난다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROBABILITY_PREFIX = "p_"


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """대회 평가 산식."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_f1(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """클래스별 F1. Macro F1 이 어느 클래스에서 깎이는지 보려면 이걸 본다."""
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()))
    scores = f1_score(y_true, y_pred, average=None, labels=list(labels), zero_division=0)
    return {str(label): float(score) for label, score in zip(labels, scores)}


def evaluate_classification(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence[str] | None = None,
) -> dict:
    """기존 `artifacts/logs/lgbm_cv_result.json` 스키마와 맞춘 요약."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()))
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class_f1": per_class_f1(y_true, y_pred, labels),
        "support": {str(label): int((y_true == label).sum()) for label in labels},
    }


def probability_columns(classes: Sequence[str]) -> list[str]:
    """클래스 이름 -> 확률 컬럼 이름.

    >>> probability_columns(["ACC", "BRCA"])
    ['p_ACC', 'p_BRCA']
    """
    return [f"{PROBABILITY_PREFIX}{label}" for label in classes]


def build_prediction_frame(
    ids: Sequence,
    proba: np.ndarray,
    classes: Sequence[str],
    y_true: Sequence | None = None,
) -> pd.DataFrame:
    """`ID, p_*, (y_true,) y_pred` 프레임.

    `proba` 의 열 순서가 `classes` 와 같다고 **가정하지 않고 검사한다.** 열이 어긋난
    확률로 argmax 를 하면 예외 없이 조용히 틀린 라벨이 나온다.

    >>> f = build_prediction_frame(["a"], [[0.2, 0.8]], ["ACC", "BRCA"])
    >>> f["y_pred"].iloc[0]
    'BRCA'
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(
            f"proba 형상 {proba.shape} 가 클래스 수 {len(classes)} 와 맞지 않는다"
        )
    if len(ids) != len(proba):
        raise ValueError(f"ID {len(ids)}개 vs 확률 {len(proba)}행")

    classes = np.asarray(classes)
    frame = pd.DataFrame(proba, columns=probability_columns(classes))
    frame.insert(0, "ID", np.asarray(ids, dtype=str))
    if y_true is not None:
        frame["y_true"] = np.asarray(y_true, dtype=str)
    frame["y_pred"] = classes[proba.argmax(axis=1)]
    return frame


def validate_prediction_frame_schema(
    frame: pd.DataFrame,
    ids: Sequence,
    classes: Sequence[str],
    *,
    oof: bool,
) -> None:
    """Validate the shared OOF/test probability artifact contract."""
    expected = ["ID", *probability_columns(classes)]
    if oof:
        expected.append("y_true")
    expected.append("y_pred")
    if list(frame.columns) != expected:
        raise ValueError(f"prediction columns differ: {list(frame.columns)}")
    expected_ids = np.asarray(ids, dtype=str)
    if not np.array_equal(frame["ID"].astype(str).to_numpy(), expected_ids):
        raise ValueError("prediction IDs or order differ")
    if frame[probability_columns(classes)].isna().any().any():
        raise ValueError("prediction probabilities contain NaN")


def read_prediction_frame(path) -> tuple[pd.DataFrame, list[str]]:
    """예측 csv 를 읽어 (프레임, 클래스 목록) 을 돌려준다.

    기존 아티팩트 일부가 BOM 을 달고 저장돼 있어서 그냥 읽으면 첫 컬럼이 `﻿ID`
    가 된다. `utf-8-sig` 로 읽어 두 경우를 모두 흡수한다.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    classes = [
        column[len(PROBABILITY_PREFIX) :]
        for column in frame.columns
        if column.startswith(PROBABILITY_PREFIX)
    ]
    if not classes:
        raise ValueError(f"{path} 에 {PROBABILITY_PREFIX}* 확률 컬럼이 없다")
    return frame, classes


def label_distribution(labels: Sequence, classes: Sequence[str]) -> dict[str, float]:
    """고정된 클래스 순서로 라벨 비율을 계산한다.

    예측 분포 진단은 실제 test 정답을 추정하는 장치가 아니다. train 실제 분포와 test
    예측 분포가 크게 벌어지는지를 보는 경보용이라, 등장하지 않은 클래스도 0으로 남긴다.
    """

    labels = np.asarray(labels, dtype=str)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels 는 비어 있지 않은 1차원 배열이어야 한다")
    classes = [str(label) for label in classes]
    unknown = sorted(set(labels.tolist()) - set(classes))
    if unknown:
        raise ValueError(f"classes 에 없는 라벨: {unknown[:5]}")
    return {label: float((labels == label).mean()) for label in classes}


def total_variation_distance(
    reference: dict[str, float], observed: dict[str, float]
) -> float:
    """두 이산 분포의 총변동거리(TVD)를 계산한다."""

    if set(reference) != set(observed):
        missing = sorted(set(reference) ^ set(observed))
        raise ValueError(f"분포의 클래스 구성이 다르다: {missing[:5]}")
    for name, distribution in (("reference", reference), ("observed", observed)):
        values = np.asarray(list(distribution.values()), dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} 분포에 유효하지 않은 값이 있다")
        if not np.isclose(values.sum(), 1.0, atol=1e-8):
            raise ValueError(f"{name} 분포의 합이 1이 아니다: {values.sum()}")
    return float(
        0.5 * sum(abs(reference[label] - observed[label]) for label in reference)
    )


def prediction_distribution_report(
    reference_labels: Sequence,
    predicted_labels: Sequence,
    classes: Sequence[str],
    *,
    warning_threshold: float = 0.10,
) -> dict:
    """train 실제 분포와 예측 분포의 차이를 경보 형태로 요약한다.

    `warning_threshold` 는 제출 선택 기준이 아니라 운영 경보다. test 의 실제 클래스
    사전확률을 모르므로 TVD를 낮추기 위해 test 예측을 강제로 보정하면 안 된다.
    """

    if not 0 <= warning_threshold <= 1:
        raise ValueError("warning_threshold 는 0~1이어야 한다")
    reference = label_distribution(reference_labels, classes)
    predicted = label_distribution(predicted_labels, classes)
    tvd = total_variation_distance(reference, predicted)
    return {
        "tvd": tvd,
        "warning_threshold": float(warning_threshold),
        "warning": bool(tvd > warning_threshold),
        "reference_distribution": reference,
        "predicted_distribution": predicted,
        "interpretation": (
            "guardrail_only: test 실제 클래스 비율은 알 수 없으므로 "
            "TVD를 직접 최소화하거나 train 분포에 강제로 맞추지 않는다"
        ),
    }
