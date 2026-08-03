from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


# 인코딩 규약 — 기존 아티팩트가 뒤섞여 있어 여기서 정한다.
#   OOF·test_predictions : plain UTF-8 (BOM 없음)
#   제출 파일            : UTF-8-sig  (대회 베이스라인 노트북 규약)
# BOM 을 단 csv 를 그냥 읽으면 첫 컬럼이 `﻿ID` 가 되므로 읽을 때는
# `metrics.read_prediction_frame` 을 쓴다.
PREDICTION_ENCODING = "utf-8"
SUBMISSION_ENCODING = "UTF-8-sig"


# DataFrame을 Parquet 파일로 저장하는 공통 저장 함수
def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

# 저장된 Parquet 파일을 읽어 DataFrame으로 반환하는 로딩 함수
def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def save_csv(df: pd.DataFrame, path: str | Path, *, encoding: str = PREDICTION_ENCODING) -> Path:
    """중간에 죽어도 반쪽 파일이 남지 않게 임시 파일에 쓰고 옮긴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    df.to_csv(temporary, index=False, encoding=encoding)
    temporary.replace(path)
    return path


def build_submission(
    predictions: pd.DataFrame,
    sample_submission_path: str | Path,
    *,
    id_column: str = "ID",
    label_column: str = "SUBCLASS",
) -> pd.DataFrame:
    """예측 프레임을 제출 스키마로 맞춘다.

    `predictions` 는 `ID` 와 `y_pred` 를 가진 프레임이다(`metrics.build_prediction_frame`
    의 출력). 행 순서가 `sample_submission.csv` 와 같다는 걸 **가정하지 않고 ID 로
    병합한다.** 두 파일의 순서가 실제로 같다는 건 확인됐지만, 순서 가정에 기대면
    나중에 어느 한쪽 생성 경로가 바뀔 때 조용히 어긋난다.

    병합 후 세 가지를 검사한다 — 행 수, 결측, ID 집합. 하나라도 어긋나면 예외를 낸다.
    """
    sample = pd.read_csv(sample_submission_path, dtype=str)
    if id_column not in sample.columns or label_column not in sample.columns:
        raise ValueError(
            f"sample_submission 에 {id_column}/{label_column} 이 없다: {list(sample.columns)}"
        )
    if "y_pred" not in predictions.columns:
        raise ValueError(f"예측 프레임에 y_pred 가 없다: {list(predictions.columns)[:8]}")

    source = predictions[[id_column, "y_pred"]].copy()
    source[id_column] = source[id_column].astype(str)

    missing_ids = set(sample[id_column]) - set(source[id_column])
    if missing_ids:
        raise ValueError(f"예측에 없는 ID {len(missing_ids)}개 (예: {sorted(missing_ids)[:5]})")

    merged = sample[[id_column]].merge(source, on=id_column, how="left", validate="one_to_one")
    if len(merged) != len(sample):
        raise ValueError(f"병합 후 행 수가 {len(merged)}, 기대 {len(sample)}")
    if merged["y_pred"].isna().any():
        raise ValueError(f"라벨이 비어 있는 행 {int(merged['y_pred'].isna().sum())}개")

    return merged.rename(columns={"y_pred": label_column})


def write_submission(
    predictions: pd.DataFrame,
    sample_submission_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """제출 csv 를 쓰고 요약을 돌려준다. 파일을 만들 뿐 어디에도 올리지 않는다."""
    submission = build_submission(predictions, sample_submission_path)
    path = save_csv(submission, output_path, encoding=SUBMISSION_ENCODING)
    counts = submission["SUBCLASS"].value_counts()
    return {
        "output_path": str(path),
        "rows": len(submission),
        "n_classes": int(counts.size),
        "top_class": f"{counts.index[0]} ({int(counts.iloc[0])})",
        "rarest_class": f"{counts.index[-1]} ({int(counts.iloc[-1])})",
    }


def average_probabilities(frames: list[pd.DataFrame], classes: list[str]) -> np.ndarray:
    """여러 예측 프레임의 확률을 ID 기준으로 정렬해 평균한다."""
    if not frames:
        raise ValueError("평균낼 프레임이 없다")
    columns = [f"p_{label}" for label in classes]
    base_ids = frames[0]["ID"].astype(str).to_numpy()
    stacked = []
    for frame in frames:
        aligned = frame.set_index(frame["ID"].astype(str)).reindex(base_ids)
        if aligned[columns].isna().any().any():
            raise ValueError("프레임 간 ID 집합이 다르다")
        stacked.append(aligned[columns].to_numpy(dtype=np.float64))
    return np.mean(stacked, axis=0)
