from __future__ import annotations
from pathlib import Path

import pandas as pd


# DataFrame을 Parquet 파일로 저장하는 공통 저장 함수
def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

# 저장된 Parquet 파일을 읽어 DataFrame으로 반환하는 로딩 함수
def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))
