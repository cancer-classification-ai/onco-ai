"""캐시된 산출물이 **같은 입력에서 나온 것인지** 확인하는 지문.

왜 필요한가
-----------
`artifacts/oof/` 에 쌓인 예측을 나중에 블렌딩하는데, 그 예측들은 각자 만들어질 때의
fold 분할에 묶여 있다. `data/process/train_folds.parquet` 를 다시 만들면 fold 경계가
바뀌고, 그러면 **예전 OOF 와 새 OOF 를 섞는 순간 교차적합 보정이 valid fold 를 보게 된다.**

이 사고는 조용하다. 파일 이름도 그대로고 행 수도 그대로라 점수만 살짝 좋아진다.
그래서 fold 를 읽는 자리마다 지문을 대조한다.

파일 바이트가 아니라 **내용**의 지문이다. parquet 은 같은 데이터라도 압축·메타데이터가
달라지면 바이트가 바뀌므로, 바이트 해시를 박아 두면 멀쩡한 재생성에도 깨진다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

#: 현재 `artifacts/` 의 모든 OOF·test 예측이 이 fold 분할에서 나왔다.
#: 2026-07-31 생성(seed 42 · n_splits 5 · 6,201행 · 5,636그룹) 이후 바뀐 적이 없다.
#: 값이 달라졌다면 fold 를 다시 만든 것이고, 그 순간 기존 예측과는 섞으면 안 된다.
EXPECTED_FOLD_FINGERPRINT = "997d89a20595cc23"

FOLD_COLUMNS = ("ID", "fold_skf5", "fold_group5")


def frame_fingerprint(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """열 몇 개의 내용을 16자 지문으로. 행 순서에 의존하지 않는다."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"지문에 필요한 열이 없다: {missing}")
    ordered = frame[list(columns)].astype(str).sort_values(list(columns), kind="stable")
    payload = "\n".join(",".join(row) for row in ordered.itertuples(index=False, name=None))
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def fold_fingerprint(folds_path: str | Path) -> str:
    return frame_fingerprint(pd.read_parquet(folds_path), FOLD_COLUMNS)


def check_fold_fingerprint(folds_path: str | Path, expected: str | None = None) -> str:
    """fold 파일이 기대한 분할인지 확인한다. 다르면 예외를 낸다.

    캐시된 예측을 재사용하기 전에 부른다. 새로 다 학습할 거라면 굳이 막을 이유는
    없지만, 그때도 기존 기록과 점수를 비교할 수 없다는 건 알아야 한다.
    """
    expected = expected or EXPECTED_FOLD_FINGERPRINT
    actual = fold_fingerprint(folds_path)
    if actual != expected:
        raise ValueError(
            f"fold 분할이 기대와 다르다 (기대 {expected} · 현재 {actual}).\n"
            f"  {folds_path}\n"
            "  기존 artifacts/oof 의 예측은 예전 분할에서 나왔으므로 섞으면 안 된다. "
            "전부 다시 학습하거나 fold 파일을 되돌린다."
        )
    return actual
