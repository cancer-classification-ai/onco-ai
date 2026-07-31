from __future__ import annotations

from typing import Iterator

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

FoldIndex = tuple[pd.Index, pd.Index]


def make_profile_hash(df: pd.DataFrame, gene_columns: list[str]) -> pd.Series:
    """유전자 변이 벡터를 해시로 변환해 동일 변이 프로파일을 식별한다.

    동일한 유전자 변이 벡터를 가진 행은 같은 해시값을 가진다.
    StratifiedGroupKFold의 groups 인자로 직접 사용할 수 있다.
    """
    return (
        pd.util.hash_pandas_object(df[gene_columns], index=False)
        .astype(str)
        .rename("profile_hash")
        .reset_index(drop=True)
    )


def make_stratified_kfold(
    df: pd.DataFrame,
    gene_columns: list[str],
    label_column: str = "SUBCLASS",
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[FoldIndex]:
    """일반 StratifiedKFold를 생성한다.

    샘플 단위 예측 성능 측정에 사용한다.
    Profile-Grouped CV와 점수 차이를 비교해 데이터 누수 규모를 확인한다.

    Yields:
        (train_index, valid_index) — DataFrame의 정수 위치 인덱스 쌍
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = df[gene_columns].reset_index(drop=True)
    y = df[label_column].reset_index(drop=True)

    for train_pos, valid_pos in skf.split(X, y):
        yield X.index[train_pos], X.index[valid_pos]


def make_profile_group_kfold(
    df: pd.DataFrame,
    gene_columns: list[str],
    label_column: str = "SUBCLASS",
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[FoldIndex]:
    """Profile Hash 기반 StratifiedGroupKFold를 생성한다.

    동일 변이 벡터(profile_hash)를 가진 행들을 항상 같은 Fold에 배치한다.
    Train Fold와 Validation Fold 사이에 동일 변이 벡터가 섞이지 않아
    CV 점수의 낙관적 편향을 방지한다.

    모든-WT 그룹(94행)처럼 매우 큰 그룹이 존재하므로
    Fold별 클래스 분포가 완벽히 균등하지 않을 수 있다.
    반드시 fold_class_distribution()으로 실제 분포를 확인해야 한다.

    Yields:
        (train_index, valid_index) — DataFrame의 정수 위치 인덱스 쌍
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    X = df[gene_columns].reset_index(drop=True)
    y = df[label_column].reset_index(drop=True)
    groups = make_profile_hash(df, gene_columns)

    for train_pos, valid_pos in sgkf.split(X, y, groups=groups):
        yield X.index[train_pos], X.index[valid_pos]


def assign_fold_column(
    df: pd.DataFrame,
    gene_columns: list[str],
    label_column: str = "SUBCLASS",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """각 행에 fold 번호(1-based)를 부여한 DataFrame을 반환한다.

    반환된 DataFrame은 원본과 같은 행 순서를 유지하며
    'fold' 컬럼(int, 1~n_splits)이 추가된다.
    make_folds.py 스크립트에서 Parquet로 저장하는 용도로 사용한다.
    """
    result = df.reset_index(drop=True).copy()
    result["fold"] = -1

    for fold_num, (_, valid_pos) in enumerate(
        make_profile_group_kfold(
            df,
            gene_columns=gene_columns,
            label_column=label_column,
            n_splits=n_splits,
            random_state=random_state,
        ),
        start=1,
    ):
        result.loc[valid_pos, "fold"] = fold_num

    return result


def fold_class_distribution(
    df: pd.DataFrame,
    label_column: str = "SUBCLASS",
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Fold별 클래스 분포를 비율(%)로 반환한다.

    assign_fold_column() 이후 반드시 호출해 Fold 균형을 확인한다.
    반환값: MultiIndex(fold, SUBCLASS) → count, ratio(%) DataFrame
    """
    counts = (
        df.groupby([fold_column, label_column])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["ratio"] = counts.groupby(fold_column)["count"].transform(
        lambda s: s / s.sum() * 100
    )
    return counts.set_index([fold_column, label_column])
