"""Profile Hash 기반 Group CV 무결성 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from cancer_hack.validation import (
    assign_fold_column,
    fold_class_distribution,
    make_profile_group_kfold,
    make_profile_hash,
    make_stratified_kfold,
)

GENE_COLS = ["TP53", "KRAS", "EGFR"]


# ---------------------------------------------------------------------------
# make_profile_hash
# ---------------------------------------------------------------------------


def test_same_vector_same_hash(toy_frame: pd.DataFrame) -> None:
    """동일 변이 벡터는 동일한 해시를 갖는다."""
    dup = pd.concat([toy_frame, toy_frame.iloc[[0]]], ignore_index=True)
    h = make_profile_hash(dup, GENE_COLS)
    assert h.iloc[0] == h.iloc[-1]


def test_different_vector_different_hash(toy_frame: pd.DataFrame) -> None:
    """변이 벡터가 다르면 해시도 달라야 한다."""
    h = make_profile_hash(toy_frame, GENE_COLS)
    assert h.nunique() == len(toy_frame)


def test_hash_length_matches_dataframe(toy_frame: pd.DataFrame) -> None:
    h = make_profile_hash(toy_frame, GENE_COLS)
    assert len(h) == len(toy_frame)


# ---------------------------------------------------------------------------
# make_stratified_kfold
# ---------------------------------------------------------------------------


def _big_frame(n: int = 120) -> pd.DataFrame:
    """StratifiedKFold가 동작하는 최소 크기 프레임(클래스당 ≥ n_splits 행)."""
    classes = ["BRCA", "ACC", "DLBC", "THYM", "LUAD", "STAD"]
    rows = []
    for i in range(n):
        cls = classes[i % len(classes)]
        rows.append(
            {
                "ID": f"s{i}",
                "SUBCLASS": cls,
                "TP53": f"M{i}",
                "KRAS": "WT",
                "EGFR": "WT",
            }
        )
    return pd.DataFrame(rows)


def test_stratified_kfold_covers_all_rows() -> None:
    """모든 행이 정확히 1번 valid Fold에 속해야 한다."""
    df = _big_frame()
    valid_indices: list[pd.Index] = []
    for _, valid_idx in make_stratified_kfold(df, GENE_COLS):
        valid_indices.append(valid_idx)

    all_valid = pd.Index(sorted(idx for fold in valid_indices for idx in fold))
    assert list(all_valid) == list(range(len(df)))


def test_stratified_kfold_no_overlap() -> None:
    """Fold 간 valid 인덱스는 겹치면 안 된다."""
    df = _big_frame()
    seen: set[int] = set()
    for _, valid_idx in make_stratified_kfold(df, GENE_COLS):
        assert seen.isdisjoint(valid_idx), "valid 인덱스가 다른 Fold와 겹침"
        seen.update(valid_idx)


# ---------------------------------------------------------------------------
# make_profile_group_kfold
# ---------------------------------------------------------------------------


def _frame_with_duplicates() -> pd.DataFrame:
    """동일 변이 벡터가 여러 라벨에 걸쳐 반복되는 프레임."""
    base = _big_frame(120)
    # 처음 10행과 같은 변이 벡터를 다른 라벨로 추가
    extra = base.iloc[:10].copy()
    extra["ID"] = [f"dup_{i}" for i in range(10)]
    extra["SUBCLASS"] = "LUAD"
    return pd.concat([base, extra], ignore_index=True)


def test_group_kfold_same_hash_in_same_fold() -> None:
    """동일 profile_hash를 가진 행들은 항상 같은 Fold에 있어야 한다."""
    df = _frame_with_duplicates()
    gene_cols = GENE_COLS
    profile_hash = make_profile_hash(df, gene_cols)

    for fold_num, (train_idx, valid_idx) in enumerate(
        make_profile_group_kfold(df, gene_cols), start=1
    ):
        train_hashes = set(profile_hash.iloc[train_idx])
        valid_hashes = set(profile_hash.iloc[valid_idx])
        overlap = train_hashes & valid_hashes
        assert not overlap, (
            f"Fold {fold_num}: 동일 profile_hash가 Train과 Valid에 동시에 존재 → {overlap}"
        )


def test_group_kfold_covers_all_rows() -> None:
    """모든 행이 정확히 1번 valid Fold에 속해야 한다."""
    df = _frame_with_duplicates()
    valid_indices: list[pd.Index] = []
    for _, valid_idx in make_profile_group_kfold(df, GENE_COLS):
        valid_indices.append(valid_idx)

    all_valid = sorted(idx for fold in valid_indices for idx in fold)
    assert all_valid == list(range(len(df)))


def test_group_kfold_no_overlap() -> None:
    """Fold 간 valid 인덱스는 겹치면 안 된다."""
    df = _frame_with_duplicates()
    seen: set[int] = set()
    for _, valid_idx in make_profile_group_kfold(df, GENE_COLS):
        assert seen.isdisjoint(valid_idx), "valid 인덱스가 다른 Fold와 겹침"
        seen.update(valid_idx)


# ---------------------------------------------------------------------------
# assign_fold_column
# ---------------------------------------------------------------------------


def test_assign_fold_column_adds_fold_column() -> None:
    df = _big_frame()
    result = assign_fold_column(df, GENE_COLS)
    assert "fold" in result.columns


def test_assign_fold_column_range() -> None:
    """fold 값은 1 이상 n_splits 이하여야 한다."""
    df = _big_frame()
    n_splits = 5
    result = assign_fold_column(df, GENE_COLS, n_splits=n_splits)
    assert result["fold"].between(1, n_splits).all()


def test_assign_fold_column_no_unassigned() -> None:
    """-1로 남아있는 행이 없어야 한다."""
    df = _big_frame()
    result = assign_fold_column(df, GENE_COLS)
    assert (result["fold"] == -1).sum() == 0


def test_assign_fold_column_row_count_preserved() -> None:
    """원본 행 수가 유지되어야 한다."""
    df = _big_frame()
    result = assign_fold_column(df, GENE_COLS)
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# fold_class_distribution
# ---------------------------------------------------------------------------


def test_fold_class_distribution_ratio_sums_to_100() -> None:
    """각 Fold 내 클래스 비율의 합은 100이어야 한다."""
    df = _big_frame()
    df_folds = assign_fold_column(df, GENE_COLS)
    dist = fold_class_distribution(df_folds)
    per_fold = dist["ratio"].reset_index().groupby("fold")["ratio"].sum()
    assert (per_fold.round(6) == 100.0).all()
