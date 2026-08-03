"""Profile Hash 기반 Group CV 무결성 테스트."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from cancer_hack.validation import (
    FOLD_META_COLUMNS,
    assign_fold_column,
    build_fold_frame,
    fold_class_distribution,
    fold_column,
    make_profile_group_kfold,
    make_profile_hash,
    make_stratified_kfold,
)

GENE_COLS = ["TP53", "KRAS", "EGFR"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


# ---------------------------------------------------------------------------
# fold_column · build_fold_frame — fold 파일의 계약
#
# `make_folds.py` 가 쓰고 `train_gbdt.py` 가 읽는다. 두 스크립트가 같은 열 이름을
# 보게 하는 게 이 블록의 목적이다.
# ---------------------------------------------------------------------------


def test_fold_column_matches_existing_artifact_names() -> None:
    """기본 5-fold 이름은 이미 쌓인 아티팩트 규약과 같아야 한다.

    `artifacts/oof/` 의 예측이 전부 이 열 이름으로 만들어졌다. 바뀌면 과거 점수와
    비교가 끊긴다.
    """
    assert fold_column("skf") == "fold_skf5"
    assert fold_column("sgkf") == "fold_group5"


def test_fold_column_follows_n_splits() -> None:
    """분할 수가 이름에 반영돼야 10-fold 파일이 5-fold 인 척하지 않는다."""
    assert fold_column("skf", 10) == "fold_skf10"
    assert fold_column("sgkf", 3) == "fold_group3"


def test_fold_column_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        fold_column("kfold")  # type: ignore[arg-type]


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """`build_fold_frame` 이 읽을 최소 원본 csv."""
    path = tmp_path / "train.csv"
    _frame_with_duplicates().to_csv(path, index=False)
    return path


def test_build_fold_frame_columns(raw_csv: Path) -> None:
    """열 구성과 순서가 계약이다 — train_gbdt 가 이 이름으로 찾는다."""
    frame = build_fold_frame(raw_csv)
    assert list(frame.columns) == [
        *FOLD_META_COLUMNS,
        fold_column("skf"),
        fold_column("sgkf"),
    ]


def test_build_fold_frame_is_zero_based_and_covers_every_row(raw_csv: Path) -> None:
    """0-based 여야 `train_gbdt` 의 `range(n_splits)` 루프가 맞물린다."""
    n_splits = 5
    frame = build_fold_frame(raw_csv, n_splits=n_splits)
    for kind in ("skf", "sgkf"):
        values = frame[fold_column(kind, n_splits)]
        assert set(values) == set(range(n_splits)), f"{kind} 가 0..{n_splits - 1} 이 아니다"


def test_build_fold_frame_preserves_csv_row_order(raw_csv: Path) -> None:
    """피처 parquet 들이 원본 순서라 fold 도 같은 순서여야 한다."""
    frame = build_fold_frame(raw_csv)
    expected = pd.read_csv(raw_csv, usecols=["ID"], dtype=str)["ID"].tolist()
    assert frame["ID"].tolist() == expected


def test_build_fold_frame_group_fold_has_no_leakage(raw_csv: Path) -> None:
    """같은 프로파일이 train 과 valid 로 갈리면 안 된다."""
    frame = build_fold_frame(raw_csv)
    column = fold_column("sgkf")
    for fold in sorted(frame[column].unique()):
        valid = set(frame.loc[frame[column] == fold, "group_key"])
        train = set(frame.loc[frame[column] != fold, "group_key"])
        assert not (valid & train), f"fold {fold} 에 걸친 그룹 {valid & train}"


def test_build_fold_frame_is_deterministic(raw_csv: Path) -> None:
    """같은 시드면 같은 분할. 재현이 안 되면 fold 파일을 정본으로 둘 이유가 없다."""
    first = build_fold_frame(raw_csv, seed=42)
    second = build_fold_frame(raw_csv, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_build_fold_frame_seed_changes_split(raw_csv: Path) -> None:
    """시드가 실제로 먹는지 — 안 먹으면 위 결정성 테스트가 무의미하다."""
    first = build_fold_frame(raw_csv, seed=42)
    other = build_fold_frame(raw_csv, seed=7)
    column = fold_column("skf")
    assert not first[column].equals(other[column])


def test_train_gbdt_does_not_create_folds() -> None:
    """학습 스크립트는 fold 를 **읽기만** 해야 한다.

    예전에는 파일이 없으면 제 손으로 만들어 저장했다. 그러면 같은 이름의 파일이 두
    경로에서 나오고 `artifacts/oof/` 의 예측이 어느 분할에서 나왔는지 사후에
    확인할 수 없다. 소스에서 생성 함수 호출을 직접 막는다.
    """
    source = (PROJECT_ROOT / "scripts/train_gbdt.py").read_text(encoding="utf-8")
    for forbidden in ("fold_assignment", "build_fold_frame", "assign_fold_column"):
        assert forbidden not in source, (
            f"train_gbdt.py 가 {forbidden} 를 부른다 — fold 생성은 make_folds.py 담당이다"
        )


def test_make_folds_cli_handles_external_input_path(tmp_path: Path, monkeypatch) -> None:
    """`--input` 이 저장소 밖 경로면 크래시하지 않고 provenance `source` 에 파일명만 남는다.

    RF 작업(Ticket 1b)의 원본 데이터는 저장소 밖 외부 디렉터리에 있다. 예전에는
    `args.input.resolve().relative_to(PROJECT_ROOT)` 가 저장소 밖 경로에서 `ValueError`
    를 내(parquet 저장 후, meta json 작성 전) 크래시했다 — 이 회귀를 잡는다.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import make_folds

    raw_path = tmp_path / "train.csv"
    _frame_with_duplicates().to_csv(raw_path, index=False)
    out_path = tmp_path / "train_folds.parquet"

    # `GROUP_CACHE` 는 CLI 인자가 아니라 모듈 상수(실제 repo 의 data/process/ 를
    # 가리킨다) — 여기서 리다이렉트하지 않으면 테스트가 실제 데이터용 캐시를
    # 합성 ID(`s0`..)로 덮어써 버린다.
    monkeypatch.setattr(make_folds, "GROUP_CACHE", tmp_path / "group_keys_cache.parquet")
    monkeypatch.setattr(
        sys, "argv", ["make_folds.py", "--input", str(raw_path), "--out", str(out_path)]
    )
    make_folds.main()

    assert out_path.exists()
    meta = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["source"] == "train.csv"
    assert str(tmp_path) not in json.dumps(meta)


def test_train_gbdt_does_not_write_full_train_module_map() -> None:
    """모듈맵은 **fold 별로만** 쓴다.

    전체 train 으로 fit 한 모듈맵 자체는 규정 위반이 아니다(test 를 안 본다). 다만
    fold 를 넘는 산출물이라 나중에 누가 무심코 test 예측에 재사용하기 쉽고, 그 순간
    fold-fit-only 규율이 조용히 깨진다. fold 생성 금지와 같은 방식으로 소스에서 막고,
    진단용 전체 맵이 필요하면 scripts/inspect_latent.py 가 따로 만든다.
    """
    source = (PROJECT_ROOT / "scripts/train_gbdt.py").read_text(encoding="utf-8")
    assert "full_train_module_map" not in source, (
        "train_gbdt.py 가 전체 train 모듈맵을 쓴다 — fold 별 맵만 남긴다"
    )
