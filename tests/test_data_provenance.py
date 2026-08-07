"""캐시된 예측이 **같은 fold 분할**에서 나온 것인지 지킨다.

`artifacts/oof/` 의 예측 100여 개는 각자 만들어질 때의 fold 분할에 묶여 있다.
`scripts/make_folds.py` 를 다시 돌려 `data/process/train_folds.parquet` 가 바뀌면 fold
경계가 달라지고, 그때부터 **예전 OOF 와 새 OOF 를 섞는 순간 교차적합 보정이 valid fold 를
보게 된다.** 파일 이름도 행 수도 그대로라 점수만 조용히 좋아진다.

그래서 fold 파일의 내용 지문을 여기 박아 둔다. 이 테스트가 깨졌다면 둘 중 하나다 —
fold 를 다시 만들었거나(그러면 기존 예측을 전부 버리고 다시 학습해야 한다), 아니면
다른 사람의 fold 파일을 받아 온 것이다. 어느 쪽이든 그냥 지문만 고치면 안 된다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cancer_hack.provenance import (
    EXPECTED_FEATURE_FINGERPRINTS,
    EXPECTED_FOLD_FINGERPRINT,
    check_feature_fingerprints,
    check_fold_fingerprint,
    fold_fingerprint,
    frame_fingerprint,
    parquet_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS = PROJECT_ROOT / "data" / "process"
FOLDS = PROCESS / "train_folds.parquet"


def _require_folds() -> Path:
    if not FOLDS.exists():
        pytest.skip("train_folds.parquet 없음 — scripts/make_folds.py 를 먼저 돌린다")
    return FOLDS


def test_fold_split_is_the_one_the_artifacts_were_built_with():
    assert fold_fingerprint(_require_folds()) == EXPECTED_FOLD_FINGERPRINT, (
        "fold 분할이 바뀌었다. artifacts/oof 의 예측을 재사용하면 안 된다 — "
        "docstring 을 읽고 결정한다."
    )


def test_fold_file_shape():
    folds = pd.read_parquet(_require_folds())
    assert len(folds) == 6201
    assert {"ID", "fold_skf5", "fold_group5"} <= set(folds.columns)
    assert sorted(folds["fold_group5"].unique().tolist()) == [0, 1, 2, 3, 4]
    assert folds["ID"].is_unique


def test_cached_oof_ids_match_the_fold_file():
    """OOF 와 fold 의 ID 집합이 다르면 정렬 단계에서 조용히 NaN 이 섞인다."""
    folds = pd.read_parquet(_require_folds())
    oof_dir = PROJECT_ROOT / "artifacts" / "oof"
    samples = sorted(oof_dir.glob("oof_*_f16_group5_*.csv"))[:3]
    if not samples:
        pytest.skip("f16 group5 OOF 캐시 없음")
    fold_ids = set(folds["ID"].astype(str))
    for path in samples:
        ids = set(pd.read_csv(path, encoding="utf-8-sig")["ID"].astype(str))
        assert ids == fold_ids, f"{path.name} 의 ID 집합이 fold 파일과 다르다"


def test_fingerprint_ignores_row_order():
    """행 순서로 지문이 바뀌면 parquet 을 다시 써 넣기만 해도 깨진다."""
    frame = pd.DataFrame({"ID": ["b", "a", "c"], "fold_skf5": [1, 0, 2], "fold_group5": [2, 1, 0]})
    columns = ("ID", "fold_skf5", "fold_group5")
    assert frame_fingerprint(frame, columns) == frame_fingerprint(frame.iloc[::-1], columns)


def test_fingerprint_changes_when_a_fold_moves():
    """한 행의 fold 가 바뀌면 지문도 바뀌어야 한다 — 안 그러면 감시 장치가 아니다."""
    frame = pd.DataFrame({"ID": ["a", "b"], "fold_skf5": [0, 1], "fold_group5": [0, 1]})
    moved = frame.copy()
    moved.loc[0, "fold_group5"] = 3
    columns = ("ID", "fold_skf5", "fold_group5")
    assert frame_fingerprint(frame, columns) != frame_fingerprint(moved, columns)


def test_feature_parquets_are_the_ones_the_artifacts_were_built_with():
    """피처 캐시가 바뀌면 기존 OOF 와 새 예측을 섞을 수 없다.

    특히 `mutation_encoded` 는 **지금 코드와 이미 어긋나 있다.** develop 을 머지하면서
    `features_basic.encode_mutation` 이 `*931*` 같은 동의 정지코돈을 2 가 아니라 1 로
    세도록 바뀌었는데(정정이 맞다) 이 파켓은 그 전 코드로 만들어졌다. 그래서
    `make_features.py` 를 다시 돌리면 enc3·comut·lsvd·lnmf·gmod 가 전부 달라진다.
    재생성은 전부 다시 학습할 때만 한다.
    """
    if not PROCESS.exists():
        pytest.skip("data/process 없음")
    missing = [n for n in EXPECTED_FEATURE_FINGERPRINTS if not (PROCESS / n).exists()]
    if missing:
        pytest.skip(f"피처 캐시 {len(missing)}개 없음 (예: {missing[0]})")
    check_feature_fingerprints(PROCESS)


def test_parquet_fingerprint_ignores_row_order(tmp_path: Path):
    frame = pd.DataFrame({"ID": ["b", "a", "c"], "x": [1, 2, 3], "y": [4, 5, 6]})
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    frame.to_parquet(a)
    frame.iloc[::-1].to_parquet(b)
    assert parquet_fingerprint(a) == parquet_fingerprint(b)


def test_parquet_fingerprint_catches_a_changed_value(tmp_path: Path):
    frame = pd.DataFrame({"ID": ["a", "b"], "x": [1, 2]})
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    frame.to_parquet(a)
    frame.assign(x=[1, 3]).to_parquet(b)
    assert parquet_fingerprint(a) != parquet_fingerprint(b)


def test_feature_check_raises_with_an_actionable_message(tmp_path: Path):
    pd.DataFrame({"ID": ["a"], "x": [1]}).to_parquet(tmp_path / "toy.parquet")
    with pytest.raises(ValueError, match="피처 파켓이 예측을 만들 때와 다르다"):
        check_feature_fingerprints(tmp_path, {"toy.parquet": "0000000000000000"})


def test_check_raises_with_an_actionable_message(tmp_path: Path):
    folds = pd.read_parquet(_require_folds()).copy()
    folds.loc[0, "fold_group5"] = (folds.loc[0, "fold_group5"] + 1) % 5
    path = tmp_path / "train_folds.parquet"
    folds.to_parquet(path)
    with pytest.raises(ValueError, match="fold 분할이 기대와 다르다"):
        check_fold_fingerprint(path)
