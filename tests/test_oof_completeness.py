"""`artifacts/oof/` 의 OOF 가 블렌딩에 쓸 수 있는 상태인지.

OOF 는 스태킹·그리디의 **입력**이다. 한 fold 가 빠졌거나 ID 가 어긋난 파일이 섞이면
블렌드가 그 행에서 0 확률을 보고 argmax 로 첫 클래스를 고른다 — 에러 없이 점수만 떨어진다.
실제로 `greedy_blend.py` 가 "test 예측이 없는 멤버는 선택 전에 뺀다"는 가드를 갖고 있는 것도
같은 이유다.

파일이 100개가 넘어 전부 읽으면 느리다. 최근 수정된 것부터 표본으로 본다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = PROJECT_ROOT / "artifacts" / "oof"
TEST_DIR = PROJECT_ROOT / "artifacts" / "test_predictions"
FOLDS = PROJECT_ROOT / "data" / "process" / "train_folds.parquet"

#: 매번 100개를 다 읽으면 테스트가 분 단위가 된다. 최근 것부터 이만큼만 본다.
SAMPLE = 12
N_TRAIN = 6201
N_TEST = 2546


def _recent(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:SAMPLE]


def _oof_files() -> list[Path]:
    return _recent(OOF_DIR, "oof_*.csv")


@pytest.mark.parametrize("path", _oof_files(), ids=lambda p: p.name[:44])
def test_oof_covers_every_train_row_once(path: Path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    assert "ID" in frame.columns, f"{path.name} 에 ID 가 없다"
    assert len(frame) == N_TRAIN, f"{path.name} 행 수 {len(frame)}"
    assert frame["ID"].is_unique, f"{path.name} 에 중복 ID"


@pytest.mark.parametrize("path", _oof_files(), ids=lambda p: p.name[:44])
def test_oof_probabilities_are_valid(path: Path):
    """확률이 아니면 블렌드 가중평균이 의미를 잃는다."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    columns = [c for c in frame.columns if c.startswith("p_")]
    assert columns, f"{path.name} 에 p_* 열이 없다"

    values = frame[columns].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), f"{path.name} 에 NaN/inf"
    assert (values >= 0).all(), f"{path.name} 에 음수 확률"
    # 행 합이 1 이어야 한다. 일부 예전 파일이 float32 로 저장돼 있어 여유를 둔다.
    assert np.allclose(values.sum(axis=1), 1.0, atol=1e-4), f"{path.name} 행 합이 1 이 아니다"


@pytest.mark.parametrize("path", _oof_files(), ids=lambda p: p.name[:44])
def test_oof_argmax_matches_recorded_prediction(path: Path):
    """`y_pred` 가 확률과 어긋나면 어느 쪽을 믿어야 할지 알 수 없다."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "y_pred" not in frame.columns:
        pytest.skip(f"{path.name} 에 y_pred 가 없다")
    columns = [c for c in frame.columns if c.startswith("p_")]
    classes = np.asarray([c[len("p_"):] for c in columns])
    recomputed = classes[frame[columns].to_numpy(dtype=np.float64).argmax(axis=1)]
    mismatch = int((recomputed != frame["y_pred"].to_numpy()).sum())
    # 확률이 동점이면 argmax 가 앞 클래스를 고른다 — 그런 행은 몇 개 있을 수 있다.
    assert mismatch <= 5, f"{path.name} 에서 argmax 와 y_pred 가 {mismatch}행 다르다"


def test_oof_ids_match_the_fold_file():
    """OOF 와 fold 의 ID 집합이 다르면 정렬 단계에서 조용히 NaN 이 섞인다."""
    if not FOLDS.exists():
        pytest.skip("train_folds.parquet 없음")
    files = _oof_files()
    if not files:
        pytest.skip("OOF 없음")
    fold_ids = set(pd.read_parquet(FOLDS)["ID"].astype(str))
    for path in files[:4]:
        ids = set(pd.read_csv(path, encoding="utf-8-sig")["ID"].astype(str))
        assert ids == fold_ids, f"{path.name} 의 ID 집합이 fold 파일과 다르다"


@pytest.mark.parametrize("path", _recent(TEST_DIR, "test_*.csv"), ids=lambda p: p.name[:44])
def test_test_predictions_cover_every_test_row(path: Path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    assert len(frame) == N_TEST, f"{path.name} 행 수 {len(frame)}"
    assert frame["ID"].is_unique


def test_every_sampled_oof_has_a_test_counterpart():
    """짝이 없으면 그 멤버가 뽑혀도 제출 파일을 못 만든다 — 그리디가 선택 전에 거른다."""
    files = _oof_files()
    if not files:
        pytest.skip("OOF 없음")
    missing = [p.name for p in files
               if not (TEST_DIR / f"test_{p.stem[len('oof_'):]}.csv").exists()]
    # 블렌드 산출물(raw_blend 등)은 test 짝을 안 만드는 경우가 있어 전부를 강제하진 않는다.
    assert len(missing) <= len(files) // 2, f"test 짝이 없는 OOF 가 많다: {missing[:5]}"
