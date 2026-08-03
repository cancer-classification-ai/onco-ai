"""`run_seed_sweep.py` 의 계약.

이 스크립트의 두 가지 위험은 (1) 기준선 v002 와 다른 설정으로 돌아 비교가 깨지는 것,
(2) 중간에 끊긴 반쪽짜리 파일을 "있음" 으로 세어 조용히 틀린 앙상블을 만드는 것이다.
둘 다 실행 몇 시간 뒤에야 드러나므로 여기서 막는다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_seed_sweep_mod", PROJECT_ROOT / "scripts" / "run_seed_sweep.py"
)
ss = importlib.util.module_from_spec(_spec)
sys.modules["run_seed_sweep_mod"] = ss
_spec.loader.exec_module(ss)


def test_settings_match_the_v002_baseline():
    """v002 = f16 · topk 500 · balanced · 5 fold · xgb+cat+rf · 0.45/0.45/0.10."""
    assert ss.CONFIG == "f16"
    assert ss.TOPK == 500
    assert ss.N_SPLITS == 5
    assert ss.MODELS == ["xgb", "catboost", "rf"]
    assert ss.FIXED_WEIGHTS == [0.45, 0.45, 0.10]
    assert ss.FOLDS_FILE.name == "train_folds.parquet"


def test_fixed_weights_align_with_model_order():
    """가중치가 모델 순서와 어긋나면 RF 에 0.45 가 들어간다 — 조용히 다른 앙상블이 된다."""
    assert len(ss.FIXED_WEIGHTS) == len(ss.MODELS)
    assert abs(sum(ss.FIXED_WEIGHTS) - 1.0) < 1e-9
    assert ss.FIXED_WEIGHTS[ss.MODELS.index("rf")] == 0.10


def test_seed_list_starts_with_the_baseline_seed():
    """seed 3개를 쓸 때 v002 의 42 가 반드시 포함돼야 기준선과 이어진다."""
    assert ss.SEEDS[0] == 42
    assert len(ss.SEEDS) == 5
    assert len(set(ss.SEEDS)) == 5


def test_both_cv_splits_are_covered():
    assert set(ss.CV_KEYS) == {"skf5", "group5"}
    assert ss.CV_KEYS["group5"] == "sgkf"  # train_gbdt 의 --cv 값
    assert ss.CV_KEYS["skf5"] == "skf"


def _write(path: Path, rows: int, cols: int = 26, id_prefix: str = "TRAIN") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = [f"p_C{i}" for i in range(cols)]
    frame = pd.DataFrame({"ID": [f"{id_prefix}_{i:04d}" for i in range(rows)]})
    for c in classes:
        frame[c] = 1.0 / cols
    frame.to_csv(path, index=False)


@pytest.fixture
def fake_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "ARTIFACTS", tmp_path)
    return tmp_path


def test_complete_pair_is_detected(fake_artifacts):
    oof, test = ss.paths_for("xgb", "group5", 42)
    _write(oof, 6201)
    _write(test, 2546, id_prefix="TEST")
    assert ss.is_complete("xgb", "group5", 42) is True


def test_truncated_oof_is_rejected(fake_artifacts):
    """끊긴 실행이 남긴 반쪽 파일 — 이걸 '있음' 으로 세면 앙상블이 조용히 틀어진다."""
    oof, test = ss.paths_for("xgb", "group5", 42)
    _write(oof, 3000)
    _write(test, 2546, id_prefix="TEST")
    assert ss.is_complete("xgb", "group5", 42) is False


def test_missing_test_prediction_is_rejected(fake_artifacts):
    """OOF 만 있고 test 예측이 없으면 제출을 못 만든다."""
    oof, _ = ss.paths_for("xgb", "group5", 42)
    _write(oof, 6201)
    assert ss.is_complete("xgb", "group5", 42) is False


def test_wrong_class_count_is_rejected(fake_artifacts):
    oof, test = ss.paths_for("xgb", "group5", 42)
    _write(oof, 6201, cols=20)
    _write(test, 2546, cols=20, id_prefix="TEST")
    assert ss.is_complete("xgb", "group5", 42) is False


def test_nothing_present_means_all_missing(fake_artifacts):
    have, missing = ss.inventory()
    assert have == {}
    assert len(missing) == len(ss.SEEDS) * len(ss.MODELS) * len(ss.CV_KEYS) == 30


def test_paths_encode_model_split_and_seed(fake_artifacts):
    oof, test = ss.paths_for("catboost", "skf5", 2024)
    assert "catboost" in oof.name and "skf5" in oof.name and "_s2024." in oof.name
    assert oof.name.startswith("oof_") and test.name.startswith("test_")
