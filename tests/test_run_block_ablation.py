"""대형 피처 블록 drop/add 실험의 비교 계약."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_block_ablation_mod", PROJECT_ROOT / "scripts" / "run_block_ablation.py"
)
ba = importlib.util.module_from_spec(_spec)
sys.modules["run_block_ablation_mod"] = ba
_spec.loader.exec_module(ba)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_gbdt as tg  # noqa: E402


def specs():
    return ba.experiment_configs(tuple(tg.CONFIGS["f16"]["blocks"]))


def test_full_is_exactly_f16_and_uses_the_same_weighting():
    matrix = specs()
    assert matrix["ba_full"]["blocks"] == tg.CONFIGS["f16"]["blocks"]
    assert matrix["ba_full"]["weight"] == tg.CONFIGS["f16"]["weight"] == "balanced"


def test_baseline_removes_only_the_six_declared_large_blocks():
    matrix = specs()
    full = matrix["ba_full"]["blocks"]
    baseline = matrix["ba_base"]["blocks"]
    assert set(full) - set(baseline) == set(ba.TARGET_BLOCKS)
    assert [block for block in full if block not in ba.TARGET_BLOCKS] == list(baseline)


def test_each_drop_changes_exactly_one_block():
    matrix = specs()
    full = matrix["ba_full"]["blocks"]
    for block in ba.TARGET_BLOCKS:
        dropped = matrix[f"ba_d_{block}"]["blocks"]
        assert set(full) - set(dropped) == {block}
        assert [value for value in full if value != block] == list(dropped)


def test_each_add_changes_exactly_one_block_and_preserves_full_order():
    matrix = specs()
    full = matrix["ba_full"]["blocks"]
    baseline = matrix["ba_base"]["blocks"]
    for block in ba.TARGET_BLOCKS:
        added = matrix[f"ba_a_{block}"]["blocks"]
        assert set(added) - set(baseline) == {block}
        assert [value for value in full if value in set(baseline) | {block}] == list(added)


def test_stage_sizes_and_order_are_fixed():
    matrix = specs()
    assert len(matrix) == 14
    assert ba.phase_names(matrix, "drop")[0] == "ba_full"
    assert ba.phase_names(matrix, "add")[0] == "ba_base"
    assert len(ba.phase_names(matrix, "drop")) == 7
    assert len(ba.phase_names(matrix, "add")) == 7
    assert len(ba.phase_names(matrix, "both")) == 14


def _result(score: float, folds: list[float], classes: dict[str, float]) -> dict:
    return {
        "oof_macro_f1": score,
        "oof_macro_f1_singleton": score - 0.01,
        "fold_macro_f1": folds,
        "per_class_f1": classes,
        "n_features": 100,
        "elapsed_seconds": 10.0,
    }


def test_summary_uses_full_minus_drop_and_add_minus_baseline():
    results = {
        "ba_full": _result(0.57, [0.56, 0.58], {"A": 0.6}),
        "ba_base": _result(0.54, [0.53, 0.55], {"A": 0.5}),
    }
    for index, block in enumerate(ba.TARGET_BLOCKS):
        results[f"ba_d_{block}"] = _result(
            0.56 - index * 0.001, [0.55, 0.57], {"A": 0.55}
        )
        results[f"ba_a_{block}"] = _result(
            0.545 + index * 0.001, [0.535, 0.555], {"A": 0.52}
        )
    summary = ba.build_summary(results, model="catboost", tag="t", cv="sgkf", seed=42)
    by_block = {row["block"]: row for row in summary["rows"]}
    first = by_block[ba.TARGET_BLOCKS[0]]
    assert abs(first["removal_importance"] - 0.01) < 1e-12
    assert abs(first["addition_gain"] - 0.005) < 1e-12
    assert abs(first["composite"] - 0.015) < 1e-12
    assert first["removal_fold_delta"] == pytest.approx([0.01, 0.01])
    assert first["addition_fold_delta"] == pytest.approx([0.005, 0.005])
