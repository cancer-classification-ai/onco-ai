from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_xgb_feature_combinations as runner  # noqa: E402


def _result(config: str, cv: str, score: float) -> dict:
    return {
        "config": config,
        "cv": cv,
        "blocks": list(runner.tg.CONFIGS[config]["blocks"]),
        "n_features": 100,
        "oof_macro_f1": score,
        "oof_macro_f1_singleton": score - 0.01,
        "oof_accuracy": score - 0.02,
        "elapsed_seconds": 1.0,
        "device": "cpu",
        "stem": f"{config}_{cv}",
    }


def test_suites_have_expected_unique_case_counts() -> None:
    assert len(runner.FACTORIAL_CONFIGS) == 8
    assert len(runner.CANDIDATE_CONFIGS) == 12
    assert len(runner.SUITES["all"]) == 19
    assert len(set(runner.SUITES["all"])) == 19


def test_comparison_reports_factorial_and_candidate_deltas() -> None:
    configs = runner.SUITES["all"]
    scores = {
        config: 0.40 + index / 100
        for index, config in enumerate(configs)
    }
    results = [
        _result(config, cv, scores[config] + (0.001 if cv == "sgkf" else 0.0))
        for config in configs
        for cv in runner.CVS
    ]
    frame = runner.comparison_frame(results)
    assert len(frame) == 38

    factorial = frame[
        (frame["config"] == "f4r_ebbnb") & (frame["cv"] == "skf")
    ].iloc[0]
    expected_factorial = scores["f4r_ebbnb"] - scores["r16_ebbnb"]
    assert factorial["delta_enc3_same_evidence"] == pytest.approx(
        expected_factorial
    )

    candidate = frame[
        (frame["config"] == "full_all_ebbnb") & (frame["cv"] == "sgkf")
    ].iloc[0]
    expected_candidate = scores["full_all_ebbnb"] - scores["full_all"]
    assert candidate["anchor_config"] == "full_all"
    assert candidate["delta_vs_anchor"] == pytest.approx(expected_candidate)
