"""`tune_sweep.py` 의 채택 판정 계약.

이 스크립트의 존재 이유는 하나다 — **단일 seed 최고점을 그대로 채택하지 않는 것.**
`research/10` §6 에서 저장된 study 37 trial 의 group5 최대 이득이 +0.0021 이었는데,
seed 표준편차 0.0036 으로 37번 뽑으면 실제 개선이 0 이어도 0.0097 이 나온다.
판정 규칙이 조용히 느슨해지면 그 함정으로 되돌아간다.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tune_sweep_mod", PROJECT_ROOT / "scripts" / "tune_sweep.py"
)
sw = importlib.util.module_from_spec(_spec)
sys.modules["tune_sweep_mod"] = sw
_spec.loader.exec_module(sw)


def _row(trial, mean, sd, baseline=False):
    return {"trial": trial, "params": {"x": trial}, "is_baseline": baseline,
            "sgkf_mean": mean, "sgkf_sd": sd, "skf_mean": mean - 0.02, "skf_sd": sd,
            "gap_mean": 0.3, "search_objective": mean}


def test_rejects_gain_smaller_than_seed_noise():
    """이게 이 스크립트의 핵심 — 델타가 seed 표준편차 안이면 기각한다."""
    rows = [_row(0, 0.4786, 0.0036, baseline=True), _row(11, 0.4807, 0.0036)]
    v = sw.verdict(rows)
    assert v["adopt"] is False
    assert v["sgkf_delta"] < v["threshold_sd"]


def test_adopts_gain_clearly_above_seed_noise():
    rows = [_row(0, 0.4786, 0.0010, baseline=True), _row(9, 0.4900, 0.0010)]
    v = sw.verdict(rows)
    assert v["adopt"] is True
    assert v["trial"] == 9


def test_uses_the_larger_of_the_two_standard_deviations():
    """후보가 유난히 불안정하면 그 불안정성으로 판정해야 한다."""
    rows = [_row(0, 0.4786, 0.0010, baseline=True), _row(5, 0.4850, 0.0090)]
    v = sw.verdict(rows)
    assert v["threshold_sd"] == 0.0090
    assert v["adopt"] is False


def test_picks_best_candidate_by_sgkf_not_search_objective():
    """탐색 목적값은 페널티가 섞여 있다. 재검증 판정은 sgkf 평균으로 한다."""
    rows = [
        _row(0, 0.4700, 0.0010, baseline=True),
        _row(1, 0.4900, 0.0010),
        _row(2, 0.4750, 0.0010),
    ]
    rows[2]["search_objective"] = 99.0  # 탐색에서는 2번이 1등이었다고 치자
    assert sw.verdict(rows)["trial"] == 1


def test_no_baseline_is_reported_not_silently_adopted():
    v = sw.verdict([_row(1, 0.49, 0.001)])
    assert v["adopt"] is None
    assert "기준선" in v["reason"]


def test_no_candidates_is_reported():
    v = sw.verdict([_row(0, 0.4786, 0.001, baseline=True)])
    assert v["adopt"] is None


def test_top_trials_always_keeps_baseline(tmp_path: Path):
    """기준선이 상위 K 밖으로 밀려도 빠지면 안 된다 — 비교 대상이 사라진다."""
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"trials": [
        {"number": 0, "value": 0.40, "params": {"a": 1}, "user_attrs": {"note": "baseline"}},
        {"number": 1, "value": 0.50, "params": {"a": 2}, "user_attrs": {}},
        {"number": 2, "value": 0.49, "params": {"a": 3}, "user_attrs": {}},
    ]}), encoding="utf-8")
    picked = sw.top_trials(summary, k=2)
    assert [t["number"] for t in picked] == [1, 2, 0]


def test_top_trials_skips_failed_trials(tmp_path: Path):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"trials": [
        {"number": 0, "value": None, "params": {}, "user_attrs": {}},
        {"number": 1, "value": 0.5, "params": {"a": 1}, "user_attrs": {}},
    ]}), encoding="utf-8")
    assert [t["number"] for t in sw.top_trials(summary, k=5)] == [1]


def test_missing_summary_is_not_fatal(tmp_path: Path):
    """한 모델의 탐색이 죽어도 나머지 모델은 계속 가야 한다."""
    assert sw.top_trials(tmp_path / "nope.json", k=3) == []


def test_default_objective_is_sgkf():
    """v002 가 group5 로 선택됐다. 두 분할 평균은 skf/sgkf 상충을 가린다."""
    assert sw.build_parser().parse_args([]).objective == "sgkf"


def test_default_config_is_the_v002_feature_set():
    assert sw.build_parser().parse_args([]).config == "f16"


def test_gap_penalty_is_on_by_default_here():
    """tune_optuna 는 기본 0(기존 동작 보존)이지만, 이 스윕은 과적합 억제가 목적이다."""
    assert sw.build_parser().parse_args([]).gap_penalty > 0
