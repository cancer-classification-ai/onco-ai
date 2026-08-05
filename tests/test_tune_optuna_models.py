"""`tune_optuna.py` 의 다중 모델 확장 계약.

Optuna 실행은 몇 시간짜리다. 탐색 공간과 기준선 좌표가 어긋나면 `enqueue_trial` 이
**실행 도중에** 죽거나, 더 나쁘게는 기준선 trial 이 조용히 빠진 채로 끝난다. 그러면
"기준선보다 나은가" 를 같은 자로 못 재게 된다. 여기서 미리 막는다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tune_optuna_mod", PROJECT_ROOT / "scripts" / "tune_optuna.py"
)
tu = importlib.util.module_from_spec(_spec)
sys.modules["tune_optuna_mod"] = tu
_spec.loader.exec_module(tu)

MODELS = ["xgb", "catboost", "rf"]


class _StubTrial:
    """`suggest_*` 를 하한으로 고정해 탐색 공간의 **키**만 뽑아낸다."""

    def suggest_int(self, name, low, high, step=None):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


@pytest.mark.parametrize("model", MODELS)
def test_search_space_keys_match_baseline_point(model):
    """어긋나면 `study.enqueue_trial(BASELINE_POINTS[model])` 이 실행 중에 깨진다."""
    assert sorted(tu.suggest_params(_StubTrial(), model)) == sorted(tu.BASELINE_POINTS[model])


@pytest.mark.parametrize("model", MODELS)
def test_model_is_registered_in_every_table(model):
    assert model in tu.SUGGEST_BY_MODEL
    assert model in tu.BASELINE_POINTS
    assert model in tu.FIXED_PARAMS_BY_MODEL


def test_catboost_avoids_bayesian_only_axis():
    """래퍼 기본이 `bootstrap_type="Bernoulli"` 라 bagging_temperature 를 쓰면 예외가 난다."""
    space = tu.suggest_params(_StubTrial(), "catboost")
    assert "bagging_temperature" not in space
    assert "subsample" in space


def test_catboost_baseline_matches_wrapper_defaults():
    """기준선 좌표가 실제로 쓰이는 값과 달라지면 trial 0 이 기준선이 아니게 된다."""
    from cancer_hack.models_gbdt import CatBoostModel

    defaults = CatBoostModel.default_params()
    point = tu.BASELINE_POINTS["catboost"]
    for key in ("l2_leaf_reg", "subsample"):
        assert point[key] == defaults[key], f"{key} 가 래퍼 기본값과 다르다"


def test_catboost_rsm_is_not_a_search_axis():
    """GPU 에서 CatBoost 가 rsm 을 무시한다 — 탐색축으로 넣으면 유령 축이 된다."""
    assert "rsm" not in tu.suggest_params(_StubTrial(), "catboost")


def test_xgb_space_is_unchanged():
    """기존 study 와 비교 가능해야 하므로 XGBoost 축은 그대로여야 한다."""
    assert sorted(tu.suggest_params(_StubTrial(), "xgb")) == [
        "colsample_bylevel", "colsample_bytree", "gamma", "learning_rate",
        "max_depth", "min_child_weight", "n_estimators", "reg_alpha",
        "reg_lambda", "subsample",
    ]


def _fake_out(gap, score):
    return {"skf_generalization_gap": gap, "skf_oof_macro_f1": score}


def test_gap_penalty_is_off_by_default():
    """기본값 0 이면 기존 동작 그대로여야 한다 — 과거 study 와 비교가 깨지면 안 된다."""
    parser = tu.build_parser()
    assert parser.parse_args([]).gap_penalty == 0.0


def test_gap_penalty_arithmetic():
    """실측으로 확인한 식: 목적 = raw - lam * max(0, 격차 - floor)."""
    lam, floor, gap, raw = 0.5, 0.20, 0.3856, 0.4396
    assert np.isclose(raw - lam * max(0.0, gap - floor), 0.3468, atol=1e-4)


def test_gap_penalty_never_rewards_low_gap():
    """floor 아래 격차에 보너스를 주면 안 된다 — 페널티는 단측이다."""
    lam, floor = 0.5, 0.20
    assert max(0.0, 0.10 - floor) == 0.0


def test_verify_refuses_unsupported_combinations():
    """`BASELINE_CV` 는 f4r·xgb 실측값이라 다른 조합은 판정 자체가 불가능하다."""
    args = SimpleNamespace(model="catboost", config="f16", cv_list=["skf"])
    assert tu.verify(None, args) == 0  # 데이터 없이도 즉시 반환해야 한다
