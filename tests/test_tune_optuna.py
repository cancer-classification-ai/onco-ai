"""탐색기가 "기준선과 같은 것을 잰다"는 전제를 못 박는다.

`tune_optuna.py` 의 결론은 전부 상대값이다 — trial 0 에 넣은 기준선보다 얼마나
높은가로 후보를 고른다. 그래서 기준선이 진짜 기준선이 아니게 되는 순간 표 전체가
거짓말을 시작하는데, 그 어긋남은 예외를 던지지 않고 그냥 숫자로만 나온다.
여기 있는 테스트는 그 조용한 어긋남 네 가지를 잡는다.

원본 csv 없이 돈다. 모델을 학습하지 않고 상수·문자열·탐색 공간만 본다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

optuna = pytest.importorskip("optuna", reason="requirements.txt 의 optuna 미설치")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_gbdt import CONFIGS, MODEL_PARAMS  # noqa: E402
from tune_optuna import (  # noqa: E402
    BASELINE_CV,
    BASELINE_POINT,
    FIXED_PARAMS,
    _objective_value,
    _train_command,
    cross_validate,
    suggest_params,
)


def test_baseline_point_agrees_with_train_gbdt():
    """`BASELINE_POINT` 는 `MODEL_PARAMS["xgb"]` 를 탐색 공간 좌표로 옮긴 것이다.

    train_gbdt 쪽 기본 파라미터를 누가 바꾸면 trial 0 은 더 이상 f4r 기준선이
    아니지만, 탐색은 멀쩡히 돌고 "기준선 대비 +0.003" 같은 숫자를 계속 뱉는다.
    비교 대상이 바뀐 걸 아무도 모른 채로.
    """
    for key, value in MODEL_PARAMS["xgb"].items():
        assert key in BASELINE_POINT, f"{key} 가 BASELINE_POINT 에 없다"
        assert BASELINE_POINT[key] == pytest.approx(value), f"{key} 가 어긋났다"


def test_baseline_point_survives_a_round_trip_through_the_search_space():
    """기준선을 `enqueue_trial` 로 넣었을 때 그대로 나와야 한다.

    탐색 범위를 좁히다가 기준선이 밖으로 밀려나면 Optuna 는 경고만 남기고 값을
    범위 안으로 끌어당긴다. 그러면 trial 0 은 기준선 근처의 다른 점이 되고,
    `report()` 가 그걸 기준선이라 부르며 표를 그린다.
    """
    seen: dict = {}

    def objective(trial):
        seen.update(suggest_params(trial))
        return 0.0

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(BASELINE_POINT)
    study.optimize(objective, n_trials=1)

    assert set(seen) == set(BASELINE_POINT)
    for key, value in BASELINE_POINT.items():
        assert seen[key] == pytest.approx(value), f"{key} 가 탐색 공간 밖이라 끌려왔다"


def test_fixed_params_do_not_overlap_the_search_space():
    """고정값과 탐색 대상이 겹치면 `{**FIXED_PARAMS, **params}` 에서 조용히 덮인다."""
    assert not set(FIXED_PARAMS) & set(BASELINE_POINT)


def test_train_command_keeps_full_float_precision():
    """재실행 커맨드의 float 은 왕복 가능해야 한다.

    subsample·colsample 은 그 값으로 행·열을 뽑는다. 유효숫자 6자리로 자르면
    표시 오차가 아니라 **다른 설정**이 되고, 실제로 잘린 값으로 재실행한 trial 이
    0.4672 대신 0.4655 를 냈다.
    """
    params = {"subsample": 0.6037440222139066, "colsample_bytree": 0.23456789012345678}
    args = SimpleNamespace(config="f4r", topk=500, seed=42)
    command = _train_command(params, args)

    for key, value in params.items():
        token = next(t for t in command.split() if t.startswith(f"{key}="))
        assert float(token.split("=", 1)[1]) == value, f"{key} 가 잘렸다"


def test_train_command_carries_the_tuned_config():
    """f4r 을 튜닝하고 f4rl 재실행 커맨드를 받으면 다른 모델을 제출하게 된다."""
    args = SimpleNamespace(config="f4rsig", topk=300, seed=7)
    command = _train_command({"max_depth": 5}, args)
    assert "--configs f4rsig" in command
    assert "--topk 300" in command
    assert "--seed 7" in command


@pytest.mark.parametrize(
    "objective,expected",
    [("mean", 0.45), ("min", 0.40), ("skf", 0.40), ("sgkf", 0.50)],
)
def test_objective_value_picks_the_right_number(objective, expected):
    assert _objective_value({"skf": 0.40, "sgkf": 0.50}, objective) == pytest.approx(expected)


def test_objective_value_rejects_a_split_it_did_not_measure():
    """`--cv skf --objective sgkf` 는 재지도 않은 값을 최대화하라는 요청이다."""
    with pytest.raises(ValueError, match="sgkf"):
        _objective_value({"skf": 0.40}, "sgkf")


def test_cross_validate_refuses_configs_it_cannot_build():
    """fold 안에서 새로 fit 하는 블록은 이 튜너가 만들지 않는다.

    막지 않으면 f4rl 을 튜닝했는데 lsvd 가 빠진 채 학습돼 f4r 의 점수가 나온다.
    예외가 아니라 그럴듯한 숫자로 나오는 게 문제라 가드를 테스트로 고정한다.
    `data` 를 만지기 전에 걸러야 하므로 None 을 넘겨도 통과하면 안 된다.
    """
    with pytest.raises(ValueError, match="lsvd"):
        cross_validate(
            None,
            {},
            config="f4rl",
            cv="skf",
            topk=500,
            n_splits=5,
            seed=42,
            use_gpu=False,
        )


def test_supported_config_is_not_rejected_before_it_touches_data():
    """f4r 은 가드를 통과해야 한다 — 통과 후 data 를 쓰다 죽는 건 이 테스트 밖이다."""
    with pytest.raises(AttributeError):
        cross_validate(
            None,
            {},
            config="f4r",
            cv="skf",
            topk=500,
            n_splits=5,
            seed=42,
            use_gpu=False,
        )


def test_baseline_cv_names_are_the_ones_the_cli_accepts():
    """`main` 은 `BASELINE_CV` 로 `--cv` 를 검증한다. 여기 없는 이름은 CLI 가 거절한다."""
    assert set(BASELINE_CV) == {"skf", "sgkf"}
    for split in BASELINE_CV.values():
        assert len(split["fold_macro_f1"]) == 5


def test_tuned_config_exists_in_train_gbdt():
    """기본 `--config f4r` 이 train_gbdt 에서 사라지면 CLI 가 즉시 죽어야 한다."""
    assert "f4r" in CONFIGS
