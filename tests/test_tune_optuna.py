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

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna", reason="requirements.txt 의 optuna 미설치")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_gbdt import (  # noqa: E402
    CONFIGS,
    MODEL_PARAMS,
    build_parser as build_train_parser,
)
import tune_optuna as tune_module  # noqa: E402
from tune_optuna import (  # noqa: E402
    BASELINE_CV,
    BASELINE_POINT,
    FOLD_FEATURE_DEFAULTS,
    FIXED_PARAMS,
    _build_fold_feature_matrices,
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


@pytest.mark.parametrize(
    "config",
    [
        "f4r",
        "f4r_freq",
        "f4r_aatrans",
        "f5",
        "f4rc",
        "f4rl",
        "f4rm",
        "f4rsig",
        "full_all",
    ],
)
def test_cross_validate_accepts_every_fold_local_feature_family(config):
    """모든 fold-local 가족이 가드를 통과해 실제 data 접근 단계까지 가야 한다."""
    with pytest.raises(AttributeError):
        cross_validate(
            None,
            {},
            config=config,
            cv="skf",
            topk=500,
            n_splits=5,
            seed=42,
            use_gpu=False,
        )


def test_optuna_fold_feature_defaults_match_train_command_defaults():
    """Optuna에서 고정한 feature 축과 최종 train_gbdt 재실행이 같아야 한다."""
    train = build_train_parser().parse_args([])
    assert FOLD_FEATURE_DEFAULTS["sparse_topk"] == train.sparse_topk
    assert FOLD_FEATURE_DEFAULTS["tfidf_min_df"] == train.tfidf_min_df
    assert FOLD_FEATURE_DEFAULTS["parsed_min_df"] == train.parsed_min_df
    assert FOLD_FEATURE_DEFAULTS["comut"] == {
        "pool": train.comut_pool,
        "pool_topk": train.comut_pool_topk,
        "mode": train.comut_mode,
        "value": train.comut_value,
        "topk": train.comut_topk,
        "min_support": train.comut_min_support,
        "min_class_support": train.comut_min_class_support,
        "min_purity": train.comut_min_purity,
        "min_lift": train.comut_min_lift,
        "max_hyper_fraction": train.comut_max_hyper,
        "max_pairs_per_gene": train.comut_max_per_gene,
    }
    assert FOLD_FEATURE_DEFAULTS["latent"] == {
        "method": train.latent_method,
        "n_components": train.latent_components,
        "row_norm": train.latent_row_norm,
        "value": train.latent_value,
        "mode": train.latent_mode,
        "gene_weight": train.latent_gene_weight,
        "min_gene_support": train.latent_min_support,
        "random_state": train.latent_random_state,
    }
    assert FOLD_FEATURE_DEFAULTS["module"] == {
        "value": train.module_value,
        "n_modules": train.module_n,
        "svd_components": train.module_svd_components,
        "mode": train.module_mode,
        "min_gene_support": train.module_min_support,
        "random_state": train.module_random_state,
    }
    assert FOLD_FEATURE_DEFAULTS["signature"] == {
        "value": train.signature_value,
        "topk": train.signature_topk,
        "mode": train.signature_mode,
        "min_class_support": train.signature_min_class_support,
        "min_lift": train.signature_min_lift,
        "max_hyper_fraction": train.signature_max_hyper,
    }


def test_full_all_optuna_builder_appends_every_fold_local_family(monkeypatch):
    """full_all이 실제 Optuna 행렬에서 어느 가족도 조용히 빠뜨리지 않는다."""

    class FakeData:
        y = np.array(["A", "B", "A", "B"])
        classes = np.array(["A", "B"])
        folds = pd.DataFrame({"fold_skf2": [0, 0, 1, 1]})
        rollup_train = None
        gene = {
            block: (
                ["G1", "G2"],
                np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )
            for block in ("enc3", "gtype")
        }
        docs = {
            block: (
                np.array(["a", "b", "a b", ""], dtype=object),
                np.empty(0, dtype=object),
            )
            for block in ("sigtok", "ptok")
        }
        raw_train = pd.DataFrame(
            {"ID": ["0", "1", "2", "3"], "G1": ["WT", "A1C", "WT", "A1C"]}
        )
        raw_gene_columns = ["G1"]
        pairs = {
            block: (
                ["G1", "G2"],
                np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )
            for block in ("comut", "lsvd", "gmod", "csig")
        }

        @staticmethod
        def assemble(config):
            assert config == "full_all"
            return (
                ["dense"],
                np.ones((4, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )

    def sparse_stub(train, test, *args, **kwargs):
        return (
            ["s0", "s1"],
            np.ones((len(train), 2), np.float32),
            np.ones((len(test), 2), np.float32),
        )

    def frequency_stub(train, test, *args, **kwargs):
        return (
            ["f0", "f1", "f2"],
            np.ones((len(train), 3), np.float32),
            np.ones((len(test), 3), np.float32),
        )

    def four_output_stub(train, test, *args, **kwargs):
        return (
            ["x0", "x1"],
            np.ones((len(train), 2), np.float32),
            np.ones((len(test), 2), np.float32),
            [],
        )

    def five_output_stub(train, test, *args, **kwargs):
        return (*four_output_stub(train, test, *args, **kwargs), object())

    monkeypatch.setattr(tune_module, "build_fold_tfidf_block", sparse_stub)
    monkeypatch.setattr(tune_module, "build_fold_parsed_token_block", sparse_stub)
    monkeypatch.setattr(tune_module, "build_fold_frequency_blocks", frequency_stub)
    monkeypatch.setattr(tune_module, "build_fold_comutation_block", four_output_stub)
    monkeypatch.setattr(tune_module, "build_fold_latent_block", five_output_stub)
    monkeypatch.setattr(tune_module, "build_fold_module_block", five_output_stub)
    monkeypatch.setattr(tune_module, "build_fold_signature_block", four_output_stub)

    data = FakeData()
    first = _build_fold_feature_matrices(
        data, config="full_all", cv="skf", topk=2, n_splits=2
    )
    second = _build_fold_feature_matrices(
        data, config="full_all", cv="skf", topk=2, n_splits=2
    )

    assert [matrix.shape for matrix in first] == [(4, 20), (4, 20)]
    assert second is first


def test_baseline_cv_names_are_the_ones_the_cli_accepts():
    """`main` 은 `BASELINE_CV` 로 `--cv` 를 검증한다. 여기 없는 이름은 CLI 가 거절한다."""
    assert set(BASELINE_CV) == {"skf", "sgkf"}
    for split in BASELINE_CV.values():
        assert len(split["fold_macro_f1"]) == 5


def test_tuned_config_exists_in_train_gbdt():
    """기본 `--config f4r` 이 train_gbdt 에서 사라지면 CLI 가 즉시 죽어야 한다."""
    assert "f4r" in CONFIGS
