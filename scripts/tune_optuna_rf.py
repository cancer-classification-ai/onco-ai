#!/usr/bin/env python
"""RF-B Optuna smoke + 본 탐색 예산 산정 (Ticket 2) — `scripts/tune_optuna.py`(GBDT) 관례 재사용.

    python scripts/tune_optuna_rf.py --data-dir <external-data-dir>

이번 스크립트가 **이번 티켓에서 실제로 실행하는 것은 smoke study 뿐**이다(spec
§5.2, tickets Ticket 2). RF-B 본 탐색의 탐색 공간(`suggest_rf_b_params`)과
objective(`objective_main`)는 Ticket 3 가 그대로 재사용할 수 있게 여기 준비해
두되, study 를 만들고 실행하는 CLI 경로는 노출하지 않는다 — 그 배선(러너,
`--mode search` 같은 진입점)은 사용자 승인된 trial 수를 받은 Ticket 3 세션이
직접 추가한다.

## 재사용 지점(재구현 금지)

- `train_rf.load_f4r_features`/`train_rf.load_folds`: f4r 조립·fold 읽기
  (Ticket 1b 정의 그대로).
- `cancer_hack.validation.BurdenBinner`/`Chi2TopKSelector`(features_basic 의
  BurdenBinner): fold-local fit. 이 스크립트는 test 행렬을 만들지 않는 오케스트
  레이션만 새로 짠다(`build_f4r_train_matrix`) — GBDT 의 `tune_optuna.py
  cross_validate` 와 같은 패턴(test 미생성)을 RF 에 적용한 것이다.
- `cancer_hack.models_rf.create_model`: RF fit/predict_proba.
- `cancer_hack.rf_artifact_validator.macro_f1_with_labels`: labels= 명시 Macro F1.
- `cancer_hack.metrics.per_class_f1`.
- `train_rf._sha256`/`_git_commit_sha`/`peak_memory_bytes`: provenance 유틸.

## fold 행렬 캐시

f4r fold 행렬(burden 경계 + chi2 top500)은 모델 파라미터와 무관하고 fold-local
로만 결정된다 — 그래서 study 시작 전 **fold 당 한 번만** 만들고
(`precompute_fold_cache`), 이후 모든 trial(smoke 3개, 나중 본 탐색 수십 개)이
그 캐시를 그대로 재사용한다. test 행렬은 이 파일 어디에서도 만들지 않는다.

## smoke 는 탐색이 아니다

smoke 의 3개 trial 은 TPE 가 고르는 게 아니라 spec 이 지정한 고정 설정
(`SMOKE_TRIALS_SPEC`)이다. Optuna 의 `CategoricalDistribution` 은 같은 파라미터
이름에 study 전체에서 같은 선택지 집합을 요구하므로(다르면
`ValueError: ... does not support dynamic value space`), 각 파라미터의 선택지를
smoke 3개 설정에 실제로 등장하는 값의 합집합으로 잡는다
(`_smoke_param_choices`) — trial 마다 그중 정확히 하나를 `enqueue_trial` 로
강제한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cancer_hack.features_basic import BurdenBinner  # noqa: E402
from cancer_hack.io import save_csv  # noqa: E402
from cancer_hack.metrics import per_class_f1  # noqa: E402
from cancer_hack.models_rf import create_model, default_n_jobs  # noqa: E402
from cancer_hack.rf_artifact_validator import macro_f1_with_labels  # noqa: E402
from cancer_hack.validation import Chi2TopKSelector, check_all_classes_present  # noqa: E402

import train_rf  # noqa: E402 — f4r/fold/데이터 로딩 재사용(재구현 금지)

TUNING_DIRNAME = "tuning"
SMOKE_STUDY_NAME = "rf_b_smoke"
MAIN_STUDY_NAME = "rf_b_optuna"
SMOKE_DB_FILENAME = "rf_b_smoke_study.db"
MAIN_DB_FILENAME = "rf_b_optuna_study.db"
HARD_TIMEOUT_SECONDS = 3 * 60 * 60
BUDGET_SAFETY_FACTOR = 1.3  # smoke 3개뿐인 추정이라 30% 여유를 둔다(측정 아님, 판단값)
TRIAL_COUNT_GRID = (20, 30, 40)

#: spec §5.4 — 문자열 key 로 저장하고 sklearn 값으로 매핑(SQLite categorical 에
#: 문자열·실수를 섞지 않는다).
MAX_FEATURES_KEYS: dict[str, str | float] = {
    "sqrt": "sqrt",
    "log2": "log2",
    "frac_005": 0.05,
    "frac_010": 0.10,
    "frac_020": 0.20,
}

#: spec §4 RF-A 고정 설정을 §5.1 탐색 공간 좌표로 옮긴 것 — Ticket 3 가 trial 0 으로
#: enqueue 할 기준선(이번 티켓에서는 실행하지 않는다).
RF_A_BASELINE_POINT: dict = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features_key": "sqrt",
    "max_samples": 1.0,
    "criterion": "gini",
    "class_weight": "balanced_subsample",
}

#: spec §5.2 — RF-A 실측 n_estimators(500)와 탐색 공간 n_estimators 중앙값 후보.
#: smoke 설정별로 다른 목표 n_estimators 에 외삽해 "n_estimators 만 비례"가 아니라
#: 파라미터별 실제 비용 차이를 반영한다(§6 요구).
SMOKE_ROLE_TARGET_N_ESTIMATORS = {
    "baseline_cost": 400,  # 탐색 공간 하한
    "search_space_center_cost": 700,  # {400..1000 step100} 의 중앙값
    "compute_upper_bound_diagnostic": 1000,  # 탐색 공간 상한
}

#: Ticket 2 §4 — smoke 3개는 고정 설정이다(TPE 가 고르지 않는다). 모델 선택에
#: 쓰지 않는다(smoke 전용, audit 목적).
SMOKE_TRIALS_SPEC: list[dict] = [
    {
        "label": "baseline_cost",
        "params": {
            "n_estimators": 100,
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_samples": 1.0,
            "criterion": "gini",
            "class_weight": "balanced_subsample",
        },
    },
    {
        "label": "search_space_center_cost",
        "params": {
            "n_estimators": 100,
            "max_features": 0.10,
            "max_depth": 16,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_samples": 0.80,
            "criterion": "gini",
            "class_weight": "balanced_subsample",
        },
    },
    {
        "label": "compute_upper_bound_diagnostic",
        "params": {
            "n_estimators": 100,
            "max_features": 0.20,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_samples": 1.0,
            "criterion": "log_loss",
            "class_weight": "balanced",
        },
    },
]


def log(message: str) -> None:
    print(message, flush=True)


def _smoke_param_choices() -> dict[str, list]:
    """파라미터 이름마다 smoke 3개 설정에 실제 등장하는 값의 합집합(순서 보존, 중복 제거).

    Optuna 의 `CategoricalDistribution` 은 같은 study 안에서 같은 파라미터 이름에
    다른 선택지 집합을 허용하지 않는다 — 선택지를 미리 합집합으로 잡아야 trial
    마다 다른 고정값을 `enqueue_trial` 로 넣어도 충돌하지 않는다.
    """
    keys = SMOKE_TRIALS_SPEC[0]["params"].keys()
    choices: dict[str, list] = {key: [] for key in keys}
    for spec in SMOKE_TRIALS_SPEC:
        for key, value in spec["params"].items():
            if value not in choices[key]:
                choices[key].append(value)
    return choices


SMOKE_PARAM_CHOICES = _smoke_param_choices()


# ---------------------------------------------------------------- RF-B 탐색 공간(Ticket 3 용)
def suggest_rf_b_params(trial: optuna.Trial) -> dict:
    """spec §5.1 8-파라미터 탐색 공간. Ticket 3(본 탐색)이 쓴다 — 이 티켓은 호출하지 않는다."""
    max_features_key = trial.suggest_categorical("max_features_key", list(MAX_FEATURES_KEYS))
    return {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=100),
        "max_depth": trial.suggest_categorical("max_depth", [8, 12, 16, 20, 24, None]),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10, 20]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8]),
        "max_features": MAX_FEATURES_KEYS[max_features_key],
        "max_samples": trial.suggest_categorical("max_samples", [0.65, 0.80, 1.0]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "log_loss"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
    }


def _resolve_baseline_point(baseline: dict) -> dict:
    """`RF_A_BASELINE_POINT`(max_features_key 포함)를 `suggest_rf_b_params` 값과 맞춘다."""
    resolved = dict(baseline)
    key = resolved.pop("max_features_key")
    resolved["max_features"] = MAX_FEATURES_KEYS[key]
    return resolved


# ---------------------------------------------------------------- f4r train-only fold 행렬
def build_f4r_train_matrix(f4r: dict, train_index: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """f4r fold 행렬 중 **train 쪽만** 만든다 — test 행렬은 여기서 절대 만들지 않는다.

    `train_rf.build_f4r_fold_matrices` 와 fold-local fit 로직(BurdenBinner/
    Chi2TopKSelector)은 동일하되, `x_test`/`gene_test` 조립을 아예 하지 않는다
    (spec Ticket 2 §4 "test matrix는 smoke objective에 만들거나 사용하지 않는다").
    """
    dataset = f4r["dataset"]
    names = f4r["names"]
    burden_positions = f4r["burden_positions"]

    x_train = f4r["dense_train"].copy()
    if burden_positions:
        binner = BurdenBinner().fit(dataset.rollup_train.iloc[train_index])
        wanted = [names[i] for i in burden_positions]
        x_train[:, burden_positions] = binner.transform(dataset.rollup_train)[wanted].to_numpy(np.float32)

    selector = Chi2TopKSelector(k=train_rf.F4R_ENC3_TOPK).fit(
        f4r["gene_train"][train_index], dataset.y[train_index]
    )
    picked = selector.indices_
    x_train = np.hstack([x_train, f4r["gene_train"][:, picked]])
    feature_columns = names + [f4r["gene_columns"][i] for i in picked]
    return x_train, feature_columns


def precompute_fold_cache(f4r: dict, fold_ids: np.ndarray, n_splits: int) -> list[dict]:
    """fold 당 한 번만 train 행렬을 만들어 이후 모든 trial 이 재사용하게 한다.

    burden/chi2 fit 은 fold 의 train 부분에서만, 모델 파라미터와 무관하게 결정되므로
    trial 마다 다시 만들 이유가 없다 — 여기서 한 번 계산해 리스트로 캐시한다.
    """
    dataset = f4r["dataset"]
    y = dataset.y
    classes = f4r["classes"]
    cache = []
    for fold in range(n_splits):
        valid_index = np.where(fold_ids == fold)[0]
        train_index = np.where(fold_ids != fold)[0]
        check_all_classes_present(y, train_index, classes)

        x_full, feature_columns = build_f4r_train_matrix(f4r, train_index)
        if len(feature_columns) != train_rf.F4R_EXPECTED_FEATURES:
            raise ValueError(
                f"fold {fold} 피처 열 수가 {len(feature_columns)}개다 — "
                f"기대값 {train_rf.F4R_EXPECTED_FEATURES}개와 다르다. 중단한다."
            )
        cache.append(
            {
                "fold": fold,
                "train_index": train_index,
                "valid_index": valid_index,
                "x_fit": x_full[train_index],
                "y_fit": y[train_index],
                "x_val": x_full[valid_index],
                "y_val": y[valid_index],
                "n_features": len(feature_columns),
            }
        )
    return cache


# ---------------------------------------------------------------- 공통 objective 코어
def evaluate_params_on_cached_folds(
    fold_cache: list[dict],
    classes: np.ndarray,
    params: dict,
    *,
    seed: int,
    n_jobs: int,
) -> dict:
    """캐시된 fold 행렬로 RF 를 학습·평가한다. Optuna 를 몰라도 되는 순수 함수.

    반환값은 spec §5.5 진단값(전체/fold별 OOF, fold별 train, 격차, 시간, 클래스별
    F1)을 전부 담는다. test 행렬은 인자로 받지도 않는다 — 구조적으로 만들 수 없다.
    """
    class_order = classes.tolist()
    n = sum(len(entry["valid_index"]) for entry in fold_cache)
    oof = np.zeros((n, len(classes)), dtype=np.float64)
    y_all = np.empty(n, dtype=object)
    fold_scores: list[float] = []
    fold_train_scores: list[float] = []
    fold_seconds: list[float] = []

    started = time.perf_counter()
    for entry in fold_cache:
        t0 = time.perf_counter()
        model = create_model(
            "rf", class_order=class_order, random_state=seed, n_jobs=n_jobs, **params
        )
        model.fit(entry["x_fit"], entry["y_fit"])

        train_proba = model.predict_proba(entry["x_fit"])
        valid_proba = model.predict_proba(entry["x_val"])

        valid_index = entry["valid_index"]
        oof[valid_index] = valid_proba
        y_all[valid_index] = entry["y_val"]

        train_pred = classes[train_proba.argmax(axis=1)]
        valid_pred = classes[valid_proba.argmax(axis=1)]
        fold_train_scores.append(macro_f1_with_labels(entry["y_fit"], train_pred, classes))
        fold_scores.append(macro_f1_with_labels(entry["y_val"], valid_pred, classes))
        fold_seconds.append(time.perf_counter() - t0)

    oof_pred = classes[oof.argmax(axis=1)]
    oof_macro_f1 = macro_f1_with_labels(y_all, oof_pred, classes)

    return {
        "oof_macro_f1": oof_macro_f1,
        "oof_per_class_f1": per_class_f1(y_all, oof_pred, labels=classes),
        "fold_macro_f1": fold_scores,
        "fold_train_macro_f1": fold_train_scores,
        "fold_std": float(np.std(fold_scores)),
        "train_validation_gap": float(np.mean(fold_train_scores) - np.mean(fold_scores)),
        "fold_elapsed_seconds": fold_seconds,
        "trial_elapsed_seconds": time.perf_counter() - started,
        "n_features": fold_cache[0]["n_features"],
        "n_splits": len(fold_cache),
    }


def _objective_core(
    trial: optuna.Trial,
    params: dict,
    *,
    fold_cache: list[dict],
    classes: np.ndarray,
    seed: int,
    n_jobs: int,
    extra_user_attrs: dict | None = None,
) -> float:
    """`evaluate_params_on_cached_folds` 를 감싸 trial user_attr 로 진단값을 남긴다.

    실패하면 `failure_reason` 을 먼저 기록한 뒤 다시 던진다 — 호출부가
    `study.optimize(..., catch=(Exception,))` 로 받아 trial 을 FAIL 로 남기고
    연속 실패 횟수를 센다.
    """
    trial.set_user_attr("params", json.loads(json.dumps(params, default=str)))
    if extra_user_attrs:
        for key, value in extra_user_attrs.items():
            trial.set_user_attr(key, value)
    trial.set_user_attr("failure_reason", None)
    try:
        result = evaluate_params_on_cached_folds(
            fold_cache, classes, params, seed=seed, n_jobs=n_jobs
        )
    except Exception as exc:  # noqa: BLE001 — 원인을 기록하고 다시 던진다
        trial.set_user_attr("failure_reason", f"{type(exc).__name__}: {str(exc)[:300]}")
        raise
    for key, value in result.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("peak_memory_bytes", train_rf.peak_memory_bytes())
    return result["oof_macro_f1"]


def objective_smoke(
    trial: optuna.Trial, *, fold_cache: list[dict], classes: np.ndarray, seed: int, n_jobs: int
) -> float:
    if trial.number >= len(SMOKE_TRIALS_SPEC):
        raise RuntimeError(
            f"smoke study 는 고정 trial {len(SMOKE_TRIALS_SPEC)}개만 정의돼 있다 "
            f"(받은 trial.number={trial.number}) — TPE 가 새 trial 을 만들려 한 것이다."
        )
    spec = SMOKE_TRIALS_SPEC[trial.number]
    params = {
        key: trial.suggest_categorical(key, SMOKE_PARAM_CHOICES[key])
        for key in spec["params"]
    }
    return _objective_core(
        trial, params, fold_cache=fold_cache, classes=classes, seed=seed, n_jobs=n_jobs,
        extra_user_attrs={"smoke_label": spec["label"], "is_smoke": True},
    )


def objective_main(
    trial: optuna.Trial, *, fold_cache: list[dict], classes: np.ndarray, seed: int, n_jobs: int
) -> float:
    """RF-B 본 탐색 objective. Ticket 3 가 호출한다 — 이 스크립트는 실행하지 않는다."""
    params = suggest_rf_b_params(trial)
    return _objective_core(
        trial, params, fold_cache=fold_cache, classes=classes, seed=seed, n_jobs=n_jobs,
        extra_user_attrs={"is_smoke": False},
    )


# ---------------------------------------------------------------- study 실행(연속 실패 Stop 포함)
def _build_storage_url(db_path: Path) -> str:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def _run_waiting_trials(
    study: optuna.Study,
    objective,
    *,
    max_new_trials: int | None,
    consecutive_failure_limit: int = 3,
) -> dict:
    """WAITING trial 을 하나씩 처리하며 연속 동일 원인 실패를 센다.

    `study.optimize(objective, n_trials=1, catch=(Exception,))` 를 반복 호출한다
    — Optuna 는 `n_trials` 를 요청받으면 대기 중(enqueue 된) trial 을 새로 표본
    추출하기 전에 먼저 소비하므로, 이렇게 하나씩 실행하면 큐를 정확히 그만큼만
    비운다. `catch` 를 줘서 실패해도 study.optimize 가 예외를 올리지 않게 하고,
    이 함수가 직접 trial 상태를 보고 연속 실패를 판단한다.
    """
    def waiting_numbers() -> set[int]:
        return {t.number for t in study.trials if t.state == optuna.trial.TrialState.WAITING}

    waiting_before_all = waiting_numbers()
    to_run = (
        len(waiting_before_all)
        if max_new_trials is None
        else min(max_new_trials, len(waiting_before_all))
    )

    consecutive_failures = 0
    last_reason: str | None = None
    ran = 0
    for _ in range(to_run):
        # `study.trials[-1]` 은 trial **번호**가 가장 큰 항목이다 — enqueue 로 미리
        # 만든 WAITING trial 들은 실행 전부터 전부 존재하므로, 방금 실행된 trial 이
        # 아니라 큐의 마지막 항목(아직 WAITING)을 가리킬 수 있다. 그래서 실행 전후
        # WAITING 집합의 차집합으로 "방금 처리된 trial 번호"를 직접 찾는다.
        before = waiting_numbers()
        study.optimize(objective, n_trials=1, catch=(Exception,))
        after = waiting_numbers()
        settled = before - after
        if len(settled) != 1:
            raise RuntimeError(
                f"trial 1개를 처리했는데 WAITING 집합 변화가 {len(settled)}개다: {settled} "
                "(FIFO 소비 가정이 깨졌다)"
            )
        ran += 1
        last_trial = study.trials[next(iter(settled))]
        if last_trial.state == optuna.trial.TrialState.FAIL:
            reason = last_trial.user_attrs.get("failure_reason", "unknown")
            consecutive_failures = consecutive_failures + 1 if reason == last_reason else 1
            last_reason = reason
            if consecutive_failures >= consecutive_failure_limit:
                raise SystemExit(
                    f"연속 {consecutive_failure_limit} trial 동일 원인 실패 — {reason}. "
                    "study 를 중단한다(Stop 조건, spec §5.2/tickets Ticket 2 §9)."
                )
        else:
            consecutive_failures = 0
            last_reason = None
    return {"waiting_before": len(waiting_before_all), "ran_this_call": ran}


def run_smoke_study(
    fold_cache: list[dict],
    classes: np.ndarray,
    *,
    db_path: Path,
    seed: int,
    sampler_seed: int,
    n_jobs: int,
    max_new_trials: int | None = None,
) -> tuple[optuna.Study, dict]:
    storage = _build_storage_url(db_path)
    study = optuna.create_study(
        study_name=SMOKE_STUDY_NAME,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    if not study.trials:
        for spec in SMOKE_TRIALS_SPEC:
            study.enqueue_trial(spec["params"], user_attrs={"smoke_label": spec["label"]})
        log(f"smoke study: 고정 trial {len(SMOKE_TRIALS_SPEC)}개를 enqueue 했다.")

    before = {
        t.number: (t.params, t.value)
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    }

    def objective(trial: optuna.Trial) -> float:
        return objective_smoke(trial, fold_cache=fold_cache, classes=classes, seed=seed, n_jobs=n_jobs)

    run_info = _run_waiting_trials(study, objective, max_new_trials=max_new_trials)

    for number, (params, value) in before.items():
        replayed = study.trials[number]
        if replayed.state != optuna.trial.TrialState.COMPLETE:
            raise RuntimeError(f"trial {number} 이 재개 후 COMPLETE 가 아니다: {replayed.state}")
        if replayed.params != params or replayed.value != value:
            raise RuntimeError(
                f"trial {number} 이 재개 후 값이 바뀌었다 — params/value 가 보존돼야 한다"
            )
    run_info["preserved_trials_checked"] = sorted(before)
    return study, run_info


# ---------------------------------------------------------------- 예산 계산
def compute_budget_report(
    *,
    rf_a_measurement: dict,
    smoke_summaries: list[dict],
    resume_verification: dict,
    fold_cache_build_seconds: float,
    data_hashes: dict,
    folds_file_info: dict,
    git_commit: str | None,
    hard_timeout_seconds: int = HARD_TIMEOUT_SECONDS,
    safety_factor: float = BUDGET_SAFETY_FACTOR,
    trial_counts: tuple[int, ...] = TRIAL_COUNT_GRID,
) -> dict:
    """spec §5.2/tickets Ticket 2 §6 이 요구하는 본 탐색 예산 산정.

    단순 n_estimators 비례가 아니라, smoke 3개(가장 싼 설정/탐색공간 중앙 비용/
    계산비용 상단)를 각각 자신의 역할에 맞는 목표 n_estimators 로 외삽해
    파라미터별 실제 비용 차이를 반영한다.
    """
    per_smoke: dict[str, dict] = {}
    for summary in smoke_summaries:
        role = summary["label"]
        target_n = SMOKE_ROLE_TARGET_N_ESTIMATORS[role]
        measured_n = summary["params"]["n_estimators"]
        measured_seconds = summary["trial_elapsed_seconds"]
        extrapolated_seconds = measured_seconds / measured_n * target_n
        per_smoke[role] = {
            "params": summary["params"],
            "oof_macro_f1": summary["oof_macro_f1"],
            "fold_macro_f1": summary["fold_macro_f1"],
            "fold_train_macro_f1": summary["fold_train_macro_f1"],
            "train_validation_gap": summary["train_validation_gap"],
            "measured_trial_elapsed_seconds": measured_seconds,
            "measured_n_estimators": measured_n,
            "peak_memory_bytes": summary["peak_memory_bytes"],
            "target_n_estimators_for_extrapolation": target_n,
            "extrapolated_seconds_at_target_n_estimators": extrapolated_seconds,
        }

    extrapolated = {
        role: v["extrapolated_seconds_at_target_n_estimators"] for role, v in per_smoke.items()
    }
    min_trial_seconds = min(extrapolated.values())
    median_trial_seconds = extrapolated.get(
        "search_space_center_cost", sorted(extrapolated.values())[len(extrapolated) // 2]
    )
    max_trial_seconds = max(extrapolated.values())

    ordering_as_expected = (
        extrapolated.get("baseline_cost", 0) <= extrapolated.get("search_space_center_cost", 0) <= extrapolated.get(
            "compute_upper_bound_diagnostic", 0
        )
    )

    projected_wall_time = {}
    for n in trial_counts:
        projected_wall_time[str(n)] = {
            "wall_seconds_no_safety": fold_cache_build_seconds + n * median_trial_seconds,
            "wall_seconds_with_safety_factor": fold_cache_build_seconds
            + n * median_trial_seconds * safety_factor,
            "wall_seconds_worst_case_upper_bound": fold_cache_build_seconds
            + n * max_trial_seconds * safety_factor,
        }

    max_trials_within_timeout = 0
    for n in range(1, 41):
        if fold_cache_build_seconds + n * median_trial_seconds * safety_factor <= hard_timeout_seconds:
            max_trials_within_timeout = n
        else:
            break
    recommended_trial_count = min(40, max_trials_within_timeout)

    if recommended_trial_count < 10:
        proceed_decision = "STOP_INSUFFICIENT_BUDGET_FOR_TPE"
    elif recommended_trial_count < 20:
        proceed_decision = "PROCEED_BELOW_TARGET_NEEDS_EXPLICIT_APPROVAL"
    else:
        proceed_decision = "PROCEED_MEETS_TARGET"

    max_observed_peak = max(
        (s["peak_memory_bytes"] for s in smoke_summaries if s["peak_memory_bytes"]),
        default=None,
    )
    rf_a_peak = rf_a_measurement.get("peak_memory_bytes")
    estimated_peak_bytes_at_1000_trees = None
    if max_observed_peak is not None or rf_a_peak is not None:
        candidates = []
        if rf_a_peak is not None:
            candidates.append(rf_a_peak / rf_a_measurement["params"]["n_estimators"] * 1000)
        if max_observed_peak is not None:
            candidates.append(max_observed_peak / 100 * 1000)
        estimated_peak_bytes_at_1000_trees = max(candidates)

    return {
        "schema_version": 1,
        "git_commit": git_commit,
        "data_files": data_hashes,
        "folds_file": folds_file_info,
        "rf_a_reference_measurement": rf_a_measurement,
        "smoke_trials": per_smoke,
        "smoke_used_for_model_selection": False,
        "smoke_audit_only_note": (
            "smoke DB/결과는 파이프라인 동작 증거로만 보존한다 — 모델 선택과 "
            "Ticket 5 최종 reference result 에는 사용하지 않는다."
        ),
        "sqlite_resume_verification": resume_verification,
        "estimation_methodology": {
            "description": (
                "smoke 3개를 각자 역할(가장 싼 설정/탐색공간 중앙 비용/계산비용 상단)에 맞는 "
                "목표 n_estimators 로 개별 외삽한다(단순 n_estimators 비례 전체 적용이 아님). "
                "fold 행렬 생성은 study 당 1회 고정비용으로 별도 가산한다."
            ),
            "fold_cache_build_seconds": fold_cache_build_seconds,
            "safety_factor": safety_factor,
            "safety_factor_rationale": (
                "smoke n=3 로만 외삽한 추정이라 불확실성이 커서 30% 여유를 더한다 "
                "(spec 이 지정한 값이 아니라 이 티켓에서 선택한 보수적 계수)."
            ),
            "role_target_n_estimators": SMOKE_ROLE_TARGET_N_ESTIMATORS,
            "min_trial_seconds": min_trial_seconds,
            "median_trial_seconds": median_trial_seconds,
            "max_trial_seconds": max_trial_seconds,
            "cost_ordering_as_expected_baseline_le_center_le_upper": ordering_as_expected,
        },
        "projected_wall_time_seconds": projected_wall_time,
        "hard_timeout_seconds": hard_timeout_seconds,
        "max_trials_within_timeout": max_trials_within_timeout,
        "recommended_trial_count": recommended_trial_count,
        "recommended_trial_count_meets_minimum_20": recommended_trial_count >= 20,
        "proceed_decision": proceed_decision,
        "estimated_peak_memory_bytes_at_1000_trees": estimated_peak_bytes_at_1000_trees,
        "peak_memory_measurement_note": (
            "peak_memory_bytes 는 트라이얼 전용 값이 아니라 "
            "resource.getrusage(RUSAGE_SELF).ru_maxrss — 프로세스 시작 이후 누적 "
            "고점(단조 비감소)이다. 같은 프로세스에서 여러 trial 을 이어 돌리면 뒤쪽 "
            "trial 일수록 앞 trial 의 고점을 그대로 물려받거나 넘길 뿐이다(train_rf.py "
            "provenance 와 동일 관례)."
        ),
        "requires_explicit_user_trial_count_approval": True,
        "main_search_executed_in_this_ticket": False,
    }


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--folds-path", type=Path, default=PROJECT_ROOT / "data/process/train_folds.parquet"
    )
    parser.add_argument(
        "--class-order-path", type=Path, default=PROJECT_ROOT / "artifacts/metadata/class_order.json"
    )
    parser.add_argument(
        "--rf-a-log-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts/logs/rf_a_f4r_group5_s42.json",
        help="budget 계산의 RF-A 실측 기준(Ticket 1b 산출물)",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="RF 모델 시드")
    parser.add_argument("--sampler-seed", type=int, default=42, help="TPE 시드")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--max-new-trials",
        type=int,
        default=None,
        help="이번 호출에서 처리할 대기 trial 수 상한(재개 시나리오 검증용). 기본은 무제한.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)

    data_dir = train_rf.resolve_data_dir(str(args.data_dir) if args.data_dir else None)
    data_hashes = train_rf.validate_data_dir(data_dir)

    f4r = train_rf.load_f4r_features(
        data_dir, n_splits=args.n_splits, class_order_path=args.class_order_path
    )
    classes = f4r["classes"]

    fold_ids, fold_col = train_rf.load_folds(
        args.folds_path, cv="sgkf", n_splits=args.n_splits, train_ids=f4r["dataset"].train_ids
    )

    n_jobs = args.n_jobs if args.n_jobs is not None else default_n_jobs()

    t0 = time.perf_counter()
    fold_cache = precompute_fold_cache(f4r, fold_ids, args.n_splits)
    fold_cache_build_seconds = time.perf_counter() - t0
    log(f"fold 행렬 캐시 생성: {fold_cache_build_seconds:.2f}s ({args.n_splits} folds)")

    tuning_dir = args.out_dir / TUNING_DIRNAME
    db_path = tuning_dir / SMOKE_DB_FILENAME
    study, resume_info = run_smoke_study(
        fold_cache,
        classes,
        db_path=db_path,
        seed=args.seed,
        sampler_seed=args.sampler_seed,
        n_jobs=n_jobs,
        max_new_trials=args.max_new_trials,
    )

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    log(
        f"smoke study: complete={len(complete)}/{len(SMOKE_TRIALS_SPEC)}  "
        f"failed={len(failed)}  이번 호출에서 실행={resume_info['ran_this_call']}"
    )

    same_reason_failures = _count_same_reason_failures(failed)
    if same_reason_failures >= 2:
        raise SystemExit(
            f"smoke 3개 중 {same_reason_failures}개가 동일 원인으로 실패했다 — "
            "Stop 조건(tickets Ticket 2 §9). 진행하지 않는다."
        )

    if len(complete) < len(SMOKE_TRIALS_SPEC):
        log(
            f"아직 {len(SMOKE_TRIALS_SPEC) - len(complete)}개 trial 이 남았다 — "
            "동일 명령을 다시 실행해 재개한다(SQLite 재개, 강제 종료 없이)."
        )
        return {"status": "partial", "complete": len(complete), "resume_info": resume_info}

    trials_frame = _trials_to_frame(study)
    trials_csv_path = tuning_dir / "rf_b_smoke_trials.csv"
    save_csv(trials_frame, trials_csv_path)
    log(f"smoke trials CSV -> {trials_csv_path}")

    with open(args.rf_a_log_path, encoding="utf-8") as handle:
        rf_a_log = json.load(handle)
    rf_a_measurement = {
        "stem": rf_a_log["stem"],
        "params": rf_a_log["params"],
        "fold_elapsed_seconds": rf_a_log["fold_elapsed_seconds"],
        "elapsed_seconds": rf_a_log["elapsed_seconds"],
        "peak_memory_bytes": rf_a_log["peak_memory_bytes"],
        "oof_macro_f1": rf_a_log["oof_macro_f1"],
        "n_features": rf_a_log["n_features"],
        "n_jobs": rf_a_log["n_jobs"],
    }

    smoke_summaries = [
        {
            "label": t.user_attrs["smoke_label"],
            "params": t.user_attrs["params"],
            "oof_macro_f1": t.user_attrs["oof_macro_f1"],
            "fold_macro_f1": t.user_attrs["fold_macro_f1"],
            "fold_train_macro_f1": t.user_attrs["fold_train_macro_f1"],
            "train_validation_gap": t.user_attrs["train_validation_gap"],
            "trial_elapsed_seconds": t.user_attrs["trial_elapsed_seconds"],
            "peak_memory_bytes": t.user_attrs.get("peak_memory_bytes"),
        }
        for t in sorted(complete, key=lambda t: t.number)
    ]

    budget_report = compute_budget_report(
        rf_a_measurement=rf_a_measurement,
        smoke_summaries=smoke_summaries,
        resume_verification=resume_info,
        fold_cache_build_seconds=fold_cache_build_seconds,
        data_hashes=data_hashes,
        folds_file_info={"name": args.folds_path.name, "fold_column": fold_col},
        git_commit=train_rf._git_commit_sha(),
    )
    budget_path = tuning_dir / "rf_b_budget_report.json"
    with open(budget_path, "w", encoding="utf-8") as handle:
        json.dump(budget_report, handle, ensure_ascii=False, indent=2)
    log(f"budget report -> {budget_path}")
    log(
        f"권장 trial 수 = {budget_report['recommended_trial_count']} "
        f"({budget_report['proceed_decision']}) — 사용자 승인 필요"
    )

    return {"status": "complete", "budget_report": budget_report, "trials_csv": str(trials_csv_path)}


def _count_same_reason_failures(failed_trials: list) -> int:
    reasons: dict[str, int] = {}
    for t in failed_trials:
        reason = t.user_attrs.get("failure_reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    return max(reasons.values(), default=0)


def _trials_to_frame(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        row = {"number": t.number, "state": str(t.state.name), "objective_value": t.value}
        for key, value in t.user_attrs.items():
            row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
