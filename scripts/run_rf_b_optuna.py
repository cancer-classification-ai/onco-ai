#!/usr/bin/env python
"""RF-B 본 Optuna 탐색(최대 40 trial) + best trial 재학습 + RF-A 대비 판정 (Ticket 3).

    python scripts/run_rf_b_optuna.py --data-dir <external-data-dir> --target-trials 40

`scripts/tune_optuna_rf.py`(Ticket 2)의 `precompute_fold_cache`/`run_main_study`/
`objective_main`/`suggest_rf_b_params`/`RF_A_BASELINE_POINT`를 그대로 재사용한다
(재구현 금지). 이 스크립트가 새로 하는 일은 다음 오케스트레이션뿐이다.

1. trial 0(RF-A 기준선) 완료 후, RF-A 저장 산출물(`artifacts/oof/oof_rf_a_...csv`)
   과 행 단위로 비교해 재현을 검증한다(§3). 실패하면 즉시 중단한다.
2. `run_main_study`로 study 전체 COMPLETE 가 `--target-trials`(≤40)가 될 때까지
   진행한다(재실행 시 남은 trial만).
3. trials/best_params/search_summary를 내보낸다.
4. COMPLETE ≥20 이면 best trial 파라미터로 `scripts/train_rf.py` 를 그대로 호출해
   canonical OOF/test/submission/log 를 만든다(이 단계에서만 test 를 만진다).
5. 저장된 파일을 다시 읽어 strict validator 를 재검증하고, per-class F1/confusion
   matrix CSV 를 내보낸 뒤 RF-A 대비 채택/보류/기각을 계산한다.
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

from cancer_hack.io import save_csv  # noqa: E402
from cancer_hack.models_rf import default_n_jobs  # noqa: E402
from cancer_hack.rf_artifact_validator import (  # noqa: E402
    macro_f1_with_labels,
    validate_oof_frame,
    validate_submission_frame,
    validate_test_probability_frame,
)

import train_rf  # noqa: E402 — canonical 산출물 생성(재구현 금지)
import tune_optuna_rf as tune_rf  # noqa: E402 — fold 캐시/본 탐색/탐색공간(재구현 금지)

MAX_TARGET_TRIALS = 40  # spec 금지 §12 "40 trial 초과 실행"
MIN_TRIALS_FOR_DECISION = 20
TRIAL0_ATOL_PROBA = 1e-6
TRIAL0_ATOL_MACRO_F1 = 1e-6
TRIAL0_ATOL_FOLD_SCORE = 1e-6

#: spec §9 — 클래스 붕괴 진단 임계값(고정, 결과를 보고 바꾸지 않는다).
CLASS_HARD_ZERO_MIN_RF_A_F1 = 0.10
CLASS_HARD_DROP = 0.15
CLASS_SOFT_DROP = 0.10
FOLD_STD_WARNING_MULTIPLIER = 2.0
ACCEPT_MIN_DELTA = 0.005
ACCEPT_MIN_FOLDS_IMPROVED = 3

#: spec §4 — RF-B 탐색 공간(§9 경계값 판정에 재사용). `suggest_rf_b_params` 와
#: 반드시 같은 값을 가리켜야 한다(따로 하드코딩하지 않고 `tune_optuna_rf` 상수만
#: 참조한다).
SEARCH_SPACE_BOUNDARIES: dict[str, list] = {
    "n_estimators": [400, 1000],
    "max_depth": [8, 12, 16, 20, 24, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features_key": list(tune_rf.MAX_FEATURES_KEYS),
    "max_samples": [0.65, 0.80, 1.0],
    "criterion": ["gini", "log_loss"],
    "class_weight": ["balanced", "balanced_subsample"],
}


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------- trial 0 재현 검증
def verify_trial0_reproduces_rf_a(
    fold_cache: list[dict],
    classes: np.ndarray,
    train_ids: np.ndarray,
    fold_ids: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
    rf_a_oof_path: Path,
    rf_a_log_path: Path,
    atol_proba: float = TRIAL0_ATOL_PROBA,
    atol_macro_f1: float = TRIAL0_ATOL_MACRO_F1,
    atol_fold_score: float = TRIAL0_ATOL_FOLD_SCORE,
) -> dict:
    """RF-A 기준선을 캐시된 fold 로 다시 계산해 저장된 RF-A OOF 와 행 단위로 비교한다.

    비교는 ID 로 맞춘다(행 순서를 가정하지 않는다). `evaluate_params_on_cached_folds`
    를 `return_oof=True` 로만 호출한다 — 이 결과는 Optuna trial 에 저장되지 않는다
    (스크립트 로컬 변수일 뿐이다).
    """
    baseline_params = tune_rf._resolve_baseline_point(tune_rf.RF_A_BASELINE_POINT)
    result = tune_rf.evaluate_params_on_cached_folds(
        fold_cache, classes, baseline_params, seed=seed, n_jobs=n_jobs, return_oof=True
    )

    proba_cols = [f"p_{c}" for c in classes]
    recomputed = pd.DataFrame(result["oof_proba"], columns=proba_cols)
    recomputed.insert(0, "ID", np.asarray(train_ids, dtype=str))
    recomputed["fold"] = np.asarray(fold_ids)
    recomputed["y_true"] = np.asarray(result["y_all"], dtype=str)
    recomputed["y_pred"] = np.asarray(result["oof_pred"], dtype=str)

    saved = pd.read_csv(rf_a_oof_path, dtype={"ID": str})
    with open(rf_a_log_path, encoding="utf-8") as handle:
        rf_a_log = json.load(handle)

    checks: dict = {
        "baseline_params": baseline_params,
        "id_set_matches": set(saved["ID"]) == set(recomputed["ID"]),
        "class_order_matches": list(rf_a_log["class_order"]) == classes.tolist(),
    }

    merged = saved.merge(
        recomputed, on="ID", how="inner", suffixes=("_saved", "_recomputed"), validate="one_to_one"
    )
    checks["row_count_matches"] = len(merged) == len(saved) == len(recomputed)
    checks["fold_matches"] = bool((merged["fold_saved"] == merged["fold_recomputed"]).all())
    checks["y_true_matches"] = bool(
        (merged["y_true_saved"].astype(str) == merged["y_true_recomputed"].astype(str)).all()
    )
    checks["y_pred_matches"] = bool(
        (merged["y_pred_saved"].astype(str) == merged["y_pred_recomputed"].astype(str)).all()
    )

    saved_proba = merged[[f"{c}_saved" for c in proba_cols]].to_numpy(dtype=np.float64)
    recomputed_proba = merged[[f"{c}_recomputed" for c in proba_cols]].to_numpy(dtype=np.float64)
    max_abs_proba_diff = float(np.abs(saved_proba - recomputed_proba).max())
    checks["max_abs_proba_diff"] = max_abs_proba_diff
    checks["proba_within_tolerance"] = max_abs_proba_diff <= atol_proba

    recomputed_oof_macro_f1 = result["oof_macro_f1"]
    checks["recomputed_oof_macro_f1"] = recomputed_oof_macro_f1
    checks["saved_oof_macro_f1"] = rf_a_log["oof_macro_f1"]
    checks["oof_macro_f1_diff"] = abs(recomputed_oof_macro_f1 - rf_a_log["oof_macro_f1"])
    checks["oof_macro_f1_within_tolerance"] = checks["oof_macro_f1_diff"] <= atol_macro_f1

    fold_diffs = [
        abs(a - b) for a, b in zip(result["fold_macro_f1"], rf_a_log["fold_macro_f1"])
    ]
    checks["fold_macro_f1_recomputed"] = result["fold_macro_f1"]
    checks["fold_macro_f1_saved"] = rf_a_log["fold_macro_f1"]
    checks["fold_macro_f1_max_abs_diff"] = float(max(fold_diffs)) if fold_diffs else None
    checks["fold_macro_f1_within_tolerance"] = all(d <= atol_fold_score for d in fold_diffs)

    checks["passed"] = all(
        [
            checks["id_set_matches"],
            checks["class_order_matches"],
            checks["row_count_matches"],
            checks["fold_matches"],
            checks["y_true_matches"],
            checks["y_pred_matches"],
            checks["proba_within_tolerance"],
            checks["oof_macro_f1_within_tolerance"],
            checks["fold_macro_f1_within_tolerance"],
        ]
    )
    return checks


# ---------------------------------------------------------------- export
def _main_trials_to_frame(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        row = {"number": t.number, "state": str(t.state.name), "objective_value": t.value}
        params = t.user_attrs.get("params", t.params)
        for key, value in params.items():
            row[f"param_{key}"] = value
        for key, value in t.user_attrs.items():
            if key == "params":
                continue
            row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        rows.append(row)
    return pd.DataFrame(rows)


def _param_is_boundary(key: str, value) -> bool:
    allowed = SEARCH_SPACE_BOUNDARIES[key]
    if key == "n_estimators":
        lo, hi = allowed
        return value == lo or value == hi
    return value == allowed[0] or value == allowed[-1]


def build_search_summary(
    study: optuna.Study,
    *,
    rf_a_log: dict,
    target_total_trials: int,
    trial0_check: dict,
    fold_hash: str,
    data_hashes: dict,
    folds_file_info: dict,
    git_commit: str | None,
    cumulative_wall_seconds: float,
) -> dict:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    best = study.best_trial if complete else None
    best_params = best.user_attrs.get("params") if best is not None else None

    best_params_boundary_flags = None
    if best_params is not None:
        # user_attr "params" 는 sklearn 값(`max_features`)만 담고 문자열 key 는
        # `best.params["max_features_key"]`(Optuna 자체 기록)에서 가져온다.
        boundary_source = dict(best_params)
        boundary_source["max_features_key"] = best.params.get("max_features_key")
        best_params_boundary_flags = {
            key: {"value": boundary_source.get(key), "is_boundary": _param_is_boundary(key, boundary_source.get(key))}
            for key in SEARCH_SPACE_BOUNDARIES
            if key != "max_features"
        }

    importance = None
    if len(complete) >= 10:
        try:
            importance = {
                k: float(v) for k, v in optuna.importance.get_param_importances(study).items()
            }
        except Exception as exc:  # noqa: BLE001 — 참고용 계산이라 실패해도 탐색을 막지 않는다
            importance = {"error": f"{type(exc).__name__}: {exc}"}

    top_trials = sorted(complete, key=lambda t: -t.value)[:10]
    top_table = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.user_attrs.get("params", t.params),
            "fold_macro_f1": t.user_attrs.get("fold_macro_f1"),
            "train_validation_gap": t.user_attrs.get("train_validation_gap"),
        }
        for t in top_trials
    ]

    total_search_seconds = sum(t.user_attrs.get("trial_elapsed_seconds", 0.0) for t in complete)

    return {
        "schema_version": 1,
        "git_commit": git_commit,
        "data_files": data_hashes,
        "folds_file": folds_file_info,
        "fold_hash": fold_hash,
        "target_total_trials": target_total_trials,
        "n_complete": len(complete),
        "n_failed": len(failed),
        "n_pruned": len(pruned),
        "trial0_rf_a_reproduction_check": trial0_check,
        "best_trial_number": best.number if best is not None else None,
        "best_value_oof_macro_f1": best.value if best is not None else None,
        "rf_a_oof_macro_f1": rf_a_log["oof_macro_f1"],
        "delta_vs_rf_a": (best.value - rf_a_log["oof_macro_f1"]) if best is not None else None,
        "best_params": best_params,
        "best_params_boundary_flags": best_params_boundary_flags,
        "parameter_importance_note": (
            "참고용이다 — trial 10개 미만이면 계산하지 않는다(불안정). "
            "탐색 범위 확장 판단에 자동으로 쓰지 않는다."
        ),
        "parameter_importance": importance,
        "top_10_trials": top_table,
        "total_search_seconds_sum_of_trials": total_search_seconds,
        "cumulative_wall_seconds": cumulative_wall_seconds,
        "hard_timeout_seconds": tune_rf.HARD_TIMEOUT_SECONDS,
    }


# ---------------------------------------------------------------- 산출물 재검증/변환
def re_validate_rf_b_artifacts(
    paths: dict[str, Path],
    *,
    classes: np.ndarray,
    train_ids: np.ndarray,
    n_splits: int,
    sample_ids: list[str],
    expected_macro_f1: float,
) -> None:
    """저장된 파일을 다시 읽어 strict validator 를 재실행한다(spec Ticket 3 §8).

    `train_rf.main()` 이 저장 **전에** 이미 같은 validator 를 한 번 돌렸다 —
    여기서는 저장·재읽기 왕복 후에도 스키마·값이 그대로인지 독립적으로 다시 본다.
    """
    oof = pd.read_csv(paths["oof"], dtype={"ID": str})
    test_pred = pd.read_csv(paths["test"], dtype={"ID": str})
    submission = pd.read_csv(paths["submission"], dtype=str, encoding="utf-8-sig")

    validate_oof_frame(
        oof, class_order=classes, train_ids=train_ids, n_splits=n_splits,
        expected_macro_f1=expected_macro_f1,
    )
    validate_test_probability_frame(test_pred, class_order=classes, sample_submission_ids=sample_ids)
    validate_submission_frame(
        submission, class_order=classes, sample_submission_ids=sample_ids, test_proba_frame=test_pred
    )


def write_per_class_f1_csv(rf_b_log: dict, rf_a_log: dict, path: Path) -> None:
    rows = [
        {"class": cls, "rf_b_f1": rf_b_log["oof_per_class_f1"].get(cls), "rf_a_f1": rf_a_log["oof_per_class_f1"].get(cls)}
        for cls in rf_b_log["class_order"]
    ]
    save_csv(pd.DataFrame(rows), path)


def write_confusion_matrix_csvs(cm_json_path: Path, raw_csv_path: Path, normalized_csv_path: Path) -> None:
    with open(cm_json_path, encoding="utf-8") as handle:
        cm = json.load(handle)
    classes = cm["class_order"]
    raw = pd.DataFrame(cm["raw"], index=classes, columns=classes)
    raw.index.name = "true_class"
    save_csv(raw.reset_index(), raw_csv_path)

    normalized = pd.DataFrame(cm["normalized_by_true_row"], index=classes, columns=classes)
    normalized.index.name = "true_class"
    save_csv(normalized.reset_index(), normalized_csv_path)


# ---------------------------------------------------------------- RF-A 대비 판정(spec §9)
def compute_rf_b_vs_rf_a_decision(*, rf_a_log: dict, rf_b_log: dict) -> dict:
    class_order = rf_b_log["class_order"]
    assert class_order == rf_a_log["class_order"], "class order 가 RF-A/RF-B 사이에 다르다"

    rf_a_oof_frame_support: dict[str, int] | None = None
    # RF-A 로그에는 support 가 없다 — oof_pred_class_distribution 은 예측 분포일 뿐이라
    # 진짜 support(=y_true 분포)는 별도 파일 없이 log 만으로는 못 구한다. 대신 로그에
    # 있는 값 중 가장 가까운 대용치인 예측 분포를 참고용으로만 남긴다.
    rf_a_oof_frame_support = rf_a_log.get("oof_pred_class_distribution", {})

    class_diagnostics = []
    hard_warning = False
    soft_warning = False
    for cls in class_order:
        a_f1 = float(rf_a_log["oof_per_class_f1"].get(cls, 0.0))
        b_f1 = float(rf_b_log["oof_per_class_f1"].get(cls, 0.0))
        drop = a_f1 - b_f1
        flags: list[str] = []
        if a_f1 >= CLASS_HARD_ZERO_MIN_RF_A_F1 and b_f1 == 0.0:
            flags.append("HARD_ZERO_COLLAPSE")
            hard_warning = True
        if drop >= CLASS_HARD_DROP:
            flags.append("HARD_DROP")
            hard_warning = True
        elif drop >= CLASS_SOFT_DROP:
            flags.append("WARNING_DROP")
            soft_warning = True
        class_diagnostics.append(
            {
                "class": cls,
                "rf_a_f1": a_f1,
                "rf_b_f1": b_f1,
                "delta": b_f1 - a_f1,
                "rf_a_pred_count_reference_only": rf_a_oof_frame_support.get(cls),
                "flags": flags,
            }
        )

    rf_a_fold_scores = rf_a_log["fold_macro_f1"]
    rf_b_fold_scores = rf_b_log["fold_macro_f1"]
    rf_a_std = float(np.std(rf_a_fold_scores))
    rf_b_std = float(np.std(rf_b_fold_scores))
    fold_std_warning = rf_b_std > FOLD_STD_WARNING_MULTIPLIER * rf_a_std

    fold_comparison = [
        {"fold": i, "rf_a": a, "rf_b": b, "delta": b - a, "improved": b > a}
        for i, (a, b) in enumerate(zip(rf_a_fold_scores, rf_b_fold_scores))
    ]
    n_folds_improved = sum(1 for f in fold_comparison if f["improved"])

    overall_delta = rf_b_log["oof_macro_f1"] - rf_a_log["oof_macro_f1"]

    criteria = {
        "overall_delta_meets_0_005": overall_delta >= ACCEPT_MIN_DELTA,
        "at_least_3_of_5_folds_improved": n_folds_improved >= ACCEPT_MIN_FOLDS_IMPROVED,
        "no_hard_class_collapse": not hard_warning,
        "fold_std_not_excessive": not fold_std_warning,
    }
    all_criteria_met = all(criteria.values())

    if all_criteria_met:
        status = "채택"
        reason = "spec §9 4개 조건을 모두 만족한다."
    elif overall_delta < 0:
        status = "기각"
        reason = f"RF-B 전체 OOF Macro F1 이 RF-A 보다 {overall_delta:+.4f} 낮다."
    else:
        status = "보류"
        unmet = [k for k, v in criteria.items() if not v]
        reason = f"채택 조건 중 {unmet} 미충족 — 명확한 개선이 아니다."

    return {
        "rf_a_oof_macro_f1": rf_a_log["oof_macro_f1"],
        "rf_b_oof_macro_f1": rf_b_log["oof_macro_f1"],
        "overall_delta": overall_delta,
        "fold_comparison": fold_comparison,
        "n_folds_improved": n_folds_improved,
        "rf_a_fold_std": rf_a_std,
        "rf_b_fold_std": rf_b_std,
        "fold_std_warning": fold_std_warning,
        "class_diagnostics": class_diagnostics,
        "hard_class_warning": hard_warning,
        "soft_class_warning": soft_warning,
        "acceptance_criteria": criteria,
        "status": status,
        "status_reason": reason,
        "note": "OOF 상승이 DACON Public LB 상승을 보장한다고 해석하지 않는다(spec §1.1/§9).",
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
        "--rf-a-log-path", type=Path, default=PROJECT_ROOT / "artifacts/logs/rf_a_f4r_group5_s42.json"
    )
    parser.add_argument(
        "--rf-a-oof-path", type=Path, default=PROJECT_ROOT / "artifacts/oof/oof_rf_a_f4r_group5_s42.csv"
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--target-trials", type=int, default=MAX_TARGET_TRIALS)
    parser.add_argument(
        "--hard-timeout-seconds", type=float, default=tune_rf.HARD_TIMEOUT_SECONDS
    )
    parser.add_argument("--max-new-trials", type=int, default=None, help="이번 호출 상한(테스트/재개용)")
    parser.add_argument(
        "--overwrite-rf-b-artifacts", action="store_true",
        help="best trial 재학습 산출물(OOF/test/submission/log)이 이미 있어도 덮어쓴다",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.target_trials > MAX_TARGET_TRIALS:
        raise SystemExit(f"--target-trials 는 {MAX_TARGET_TRIALS} 을 넘을 수 없다(금지 §12).")

    data_dir = train_rf.resolve_data_dir(str(args.data_dir) if args.data_dir else None)
    data_hashes = train_rf.validate_data_dir(data_dir)

    f4r = train_rf.load_f4r_features(
        data_dir, n_splits=args.n_splits, class_order_path=args.class_order_path
    )
    classes = f4r["classes"]
    train_ids = f4r["dataset"].train_ids

    fold_ids, fold_col = train_rf.load_folds(
        args.folds_path, cv="sgkf", n_splits=args.n_splits, train_ids=train_ids
    )
    n_jobs = args.n_jobs if args.n_jobs is not None else default_n_jobs()
    fold_hash = tune_rf.compute_fold_hash(fold_ids)

    t0 = time.perf_counter()
    fold_cache = tune_rf.precompute_fold_cache(f4r, fold_ids, args.n_splits)
    fold_cache_build_seconds = time.perf_counter() - t0
    log(f"fold 행렬 캐시 생성: {fold_cache_build_seconds:.2f}s ({args.n_splits} folds)")

    tuning_dir = args.out_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) trial 0 RF-A 재현 검증(멱등 — 이미 통과 기록이 있으면 재사용) ----
    verification_path = tuning_dir / "rf_b_trial0_reproduction_check.json"
    if verification_path.exists():
        with open(verification_path, encoding="utf-8") as handle:
            trial0_check = json.load(handle)
        log(f"trial 0 RF-A 재현 검증: 기존 기록 재사용 (passed={trial0_check['passed']})")
    else:
        trial0_check = verify_trial0_reproduces_rf_a(
            fold_cache, classes, train_ids, fold_ids,
            seed=args.seed, n_jobs=n_jobs,
            rf_a_oof_path=args.rf_a_oof_path, rf_a_log_path=args.rf_a_log_path,
        )
        with open(verification_path, "w", encoding="utf-8") as handle:
            json.dump(trial0_check, handle, ensure_ascii=False, indent=2)
        log(f"trial 0 RF-A 재현 검증: passed={trial0_check['passed']}  "
            f"max|Δproba|={trial0_check['max_abs_proba_diff']:.2e}  "
            f"ΔOOF={trial0_check['oof_macro_f1_diff']:.2e}")

    if not trial0_check["passed"]:
        raise SystemExit(
            "trial 0(RF-A 기준선) 재현 실패 — 허용오차를 임의로 넓히지 않고 본 탐색을 "
            "중단한다(tickets Ticket 3 §3).\n" + json.dumps(trial0_check, ensure_ascii=False, indent=2)
        )

    # ---- 2) 본 탐색: study 전체 COMPLETE 가 target_total_trials 가 될 때까지 ----
    db_path = tuning_dir / tune_rf.MAIN_DB_FILENAME
    study, run_info = tune_rf.run_main_study(
        fold_cache, classes, db_path=db_path, seed=args.seed, sampler_seed=args.sampler_seed,
        n_jobs=n_jobs, target_total_trials=args.target_trials, fold_hash=fold_hash,
        hard_timeout_seconds=args.hard_timeout_seconds, max_new_trials=args.max_new_trials,
    )
    n_complete = tune_rf.count_complete(study)
    log(
        f"본 탐색: complete={n_complete}/{args.target_trials}  "
        f"이번 호출 실행={run_info['ran_this_call']}  누적={run_info['cumulative_wall_seconds']:.0f}s  "
        f"stop_reason={run_info['stop_reason']}"
    )

    # ---- 3) trials/best_params/search_summary 는 진행 상태와 무관하게 남긴다 ----
    trials_csv_path = tuning_dir / "rf_b_trials.csv"
    save_csv(_main_trials_to_frame(study), trials_csv_path)
    log(f"trials CSV -> {trials_csv_path}")

    with open(args.rf_a_log_path, encoding="utf-8") as handle:
        rf_a_log = json.load(handle)

    search_summary = build_search_summary(
        study, rf_a_log=rf_a_log, target_total_trials=args.target_trials,
        trial0_check=trial0_check, fold_hash=fold_hash, data_hashes=data_hashes,
        folds_file_info={"name": args.folds_path.name, "fold_column": fold_col},
        git_commit=train_rf._git_commit_sha(), cumulative_wall_seconds=run_info["cumulative_wall_seconds"],
    )

    if n_complete < MIN_TRIALS_FOR_DECISION:
        status_label = "INSUFFICIENT_HOLD"
    elif n_complete >= args.target_trials:
        status_label = "COMPLETE_ALL"
    else:
        status_label = "PARTIAL_ANALYZABLE"
    search_summary["status_label"] = status_label

    with open(tuning_dir / "rf_b_search_summary.json", "w", encoding="utf-8") as handle:
        json.dump(search_summary, handle, ensure_ascii=False, indent=2)
    log(f"search summary -> {tuning_dir / 'rf_b_search_summary.json'}  status={status_label}")

    if search_summary["best_params"] is not None:
        best_params_path = tuning_dir / "rf_b_best_params.json"
        with open(best_params_path, "w", encoding="utf-8") as handle:
            json.dump(search_summary["best_params"], handle, ensure_ascii=False, indent=2)
        log(f"best params -> {best_params_path}")

    if status_label == "INSUFFICIENT_HOLD":
        log(
            f"COMPLETE trial 이 {n_complete}개(<{MIN_TRIALS_FOR_DECISION}) — "
            "RF-B 채택 판단을 보류한다. 재실행해 더 진행한다."
        )
        return {"status": status_label, "search_summary": search_summary}

    if status_label == "PARTIAL_ANALYZABLE":
        log(
            f"주의: {run_info['stop_reason']} 로 {args.target_trials}-trial 을 다 못 채웠다"
            f"({n_complete}개 COMPLETE). 분석은 하되 '{args.target_trials}-trial 완료' 라고 "
            "표현하지 않는다."
        )

    # ---- 4) best trial 최종 재학습 — train_rf.py 재사용, 여기서만 test 사용 ----
    best_params_path = tuning_dir / "rf_b_best_params.json"
    train_rf_argv = [
        "--model", "rf",
        "--feature-set", "f4r",
        "--tag", "b_f4r",
        "--data-dir", str(data_dir),
        "--folds-path", str(args.folds_path),
        "--class-order-path", str(args.class_order_path),
        "--config", str(best_params_path),
        "--seed", str(args.seed),
        "--n-jobs", str(n_jobs),
        "--out-dir", str(args.out_dir),
    ]
    if args.overwrite_rf_b_artifacts:
        train_rf_argv.append("--overwrite")
    log("best trial 재학습 -> scripts/train_rf.py 재사용(canonical OOF/test/submission/log)")
    rf_b_provenance = train_rf.main(train_rf_argv)

    # ---- 5) 저장 후 재읽기 strict 재검증 ----
    stem = f"rf_b_f4r_{train_rf.CV_SLUG['sgkf']}_s{args.seed}"
    paths = train_rf.output_paths(args.out_dir, stem)
    sample_ids = train_rf.load_sample_submission_ids(data_dir)
    re_validate_rf_b_artifacts(
        paths, classes=classes, train_ids=train_ids, n_splits=args.n_splits,
        sample_ids=sample_ids, expected_macro_f1=rf_b_provenance["oof_macro_f1"],
    )
    log("저장된 OOF/test/submission 재읽기 재검증 통과.")

    # ---- 6) per-class F1 / confusion matrix CSV ----
    metrics_dir = args.out_dir / "metrics"
    write_per_class_f1_csv(
        rf_b_provenance, rf_a_log, metrics_dir / f"per_class_f1_{stem}.csv"
    )
    write_confusion_matrix_csvs(
        paths["confusion_matrix"],
        metrics_dir / f"confusion_matrix_{stem}.csv",
        metrics_dir / f"confusion_matrix_{stem}_normalized.csv",
    )

    # ---- 7) RF-A 대비 판정 ----
    decision = compute_rf_b_vs_rf_a_decision(rf_a_log=rf_a_log, rf_b_log=rf_b_provenance)
    decision_path = metrics_dir / "rf_b_vs_rf_a_decision.json"
    with open(decision_path, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    log(f"RF-A 대비 판정: {decision['status']} ({decision['status_reason']}) -> {decision_path}")

    return {
        "status": status_label,
        "search_summary": search_summary,
        "rf_b_provenance": rf_b_provenance,
        "decision": decision,
    }


if __name__ == "__main__":
    main()
