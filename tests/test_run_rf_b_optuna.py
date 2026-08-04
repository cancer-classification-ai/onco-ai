"""`scripts/run_rf_b_optuna.py` 계약 — trial 0 재현 검증, 탐색 요약, RF-A 대비 판정 (Ticket 3).

원본은 `docs/specs/random_forest_stacking_model.md` §5/§9, `..._tickets.md` Ticket 3.
전부 합성 데이터·임시 파일로 돈다. `train_rf.main()`(best trial 최종 재학습)은
`data/process/*.parquet` 실제 f4r 산출물에 고정돼 있어 합성으로 대체할 수 없으므로
이 테스트 파일에서 호출하지 않는다 — 실제 데이터로는 스크립트를 직접 실행해
검증한다(Ticket 2 와 동일 원칙).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_rf_b_optuna as r  # noqa: E402
import tune_optuna_rf as tune_rf  # noqa: E402

from cancer_hack.validation import fold_assignment  # noqa: E402

N_CLASSES = 6
CLASS_ORDER = np.array([f"C{i:02d}" for i in range(N_CLASSES)])
N_SPLITS = 5
N_FEATURES = 12
N_GENE = 20


class _FakeDataset:
    pass


def make_synthetic_f4r(*, n_samples: int = 180, random_state: int = 0):
    X, yi = make_classification(
        n_samples=n_samples, n_features=N_FEATURES, n_informative=8, n_redundant=0,
        n_classes=N_CLASSES, n_clusters_per_class=1, random_state=random_state,
    )
    y = CLASS_ORDER[yi]
    train_ids = np.array([f"tr{i:04d}" for i in range(n_samples)])

    dataset = _FakeDataset()
    dataset.y = y
    dataset.rollup_train = None
    dataset.train_ids = train_ids

    rng = np.random.RandomState(random_state)
    gene_train = np.abs(rng.randn(n_samples, N_GENE)).astype(np.float32)
    f4r = {
        "dataset": dataset,
        "classes": CLASS_ORDER,
        "names": [f"f{i}" for i in range(N_FEATURES)],
        "dense_train": X.astype(np.float32),
        "burden_positions": [],
        "gene_columns": [f"g{i}" for i in range(N_GENE)],
        "gene_train": gene_train,
    }
    fold_ids = fold_assignment(y, kind="sgkf", n_splits=N_SPLITS, seed=42, groups=np.arange(n_samples))
    return f4r, y, train_ids, fold_ids


@pytest.fixture(autouse=True)
def _synthetic_feature_dims(monkeypatch):
    monkeypatch.setattr(tune_rf.train_rf, "F4R_EXPECTED_FEATURES", N_FEATURES + N_GENE)
    monkeypatch.setattr(tune_rf.train_rf, "F4R_ENC3_TOPK", N_GENE)


def _write_synthetic_rf_a_artifacts(tmp_path, cache, train_ids, fold_ids):
    """RF_A_BASELINE_POINT 로 실제 계산한 값을 "RF-A 저장 산출물"인 척 파일로 남긴다."""
    baseline_params = tune_rf._resolve_baseline_point(tune_rf.RF_A_BASELINE_POINT)
    result = tune_rf.evaluate_params_on_cached_folds(
        cache, CLASS_ORDER, baseline_params, seed=42, n_jobs=1, return_oof=True
    )
    proba_cols = [f"p_{c}" for c in CLASS_ORDER]
    oof_df = pd.DataFrame(result["oof_proba"], columns=proba_cols)
    oof_df.insert(0, "ID", train_ids)
    oof_df.insert(1, "fold", fold_ids)
    oof_df["y_true"] = result["y_all"]
    oof_df["y_pred"] = result["oof_pred"]

    oof_path = tmp_path / "oof_rf_a.csv"
    log_path = tmp_path / "rf_a_log.json"
    oof_df.to_csv(oof_path, index=False)
    log = {
        "class_order": CLASS_ORDER.tolist(),
        "oof_macro_f1": result["oof_macro_f1"],
        "fold_macro_f1": result["fold_macro_f1"],
        "oof_per_class_f1": result["oof_per_class_f1"],
    }
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log, handle)
    return oof_path, log_path, result


# ---------------------------------------------------------------- trial 0 재현 검증


def test_trial0_verification_passes_against_self_consistent_rf_a(tmp_path):
    f4r, y, train_ids, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    oof_path, log_path, _ = _write_synthetic_rf_a_artifacts(tmp_path, cache, train_ids, fold_ids)

    checks = r.verify_trial0_reproduces_rf_a(
        cache, CLASS_ORDER, train_ids, fold_ids, seed=42, n_jobs=1,
        rf_a_oof_path=oof_path, rf_a_log_path=log_path,
    )
    assert checks["passed"] is True
    assert checks["max_abs_proba_diff"] == 0.0
    assert checks["oof_macro_f1_diff"] == 0.0


def test_trial0_verification_detects_fold_mismatch(tmp_path):
    f4r, y, train_ids, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    oof_path, log_path, _ = _write_synthetic_rf_a_artifacts(tmp_path, cache, train_ids, fold_ids)

    tampered = pd.read_csv(oof_path, dtype={"ID": str})
    tampered.loc[0, "fold"] = int((tampered.loc[0, "fold"] + 1) % N_SPLITS)
    tampered_path = tmp_path / "oof_rf_a_tampered.csv"
    tampered.to_csv(tampered_path, index=False)

    checks = r.verify_trial0_reproduces_rf_a(
        cache, CLASS_ORDER, train_ids, fold_ids, seed=42, n_jobs=1,
        rf_a_oof_path=tampered_path, rf_a_log_path=log_path,
    )
    assert checks["passed"] is False
    assert checks["fold_matches"] is False


def test_trial0_verification_detects_probability_drift(tmp_path):
    f4r, y, train_ids, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    oof_path, log_path, _ = _write_synthetic_rf_a_artifacts(tmp_path, cache, train_ids, fold_ids)

    tampered = pd.read_csv(oof_path, dtype={"ID": str})
    proba_col = [c for c in tampered.columns if c.startswith("p_")][0]
    tampered.loc[0, proba_col] = tampered.loc[0, proba_col] + 0.01
    tampered_path = tmp_path / "oof_rf_a_drift.csv"
    tampered.to_csv(tampered_path, index=False)

    checks = r.verify_trial0_reproduces_rf_a(
        cache, CLASS_ORDER, train_ids, fold_ids, seed=42, n_jobs=1,
        rf_a_oof_path=tampered_path, rf_a_log_path=log_path,
        atol_proba=1e-6,
    )
    assert checks["passed"] is False
    assert checks["proba_within_tolerance"] is False
    assert checks["max_abs_proba_diff"] >= 0.009


def test_trial0_verification_detects_class_order_mismatch(tmp_path):
    f4r, y, train_ids, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    oof_path, log_path, _ = _write_synthetic_rf_a_artifacts(tmp_path, cache, train_ids, fold_ids)

    with open(log_path, encoding="utf-8") as handle:
        log = json.load(handle)
    log["class_order"] = list(reversed(log["class_order"]))
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log, handle)

    checks = r.verify_trial0_reproduces_rf_a(
        cache, CLASS_ORDER, train_ids, fold_ids, seed=42, n_jobs=1,
        rf_a_oof_path=oof_path, rf_a_log_path=log_path,
    )
    assert checks["passed"] is False
    assert checks["class_order_matches"] is False


# ---------------------------------------------------------------- trial 0 캐시 provenance
#
# 실제 RF-A 재현(모델 학습)은 이 섹션에서 돌리지 않는다 — `build_trial0_cache_context`/
# `trial0_cache_is_reusable`/`resolve_trial0_check` 는 순수 캐시 정책 함수라 합성
# fixture(작은 임시 파일)와 mock `compute` 콜백만으로 검증한다(tickets Ticket 3 §3).


def _sample_trial0_context(**overrides) -> dict:
    base = {
        "schema_version": r.TRIAL0_CACHE_SCHEMA_VERSION,
        "git_commit": "abc123",
        "data_files": {"train.csv": "h1", "test.csv": "h2", "sample_submission.csv": "h3"},
        "fold_hash": "fh1",
        "folds_file": {"name": "train_folds.parquet", "sha256": "fs1"},
        "rf_a_oof_sha256": "oof1",
        "rf_a_log_sha256": "log1",
        "class_order": ["C00", "C01"],
        "seed": 42,
        "n_splits": 5,
        "rf_a_baseline_params": {"n_estimators": 500, "max_features": "sqrt"},
    }
    base.update(overrides)
    return base


def test_build_trial0_cache_context_hashes_files_and_omits_paths(tmp_path):
    folds_path = tmp_path / "train_folds.parquet"
    folds_path.write_bytes(b"fold-bytes")
    rf_a_oof_path = tmp_path / "oof_rf_a.csv"
    rf_a_oof_path.write_text("ID,fold\n1,0\n", encoding="utf-8")
    rf_a_log_path = tmp_path / "rf_a_log.json"
    rf_a_log_path.write_text("{}", encoding="utf-8")

    context = r.build_trial0_cache_context(
        git_commit="deadbeef",
        data_hashes={"train.csv": "h1", "test.csv": "h2", "sample_submission.csv": "h3"},
        fold_hash="fh1",
        folds_path=folds_path,
        rf_a_oof_path=rf_a_oof_path,
        rf_a_log_path=rf_a_log_path,
        classes=np.array(["C01", "C00"]),
        seed=42,
        n_splits=5,
    )
    for key in r.TRIAL0_CACHE_CONTEXT_KEYS:
        assert key in context
    assert context["folds_file"] == {
        "name": "train_folds.parquet", "sha256": r.train_rf._sha256(folds_path)
    }
    assert context["class_order"] == ["C01", "C00"]
    assert context["seed"] == 42
    assert context["n_splits"] == 5
    # 비밀정보·절대경로는 담지 않는다.
    assert str(tmp_path) not in json.dumps(context)


def test_trial0_cache_reused_when_context_matches_exactly():
    context = _sample_trial0_context()
    cached = {"passed": True, "context": dict(context)}
    assert r.trial0_cache_is_reusable(cached, context) is True


@pytest.mark.parametrize(
    "override",
    [
        {"git_commit": "different-commit"},
        {"data_files": {"train.csv": "changed", "test.csv": "h2", "sample_submission.csv": "h3"}},
        {"fold_hash": "different-fold-hash"},
        {"folds_file": {"name": "train_folds.parquet", "sha256": "different-sha"}},
        {"rf_a_oof_sha256": "different-oof-sha"},
        {"rf_a_log_sha256": "different-log-sha"},
        {"class_order": ["C01", "C00"]},
        {"seed": 1},
        {"n_splits": 10},
        {"rf_a_baseline_params": {"n_estimators": 999}},
    ],
)
def test_trial0_cache_stale_when_any_provenance_field_differs(override):
    current = _sample_trial0_context()
    cached = {"passed": True, "context": _sample_trial0_context(**override)}
    assert r.trial0_cache_is_reusable(cached, current) is False


def test_trial0_cache_not_reused_when_legacy_cache_has_no_context():
    current = _sample_trial0_context()
    assert r.trial0_cache_is_reusable({"passed": True}, current) is False


def test_trial0_cache_not_reused_when_legacy_context_missing_fields():
    current = _sample_trial0_context()
    partial = _sample_trial0_context()
    del partial["fold_hash"]
    assert r.trial0_cache_is_reusable({"passed": True, "context": partial}, current) is False


def test_trial0_cache_never_reused_when_cached_passed_is_false():
    current = _sample_trial0_context()
    cached = {"passed": False, "context": dict(current)}
    assert r.trial0_cache_is_reusable(cached, current) is False


def test_resolve_trial0_check_computes_and_persists_context_when_cache_missing(tmp_path, capsys):
    context = _sample_trial0_context()
    verification_path = tmp_path / "check.json"
    calls = []

    def compute():
        calls.append(1)
        return {"passed": True, "max_abs_proba_diff": 1e-9, "oof_macro_f1_diff": 0.0}

    result = r.resolve_trial0_check(verification_path, context, compute=compute)
    assert calls == [1]
    assert result["passed"] is True
    saved = json.loads(verification_path.read_text(encoding="utf-8"))
    assert saved["context"] == context
    assert "cache missing: computing" in capsys.readouterr().out


def test_resolve_trial0_check_reuses_cache_without_recomputing_on_exact_match(tmp_path, capsys):
    context = _sample_trial0_context()
    verification_path = tmp_path / "check.json"
    verification_path.write_text(
        json.dumps({"passed": True, "max_abs_proba_diff": 0.0, "oof_macro_f1_diff": 0.0, "context": context}),
        encoding="utf-8",
    )

    def compute():
        raise AssertionError("context 가 일치하면 재계산이 호출되면 안 된다")

    result = r.resolve_trial0_check(verification_path, context, compute=compute)
    assert result["passed"] is True
    assert "cache reused: exact context match" in capsys.readouterr().out


def test_resolve_trial0_check_recomputes_and_overwrites_when_context_stale(tmp_path, capsys):
    stale_context = _sample_trial0_context(git_commit="old-commit")
    fresh_context = _sample_trial0_context(git_commit="new-commit")
    verification_path = tmp_path / "check.json"
    verification_path.write_text(
        json.dumps({"passed": True, "max_abs_proba_diff": 0.0, "oof_macro_f1_diff": 0.0, "context": stale_context}),
        encoding="utf-8",
    )
    calls = []

    def compute():
        calls.append(1)
        return {"passed": True, "max_abs_proba_diff": 2e-9, "oof_macro_f1_diff": 0.0}

    result = r.resolve_trial0_check(verification_path, fresh_context, compute=compute)
    assert calls == [1]
    assert result["passed"] is True
    saved = json.loads(verification_path.read_text(encoding="utf-8"))
    assert saved["context"] == fresh_context
    assert "cache stale: context mismatch, recomputing" in capsys.readouterr().out


def test_resolve_trial0_check_recomputes_when_cached_passed_is_false(tmp_path, capsys):
    context = _sample_trial0_context()
    verification_path = tmp_path / "check.json"
    verification_path.write_text(
        json.dumps({"passed": False, "context": context}), encoding="utf-8"
    )
    calls = []

    def compute():
        calls.append(1)
        return {"passed": True, "max_abs_proba_diff": 0.0, "oof_macro_f1_diff": 0.0}

    result = r.resolve_trial0_check(verification_path, context, compute=compute)
    assert calls == [1]
    assert result["passed"] is True
    assert "cache stale: context mismatch, recomputing" in capsys.readouterr().out


# ---------------------------------------------------------------- 탐색 요약


def test_build_search_summary_picks_best_trial_and_computes_delta(tmp_path):
    f4r, y, train_ids, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    fold_hash = tune_rf.compute_fold_hash(fold_ids)
    study, run_info = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=tmp_path / "main.db", seed=42, sampler_seed=42, n_jobs=1,
        target_total_trials=4, fold_hash=fold_hash,
    )
    rf_a_log = {"oof_macro_f1": 0.4460266317834502, "class_order": CLASS_ORDER.tolist()}

    summary = r.build_search_summary(
        study, rf_a_log=rf_a_log, target_total_trials=4, trial0_check={"passed": True},
        fold_hash=fold_hash, data_hashes={}, folds_file_info={}, git_commit="deadbeef",
        cumulative_wall_seconds=run_info["cumulative_wall_seconds"],
    )
    assert summary["n_complete"] == 4
    assert summary["best_trial_number"] is not None
    assert summary["best_value_oof_macro_f1"] == study.best_trial.value
    assert summary["delta_vs_rf_a"] == pytest.approx(study.best_trial.value - rf_a_log["oof_macro_f1"])
    assert summary["best_params"] == study.best_trial.user_attrs["params"]
    assert set(summary["best_params_boundary_flags"]) == set(r.SEARCH_SPACE_BOUNDARIES) - {"max_features"}
    assert len(summary["top_10_trials"]) <= 4


def test_param_is_boundary_flags_extremes_correctly():
    assert r._param_is_boundary("n_estimators", 400) is True
    assert r._param_is_boundary("n_estimators", 1000) is True
    assert r._param_is_boundary("n_estimators", 700) is False
    assert r._param_is_boundary("max_depth", 8) is True
    assert r._param_is_boundary("max_depth", None) is True
    assert r._param_is_boundary("max_depth", 16) is False
    assert r._param_is_boundary("max_features_key", "sqrt") is True
    assert r._param_is_boundary("max_features_key", "frac_010") is False


# ---------------------------------------------------------------- RF-A 대비 판정(§9)


def _make_log(oof_f1, fold_scores, per_class):
    return {
        "class_order": ["ACC", "BRCA", "DLBC"],
        "oof_macro_f1": oof_f1,
        "fold_macro_f1": fold_scores,
        "oof_per_class_f1": per_class,
    }


def test_decision_accepts_when_all_four_criteria_met():
    rf_a = _make_log(0.4460, [0.43, 0.44, 0.45, 0.46, 0.44], {"ACC": 0.70, "BRCA": 0.45, "DLBC": 0.30})
    rf_b = _make_log(0.4520, [0.44, 0.45, 0.46, 0.47, 0.45], {"ACC": 0.72, "BRCA": 0.46, "DLBC": 0.32})
    decision = r.compute_rf_b_vs_rf_a_decision(rf_a_log=rf_a, rf_b_log=rf_b)
    assert decision["status"] == "채택"
    assert all(decision["acceptance_criteria"].values())


def test_decision_rejects_when_overall_delta_negative():
    rf_a = _make_log(0.4460, [0.43, 0.44, 0.45, 0.46, 0.44], {"ACC": 0.70, "BRCA": 0.45, "DLBC": 0.30})
    rf_b = _make_log(0.4300, [0.40, 0.41, 0.42, 0.44, 0.42], {"ACC": 0.60, "BRCA": 0.40, "DLBC": 0.25})
    decision = r.compute_rf_b_vs_rf_a_decision(rf_a_log=rf_a, rf_b_log=rf_b)
    assert decision["status"] == "기각"


def test_decision_holds_on_hard_class_collapse_even_with_positive_delta():
    rf_a = _make_log(0.4460, [0.43, 0.44, 0.45, 0.46, 0.44], {"ACC": 0.70, "BRCA": 0.45, "DLBC": 0.30})
    rf_b = _make_log(0.4520, [0.44, 0.45, 0.46, 0.47, 0.45], {"ACC": 0.0, "BRCA": 0.46, "DLBC": 0.32})
    decision = r.compute_rf_b_vs_rf_a_decision(rf_a_log=rf_a, rf_b_log=rf_b)
    assert decision["status"] == "보류"
    assert decision["hard_class_warning"] is True
    collapsed = next(c for c in decision["class_diagnostics"] if c["class"] == "ACC")
    assert "HARD_ZERO_COLLAPSE" in collapsed["flags"]


def test_decision_holds_when_fewer_than_three_folds_improve():
    rf_a = _make_log(0.4460, [0.50, 0.50, 0.50, 0.30, 0.30], {"ACC": 0.70, "BRCA": 0.45, "DLBC": 0.30})
    rf_b = _make_log(0.4520, [0.45, 0.45, 0.45, 0.60, 0.60], {"ACC": 0.71, "BRCA": 0.46, "DLBC": 0.31})
    decision = r.compute_rf_b_vs_rf_a_decision(rf_a_log=rf_a, rf_b_log=rf_b)
    assert decision["n_folds_improved"] == 2
    assert decision["status"] == "보류"


def test_decision_holds_on_excessive_fold_std_increase():
    rf_a = _make_log(0.4460, [0.44, 0.45, 0.45, 0.44, 0.45], {"ACC": 0.70, "BRCA": 0.45, "DLBC": 0.30})
    # 평균은 더 높지만 fold 편차가 RF-A 의 2배를 훌쩍 넘는다.
    rf_b = _make_log(0.4520, [0.20, 0.80, 0.30, 0.70, 0.50], {"ACC": 0.72, "BRCA": 0.46, "DLBC": 0.32})
    decision = r.compute_rf_b_vs_rf_a_decision(rf_a_log=rf_a, rf_b_log=rf_b)
    assert decision["fold_std_warning"] is True
    assert decision["status"] == "보류"


def test_decision_soft_warning_does_not_block_acceptance_alone():
    """0.10~0.15 사이의 소프트 하락은 warning 이지만, 나머지 3개 기준을 다 만족하면 채택된다."""
    rf_a = _make_log(0.4460, [0.43, 0.44, 0.45, 0.46, 0.44], {"ACC": 0.70, "BRCA": 0.45, "DLBC": 0.30})
    rf_b = _make_log(0.4520, [0.44, 0.45, 0.46, 0.47, 0.45], {"ACC": 0.59, "BRCA": 0.46, "DLBC": 0.32})
    decision = r.compute_rf_b_vs_rf_a_decision(rf_a_log=rf_a, rf_b_log=rf_b)
    assert decision["soft_class_warning"] is True
    assert decision["hard_class_warning"] is False
    assert decision["status"] == "채택"


# ---------------------------------------------------------------- metrics export


def test_write_per_class_f1_csv_round_trip(tmp_path):
    rf_a_log = {"oof_per_class_f1": {"ACC": 0.70, "BRCA": 0.45}}
    rf_b_log = {"class_order": ["ACC", "BRCA"], "oof_per_class_f1": {"ACC": 0.72, "BRCA": 0.46}}
    path = tmp_path / "per_class_f1.csv"
    r.write_per_class_f1_csv(rf_b_log, rf_a_log, path)

    frame = pd.read_csv(path)
    assert list(frame.columns) == ["class", "rf_b_f1", "rf_a_f1"]
    assert frame.loc[frame["class"] == "ACC", "rf_b_f1"].iloc[0] == pytest.approx(0.72)
    assert frame.loc[frame["class"] == "ACC", "rf_a_f1"].iloc[0] == pytest.approx(0.70)


def test_write_confusion_matrix_csvs_round_trip(tmp_path):
    classes = ["ACC", "BRCA"]
    cm = {
        "class_order": classes,
        "raw": [[10, 2], [3, 15]],
        "normalized_by_true_row": [[0.833, 0.167], [0.167, 0.833]],
    }
    cm_path = tmp_path / "cm.json"
    with open(cm_path, "w", encoding="utf-8") as handle:
        json.dump(cm, handle)

    raw_csv = tmp_path / "cm_raw.csv"
    norm_csv = tmp_path / "cm_norm.csv"
    r.write_confusion_matrix_csvs(cm_path, raw_csv, norm_csv)

    raw_frame = pd.read_csv(raw_csv)
    assert list(raw_frame.columns) == ["true_class", "ACC", "BRCA"]
    assert raw_frame.loc[raw_frame["true_class"] == "ACC", "BRCA"].iloc[0] == 2

    norm_frame = pd.read_csv(norm_csv)
    assert norm_frame.loc[norm_frame["true_class"] == "BRCA", "ACC"].iloc[0] == pytest.approx(0.167)


# ---------------------------------------------------------------- CLI


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        r.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--target-trials" in out


def test_cli_rejects_target_trials_above_40(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_DATA_DIR", str(tmp_path))  # main() 이 이 인자보다 먼저 죽어야 한다
    with pytest.raises(SystemExit, match="40"):
        r.main(["--target-trials", "41"])


# ---------------------------------------------------------------- 저장 후 재검증 — group leakage 배선


def test_re_validate_rf_b_artifacts_catches_group_crossing_via_passed_mapping(tmp_path):
    """`group_key_by_id` 가 `re_validate_rf_b_artifacts` 안에서 실제로
    `validate_oof_frame` 에 전달되는지 확인한다. 이 배선이 제거되면(인자를 받고도
    버리면) 아래 손상된 OOF 가 통과해버려 이 테스트가 실패한다."""
    classes = np.array(["C00", "C01"])
    ids = ["tr0", "tr1", "tr2", "tr3"]
    # tr0/tr1 은 같은 group("g0")인데 서로 다른 fold(0, 1)에 있다 — 손상된 OOF.
    oof = pd.DataFrame({
        "ID": ids,
        "fold": [0, 1, 0, 1],
        "y_true": ["C00", "C00", "C01", "C01"],
        "y_pred": ["C00", "C00", "C01", "C01"],
        "p_C00": [0.9, 0.9, 0.1, 0.1],
        "p_C01": [0.1, 0.1, 0.9, 0.9],
    })
    test_pred = pd.DataFrame({"ID": ["te0", "te1"], "p_C00": [0.5, 0.5], "p_C01": [0.5, 0.5]})
    submission = pd.DataFrame({"ID": ["te0", "te1"], "SUBCLASS": ["C00", "C01"]})

    paths = {
        "oof": tmp_path / "oof.csv",
        "test": tmp_path / "test.csv",
        "submission": tmp_path / "submission.csv",
    }
    oof.to_csv(paths["oof"], index=False)
    test_pred.to_csv(paths["test"], index=False)
    submission.to_csv(paths["submission"], index=False, encoding="utf-8-sig")

    expected_macro_f1 = r.macro_f1_with_labels(oof["y_true"], oof["y_pred"], classes)

    with pytest.raises(ValueError, match="동일 group 이 fold 를 넘는 사례"):
        r.re_validate_rf_b_artifacts(
            paths, classes=classes, train_ids=np.array(ids), n_splits=2,
            sample_ids=["te0", "te1"], expected_macro_f1=expected_macro_f1,
            group_key_by_id={"tr0": "g0", "tr1": "g0", "tr2": "g1", "tr3": "g2"},
        )
