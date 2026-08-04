"""`scripts/tune_optuna_rf.py` 계약 — Optuna smoke/재개, 탐색 공간, budget 계산 (Ticket 2).

원본은 `docs/specs/random_forest_stacking_model.md` §5, `..._tickets.md` Ticket 2.
전부 합성 데이터·임시 SQLite 파일로 돈다. f4r 조립(`train_rf.load_f4r_features`)은
`data/process/*.parquet` 실제 파일에 고정돼 있어 합성으로 대체할 수 없으므로,
그 아래 단계(`precompute_fold_cache`/`evaluate_params_on_cached_folds`)부터를
합성 "f4r-모양" dict 로 검증한다 — 이 dict 가 요구하는 키만 채우면 f4r 여부와
무관하게 캐시·objective·재개 로직을 전부 재현할 수 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tune_optuna_rf as tune_rf  # noqa: E402

from cancer_hack.validation import Chi2TopKSelector, fold_assignment  # noqa: E402

N_CLASSES = 6
CLASS_ORDER = np.array([f"C{i:02d}" for i in range(N_CLASSES)])
N_SPLITS = 5
N_FEATURES = 12
N_GENE = 20


class _FakeDataset:
    """`train_gbdt.Dataset` 흉내 — `precompute_fold_cache` 가 쓰는 속성만 가진다."""


def make_synthetic_f4r(*, n_samples: int = 180, random_state: int = 0, with_test_sentinel: bool = False):
    X, yi = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=8,
        n_redundant=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    y = CLASS_ORDER[yi]

    dataset = _FakeDataset()
    dataset.y = y
    dataset.rollup_train = None  # burden_positions 가 비어 있으니 안 쓰인다

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
    if with_test_sentinel:
        f4r["dense_test"] = _BoomOnAccess()
        f4r["gene_test"] = _BoomOnAccess()
        f4r["test_ids"] = _BoomOnAccess()

    fold_ids = fold_assignment(
        y, kind="sgkf", n_splits=N_SPLITS, seed=42, groups=np.arange(n_samples)
    )
    return f4r, y, fold_ids


class _BoomOnAccess:
    """test 데이터가 objective 경로에서 참조되면 즉시 실패하게 만드는 감시용 sentinel."""

    def __getitem__(self, item):
        raise AssertionError("test 데이터(dense_test/gene_test/test_ids)가 objective 에서 접근됐다")

    def __getattr__(self, item):
        raise AssertionError("test 데이터(dense_test/gene_test/test_ids)가 objective 에서 접근됐다")


@pytest.fixture(autouse=True)
def _synthetic_feature_dims(monkeypatch):
    """실제 f4r 은 1,055열 고정이지만 합성 데이터는 훨씬 작다 — 기대 열 수를 맞춰 둔다."""
    monkeypatch.setattr(tune_rf.train_rf, "F4R_EXPECTED_FEATURES", N_FEATURES + N_GENE)
    monkeypatch.setattr(tune_rf.train_rf, "F4R_ENC3_TOPK", N_GENE)


# ---------------------------------------------------------------- max_features 문자열 key 매핑


def test_max_features_keys_cover_spec_5_4_mapping():
    assert tune_rf.MAX_FEATURES_KEYS == {
        "sqrt": "sqrt",
        "log2": "log2",
        "frac_005": 0.05,
        "frac_010": 0.10,
        "frac_020": 0.20,
    }


def test_suggest_rf_b_params_maps_max_features_key_to_sklearn_value():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = tune_rf.suggest_rf_b_params(trial)

    assert params["max_features"] == tune_rf.MAX_FEATURES_KEYS[trial.params["max_features_key"]]
    # SQLite 에 문자열/실수를 같은 이름으로 섞지 않는다 — 실제 저장되는 파라미터 이름은 "키"뿐.
    assert "max_features_key" in trial.params
    assert "max_features" not in trial.params


def test_suggest_rf_b_params_search_space_matches_spec_5_1():
    study = optuna.create_study(direction="maximize")
    seen_n_estimators = set()
    seen_max_depth = set()
    for _ in range(30):
        trial = study.ask()
        params = tune_rf.suggest_rf_b_params(trial)
        assert params["n_estimators"] in range(400, 1001, 100)
        assert params["max_depth"] in (8, 12, 16, 20, 24, None)
        assert params["min_samples_split"] in (2, 5, 10, 20)
        assert params["min_samples_leaf"] in (1, 2, 4, 8)
        assert params["max_samples"] in (0.65, 0.80, 1.0)
        assert params["criterion"] in ("gini", "log_loss")
        assert params["class_weight"] in ("balanced", "balanced_subsample")
        seen_n_estimators.add(params["n_estimators"])
        seen_max_depth.add(params["max_depth"])
        study.tell(trial, 0.0)
    assert len(seen_n_estimators) > 1  # 실제로 표본이 다양해야 한다(퇴화 아님)


def test_rf_a_baseline_point_replays_exactly_through_search_space():
    """RF-A 설정이 §5.1 탐색 공간 좌표로 정확히 표현되는지 확인한다(Ticket 3 trial 0 전제)."""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial(tune_rf.RF_A_BASELINE_POINT)

    captured = {}

    def objective(trial):
        captured.update(tune_rf.suggest_rf_b_params(trial))
        return 0.0

    study.optimize(objective, n_trials=1)
    resolved = tune_rf._resolve_baseline_point(tune_rf.RF_A_BASELINE_POINT)
    assert captured == resolved


# ---------------------------------------------------------------- sampler seed / pruner


def test_tpe_sampler_seed_42_is_deterministic_across_independent_studies():
    def run_once():
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        history = []

        def objective(trial):
            params = tune_rf.suggest_rf_b_params(trial)
            history.append(params)
            return float(params["n_estimators"])

        study.optimize(objective, n_trials=8)
        return history

    assert run_once() == run_once()


def test_nop_pruner_is_default_for_smoke_study(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    study, _ = tune_rf.run_smoke_study(
        cache, CLASS_ORDER, db_path=tmp_path / "smoke.db", seed=42, sampler_seed=42, n_jobs=1
    )
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    assert isinstance(study.sampler, optuna.samplers.TPESampler)


def test_smoke_and_main_study_identity_are_kept_separate():
    assert tune_rf.SMOKE_STUDY_NAME != tune_rf.MAIN_STUDY_NAME
    assert tune_rf.SMOKE_DB_FILENAME != tune_rf.MAIN_DB_FILENAME


# ---------------------------------------------------------------- objective 계약


def test_objective_returns_group5_oof_macro_f1_with_canonical_class_order():
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    result = tune_rf.evaluate_params_on_cached_folds(
        cache, CLASS_ORDER, tune_rf.SMOKE_TRIALS_SPEC[0]["params"], seed=42, n_jobs=1
    )
    assert 0.0 <= result["oof_macro_f1"] <= 1.0
    assert len(result["fold_macro_f1"]) == N_SPLITS
    assert list(result["oof_per_class_f1"]) == [str(c) for c in CLASS_ORDER]


def test_test_data_is_never_touched_by_objective():
    """test 행렬/ID 에 접근하면 즉시 실패하는 sentinel 을 심어 구조적으로 확인한다.

    정적 문자열 검색(`grep`)이 아니라, 실제로 objective 실행 경로가 그 값을
    읽으려 시도하면 예외가 나게 만들어 검증한다.
    """
    f4r, y, fold_ids = make_synthetic_f4r(with_test_sentinel=True)
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)  # 여기서 이미 건드리면 실패
    for spec in tune_rf.SMOKE_TRIALS_SPEC:
        tune_rf.evaluate_params_on_cached_folds(cache, CLASS_ORDER, spec["params"], seed=42, n_jobs=1)


def test_fold_cache_is_built_once_and_reused_across_trials(monkeypatch):
    """chi2 fit 이 fold 당 정확히 1번만 일어나야 한다 — trial 수와 무관하게."""
    f4r, y, fold_ids = make_synthetic_f4r()
    call_count = {"n": 0}
    original_fit = Chi2TopKSelector.fit

    def counting_fit(self, X, y_):
        call_count["n"] += 1
        return original_fit(self, X, y_)

    monkeypatch.setattr(Chi2TopKSelector, "fit", counting_fit)
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    assert call_count["n"] == N_SPLITS

    for spec in tune_rf.SMOKE_TRIALS_SPEC:
        tune_rf.evaluate_params_on_cached_folds(cache, CLASS_ORDER, spec["params"], seed=42, n_jobs=1)
    assert call_count["n"] == N_SPLITS  # 3개 trial 을 더 돌려도 늘지 않는다


def test_build_f4r_train_matrix_output_matches_expected_feature_count():
    f4r, y, fold_ids = make_synthetic_f4r()
    train_index = np.where(fold_ids != 0)[0]
    x_train, feature_columns = tune_rf.build_f4r_train_matrix(f4r, train_index)
    assert x_train.shape == (len(y), N_FEATURES + N_GENE)
    assert len(feature_columns) == N_FEATURES + N_GENE


# ---------------------------------------------------------------- SQLite 재개


def test_sqlite_study_created_at_given_path(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    db_path = tmp_path / "rf_b_smoke_study.db"
    tune_rf.run_smoke_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1
    )
    assert db_path.exists()


def test_resume_after_clean_exit_preserves_earlier_trials_and_completes_the_rest(tmp_path):
    """강제 종료 없이: 1개 완료 후 정상 종료 -> 재오픈 -> 나머지 완료. 번호/params/value 보존."""
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    db_path = tmp_path / "smoke.db"

    study1, info1 = tune_rf.run_smoke_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1, max_new_trials=1
    )
    assert info1 == {"waiting_before": 3, "ran_this_call": 1, "preserved_trials_checked": []}
    assert [t.state for t in study1.trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.WAITING,
        optuna.trial.TrialState.WAITING,
    ]
    trial0_params = study1.trials[0].params
    trial0_value = study1.trials[0].value

    # 새 Study 객체로 다시 연다 — 프로세스 재시작을 흉내낸다(강제 종료 아님, 정상 반환 후).
    study2, info2 = tune_rf.run_smoke_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1
    )
    assert info2["waiting_before"] == 2
    assert info2["ran_this_call"] == 2
    assert info2["preserved_trials_checked"] == [0]
    assert [t.state for t in study2.trials] == [optuna.trial.TrialState.COMPLETE] * 3
    assert study2.trials[0].params == trial0_params
    assert study2.trials[0].value == trial0_value
    # 중복·번호 밀림 없음 — 정확히 3개.
    assert len(study2.trials) == 3
    labels = [t.user_attrs["smoke_label"] for t in study2.trials]
    assert labels == [spec["label"] for spec in tune_rf.SMOKE_TRIALS_SPEC]


def test_resume_reload_from_disk_sees_identical_trials(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    db_path = tmp_path / "smoke.db"
    tune_rf.run_smoke_study(cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1)

    reloaded = optuna.load_study(
        study_name=tune_rf.SMOKE_STUDY_NAME, storage=tune_rf._build_storage_url(db_path)
    )
    assert len(reloaded.trials) == 3
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in reloaded.trials)


# ---------------------------------------------------------------- 진단값 저장


def test_trial_diagnostics_are_recorded_for_every_completed_trial(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    study, _ = tune_rf.run_smoke_study(
        cache, CLASS_ORDER, db_path=tmp_path / "smoke.db", seed=42, sampler_seed=42, n_jobs=1
    )
    required_keys = {
        "params", "fold_macro_f1", "fold_train_macro_f1", "fold_std",
        "train_validation_gap", "fold_elapsed_seconds", "trial_elapsed_seconds",
        "oof_macro_f1", "oof_per_class_f1", "n_features", "n_splits",
        "failure_reason", "peak_memory_bytes", "smoke_label",
    }
    for trial in study.trials:
        assert required_keys.issubset(trial.user_attrs), sorted(required_keys - set(trial.user_attrs))
        assert trial.user_attrs["failure_reason"] is None


# ---------------------------------------------------------------- 연속 실패 Stop


def test_three_consecutive_same_reason_failures_stop(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("합성 강제 실패")

    monkeypatch.setattr(tune_rf, "create_model", boom)
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)

    with pytest.raises(SystemExit, match="연속 3 trial 동일 원인 실패"):
        tune_rf.run_smoke_study(
            cache, CLASS_ORDER, db_path=tmp_path / "smoke.db", seed=42, sampler_seed=42, n_jobs=1
        )


def test_two_of_three_same_reason_failures_are_counted_even_if_not_consecutive():
    class _Trial:
        def __init__(self, reason):
            self.user_attrs = {"failure_reason": reason}

    failures = [_Trial("A"), _Trial("B"), _Trial("A")]
    assert tune_rf._count_same_reason_failures(failures) == 2


def test_single_failure_does_not_trigger_stop(tmp_path, monkeypatch):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)

    real_evaluate = tune_rf.evaluate_params_on_cached_folds
    call_count = {"n": 0}

    def flaky(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("일시적 실패")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(tune_rf, "evaluate_params_on_cached_folds", flaky)
    study, _ = tune_rf.run_smoke_study(
        cache, CLASS_ORDER, db_path=tmp_path / "smoke.db", seed=42, sampler_seed=42, n_jobs=1
    )
    states = [t.state for t in study.trials]
    assert states == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.COMPLETE,
    ]


def test_smoke_objective_rejects_trial_beyond_fixed_configs():
    """`trial.number` 확인은 fold_cache 를 건드리기 전에 일어난다 — 가짜 trial 로 충분하다."""

    class _FakeTrialWithNumberOnly:
        number = len(tune_rf.SMOKE_TRIALS_SPEC)  # smoke 가 정의한 개수를 넘어선 번호

    with pytest.raises(RuntimeError, match="고정 trial 3개만 정의"):
        tune_rf.objective_smoke(
            _FakeTrialWithNumberOnly(), fold_cache=[], classes=CLASS_ORDER, seed=42, n_jobs=1
        )


# ---------------------------------------------------------------- budget 계산


def _fake_rf_a_measurement(*, seconds=11.0, peak=2_000_000_000, n_estimators=500):
    return {
        "stem": "rf_a_f4r_group5_s42",
        "params": {"n_estimators": n_estimators, "max_features": "sqrt",
                   "class_weight": "balanced_subsample"},
        "fold_elapsed_seconds": [seconds / 5] * 5,
        "elapsed_seconds": seconds,
        "peak_memory_bytes": peak,
        "oof_macro_f1": 0.446,
        "n_features": 1055,
        "n_jobs": 7,
    }


def _fake_smoke_summary(label, *, seconds, n_estimators=100, peak=1_500_000_000):
    return {
        "label": label,
        "params": {**tune_rf.SMOKE_TRIALS_SPEC[
            [s["label"] for s in tune_rf.SMOKE_TRIALS_SPEC].index(label)
        ]["params"], "n_estimators": n_estimators},
        "oof_macro_f1": 0.4,
        "fold_macro_f1": [0.4] * 5,
        "fold_train_macro_f1": [0.6] * 5,
        "train_validation_gap": 0.2,
        "trial_elapsed_seconds": seconds,
        "peak_memory_bytes": peak,
    }


def test_budget_report_recommends_40_when_trials_are_cheap():
    smoke_summaries = [
        _fake_smoke_summary("baseline_cost", seconds=0.5),
        _fake_smoke_summary("search_space_center_cost", seconds=1.0),
        _fake_smoke_summary("compute_upper_bound_diagnostic", seconds=2.0),
    ]
    report = tune_rf.compute_budget_report(
        rf_a_measurement=_fake_rf_a_measurement(),
        smoke_summaries=smoke_summaries,
        resume_verification={"ok": True},
        fold_cache_build_seconds=5.0,
        data_hashes={"train.csv": {"sha256": "x"}},
        folds_file_info={"name": "train_folds.parquet", "fold_column": "fold_group5"},
        git_commit="deadbeef",
    )
    assert report["recommended_trial_count"] == 40
    assert report["proceed_decision"] == "PROCEED_MEETS_TARGET"
    assert report["recommended_trial_count_meets_minimum_20"] is True
    assert report["requires_explicit_user_trial_count_approval"] is True
    assert report["main_search_executed_in_this_ticket"] is False


def test_budget_report_flags_insufficient_budget_when_trials_are_expensive():
    # 목표 n_estimators 외삽 후 median 이 3시간/9 trial 도 안 되게 극단적으로 비싼 경우.
    huge = 3 * 60 * 60  # 1 trial 만으로 hard timeout 을 다 쓰는 costs
    smoke_summaries = [
        _fake_smoke_summary("baseline_cost", seconds=huge / 4 * (100 / 400)),
        _fake_smoke_summary("search_space_center_cost", seconds=huge * (100 / 700)),
        _fake_smoke_summary("compute_upper_bound_diagnostic", seconds=huge * (100 / 1000)),
    ]
    report = tune_rf.compute_budget_report(
        rf_a_measurement=_fake_rf_a_measurement(),
        smoke_summaries=smoke_summaries,
        resume_verification={"ok": True},
        fold_cache_build_seconds=1.0,
        data_hashes={},
        folds_file_info={},
        git_commit=None,
    )
    assert report["recommended_trial_count"] < 10
    assert report["proceed_decision"] == "STOP_INSUFFICIENT_BUDGET_FOR_TPE"


def test_budget_report_no_absolute_paths_in_output(tmp_path):
    smoke_summaries = [
        _fake_smoke_summary("baseline_cost", seconds=0.5),
        _fake_smoke_summary("search_space_center_cost", seconds=1.0),
        _fake_smoke_summary("compute_upper_bound_diagnostic", seconds=2.0),
    ]
    report = tune_rf.compute_budget_report(
        rf_a_measurement=_fake_rf_a_measurement(),
        smoke_summaries=smoke_summaries,
        resume_verification={"ok": True},
        fold_cache_build_seconds=5.0,
        data_hashes={"train.csv": {"sha256": "x"}},
        folds_file_info={"name": "train_folds.parquet", "fold_column": "fold_group5"},
        git_commit="deadbeef",
    )
    import json

    text = json.dumps(report)
    assert str(tmp_path) not in text
    assert str(Path.home()) not in text or str(Path.home()) == "/"


# ---------------------------------------------------------------- CLI


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        tune_rf.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--data-dir" in out
    assert "--max-new-trials" in out


# ================================================================== Ticket 3
# 본 탐색(`run_main_study`) — smoke(Ticket 2, 위)와 별개 경로. 전부 합성 데이터.


def test_evaluate_returns_predicted_class_distribution():
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    result = tune_rf.evaluate_params_on_cached_folds(
        cache, CLASS_ORDER, tune_rf.SMOKE_TRIALS_SPEC[0]["params"], seed=42, n_jobs=1
    )
    dist = result["oof_pred_class_distribution"]
    assert sum(dist.values()) == len(y)
    assert set(dist) <= set(CLASS_ORDER.tolist())


def test_return_oof_includes_full_arrays_without_mutating_default_path():
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    params = tune_rf.SMOKE_TRIALS_SPEC[0]["params"]

    default_result = tune_rf.evaluate_params_on_cached_folds(cache, CLASS_ORDER, params, seed=42, n_jobs=1)
    assert "oof_proba" not in default_result

    full_result = tune_rf.evaluate_params_on_cached_folds(
        cache, CLASS_ORDER, params, seed=42, n_jobs=1, return_oof=True
    )
    assert full_result["oof_proba"].shape == (len(y), N_CLASSES)
    assert len(full_result["oof_pred"]) == len(y)
    assert len(full_result["y_all"]) == len(y)
    # 나머지 진단값은 return_oof 여부와 무관하게 동일해야 한다(같은 계산의 부산물일 뿐).
    assert full_result["oof_macro_f1"] == default_result["oof_macro_f1"]


def test_diagnostic_user_attr_allowlist_excludes_oof_arrays(tmp_path):
    """`return_oof=True` 로 얻는 큰 배열이 실수로라도 Optuna user_attrs 에 안 들어가야 한다."""
    assert "oof_proba" not in tune_rf.DIAGNOSTIC_USER_ATTR_KEYS
    assert "oof_pred" not in tune_rf.DIAGNOSTIC_USER_ATTR_KEYS
    assert "y_all" not in tune_rf.DIAGNOSTIC_USER_ATTR_KEYS

    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    study, _ = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=tmp_path / "main.db", seed=42, sampler_seed=42, n_jobs=1,
        target_total_trials=1,
    )
    for key in ("oof_proba", "oof_pred", "y_all"):
        assert key not in study.trials[0].user_attrs


def test_compute_fold_hash_is_stable_and_sensitive_to_change():
    a = np.array([0, 1, 2, 3, 4, 0, 1])
    b = np.array([0, 1, 2, 3, 4, 0, 1])
    c = np.array([0, 1, 2, 3, 4, 0, 2])
    assert tune_rf.compute_fold_hash(a) == tune_rf.compute_fold_hash(b)
    assert tune_rf.compute_fold_hash(a) != tune_rf.compute_fold_hash(c)


def test_objective_main_records_fold_hash_when_given(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    fold_hash = tune_rf.compute_fold_hash(fold_ids)
    study, _ = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=tmp_path / "main.db", seed=42, sampler_seed=42, n_jobs=1,
        target_total_trials=1, fold_hash=fold_hash,
    )
    assert study.trials[0].user_attrs["fold_hash"] == fold_hash


def test_run_main_study_enqueues_rf_a_baseline_as_trial_0_only_once(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    db_path = tmp_path / "main.db"

    study, _ = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1, target_total_trials=1
    )
    resolved = tune_rf._resolve_baseline_point(tune_rf.RF_A_BASELINE_POINT)
    trial0_params = dict(study.trials[0].params)
    trial0_params["max_features"] = tune_rf.MAX_FEATURES_KEYS[trial0_params.pop("max_features_key")]
    assert trial0_params == resolved

    # 다시 호출해도 trial 0 이 중복 enqueue 되지 않는다(스터디에 이미 trial 이 있다).
    study2, info2 = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1, target_total_trials=1
    )
    assert info2["ran_this_call"] == 0
    assert len(study2.trials) == 1


def test_run_main_study_target_total_semantics_resume_runs_only_remaining(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    db_path = tmp_path / "main.db"

    study1, info1 = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1,
        target_total_trials=5, max_new_trials=2,
    )
    assert info1["ran_this_call"] == 2
    assert info1["complete_total"] == 2

    study2, info2 = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1, target_total_trials=5
    )
    assert info2["ran_this_call"] == 3  # 5 - 2 만 더 실행
    assert info2["complete_total"] == 5
    assert len(study2.trials) == 5

    # 이미 목표에 도달했으면 세 번째 호출은 아무것도 더 실행하지 않는다.
    study3, info3 = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1, target_total_trials=5
    )
    assert info3["ran_this_call"] == 0
    assert info3["complete_total"] == 5


def test_run_main_study_cumulative_wall_seconds_persists_across_resumes(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    db_path = tmp_path / "main.db"

    _, info1 = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1,
        target_total_trials=3, max_new_trials=1,
    )
    _, info2 = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=db_path, seed=42, sampler_seed=42, n_jobs=1, target_total_trials=3
    )
    assert info2["cumulative_wall_seconds"] > info1["cumulative_wall_seconds"]


def test_run_main_study_stops_before_starting_new_trial_past_hard_timeout(tmp_path):
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    study, info = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=tmp_path / "main.db", seed=42, sampler_seed=42, n_jobs=1,
        target_total_trials=5, hard_timeout_seconds=0.0001,
    )
    assert info["stop_reason"] == "HARD_TIMEOUT_REACHED_BEFORE_STARTING_NEW_TRIAL"
    # trial 0 은 이미 시작된 뒤라 끝까지 실행됐어야 한다(중간에 끊지 않는다).
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE


def test_run_main_study_three_consecutive_same_reason_failures_stop(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("합성 강제 실패")

    monkeypatch.setattr(tune_rf, "create_model", boom)
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)

    with pytest.raises(SystemExit, match="연속 3 trial 동일 원인 실패"):
        tune_rf.run_main_study(
            cache, CLASS_ORDER, db_path=tmp_path / "main.db", seed=42, sampler_seed=42, n_jobs=1,
            target_total_trials=5,
        )


def test_run_one_trial_identifies_waiting_trial_correctly(tmp_path):
    """Ticket 2 에서 실제로 겪은 버그(`study.trials[-1]` 오판) 회귀 테스트."""
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"x": 1})
    study.enqueue_trial({"x": 2})
    study.enqueue_trial({"x": 3})

    def objective(trial):
        return float(trial.suggest_categorical("x", [1, 2, 3]))

    first = tune_rf._run_one_trial(study, objective)
    assert first.number == 0
    assert first.params["x"] == 1

    second = tune_rf._run_one_trial(study, objective)
    assert second.number == 1
    assert second.params["x"] == 2


def test_run_one_trial_identifies_freshly_sampled_trial():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial):
        return float(trial.suggest_int("x", 1, 100))

    t1 = tune_rf._run_one_trial(study, objective)
    t2 = tune_rf._run_one_trial(study, objective)
    assert t1.number == 0
    assert t2.number == 1


def test_main_study_search_space_matches_suggest_rf_b_params(tmp_path):
    """본 탐색이 실제로 `suggest_rf_b_params` 의 탐색 공간을 쓰는지(TPE 로 뽑은 값 검증)."""
    f4r, y, fold_ids = make_synthetic_f4r()
    cache = tune_rf.precompute_fold_cache(f4r, fold_ids, N_SPLITS)
    study, _ = tune_rf.run_main_study(
        cache, CLASS_ORDER, db_path=tmp_path / "main.db", seed=42, sampler_seed=42, n_jobs=1,
        target_total_trials=3,
    )
    for trial in study.trials:
        params = trial.user_attrs["params"]
        assert params["n_estimators"] in range(400, 1001, 100)
        assert params["criterion"] in ("gini", "log_loss")
