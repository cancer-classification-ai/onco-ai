"""`scripts/train_rf.py` CLI 계약 — 합성 데이터 end-to-end (Ticket 1a).

실제 대회 데이터나 개인 경로 없이, 작은 합성 `train.csv`/`test.csv`/
`sample_submission.csv` 와 미리 만든 합성 fold 파일만으로 CLI 전체 경로를
검증한다. 원본은 `docs/specs/random_forest_stacking_model.md`,
`docs/specs/random_forest_stacking_model_tickets.md` Ticket 1a.
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

import train_rf  # noqa: E402

from cancer_hack.rf_artifact_validator import (  # noqa: E402
    validate_oof_frame,
    validate_submission_frame,
    validate_test_probability_frame,
)
from cancer_hack.validation import fold_assignment  # noqa: E402

N_CLASSES = 6
N_PER_CLASS = 20
CLASS_ORDER = [f"C{i:02d}" for i in range(N_CLASSES)]
N_SPLITS = 5
N_FEATURES = 10
FEATURE_COLUMNS = [f"feat_{i}" for i in range(N_FEATURES)]


def _make_frame(n_samples: int, id_prefix: str, random_state: int, *, with_label: bool):
    X, yi = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=6,
        n_redundant=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    y = np.array([CLASS_ORDER[i] for i in yi])
    ids = [f"{id_prefix}{i:04d}" for i in range(n_samples)]
    frame = pd.DataFrame(X.astype(np.float64), columns=FEATURE_COLUMNS)
    frame.insert(0, "ID", ids)
    if with_label:
        frame["SUBCLASS"] = y
    return frame, y


def _duplicate_a_few_profiles(frame: pd.DataFrame, y: np.ndarray, pairs: list[tuple[int, int]]):
    """profile_hash 중복 그룹 구조를 흉내 낸다 — 동일 변이 프로파일 = 동일 feature 벡터."""
    group_keys = np.arange(len(frame))
    for a, b in pairs:
        frame.loc[b, FEATURE_COLUMNS] = frame.loc[a, FEATURE_COLUMNS].to_numpy()
        y[b] = y[a]
        frame.loc[b, "SUBCLASS"] = y[b]
        group_keys[b] = group_keys[a]
    return group_keys


def build_data_dir(tmp_path: Path, *, n_estimators: int = 25, test_random_state: int = 1):
    """합성 train/test/sample_submission + fold 파일을 만들고 필요한 경로를 돌려준다."""
    data_dir = tmp_path / "external_data"
    data_dir.mkdir(parents=True)

    train_frame, y = _make_frame(
        N_CLASSES * N_PER_CLASS, "tr", random_state=0, with_label=True
    )
    group_keys = _duplicate_a_few_profiles(train_frame, y, [(0, 1), (10, 11), (50, 51)])
    train_frame.to_csv(data_dir / "train.csv", index=False)

    test_frame, _ = _make_frame(30, "te", random_state=test_random_state, with_label=False)
    test_frame.to_csv(data_dir / "test.csv", index=False)

    sample_submission = pd.DataFrame(
        {"ID": test_frame["ID"], "SUBCLASS": CLASS_ORDER[0]}
    )
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

    fold_values = fold_assignment(
        y, kind="sgkf", n_splits=N_SPLITS, seed=42, groups=group_keys
    )
    folds_path = tmp_path / "train_folds.parquet"
    pd.DataFrame(
        {"ID": train_frame["ID"], "group_key": group_keys, "fold_group5": fold_values}
    ).to_parquet(folds_path, index=False)

    config_path = tmp_path / "rf_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump({"n_estimators": n_estimators}, handle)

    return {
        "data_dir": data_dir,
        "folds_path": folds_path,
        "config_path": config_path,
        "train_ids": train_frame["ID"].tolist(),
        "test_ids": test_frame["ID"].tolist(),
        "group_keys": group_keys,
    }


def run_cli(tmp_path: Path, ctx: dict, out_dir: Path, *, extra_args: list[str] | None = None):
    argv = [
        "--data-dir", str(ctx["data_dir"]),
        "--folds-path", str(ctx["folds_path"]),
        "--config", str(ctx["config_path"]),
        "--out-dir", str(out_dir),
        "--tag", "test",
    ]
    if extra_args:
        argv += extra_args
    return train_rf.main(argv)


# ---------------------------------------------------------------- --help


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        train_rf.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--data-dir" in out
    assert "--folds-path" in out


# ---------------------------------------------------------------- 외부 데이터 계약


def test_missing_data_dir_and_env_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("RF_DATA_DIR", raising=False)
    with pytest.raises(SystemExit, match="RF_DATA_DIR"):
        train_rf.resolve_data_dir(None)


def test_env_fallback_is_used_when_cli_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_DATA_DIR", str(tmp_path))
    assert train_rf.resolve_data_dir(None) == tmp_path


def test_cli_value_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_DATA_DIR", str(tmp_path / "from_env"))
    other = tmp_path / "from_cli"
    assert train_rf.resolve_data_dir(str(other)) == other


def test_missing_required_files_raises_with_filenames(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit, match="train.csv"):
        train_rf.validate_data_dir(empty_dir)


def test_present_files_are_hashed(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    for name in train_rf.REQUIRED_DATA_FILES:
        (data_dir / name).write_text("ID,SUBCLASS\n1,ACC\n", encoding="utf-8")
    hashes = train_rf.validate_data_dir(data_dir)
    assert set(hashes) == set(train_rf.REQUIRED_DATA_FILES)
    assert all(len(v) == 64 for v in hashes.values())  # sha256 hex digest


# ---------------------------------------------------------------- fold 파일


def test_load_folds_missing_file_raises(tmp_path):
    with pytest.raises(SystemExit, match="fold 파일이 없다"):
        train_rf.load_folds(
            tmp_path / "nope.parquet", cv="sgkf", n_splits=5, train_ids=np.array(["a"])
        )


def test_load_folds_uses_fold_group5_for_sgkf(tmp_path):
    ids = np.array(["a", "b", "c", "d", "e"])
    path = tmp_path / "folds.parquet"
    pd.DataFrame(
        {"ID": ids, "group_key": [10, 11, 12, 13, 14], "fold_group5": [0, 1, 2, 3, 4]}
    ).to_parquet(path, index=False)
    fold_ids, column, group_key_by_id = train_rf.load_folds(
        path, cv="sgkf", n_splits=5, train_ids=ids
    )
    assert column == "fold_group5"
    assert fold_ids.tolist() == [0, 1, 2, 3, 4]
    assert group_key_by_id == {"a": 10, "b": 11, "c": 12, "d": 13, "e": 14}


def test_load_folds_sgkf_without_group_key_column_raises(tmp_path):
    """Group5 는 group leakage 검사에 group_key 가 필요하다 — 없으면 조용히 넘어가지 않는다."""
    ids = np.array(["a", "b", "c", "d", "e"])
    path = tmp_path / "folds.parquet"
    pd.DataFrame({"ID": ids, "fold_group5": [0, 1, 2, 3, 4]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="group_key"):
        train_rf.load_folds(path, cv="sgkf", n_splits=5, train_ids=ids)


def test_load_folds_skf_without_group_key_column_is_allowed(tmp_path):
    """skf 는 group 구조를 가정하지 않는 CV 라 group_key 가 선택적이다."""
    ids = np.array(["a", "b", "c", "d", "e"])
    path = tmp_path / "folds.parquet"
    pd.DataFrame({"ID": ids, "fold_skf5": [0, 1, 2, 3, 4]}).to_parquet(path, index=False)
    fold_ids, column, group_key_by_id = train_rf.load_folds(
        path, cv="skf", n_splits=5, train_ids=ids
    )
    assert column == "fold_skf5"
    assert group_key_by_id is None


# ---------------------------------------------------------------- 덮어쓰기 방지


def test_check_overwrite_blocks_existing_file(tmp_path):
    existing = tmp_path / "a.csv"
    existing.write_text("x", encoding="utf-8")
    paths = {"a": existing, "b": tmp_path / "b.csv"}
    with pytest.raises(SystemExit, match="overwrite"):
        train_rf.check_overwrite(paths, overwrite=False)


def test_check_overwrite_allows_with_flag(tmp_path):
    existing = tmp_path / "a.csv"
    existing.write_text("x", encoding="utf-8")
    paths = {"a": existing}
    train_rf.check_overwrite(paths, overwrite=True)  # 예외 없이 통과해야 한다


# ---------------------------------------------------------------- 합성 CLI smoke


def test_synthetic_cli_smoke_produces_valid_artifacts(tmp_path):
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    provenance = run_cli(tmp_path, ctx, out_dir)

    oof = pd.read_csv(out_dir / "oof" / "oof_rf_test_group5_s42.csv", encoding="utf-8")
    test_pred = pd.read_csv(
        out_dir / "test_predictions" / "test_rf_test_group5_s42.csv", encoding="utf-8"
    )
    submission = pd.read_csv(
        out_dir / "submissions" / "submission_rf_test_group5_s42.csv",
        encoding="utf-8-sig",
        dtype=str,
    )

    classes = np.array(sorted(set(pd.read_csv(ctx["data_dir"] / "train.csv")["SUBCLASS"])))
    sample_ids = pd.read_csv(ctx["data_dir"] / "sample_submission.csv", dtype=str)["ID"].tolist()

    validate_oof_frame(
        oof,
        class_order=classes,
        train_ids=ctx["train_ids"],
        n_splits=N_SPLITS,
        expected_macro_f1=provenance["oof_macro_f1"],
    )
    validate_test_probability_frame(test_pred, class_order=classes, sample_submission_ids=sample_ids)
    validate_submission_frame(
        submission,
        class_order=classes,
        sample_submission_ids=sample_ids,
        test_proba_frame=test_pred,
    )

    log_path = out_dir / "logs" / "rf_test_group5_s42.json"
    assert log_path.exists()
    with open(log_path, encoding="utf-8") as handle:
        logged = json.load(handle)
    assert logged["fold_macro_f1"] and len(logged["fold_macro_f1"]) == N_SPLITS
    assert 0.0 <= logged["oof_macro_f1"] <= 1.0


def test_saved_test_predictions_columns_are_exactly_id_and_probabilities(tmp_path):
    """spec §8.2 — test_predictions.csv 는 `ID` + `p_{class}` 만 가진다(`y_pred` 없음)."""
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    run_cli(tmp_path, ctx, out_dir)

    test_pred = pd.read_csv(out_dir / "test_predictions" / "test_rf_test_group5_s42.csv")
    classes = sorted(set(pd.read_csv(ctx["data_dir"] / "train.csv")["SUBCLASS"]))
    assert list(test_pred.columns) == ["ID", *[f"p_{c}" for c in classes]]
    assert "y_pred" not in test_pred.columns


def test_saved_oof_y_pred_matches_persisted_probability_argmax_exactly(tmp_path):
    """저장·재읽은 OOF `y_pred` 가 그 파일 자체의 확률 argmax 와 정확히 일치해야 한다."""
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    run_cli(tmp_path, ctx, out_dir)

    oof = pd.read_csv(out_dir / "oof" / "oof_rf_test_group5_s42.csv")
    classes = np.array(sorted(set(pd.read_csv(ctx["data_dir"] / "train.csv")["SUBCLASS"])))
    proba_cols = [f"p_{c}" for c in classes]

    recomputed = classes[oof[proba_cols].to_numpy(dtype=np.float64).argmax(axis=1)]
    assert (oof["y_pred"].to_numpy() == recomputed).all()


def test_saved_submission_matches_persisted_test_probability_argmax_exactly(tmp_path):
    """저장·재읽은 submission 의 SUBCLASS 가 저장된 test 확률 argmax 와 정확히 일치해야 한다."""
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    run_cli(tmp_path, ctx, out_dir)

    test_pred = pd.read_csv(out_dir / "test_predictions" / "test_rf_test_group5_s42.csv")
    submission = pd.read_csv(
        out_dir / "submissions" / "submission_rf_test_group5_s42.csv",
        encoding="utf-8-sig",
        dtype=str,
    )
    classes = np.array(sorted(set(pd.read_csv(ctx["data_dir"] / "train.csv")["SUBCLASS"])))
    proba_cols = [f"p_{c}" for c in classes]

    recomputed = classes[test_pred[proba_cols].to_numpy(dtype=np.float64).argmax(axis=1)]
    merged = submission.merge(
        test_pred[["ID"]].assign(argmax=recomputed), on="ID", how="left", validate="one_to_one"
    )
    assert (merged["SUBCLASS"].to_numpy() == merged["argmax"].to_numpy()).all()


def test_et_model_also_runs_end_to_end(tmp_path):
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    run_cli(tmp_path, ctx, out_dir, extra_args=["--model", "et"])
    assert (out_dir / "submissions" / "submission_et_test_group5_s42.csv").exists()


def test_rerun_without_overwrite_is_blocked(tmp_path):
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    run_cli(tmp_path, ctx, out_dir)
    with pytest.raises(SystemExit, match="overwrite"):
        run_cli(tmp_path, ctx, out_dir)


def test_rerun_with_overwrite_flag_succeeds(tmp_path):
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    run_cli(tmp_path, ctx, out_dir)
    run_cli(tmp_path, ctx, out_dir, extra_args=["--overwrite"])  # 예외 없어야 한다


def test_provenance_has_no_absolute_path(tmp_path):
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    run_cli(tmp_path, ctx, out_dir)

    log_path = out_dir / "logs" / "rf_test_group5_s42.json"
    raw_text = log_path.read_text(encoding="utf-8")
    assert str(ctx["data_dir"]) not in raw_text
    # `write_submission` 의 `output_path` 도 절대경로다 — provenance 에는 안 남아야 한다.
    assert str(out_dir) not in raw_text
    assert str(tmp_path) not in raw_text

    with open(log_path, encoding="utf-8") as handle:
        logged = json.load(handle)
    for name, info in logged["data_files"].items():
        assert name in train_rf.REQUIRED_DATA_FILES
        assert len(info["sha256"]) == 64
    assert logged["submission"]["output_path"] == "submission_rf_test_group5_s42.csv"


def test_test_features_are_not_used_for_fitting(tmp_path):
    """test.csv 를 통째로 바꿔도 OOF 는 그대로다 — fit 은 train partition 만 본다."""
    ctx = build_data_dir(tmp_path)
    out_a = tmp_path / "artifacts_a"
    run_cli(tmp_path, ctx, out_a)

    # train/fold 는 그대로 두고 test.csv 만 다른 분포로 바꾼다(ID 는 동일하게 유지).
    new_test_frame, _ = _make_frame(30, "te", random_state=99, with_label=False)
    new_test_frame.to_csv(ctx["data_dir"] / "test.csv", index=False)

    out_b = tmp_path / "artifacts_b"
    run_cli(tmp_path, ctx, out_b)

    oof_a = pd.read_csv(out_a / "oof" / "oof_rf_test_group5_s42.csv")
    oof_b = pd.read_csv(out_b / "oof" / "oof_rf_test_group5_s42.csv")
    pd.testing.assert_frame_equal(oof_a, oof_b)

    test_a = pd.read_csv(out_a / "test_predictions" / "test_rf_test_group5_s42.csv")
    test_b = pd.read_csv(out_b / "test_predictions" / "test_rf_test_group5_s42.csv")
    assert not test_a.drop(columns=["ID"]).equals(test_b.drop(columns=["ID"]))


def test_fold_train_partition_missing_class_raises(tmp_path):
    """train partition 에서 canonical 클래스가 통째로 빠지면 명확히 중단한다."""
    ctx = build_data_dir(tmp_path)
    train_df = pd.read_csv(ctx["data_dir"] / "train.csv")
    folds = pd.read_parquet(ctx["folds_path"])

    # 클래스 하나를 몽땅 fold 0 의 validation 에만 몰아넣어 train partition 에서 빠지게 한다.
    target_class = sorted(train_df["SUBCLASS"].unique())[0]
    mask = train_df["SUBCLASS"] == target_class
    folds.loc[mask.to_numpy(), "fold_group5"] = 0
    folds.to_parquet(ctx["folds_path"], index=False)

    out_dir = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="fold train 에 없는 클래스"):
        run_cli(tmp_path, ctx, out_dir)


# ---------------------------------------------------------------- group leakage guard(실제 실행 경로)


def test_group_leakage_guard_passes_with_consistent_group_fold_file(tmp_path):
    """정상 fold 파일(group_key 포함, 동일 group 이 항상 같은 fold)은 그대로 통과한다."""
    ctx = build_data_dir(tmp_path)
    out_dir = tmp_path / "artifacts"
    run_cli(tmp_path, ctx, out_dir)  # 예외 없이 끝나야 한다

    folds = pd.read_parquet(ctx["folds_path"])
    oof = pd.read_csv(out_dir / "oof" / "oof_rf_test_group5_s42.csv", dtype={"ID": str})
    merged = oof.merge(folds[["ID", "group_key"]], on="ID", how="left")

    # build_data_dir 가 3쌍(0,1)/(10,11)/(50,51)을 중복 profile 로 만들어 뒀다 —
    # 실제로 group_key 가 겹치는 group 이 있는지, 그리고 전부 같은 fold 인지 확인한다.
    duplicated = merged["group_key"].value_counts()
    duplicated_groups = duplicated[duplicated > 1].index
    assert len(duplicated_groups) == 3
    crossing = merged.groupby("group_key")["fold"].nunique()
    assert (crossing.loc[duplicated_groups] == 1).all()


def test_group_leakage_guard_stops_before_saving_any_artifact_on_corrupted_fold_file(tmp_path):
    """같은 group_key 가 서로 다른 fold 에 배치된 손상된 fold 파일은 저장 전에 명확히 중단한다."""
    ctx = build_data_dir(tmp_path)
    folds = pd.read_parquet(ctx["folds_path"])

    # (0,1) 쌍은 group_key 가 같다 — 한쪽 fold 만 인위적으로 옮겨 손상시킨다.
    pair_a_id, pair_b_id = ctx["train_ids"][0], ctx["train_ids"][1]
    row_a = folds.index[folds["ID"] == pair_a_id][0]
    row_b = folds.index[folds["ID"] == pair_b_id][0]
    assert folds.loc[row_a, "group_key"] == folds.loc[row_b, "group_key"]
    original_fold = folds.loc[row_a, "fold_group5"]
    folds.loc[row_b, "fold_group5"] = (original_fold + 1) % N_SPLITS
    assert folds.loc[row_a, "fold_group5"] != folds.loc[row_b, "fold_group5"]
    folds.to_parquet(ctx["folds_path"], index=False)

    out_dir = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="group"):
        run_cli(tmp_path, ctx, out_dir)

    # 검증 실패는 어떤 산출물도 저장되기 전에 일어나야 한다.
    paths = train_rf.output_paths(out_dir, "rf_test_group5_s42")
    for path in paths.values():
        assert not path.exists(), f"{path} 가 검증 실패에도 저장됐다"
