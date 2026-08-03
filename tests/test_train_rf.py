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
    pd.DataFrame({"ID": train_frame["ID"], "fold_group5": fold_values}).to_parquet(
        folds_path, index=False
    )

    config_path = tmp_path / "rf_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump({"n_estimators": n_estimators}, handle)

    return {
        "data_dir": data_dir,
        "folds_path": folds_path,
        "config_path": config_path,
        "train_ids": train_frame["ID"].tolist(),
        "test_ids": test_frame["ID"].tolist(),
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
    pd.DataFrame({"ID": ids, "fold_group5": [0, 1, 2, 3, 4]}).to_parquet(path, index=False)
    fold_ids, column = train_rf.load_folds(path, cv="sgkf", n_splits=5, train_ids=ids)
    assert column == "fold_group5"
    assert fold_ids.tolist() == [0, 1, 2, 3, 4]


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
