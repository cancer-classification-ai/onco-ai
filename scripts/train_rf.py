#!/usr/bin/env python
"""Random Forest / ExtraTrees 학습 CLI — RF stacking base model (Ticket 1a).

    python scripts/train_rf.py --data-dir <external-data-dir> \\
        --folds-path data/process/train_folds.parquet --model rf

원본 `train.csv`/`test.csv`/`sample_submission.csv` 는 저장소 밖 외부 디렉터리에
있다. `--data-dir` (CLI) 또는 `RF_DATA_DIR` (환경변수, CLI 가 우선) 로 그 경로를
받는다 — 저장소 안으로 복사하거나 심볼릭 링크를 만들지 않고, 로그/provenance 에는
파일명과 SHA-256 만 남긴다(절대경로 기록 금지).

## Ticket 1a 범위

이 스크립트는 인터페이스 뼈대다. `f4r`(1,055열) 피처 조립, Group5 fold 생성은
하지 않는다 — `train.csv`/`test.csv` 의 `ID`/`SUBCLASS` 를 뺀 나머지 열을 그대로
피처로 쓰고, fold 배정은 `--folds-path` 로 미리 만들어진 파일(`ID` + fold 열)을
읽기만 한다. 둘 다 이 스크립트가 만들지 않는다 — `scripts/make_folds.py`(fold)와
피처 파이프라인(f4r)은 각각 자기 자리에서 만든다(Ticket 1b 이후 범위).

Random Forest/ExtraTrees 는 GBDT 의 `eval_set`/early stopping 같은 반복수 선택
장치가 없다 — test 는 fold 마다 `predict_proba` 에만 쓰이고 fit 에는 전혀
들어가지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.io import save_csv, write_submission  # noqa: E402
from cancer_hack.metrics import build_prediction_frame, probability_columns  # noqa: E402
from cancer_hack.models_rf import create_model, default_n_jobs  # noqa: E402
from cancer_hack.rf_artifact_validator import (  # noqa: E402
    macro_f1_with_labels,
    validate_oof_frame,
    validate_submission_frame,
    validate_test_probability_frame,
)
from cancer_hack.validation import CV_SLUG, check_all_classes_present, fold_column  # noqa: E402

SEED = 42
REQUIRED_DATA_FILES = ("train.csv", "test.csv", "sample_submission.csv")
META_COLUMNS = ("ID", "SUBCLASS")

#: spec §4(RF-A)/§6.1(RF-C) 고정값. `--config` JSON 이 있으면 이 위에 덮어쓴다.
DEFAULT_PARAMS: dict[str, dict] = {
    "rf": {"n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced_subsample"},
    "et": {
        "n_estimators": 500,
        "max_features": "sqrt",
        "bootstrap": False,
        "class_weight": "balanced",
    },
}


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------- 외부 데이터
def resolve_data_dir(cli_value: str | None) -> Path:
    """`--data-dir` > `RF_DATA_DIR` > 에러. 실제 경로는 반환값에만 있다."""
    if cli_value:
        return Path(cli_value)
    env_value = os.environ.get("RF_DATA_DIR")
    if env_value:
        return Path(env_value)
    raise SystemExit(
        "데이터 디렉터리를 지정해야 한다: --data-dir <path> 또는 RF_DATA_DIR 환경변수 "
        "중 하나가 필요하다."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data_dir(data_dir: Path) -> dict[str, str]:
    """필수 파일 존재를 확인하고 `{파일명: sha256}` 을 돌려준다(절대경로 없음)."""
    missing = [name for name in REQUIRED_DATA_FILES if not (data_dir / name).exists()]
    if missing:
        raise SystemExit(f"{data_dir} 에 없는 필수 파일: {missing}")
    return {name: _sha256(data_dir / name) for name in REQUIRED_DATA_FILES}


# ---------------------------------------------------------------- 피처(합성 경로)
def load_features(data_dir: Path):
    """`ID`/`SUBCLASS` 를 뺀 나머지 열을 그대로 피처로 쓴다(f4r 조립은 범위 밖).

    반환: (feature_columns, train_ids, test_ids, sample_submission_ids, y,
    X_train, X_test).
    """
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"ID": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"ID": str})
    sample_df = pd.read_csv(data_dir / "sample_submission.csv", dtype={"ID": str})

    feature_columns = [c for c in train_df.columns if c not in META_COLUMNS]
    missing_in_test = sorted(set(feature_columns) - set(test_df.columns))
    if missing_in_test:
        raise ValueError(f"test.csv 에 없는 피처 열: {missing_in_test[:5]}")

    train_ids = train_df["ID"].to_numpy()
    test_ids = test_df["ID"].to_numpy()
    sample_ids = sample_df["ID"].astype(str).tolist()
    y = train_df["SUBCLASS"].astype(str).to_numpy()
    X_train = train_df[feature_columns].to_numpy(dtype=np.float64)
    X_test = test_df[feature_columns].to_numpy(dtype=np.float64)
    return feature_columns, train_ids, test_ids, sample_ids, y, X_train, X_test


# ---------------------------------------------------------------- fold(사전계산본만 읽음)
def load_folds(path: Path, *, cv: str, n_splits: int, train_ids: np.ndarray):
    """미리 만든 fold 파일을 **읽기만** 한다 — 이 스크립트는 fold 를 만들지 않는다."""
    if not path.exists():
        raise SystemExit(
            f"fold 파일이 없다: {path.name}. Ticket 1a 는 fold 를 생성하지 않는다 — "
            "미리 만든 fold 파일을 --folds-path 로 전달한다."
        )
    folds = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"ID": str})

    column = fold_column(cv, n_splits)
    assert cv != "sgkf" or column == "fold_group5", (
        f"sgkf 는 fold_group5 열을 써야 하는데 {column} 이 계산됐다"
    )

    missing = [c for c in ("ID", column) if c not in folds.columns]
    if missing:
        raise ValueError(f"{path.name} 에 없는 열 {missing}: {list(folds.columns)}")

    folds = folds.copy()
    folds["ID"] = folds["ID"].astype(str)
    train_id_strs = [str(i) for i in train_ids]
    if set(folds["ID"]) != set(train_id_strs):
        raise ValueError(f"{path.name} 의 ID 집합이 train.csv 와 다르다")

    ordered = folds.set_index("ID").reindex(train_id_strs)
    fold_values = ordered[column].to_numpy()
    fold_ids = fold_values.astype(np.int64)
    if not np.array_equal(fold_ids, fold_values.astype(np.float64)):
        raise ValueError(f"{path.name} 의 {column} 값이 정수가 아니다")
    if fold_ids.min() < 0 or fold_ids.max() > n_splits - 1:
        raise ValueError(
            f"{path.name} 의 {column} 값 범위 이탈: [{fold_ids.min()}, {fold_ids.max()}], "
            f"기대 [0, {n_splits - 1}]"
        )
    return fold_ids, column


# ---------------------------------------------------------------- 산출물 경로
def output_paths(out_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "oof": out_dir / "oof" / f"oof_{stem}.csv",
        "test": out_dir / "test_predictions" / f"test_{stem}.csv",
        "submission": out_dir / "submissions" / f"submission_{stem}.csv",
        "log": out_dir / "logs" / f"{stem}.json",
    }


def check_overwrite(paths: dict[str, Path], *, overwrite: bool) -> None:
    """기존 산출물을 조용히 덮어쓰지 않는다(`make_folds.py` 관례와 동일)."""
    if overwrite:
        return
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise SystemExit(
            "이미 있는 산출물이 있다(덮어쓰려면 --overwrite): " + ", ".join(existing)
        )


def _csv_round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """실제 저장 후 다시 읽었을 때와 동일한 float 표현이 되도록 미리 한 번 왕복시킨다.

    `pandas.to_csv` 기본 float 포맷팅은 float64 완전 왕복을 보장하지 않는다
    (실측: 5,000행×6열 중 22,037개 셀이 첫 왕복에서 값이 바뀌었다 — 드문 ULP
    수준이 아니라 흔한 정밀도 절단이다). 대신 **한 번 왕복한 값을 다시
    왕복시키면 더 바뀌지 않는다**(멱등, 같은 실측으로 확인됨). 그래서 `y_pred`/
    `SUBCLASS` 를 계산하기 전에 확률을 이 함수로 한 번 정규화해 두면, 그 값을
    기준으로 계산한 `y_pred`/`SUBCLASS` 는 실제 디스크에 저장된 뒤 나중에
    다시 읽어도 항상 같은 `np.argmax` 를 낸다 — 저장 전 계산과 저장 후 재읽기가
    서로 다른 값을 보고 근접 동점의 승자가 갈리는 사고를 막는다.

    실제 파일 대신 메모리 버퍼로 왕복시킨다 — `to_csv`/`read_csv` 의 float
    포맷팅 로직은 대상이 파일이든 버퍼든 동일해서 결과가 같고, 검증 실패 시
    디스크에 중간 산출물을 남기지 않는다.
    """
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, dtype={"ID": str})


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["rf", "et"], default="rf")
    parser.add_argument(
        "--config", type=Path, default=None, help="모델 하이퍼파라미터 JSON(선택, 기본은 spec 고정값)"
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="원본 csv 3종이 있는 외부 디렉터리")
    parser.add_argument(
        "--folds-path",
        type=Path,
        required=True,
        help="ID + fold_{cv}{n_splits} 열을 가진 사전계산 fold 파일(parquet/csv)",
    )
    parser.add_argument("--cv", default="sgkf", choices=sorted(CV_SLUG))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--tag", default="v1", help="파일명에 들어가는 실험 이름")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="artifacts 루트(기본: 저장소의 artifacts/)"
    )
    parser.add_argument("--overwrite", action="store_true", help="기존 산출물을 덮어쓴다")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)

    data_dir = resolve_data_dir(str(args.data_dir) if args.data_dir else None)
    data_hashes = validate_data_dir(data_dir)

    out_dir = args.out_dir if args.out_dir is not None else PROJECT_ROOT / "artifacts"
    cv_slug = CV_SLUG[args.cv]
    stem = f"{args.model}_{args.tag}_{cv_slug}_s{args.seed}"
    paths = output_paths(out_dir, stem)
    check_overwrite(paths, overwrite=args.overwrite)

    feature_columns, train_ids, test_ids, sample_ids, y, X_train, X_test = load_features(data_dir)
    classes = np.unique(y)
    fold_ids, fold_col = load_folds(
        args.folds_path, cv=args.cv, n_splits=args.n_splits, train_ids=train_ids
    )

    params = dict(DEFAULT_PARAMS[args.model])
    if args.config is not None:
        with open(args.config, encoding="utf-8") as handle:
            params.update(json.load(handle))

    n_jobs = args.n_jobs if args.n_jobs is not None else default_n_jobs()
    class_order = classes.tolist()

    oof = np.zeros((len(y), len(classes)), dtype=np.float64)
    test_proba = np.zeros((len(test_ids), len(classes)), dtype=np.float64)
    fold_scores: list[float] = []

    started = time.perf_counter()
    for fold in range(args.n_splits):
        t0 = time.perf_counter()
        valid_index = np.where(fold_ids == fold)[0]
        train_index = np.where(fold_ids != fold)[0]
        check_all_classes_present(y, train_index, classes)

        model = create_model(
            args.model, class_order=class_order, random_state=args.seed, n_jobs=n_jobs, **params
        )
        model.fit(X_train[train_index], y[train_index])

        oof[valid_index] = model.predict_proba(X_train[valid_index])
        test_proba += model.predict_proba(X_test) / args.n_splits

        fold_pred = classes[oof[valid_index].argmax(axis=1)]
        score = macro_f1_with_labels(y[valid_index], fold_pred, classes)
        fold_scores.append(score)
        log(f"[{stem}] fold {fold + 1}/{args.n_splits}  Macro F1={score:.4f}  ({time.perf_counter() - t0:.1f}s)")

    elapsed = time.perf_counter() - started

    proba_cols = probability_columns(classes)

    # OOF: 확률(+fold/y_true)을 먼저 조립하고, 저장될 값으로 정규화(`_csv_round_trip`)
    # 한 뒤에야 그 값을 기준으로 `y_pred` 를 계산한다 — 왕복 전 값으로 계산하면
    # 나중에 저장된 파일을 다시 읽어 재계산한 argmax 와 근접 동점에서 어긋날 수 있다.
    oof_proba = build_prediction_frame(train_ids, oof, classes, y_true=y).drop(columns=["y_pred"])
    oof_proba.insert(1, "fold", fold_ids)
    oof_proba = oof_proba[["ID", "fold", "y_true", *proba_cols]]
    oof_proba = _csv_round_trip(oof_proba)
    oof_pred = classes[oof_proba[proba_cols].to_numpy(dtype=np.float64).argmax(axis=1)]
    oof_frame = oof_proba.copy()
    oof_frame.insert(3, "y_pred", oof_pred)
    oof_frame = oof_frame[["ID", "fold", "y_true", "y_pred", *proba_cols]]
    oof_macro_f1 = macro_f1_with_labels(oof_frame["y_true"], oof_frame["y_pred"], classes)

    # test 확률: spec §8.2 대로 `ID` + `p_{class}` 만 저장한다(`y_pred` 없음).
    # 마찬가지로 저장될 값으로 정규화해 둔다 — submission 을 여기서 뽑아낼 때
    # 저장된 test_predictions.csv 와 다른 값을 볼 위험이 없어야 한다.
    test_frame = build_prediction_frame(test_ids, test_proba, classes)[["ID", *proba_cols]]
    test_frame = _csv_round_trip(test_frame)

    # 저장 전에 자체 검증한다 — 문제가 있으면 파일을 남기지 않고 여기서 죽는다.
    # 위에서 이미 저장될 값으로 정규화했으므로, 이 검증은 실제로 디스크에 쓰일
    # 내용과 동일한 값을 보고 있다.
    validate_oof_frame(
        oof_frame,
        class_order=classes,
        train_ids=train_ids,
        n_splits=args.n_splits,
        expected_macro_f1=oof_macro_f1,
    )
    validate_test_probability_frame(test_frame, class_order=classes, sample_submission_ids=sample_ids)

    save_csv(oof_frame, paths["oof"])
    save_csv(test_frame, paths["test"])

    # submission 은 저장한 test 확률(정규화된 `test_frame`)에서 별도로 생성한다.
    # `write_submission` 이 `y_pred` 열을 요구하므로, 저장용 test_frame(순수
    # ID+p_*)과는 별개의 내부 frame 을 만든다 — 저장 파일 스키마에 y_pred 를
    # 다시 섞지 않기 위해서다.
    submission_source = test_frame.copy()
    submission_source["y_pred"] = classes[
        test_frame[proba_cols].to_numpy(dtype=np.float64).argmax(axis=1)
    ]
    submission_info = write_submission(
        submission_source, data_dir / "sample_submission.csv", paths["submission"]
    )

    submission_frame = pd.read_csv(paths["submission"], dtype=str, encoding="utf-8-sig")
    validate_submission_frame(
        submission_frame,
        class_order=classes,
        sample_submission_ids=sample_ids,
        test_proba_frame=test_frame,
    )

    provenance = {
        "stem": stem,
        "model": args.model,
        "params": params,
        "cv": args.cv,
        "fold_column": fold_col,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "n_jobs": n_jobs,
        "n_features": len(feature_columns),
        "n_train_rows": int(len(y)),
        "n_test_rows": int(len(test_ids)),
        "class_order": class_order,
        "fold_macro_f1": fold_scores,
        "oof_macro_f1": oof_macro_f1,
        "elapsed_seconds": elapsed,
        # 파일명 + SHA-256 만 — 절대경로는 기록하지 않는다.
        "data_files": {name: {"sha256": digest} for name, digest in data_hashes.items()},
        "folds_file": {"name": args.folds_path.name, "fold_column": fold_col},
        # `write_submission` 이 돌려주는 `output_path` 는 절대경로다 — 파일명만 남긴다.
        "submission": {**submission_info, "output_path": paths["submission"].name},
        "git_commit": _git_commit_sha(),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
    }
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["log"], "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)

    log(f"[{stem}] OOF Macro F1 = {oof_macro_f1:.4f}  ({elapsed:.1f}s)")
    log(f"  oof        -> {paths['oof']}")
    log(f"  test       -> {paths['test']}")
    log(f"  submission -> {paths['submission']}")
    log(f"  log        -> {paths['log']}")
    log("\n제출 파일은 로컬에만 만들어 뒀다. DACON 업로드는 직접 한다.")

    return provenance


if __name__ == "__main__":
    main()
