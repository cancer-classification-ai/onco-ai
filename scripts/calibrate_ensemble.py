#!/usr/bin/env python
"""OOF 전용 학습으로 확률 앙상블과 클래스 로짓 보정을 만든다.

예시 - f4r/f6x StratifiedKFold 예측 결합:

    python scripts/calibrate_ensemble.py \
      --oof artifacts/oof/oof_xgb_v2_f4r_skf5_k500_s42.csv \
             artifacts/oof/oof_xgb_v2_f6x_skf5_k500_s42.csv \
      --test artifacts/test_predictions/test_xgb_v2_f4r_skf5_k500_s42.csv \
              artifacts/test_predictions/test_xgb_v2_f6x_skf5_k500_s42.csv \
      --fold-column fold_skf5 --tag xgb_f4r_f6x_skf_cf

혼합 가중치와 로짓 오프셋은 test 확률·test 예측 분포를 보지 않고 OOF와 OOF 정답으로만
학습한다. 성능 보고는 fold 교차적합 예측으로 하고, 전체 OOF 적합은 최종 test 변환에만 쓴다.
TVD는 분포 붕괴 경보일 뿐 최적화 목표가 아니다. 파일을 만들 뿐 DACON에 업로드하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.calibration import MacroF1LogitBias  # noqa: E402
from cancer_hack.ensemble import MacroF1Blender, weighted_average  # noqa: E402
from cancer_hack.io import save_csv, write_submission  # noqa: E402
from cancer_hack.metrics import (  # noqa: E402
    build_prediction_frame,
    evaluate_classification,
    macro_f1,
    prediction_distribution_report,
    probability_columns,
    read_prediction_frame,
)

ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_FOLDS = PROJECT_ROOT / "data/process/train_folds.parquet"
DEFAULT_SAMPLE = PROJECT_ROOT / "data/raw/sample_submission.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--oof", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--fold-column", default="fold_skf5")
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "가중치를 학습하지 않고 이 값으로 고정한다. --oof 와 개수·순서가 같아야 한다. "
            "합이 1 이 아니면 정규화한다. 로짓 보정은 이 위에 그대로 얹는다"
        ),
    )
    parser.add_argument("--tvd-warning-threshold", type=float, default=0.10)
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _aligned_matrix(
    frame: pd.DataFrame, ids: np.ndarray, classes: list[str], *, source: Path
) -> np.ndarray:
    if frame["ID"].astype(str).duplicated().any():
        raise ValueError(f"{source} 에 중복 ID가 있다")
    indexed = frame.copy()
    indexed.index = indexed["ID"].astype(str)
    aligned = indexed.reindex(ids)
    columns = probability_columns(classes)
    if aligned[columns].isna().any().any():
        raise ValueError(f"{source} 의 ID 집합이 기준 파일과 다르다")
    return aligned[columns].to_numpy(dtype=np.float64)


def _load_prediction_set(paths: list[Path], *, require_y: bool):
    frames: list[pd.DataFrame] = []
    class_sets: list[tuple[str, ...]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame, classes = read_prediction_frame(path)
        if require_y and "y_true" not in frame.columns:
            raise ValueError(f"{path} 에 y_true가 없다")
        frames.append(frame)
        class_sets.append(tuple(classes))
    if len(set(class_sets)) != 1:
        raise ValueError("확률 파일마다 클래스 열 또는 순서가 다르다")
    classes = list(class_sets[0])
    ids = frames[0]["ID"].astype(str).to_numpy()
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{paths[0]} 에 중복 ID가 있다")
    arrays = [
        _aligned_matrix(frame, ids, classes, source=path)
        for frame, path in zip(frames, paths)
    ]
    labels = None
    if require_y:
        base = frames[0].copy()
        base.index = base["ID"].astype(str)
        labels = base.reindex(ids)["y_true"].astype(str).to_numpy()
        for frame, path in zip(frames[1:], paths[1:]):
            other = frame.copy()
            other.index = other["ID"].astype(str)
            candidate = other.reindex(ids)["y_true"].astype(str).to_numpy()
            if not np.array_equal(labels, candidate):
                raise ValueError(f"{path} 의 y_true가 기준 OOF와 다르다")
    return ids, labels, classes, arrays


def _load_fold_values(path: Path, ids: np.ndarray, column: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    folds = pd.read_parquet(path)
    if column not in folds.columns:
        raise ValueError(f"{path} 에 {column}이 없다: {list(folds.columns)}")
    if folds["ID"].astype(str).duplicated().any():
        raise ValueError(f"{path} 에 중복 ID가 있다")
    folds = folds.copy()
    folds.index = folds["ID"].astype(str)
    values = folds.reindex(ids)[column]
    if values.isna().any():
        raise ValueError(f"{path} 의 ID 집합이 OOF와 다르다")
    return values.to_numpy(dtype=np.int64)


def _check_output_paths(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"이미 있는 출력 파일: {existing}. --overwrite 를 준다")


def main() -> None:
    args = parse_args()
    if len(args.oof) != len(args.test):
        raise ValueError("--oof 와 --test 파일 수가 같아야 한다")

    oof_ids, y_true, classes, oof_arrays = _load_prediction_set(args.oof, require_y=True)
    test_ids, _, test_classes, test_arrays = _load_prediction_set(
        args.test, require_y=False
    )
    if classes != test_classes:
        raise ValueError("OOF와 test의 클래스 열 또는 순서가 다르다")
    fold_values = _load_fold_values(args.folds, oof_ids, args.fold_column)

    # 고정 가중치. 학습된 가중치는 fold 마다 흔들리는데(실측 폭 0.23) 그 흔들림이
    # 그대로 test 로 이전된다는 보장이 없다. 사람이 값을 박아 두면 fold 노이즈를
    # 줍지 않는다. 대신 그 값이 맞다는 근거는 사람이 져야 한다.
    fixed_weights = None
    if args.weights is not None:
        if len(args.weights) != len(args.oof):
            raise ValueError(
                f"--weights 는 {len(args.oof)}개여야 한다 (--oof 개수). "
                f"받은 값 {len(args.weights)}개: {args.weights}"
            )
        weights = np.asarray(args.weights, dtype=np.float64)
        if (weights < 0).any():
            raise ValueError(f"음수 가중치는 받지 않는다: {args.weights}")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("가중치 합이 0 이다")
        fixed_weights = weights / total
        if abs(total - 1.0) > 1e-9:
            print(f"가중치 합 {total:g} -> 1 로 정규화: {fixed_weights.round(4).tolist()}")

    crossfit_raw_blend = np.zeros_like(oof_arrays[0], dtype=np.float64)
    crossfit_proba = np.zeros_like(oof_arrays[0], dtype=np.float64)
    fold_results = []
    for fold in sorted(np.unique(fold_values).tolist()):
        train_mask = fold_values != fold
        valid_mask = ~train_mask
        if fixed_weights is None:
            blender = MacroF1Blender().fit(
                [values[train_mask] for values in oof_arrays], y_true[train_mask], classes
            )
            fold_weights = blender.weights_
            blend = blender.predict_proba
        else:
            # fold 밖에서 학습할 게 없다. 보정만 이 fold 의 train 부분에서 fit 한다.
            fold_weights = fixed_weights
            blend = lambda values: weighted_average(values, fixed_weights)  # noqa: E731

        train_blend = blend([values[train_mask] for values in oof_arrays])
        calibrator = MacroF1LogitBias().fit(train_blend, y_true[train_mask], classes)
        valid_blend = blend([values[valid_mask] for values in oof_arrays])
        crossfit_raw_blend[valid_mask] = valid_blend
        valid_adjusted = calibrator.predict_proba(valid_blend)
        crossfit_proba[valid_mask] = valid_adjusted
        valid_pred = np.asarray(classes)[valid_adjusted.argmax(axis=1)]
        fold_results.append(
            {
                "fold": int(fold),
                "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "weights": [float(w) for w in fold_weights],
                "bias": calibrator.bias_by_class(),
                "raw_blend_macro_f1": macro_f1(
                    y_true[valid_mask],
                    np.asarray(classes)[valid_blend.argmax(axis=1)],
                ),
                "calibrated_macro_f1": macro_f1(y_true[valid_mask], valid_pred),
            }
        )

    source_scores = {
        path.stem: macro_f1(y_true, np.asarray(classes)[values.argmax(axis=1)])
        for path, values in zip(args.oof, oof_arrays)
    }
    uniform_weights = np.full(len(oof_arrays), 1.0 / len(oof_arrays))
    uniform_oof = weighted_average(oof_arrays, uniform_weights)
    uniform_score = macro_f1(y_true, np.asarray(classes)[uniform_oof.argmax(axis=1)])
    crossfit_raw_pred = np.asarray(classes)[crossfit_raw_blend.argmax(axis=1)]
    crossfit_raw_summary = evaluate_classification(y_true, crossfit_raw_pred, classes)
    crossfit_pred = np.asarray(classes)[crossfit_proba.argmax(axis=1)]
    crossfit_summary = evaluate_classification(y_true, crossfit_pred, classes)

    # 최종 test 변환용. 이 학습 점수는 같은 OOF를 fit/evaluate하므로 모델 선택에 쓰지 않는다.
    if fixed_weights is None:
        final_blender = MacroF1Blender().fit(oof_arrays, y_true, classes)
        final_weights = final_blender.weights_
        final_blend = final_blender.predict_proba
    else:
        final_weights = fixed_weights
        final_blend = lambda values: weighted_average(values, fixed_weights)  # noqa: E731
    final_oof_blend = final_blend(oof_arrays)
    final_blend_pred = np.asarray(classes)[final_oof_blend.argmax(axis=1)]
    final_calibrator = MacroF1LogitBias().fit(final_oof_blend, y_true, classes)
    full_fit_oof = final_calibrator.predict_proba(final_oof_blend)
    full_fit_pred = np.asarray(classes)[full_fit_oof.argmax(axis=1)]

    raw_test_blend = final_blend(test_arrays)
    raw_test_pred = np.asarray(classes)[raw_test_blend.argmax(axis=1)]
    uniform_test = weighted_average(test_arrays, uniform_weights)
    uniform_test_pred = np.asarray(classes)[uniform_test.argmax(axis=1)]
    final_test = final_calibrator.predict_proba(raw_test_blend)
    final_test_pred = np.asarray(classes)[final_test.argmax(axis=1)]
    raw_distribution = prediction_distribution_report(
        y_true,
        raw_test_pred,
        classes,
        warning_threshold=args.tvd_warning_threshold,
    )
    uniform_distribution = prediction_distribution_report(
        y_true,
        uniform_test_pred,
        classes,
        warning_threshold=args.tvd_warning_threshold,
    )
    adjusted_distribution = prediction_distribution_report(
        y_true,
        final_test_pred,
        classes,
        warning_threshold=args.tvd_warning_threshold,
    )

    oof_path = ARTIFACTS / "oof" / f"oof_{args.tag}.csv"
    test_path = ARTIFACTS / "test_predictions" / f"test_{args.tag}.csv"
    raw_oof_path = ARTIFACTS / "oof" / f"oof_{args.tag}_raw_blend.csv"
    raw_test_path = ARTIFACTS / "test_predictions" / f"test_{args.tag}_raw_blend.csv"
    uniform_oof_path = ARTIFACTS / "oof" / f"oof_{args.tag}_uniform_blend.csv"
    uniform_test_path = (
        ARTIFACTS / "test_predictions" / f"test_{args.tag}_uniform_blend.csv"
    )
    log_path = ARTIFACTS / "logs" / f"{args.tag}.json"
    output_paths = [
        oof_path,
        test_path,
        raw_oof_path,
        raw_test_path,
        uniform_oof_path,
        uniform_test_path,
        log_path,
    ]
    submission_path = ARTIFACTS / "submissions" / f"submission_{args.tag}.csv"
    raw_submission_path = (
        ARTIFACTS / "submissions" / f"submission_{args.tag}_raw_blend.csv"
    )
    uniform_submission_path = (
        ARTIFACTS / "submissions" / f"submission_{args.tag}_uniform_blend.csv"
    )
    if args.submission:
        output_paths.extend(
            [submission_path, raw_submission_path, uniform_submission_path]
        )
    _check_output_paths(output_paths, overwrite=args.overwrite)

    oof_frame = build_prediction_frame(
        oof_ids, crossfit_proba, classes, y_true=y_true
    )
    test_frame = build_prediction_frame(test_ids, final_test, classes)
    raw_oof_frame = build_prediction_frame(
        oof_ids, crossfit_raw_blend, classes, y_true=y_true
    )
    raw_test_frame = build_prediction_frame(test_ids, raw_test_blend, classes)
    uniform_oof_frame = build_prediction_frame(
        oof_ids, uniform_oof, classes, y_true=y_true
    )
    uniform_test_frame = build_prediction_frame(test_ids, uniform_test, classes)
    save_csv(oof_frame, oof_path)
    save_csv(test_frame, test_path)
    save_csv(raw_oof_frame, raw_oof_path)
    save_csv(raw_test_frame, raw_test_path)
    save_csv(uniform_oof_frame, uniform_oof_path)
    save_csv(uniform_test_frame, uniform_test_path)

    result = {
        "tag": args.tag,
        "oof_sources": [str(path) for path in args.oof],
        "test_sources": [str(path) for path in args.test],
        "fold_file": str(args.folds),
        "fold_column": args.fold_column,
        "n_samples": len(oof_ids),
        "n_test": len(test_ids),
        "classes": classes,
        "source_oof_macro_f1": source_scores,
        "uniform_blend_oof_macro_f1": uniform_score,
        "crossfit_raw_blend": crossfit_raw_summary,
        "crossfit_calibrated": crossfit_summary,
        # 가중치를 사람이 박았는지 학습했는지. 나중에 로그만 보고 구분이 안 되면
        # "이 점수가 fold 에 맞춰 최적화된 값인가"를 되짚을 수 없다.
        "weight_source": "fixed" if fixed_weights is not None else "learned",
        "crossfit_folds": fold_results,
        "final_full_oof_fit": {
            "warning": "optimistic_not_for_model_selection",
            "raw_blend_macro_f1": macro_f1(y_true, final_blend_pred),
            "macro_f1": macro_f1(y_true, full_fit_pred),
            "weights": [float(w) for w in final_weights],
            "bias": final_calibrator.bias_by_class(),
        },
        "test_distribution_guardrail": {
            "uniform_blend": uniform_distribution,
            "before_logit_bias": raw_distribution,
            "after_logit_bias": adjusted_distribution,
        },
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_log = log_path.with_suffix(".tmp.json")
    with open(temporary_log, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    temporary_log.replace(log_path)

    if args.submission:
        write_submission(test_frame, args.sample_submission, submission_path)
        write_submission(raw_test_frame, args.sample_submission, raw_submission_path)
        write_submission(
            uniform_test_frame, args.sample_submission, uniform_submission_path
        )

    print(f"source OOF: {source_scores}")
    print(f"uniform blend OOF: {uniform_score:.6f}")
    print(f"cross-fit learned blend OOF: {crossfit_raw_summary['macro_f1']:.6f}")
    print(f"cross-fit learned blend+bias OOF: {crossfit_summary['macro_f1']:.6f}")
    print(
        "full OOF fit (낙관적·선택 금지): "
        f"{result['final_full_oof_fit']['macro_f1']:.6f}"
    )
    print(
        f"final weights: {np.asarray(final_weights).round(4).tolist()}"
        + ("  (고정)" if fixed_weights is not None else "  (학습)")
    )
    print(
        "test TVD guardrail: "
        f"{raw_distribution['tvd']:.4f} -> {adjusted_distribution['tvd']:.4f} "
        f"(warning={adjusted_distribution['warning']})"
    )
    print(f"OOF: {oof_path}")
    print(f"test: {test_path}")
    print(f"raw blend OOF: {raw_oof_path}")
    print(f"raw blend test: {raw_test_path}")
    print(f"uniform blend OOF: {uniform_oof_path}")
    print(f"uniform blend test: {uniform_test_path}")
    print(f"log: {log_path}")
    if args.submission:
        print(f"submission candidate: {submission_path}")
        print(f"raw blend submission candidate: {raw_submission_path}")
        print(f"uniform blend submission candidate: {uniform_submission_path}")
    print("로컬 파일만 만들었다. DACON 업로드는 직접 한다.")


if __name__ == "__main__":
    main()
