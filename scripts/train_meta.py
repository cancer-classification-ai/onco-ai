#!/usr/bin/env python
"""OOF 스태킹 — 멤버 확률을 입력으로 받는 메타 모델을 학습한다.

    python scripts/train_meta.py \
        --oof  artifacts/oof/oof_xgb_....csv artifacts/oof/oof_catboost_....csv \
        --test artifacts/test_predictions/test_xgb_....csv ... \
        --fold-column fold_skf5 --tag stack16_skf5

## 블렌딩과 무엇이 다른가

`calibrate_ensemble.py` 는 멤버마다 스칼라 가중치 하나를 준다(멤버 수 − 1 개
파라미터). 스태킹은 **클래스별로 다른 가중치**를 배운다 — 26개 클래스 × 멤버 수
만큼의 계수가 생긴다. "CatBoost 는 SKCM 에 강하고 XGBoost 는 KIRC 에 강하다"
같은 걸 잡아낼 수 있는 대신, 6,201행에서 그만큼 많은 파라미터를 학습해야 한다.

그래서 기본 메타 모델은 로지스틱 회귀다. 파라미터가 적고 정규화가 걸린다.

## 정직한 평가

메타 모델도 **fold 교차적합**한다. fold k 의 점수는 나머지 fold 로 학습한 메타
모델로 낸다. 전체 OOF 에 fit 하고 같은 OOF 로 평가하면 자기평가 거품이 낀다 —
이 저장소에서 실측 +0.02 였다. 그 값도 로그에 남기지만
`optimistic_not_for_model_selection` 경고를 달아 선택에 못 쓰게 한다.

멤버 OOF 자체가 이미 out-of-fold 이므로, 여기에 메타 모델을 교차적합하면
"멤버도 못 본 행 · 메타도 못 본 행"에서 점수가 나온다.

## 전제: 모든 멤버가 같은 분할이어야 한다

멤버마다 fold 가 다르면 어떤 행은 "A 는 안 본 행인데 B 는 학습에 쓴 행"이 된다.
그 행에서 B 확률이 비정상적으로 정확하니 메타 모델이 B 에 쏠리고, CV 는 오르는데
test 로는 이전되지 않는다. `--fold-column` 이 가리키는 분할로 멤버가 전부 만들어졌는지
**사람이 확인해야 한다** — OOF 파일에 fold 열이 없어서 이 스크립트는 검사할 수 없다.

## 만들지 않는 것

DACON 제출은 하지 않는다. 로컬 파일만 만든다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cancer_hack.io import save_csv, write_submission  # noqa: E402
from cancer_hack.metrics import (  # noqa: E402
    build_prediction_frame,
    evaluate_classification,
    macro_f1,
    prediction_distribution_report,
)
from calibrate_ensemble import _load_fold_values, _load_prediction_set  # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_FOLDS = PROJECT_ROOT / "data/process/train_folds.parquet"
DEFAULT_SAMPLE = PROJECT_ROOT / "data/raw/sample_submission.csv"


def log(message: str) -> None:
    print(message, flush=True)


def stack_features(arrays: list[np.ndarray], *, transform: str) -> np.ndarray:
    """멤버 확률을 옆으로 이어 붙여 메타 모델 입력을 만든다.

    `logit` 은 확률을 로그-오즈로 편다. 확률은 0~1 에 눌려 있어 선형 모델이 양 끝의
    차이를 잘 못 보는데, 로짓으로 펴면 그게 살아난다. 0 과 1 근처에서 발산하므로
    클리핑한다.
    """
    if transform == "logit":
        arrays = [np.log(a / (1 - a)) for a in (np.clip(x, 1e-6, 1 - 1e-6) for x in arrays)]
    elif transform != "prob":
        raise SystemExit(f"모르는 transform {transform!r}. 사용 가능: prob, logit")
    return np.hstack(arrays)


def build_meta(name: str, seed: int, strength: float):
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression

        # 6,201행에 멤버당 26열이다. 정규화 없이는 바로 외운다.
        return LogisticRegression(C=strength, max_iter=2000, random_state=seed)
    if name == "ridge":
        from sklearn.linear_model import RidgeClassifier

        return RidgeClassifier(alpha=1.0 / max(strength, 1e-9), random_state=seed)
    raise SystemExit(f"모르는 메타 모델 {name!r}. 사용 가능: logistic, ridge")


def meta_proba(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features), dtype=np.float64)
    # RidgeClassifier 는 확률을 안 준다. decision_function 을 softmax 로 편다.
    scores = np.asarray(model.decision_function(features), dtype=np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--oof", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--test",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "생략하면 OOF 평가만 하고 test 예측·제출 파일을 만들지 않는다. "
            "멤버 중 test 확률을 아직 못 받은 게 있을 때 쓴다"
        ),
    )
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--fold-column", default="fold_skf5")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--meta", default="logistic", help="logistic / ridge")
    parser.add_argument("--C", type=float, default=1.0, help="작을수록 정규화가 세다")
    parser.add_argument("--transform", default="prob", help="prob / logit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.test is not None and len(args.oof) != len(args.test):
        raise SystemExit(f"--oof {len(args.oof)}개 · --test {len(args.test)}개 — 같아야 한다")
    if args.test is None and args.submission:
        raise SystemExit("--test 없이 --submission 은 만들 수 없다")

    oof_path = ARTIFACTS / "oof" / f"oof_{args.tag}.csv"
    test_path = ARTIFACTS / "test_predictions" / f"test_{args.tag}.csv"
    log_path = ARTIFACTS / "logs" / f"{args.tag}.json"
    existing = [p for p in (oof_path, log_path) if p.exists()]
    if args.test is not None and test_path.exists():
        existing.append(test_path)
    if existing and not args.overwrite:
        raise SystemExit(f"이미 있다: {existing[0]} — 태그를 바꾸거나 --overwrite 를 준다")

    ids, y_true, classes, oof_arrays = _load_prediction_set(args.oof, require_y=True)
    test_ids = test_arrays = None
    if args.test is not None:
        test_ids, _, test_classes, test_arrays = _load_prediction_set(args.test, require_y=False)
        if classes != test_classes:
            raise SystemExit("OOF 와 test 의 클래스 열 또는 순서가 다르다")
    fold_values = _load_fold_values(args.folds, ids, args.fold_column)

    log(f"멤버 {len(oof_arrays)}개 · 클래스 {len(classes)}개 · train {len(ids)}행 · "
        + (f"test {len(test_ids)}행" if test_ids is not None else "test 없음 (OOF 평가만)"))
    log(f"메타 {args.meta} (C={args.C}, transform={args.transform}) · 분할 {args.fold_column}")

    member_scores = {
        path.stem: float(macro_f1(y_true, np.asarray(classes)[values.argmax(axis=1)]))
        for path, values in zip(args.oof, oof_arrays)
    }
    for name, score in member_scores.items():
        log(f"  멤버 {score:.4f}  {name[:68]}")

    features = stack_features(oof_arrays, transform=args.transform)
    test_features = (
        stack_features(test_arrays, transform=args.transform)
        if test_arrays is not None
        else None
    )

    # 메타 모델 교차적합 — fold k 는 나머지 fold 로 학습한 모델이 예측한다.
    stacked = np.zeros((len(ids), len(classes)), dtype=np.float64)
    fold_rows = []
    for fold in sorted(np.unique(fold_values).tolist()):
        train_mask = fold_values != fold
        valid_mask = ~train_mask
        model = build_meta(args.meta, args.seed, args.C)
        model.fit(features[train_mask], y_true[train_mask])
        if list(model.classes_) != list(classes):
            raise SystemExit(f"fold {fold} 의 클래스 순서가 전체와 다르다")
        stacked[valid_mask] = meta_proba(model, features[valid_mask])
        fold_rows.append({
            "fold": int(fold),
            "n_train": int(train_mask.sum()),
            "n_valid": int(valid_mask.sum()),
            "macro_f1": float(macro_f1(
                y_true[valid_mask], np.asarray(classes)[stacked[valid_mask].argmax(axis=1)]
            )),
        })
        log(f"  fold {fold}  {fold_rows[-1]['macro_f1']:.4f}")

    stacked_pred = np.asarray(classes)[stacked.argmax(axis=1)]
    summary = evaluate_classification(y_true, stacked_pred, classes)
    best_member = max(member_scores.values())
    gain = summary["macro_f1"] - best_member
    log(f"\n스태킹 OOF(교차적합) {summary['macro_f1']:.4f}  "
        f"최고 멤버 {best_member:.4f}  차이 {gain:+.4f}")

    # test 용 최종 모델은 전체 OOF 로 학습한다. 이 점수는 자기평가라 선택에 못 쓴다.
    final = build_meta(args.meta, args.seed, args.C)
    final.fit(features, y_true)
    full_fit_pred = np.asarray(classes)[meta_proba(final, features).argmax(axis=1)]
    log(f"전체 OOF 적합(낙관·선택 금지) {macro_f1(y_true, full_fit_pred):.4f}")

    guardrail = None
    if test_features is not None:
        test_proba = meta_proba(final, test_features)
        test_pred = np.asarray(classes)[test_proba.argmax(axis=1)]
        guardrail = prediction_distribution_report(y_true, test_pred, classes)
        log(f"test 분포 TVD {guardrail['tvd']:.4f} (warning={guardrail['warning']})")

    save_csv(build_prediction_frame(ids, stacked, classes, y_true=y_true), oof_path)
    if test_features is not None:
        save_csv(build_prediction_frame(test_ids, test_proba, classes), test_path)

    payload = {
        "tag": args.tag,
        "meta": args.meta,
        "C": args.C,
        "transform": args.transform,
        "seed": args.seed,
        "fold_column": args.fold_column,
        "oof_sources": [str(p) for p in args.oof],
        "test_sources": [str(p) for p in args.test] if args.test else None,
        "classes": classes,
        "n_samples": int(len(ids)),
        "n_test": int(len(test_ids)) if test_ids is not None else None,
        "member_oof_macro_f1": member_scores,
        "best_member_macro_f1": best_member,
        "crossfit_stacked": summary,
        "crossfit_folds": fold_rows,
        "gain_over_best_member": float(gain),
        "full_oof_fit": {
            "warning": "optimistic_not_for_model_selection",
            "macro_f1": float(macro_f1(y_true, full_fit_pred)),
        },
        "test_distribution_guardrail": guardrail,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temp = log_path.with_suffix(".tmp.json")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp.replace(log_path)

    log(f"\nOOF  {oof_path}")
    log(f"test {test_path}" if test_features is not None
        else "test 예측 없음 — 멤버의 test 확률이 모이면 다시 돌린다")
    log(f"로그 {log_path}")

    if args.submission:
        info = write_submission(
            build_prediction_frame(test_ids, test_proba, classes), args.sample_submission,
            ARTIFACTS / "submissions" / f"submission_{args.tag}.csv",
        )
        log(f"제출 후보 {info['output_path']} ({info['rows']}행)")
    log("로컬 파일만 만들었다. DACON 업로드는 사람이 한다.")


if __name__ == "__main__":
    main()
