#!/usr/bin/env python
"""팀 공유 드라이브 규격으로 OOF·test 예측을 내보낸다.

    python scripts/export_for_drive.py \
        --oof  artifacts/oof/oof_ens11_xcr_skf5.csv \
        --test artifacts/test_predictions/test_ens11_xcr_skf5.csv \
        --log  artifacts/logs/ens11_xcr_skf5.json \
        --version v001 --config f11 --seed 42

## 왜 변환이 필요한가

우리 파이프라인은 `ID` / `p_<클래스이름>` / `y_true` 를 쓰고 fold 번호를 안 담는다.
팀 공유 규격은 `sample_id` / `prob_class_<번호>` / `true_label` / `fold` 다. 컬럼
이름만 바꾸면 되는 게 아니라 **클래스 이름을 번호로 바꾸는 매핑**이 들어간다.

여기가 이 스크립트에서 제일 위험한 자리다. 담당자마다 번호를 다르게 매기면 확률이
엉뚱한 클래스에 붙는데, 파일만 봐서는 아무도 못 알아챈다. 그래서 매핑을
`config.json` 의 `class_mapping` 에 통째로 적어 같이 올린다 — 받는 쪽이 대조할 수
있어야 한다.

정렬 기준은 **클래스 이름 알파벳순**이고, 이건 `train_gbdt.py` 가
`np.unique(y)` 로 만든 `data.classes` 순서와 같다. 확률 컬럼도 그 순서로 저장돼
있으므로 `p_*` 를 왼쪽부터 읽으면 그대로 `prob_class_0` 부터가 된다.

## fold 컬럼

우리 OOF 에는 없어서 `data/process/train_folds.parquet` 에서 `ID` 로 병합해 붙인다.
어느 분할(`fold_skf5` / `fold_group5`)을 썼는지는 앙상블 로그의 `fold_column` 을
읽어 자동으로 맞춘다 — 손으로 고르게 두면 OOF 를 만든 분할과 다른 번호가 붙어도
아무 경고 없이 통과한다.

## 산출물

    oof.csv         sample_id, fold, true_label, prob_class_0..N
    test_pred.csv   sample_id, prob_class_0..N
    submission.csv  ID, SUBCLASS  — test_pred.csv 의 argmax
    metrics.json    OOF·fold 별 macro F1 과 멤버 점수
    config.json     클래스 매핑·피처 블록·분할

`submission.csv` 는 `test_pred.csv` 에서 파생시킨다. 기존 제출 파일을 복사해 오면
확률과 라벨이 서로 다른 실험에서 온 조합이 될 수 있는데 폴더만 봐서는 구분이 안 된다.

## 만들지 않는 것

DACON 제출은 하지 않는다. 이 스크립트는 로컬 폴더만 만들고, 드라이브 업로드도
사람이 한다. `READY.txt` 는 업로드 완료 신호이므로 **업로드가 끝난 뒤에** 만든다
(`--ready` 로 따로 붙인다).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.io import write_submission  # noqa: E402
from cancer_hack.metrics import macro_f1, read_prediction_frame  # noqa: E402

DEFAULT_FOLDS = PROJECT_ROOT / "data/process/train_folds.parquet"
DEFAULT_SAMPLE_SUBMISSION = PROJECT_ROOT / "data/raw/sample_submission.csv"
DEFAULT_OUT = PROJECT_ROOT / "artifacts/drive_export"


def log(message: str) -> None:
    print(message, flush=True)


def build_class_mapping(classes: list[str]) -> dict[str, int]:
    """클래스 이름 -> prob_class 번호. 알파벳순이 곧 번호순이다."""
    ordered = sorted(classes)
    if ordered != list(classes):
        raise SystemExit(
            "확률 컬럼이 알파벳순이 아니다. 이 스크립트는 열 순서를 그대로 번호로 "
            f"옮기므로 순서가 다르면 확률이 밀린다.\n  받은 순서: {list(classes)}"
        )
    return {name: i for i, name in enumerate(ordered)}


def convert_oof(oof_path: Path, folds_path: Path, fold_column: str) -> tuple[pd.DataFrame, dict]:
    frame, classes = read_prediction_frame(oof_path)
    mapping = build_class_mapping(classes)

    if "y_true" not in frame.columns:
        raise SystemExit(f"{oof_path.name} 에 y_true 가 없다 — OOF 파일이 맞는지 확인한다")

    folds = pd.read_parquet(folds_path)
    if fold_column not in folds.columns:
        raise SystemExit(
            f"{folds_path.name} 에 {fold_column} 이 없다. 있는 열: {list(folds.columns)}"
        )

    merged = frame.merge(folds[["ID", fold_column]], on="ID", how="left", validate="one_to_one")
    missing = int(merged[fold_column].isna().sum())
    if missing:
        raise SystemExit(
            f"{missing}개 행에 fold 번호가 안 붙었다 — OOF 의 ID 가 fold 파일에 없다"
        )

    out = pd.DataFrame({
        "sample_id": merged["ID"],
        "fold": merged[fold_column].astype(int),
        "true_label": merged["y_true"],
    })
    for name, index in mapping.items():
        out[f"prob_class_{index}"] = merged[f"p_{name}"].to_numpy(np.float64)
    return out, mapping


def convert_test(test_path: Path, mapping: dict[str, int]) -> pd.DataFrame:
    frame, classes = read_prediction_frame(test_path)
    if build_class_mapping(classes) != mapping:
        raise SystemExit("test 파일의 클래스 구성이 OOF 와 다르다 — 같은 실험이 맞는지 확인한다")

    out = pd.DataFrame({"sample_id": frame["ID"]})
    for name, index in mapping.items():
        out[f"prob_class_{index}"] = frame[f"p_{name}"].to_numpy(np.float64)
    return out


def build_submission_frame(test: pd.DataFrame, mapping: dict[str, int]) -> pd.DataFrame:
    """test 확률 -> `io.build_submission` 이 받는 프레임(`ID` / `y_pred`).

    **같이 올리는 `test_pred.csv` 에서 그대로 뽑는다.** 기존 제출 파일을 복사해 오면
    확률과 라벨이 다른 실험에서 온 조합이 될 수 있는데, 폴더만 봐서는 구분이 안 된다.
    한 파일에서 파생시키면 그 어긋남이 원천적으로 불가능하다.
    """
    inverse = {index: name for name, index in mapping.items()}
    columns = [f"prob_class_{i}" for i in range(len(mapping))]
    labels = np.array([inverse[i] for i in range(len(mapping))])
    return pd.DataFrame({
        "ID": test["sample_id"],
        "y_pred": labels[test[columns].to_numpy(np.float64).argmax(axis=1)],
    })


def build_metrics(oof: pd.DataFrame, mapping: dict[str, int], source_log: dict | None) -> dict:
    inverse = {index: name for name, index in mapping.items()}
    prob_columns = [f"prob_class_{i}" for i in range(len(mapping))]
    predicted = np.array([inverse[i] for i in oof[prob_columns].to_numpy().argmax(axis=1)])

    metrics = {
        "oof_macro_f1": float(macro_f1(oof["true_label"], predicted)),
        "n_samples": int(len(oof)),
        "n_classes": len(mapping),
        "fold_macro_f1": {},
    }
    for fold in sorted(oof["fold"].unique()):
        mask = (oof["fold"] == fold).to_numpy()
        metrics["fold_macro_f1"][str(int(fold))] = float(
            macro_f1(oof["true_label"][mask], predicted[mask])
        )
    if source_log:
        metrics["source_log"] = {
            key: source_log[key]
            for key in ("tag", "fold_column", "oof_sources", "source_oof_macro_f1",
                        "uniform_blend_oof_macro_f1")
            if key in source_log
        }
        for key in ("crossfit_raw_blend", "crossfit_calibrated"):
            if key in source_log:
                metrics["source_log"][key] = {
                    "macro_f1": source_log[key].get("macro_f1"),
                    "accuracy": source_log[key].get("accuracy"),
                }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--log", type=Path, help="앙상블/학습 로그 json — fold_column 을 여기서 읽는다")
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--fold-column", default=None, help="생략하면 --log 에서 읽는다")
    parser.add_argument("--version", required=True, help="v001 처럼")
    parser.add_argument("--config", required=True, help="f11 처럼 — 폴더 이름에 들어간다")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="ensemble", help="폴더 설명용")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ready", action="store_true",
                        help="READY.txt 를 같이 만든다. **업로드가 끝난 뒤에만** 쓴다")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_log = None
    if args.log and args.log.exists():
        with open(args.log, encoding="utf-8") as handle:
            source_log = json.load(handle)

    fold_column = args.fold_column or (source_log or {}).get("fold_column")
    if not fold_column:
        raise SystemExit("--fold-column 을 주거나 fold_column 이 든 --log 를 준다")

    log(f"fold 열: {fold_column}")
    oof, mapping = convert_oof(args.oof, args.folds, fold_column)
    test = convert_test(args.test, mapping)
    metrics = build_metrics(oof, mapping, source_log)

    score = metrics["oof_macro_f1"]
    # 폴더 이름에 분할을 박는다. 같은 모델·같은 피처라도 skf 와 sgkf 는 점수가
    # 0.01~0.02 벌어져서, 이름에 없으면 나중에 둘을 나란히 놓고 "왜 다르지" 하게 된다.
    # 스태킹은 같은 분할끼리만 섞어야 하므로 눈으로 걸러낼 수 있어야 한다.
    cv_slug = {"fold_skf5": "skf5", "fold_group5": "group5"}.get(fold_column, fold_column)
    folder = (
        args.out
        / f"{args.version}_seed{args.seed}_{args.config}_{cv_slug}_macroF1_{score:.4f}"
    )
    if folder.exists() and not args.overwrite:
        raise SystemExit(f"{folder} 가 이미 있다. 버전을 올리거나 --overwrite 를 준다")
    folder.mkdir(parents=True, exist_ok=True)

    oof.to_csv(folder / "oof.csv", index=False, encoding="utf-8")
    test.to_csv(folder / "test_pred.csv", index=False, encoding="utf-8")

    # 제출 파일도 같이 넣는다 — 확률만 보면 "그래서 뭘 낸 거냐"를 되짚기 어렵다.
    # test_pred.csv 에서 파생시키므로 둘이 어긋날 수 없다. 만들 뿐 올리지 않는다.
    summary = write_submission(
        build_submission_frame(test, mapping),
        args.sample_submission,
        folder / "submission.csv",
    )

    with open(folder / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    config = {
        "version": args.version,
        "model": args.model,
        "config": args.config,
        "seed": args.seed,
        "cv": fold_column,
        "n_splits": int(oof["fold"].nunique()),
        # 받는 쪽이 확률 컬럼을 대조할 수 있어야 한다. 이게 어긋나면 조용히 틀린다.
        "class_mapping": mapping,
        "class_order_note": "prob_class_N 의 N 은 클래스 이름 알파벳순 인덱스다",
        "source_files": {"oof": args.oof.name, "test": args.test.name},
        "test_pred_note": "각 fold 의 test 예측을 평균한 값이다",
        "submission_note": "test_pred.csv 의 argmax 다. DACON 업로드는 하지 않았다",
    }
    with open(folder / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    if args.ready:
        (folder / "READY.txt").write_text(
            "업로드 완료.\n", encoding="utf-8"
        )

    log(f"\n{folder}")
    for path in sorted(folder.iterdir()):
        log(f"  {path.name:16s} {path.stat().st_size:>10,} bytes")
    log(f"\nOOF Macro F1 = {score:.4f}  ·  fold {sorted(oof['fold'].unique().tolist())}")
    log(f"제출 {summary['rows']}행 · 클래스 {summary['n_classes']}종 · "
        f"최다 {summary['top_class']} · 최소 {summary['rarest_class']}")
    log("DACON 업로드는 하지 않는다 — 로컬 파일만 만들었다.")
    if not args.ready:
        log("READY.txt 는 안 만들었다 — 업로드가 끝난 뒤 --ready 로 붙인다.")


if __name__ == "__main__":
    main()
