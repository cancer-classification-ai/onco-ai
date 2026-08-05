#!/usr/bin/env python
r"""팀원이 다른 환경에서 뽑은 OOF 를 **검증하고** 우리 앙상블에 넣어 본다.

    # 검증만 — fold 가 우리 것과 같은지 본다 (블렌딩 전에 반드시)
    .\.venv\Scripts\python.exe scripts\external_members.py --verify `
        --member "MLP=..\Models\MLP-...\oof\oof_dl_mlp_mlp_full_s42_skf5.csv"

    # 폴더 하나를 통째로 훑는다 (oof_*.csv 를 재귀로 찾는다)
    .\.venv\Scripts\python.exe scripts\external_members.py --verify --scan ..\Models

    # 검증 + 우리 GBDT 앙상블에 섞어 가중치를 훑는다
    .\.venv\Scripts\python.exe scripts\external_members.py --scan ..\Models --cv skf5

## 왜 검증이 먼저인가

이 팀은 멤버마다 다른 기계(로컬·Colab·Kaggle)에서 OOF 를 뽑아 한데 모은다. 그때 조용히
깨지는 게 **fold 분할**이다. 다른 분할에서 나온 OOF 를 섞으면 각자의 CV 는 멀쩡해 보이는데
합쳐 놓은 숫자만 거짓말을 한다 — 예외도 안 나고 그럴듯한 값이 찍힌다.

실제로 팀원 CatBoost 는 Colab fold 가 우리 것과 **20% 밖에 안 겹쳐** 스태킹을 접었다
(`process/notion_EXP_026_f16_ensemble.md`). 반대로 DL 4종은 Kaggle 에서 나왔는데도 우리
`fold_skf5` 와 fold 별 점수가 소수 6자리까지 같아 그대로 쓸 수 있었다.

그 차이를 **눈으로 구별할 방법이 없다.** 그래서 기계로 판정한다.

## 판정 방법

`--log` 로 그쪽 학습 로그(json)를 같이 주면 가장 강한 검증이 된다.

    그쪽 OOF 를 **우리** fold 로 잘라 fold 별 macro F1 을 다시 계산 → 그쪽 로그의
    `fold_macro_f1` 과 일치하는가

일치하면 같은 분할이다. 로그가 없으면 행 수·ID 순서·확률 합만 본다(약한 검증).

## 규정

train 라벨과 OOF 확률만 쓴다. test 는 예측 파일을 그대로 옮겨 담을 뿐 통계를 내지 않는다.

## DACON 제출

이 스크립트는 제출 파일을 만들지 않는다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import json
import re
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


# 경로는 `cancer_hack.paths` 가 정한다 — 값은 쓰는 시점에 정해진다.
from cancer_hack.paths import LAZY_RAW, LazyDir, artifacts_dir, process_dir, raw_dir  # noqa: E402
from cancer_hack.metrics import macro_f1  # noqa: E402

RAW = LAZY_RAW
FOLDS = LazyDir(lambda: process_dir() / "train_folds.parquet")
#: 우리 GBDT 멤버. `run_seed_sweep.py` 와 같은 규약이다.
SLUG = "k500_sp1000m3p2_cm20drishamm83824c_lt64svdnmfl257c23c_gm24shac918bc_sg30sha71bc36"
GBDT_MODELS = ["xgb", "catboost", "rf"]
GBDT_WEIGHTS = [0.45, 0.45, 0.10]
from cancer_hack.validation import SEED_ENSEMBLE  # noqa: E402

DEFAULT_SEEDS = list(SEED_ENSEMBLE)


def log(message: str) -> None:
    print(message, flush=True)


def load_labels() -> tuple[pd.Series, list[str]]:
    labels = pd.read_csv(RAW / "train.csv", usecols=["ID", "SUBCLASS"]).set_index("ID")["SUBCLASS"]
    return labels, sorted(labels.unique())


def prob_columns(frame: pd.DataFrame, classes: list[str]) -> list[str] | None:
    wanted = [f"p_{c}" for c in classes]
    if all(c in frame.columns for c in wanted):
        return wanted
    wanted = [f"prob_class_{i}" for i in range(len(classes))]
    if all(c in frame.columns for c in wanted):
        return wanted
    return None


def guess_log(oof_path: Path) -> Path | None:
    """`oof_<stem>.csv` 옆이나 `../logs/<stem>.json` 에 있는 학습 로그를 찾는다."""
    stem = oof_path.stem
    stem = stem[4:] if stem.startswith("oof_") else stem
    for candidate in (
        oof_path.with_name(f"{stem}.json"),
        oof_path.parent / f"{stem}.json",
        oof_path.parent.parent / "logs" / f"{stem}.json",
        oof_path.parent.parent / "artifacts" / "logs" / f"{stem}.json",
    ):
        if candidate.exists():
            return candidate
    return None


def guess_test(oof_path: Path) -> Path | None:
    stem = oof_path.stem[4:] if oof_path.stem.startswith("oof_") else oof_path.stem
    for candidate in (
        oof_path.parent.parent / "test_predictions" / f"test_{stem}.csv",
        oof_path.parent / f"test_{stem}.csv",
        oof_path.parent.parent / "artifacts" / "test_predictions" / f"test_{stem}.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def verify(oof_path: Path, folds: pd.DataFrame, labels: pd.Series, classes: list[str]) -> dict:
    """한 멤버를 검증한다. `fold` 키가 'skf5'/'group5' 면 그 분할에서 나온 것이다."""
    out: dict = {"path": oof_path, "ok": False, "fold": None, "notes": []}
    try:
        frame = pd.read_csv(oof_path)
    except Exception as error:  # noqa: BLE001
        out["notes"].append(f"읽기 실패: {error}")
        return out

    cols = prob_columns(frame, classes)
    if cols is None:
        out["notes"].append("확률 열을 못 찾았다 (p_<클래스> 또는 prob_class_N 이어야 한다)")
        return out
    if len(frame) != len(folds):
        out["notes"].append(f"행 수 {len(frame)} != {len(folds)}")
        return out
    if "ID" not in frame.columns or not (frame["ID"].to_numpy() == folds["ID"].to_numpy()).all():
        out["notes"].append("ID 순서가 train_folds 와 다르다")
        return out

    probs = np.asarray(frame[cols], dtype=float)
    if not np.isfinite(probs).all():
        out["notes"].append("확률에 NaN/inf 가 있다")
        return out

    y = labels.reindex(frame["ID"]).astype(str).to_numpy()
    pred = np.asarray(classes)[probs.argmax(1)]
    out["oof_macro_f1"] = float(macro_f1(y, pred))
    out["probs"] = probs
    out["y"] = y

    # 가장 강한 검증 — 우리 fold 로 잘라 재계산한 값이 그쪽 로그와 맞는가
    log_path = guess_log(oof_path)
    if log_path is not None:
        meta = json.loads(log_path.read_text(encoding="utf-8"))
        theirs = meta.get("fold_macro_f1")
        if theirs:
            for name in ("skf5", "group5"):
                column = f"fold_{name}"
                if column not in folds.columns:
                    continue
                ours = [
                    float(macro_f1(y[folds[column].to_numpy() == f], pred[folds[column].to_numpy() == f]))
                    for f in range(len(theirs))
                ]
                if max(abs(a - b) for a, b in zip(ours, theirs)) < 1e-6:
                    out["fold"] = name
                    break
            if out["fold"] is None:
                out["notes"].append("fold 별 점수가 우리 어느 분할과도 안 맞는다 — 다른 분할이다")
                return out
        else:
            out["notes"].append("로그에 fold_macro_f1 이 없어 분할을 확정 못 했다")
    else:
        out["notes"].append("학습 로그를 못 찾아 분할을 확정 못 했다 (약한 검증)")
        match = re.search(r"_(skf5|group5|sgkf)", oof_path.stem)
        if match:
            out["fold"] = "group5" if match.group(1) in ("group5", "sgkf") else "skf5"
            out["notes"].append(f"파일명으로 {out['fold']} 로 **추정** — 근거가 약하다")

    out["test_path"] = guess_test(oof_path)
    out["ok"] = True
    return out


def our_ensemble(cv: str, seeds: list[int], classes: list[str], labels: pd.Series):
    """f16 GBDT 3종 × seed 평균 -> 0.45/0.45/0.10 블렌드."""
    oof_dir = artifacts_dir() / "oof"
    cols = [f"p_{c}" for c in classes]
    stacks, ids = [], None
    for model in GBDT_MODELS:
        mats = []
        for seed in seeds:
            path = oof_dir / f"oof_{model}_repo16_f16_{cv}_{SLUG}_s{seed}.csv"
            if not path.exists():
                return None, None, f"{path.name} 이 없다"
            frame = pd.read_csv(path)
            ids = frame["ID"].to_numpy() if ids is None else ids
            mats.append(np.asarray(frame[cols], dtype=float))
        stacks.append(np.mean(mats, axis=0))
    blended = np.average(stacks, axis=0, weights=GBDT_WEIGHTS)
    return blended, labels.reindex(ids).astype(str).to_numpy(), None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", action="append", default=[],
                        help="'이름=OOF경로' 형식. 여러 번 줄 수 있다")
    parser.add_argument("--scan", type=Path, default=None,
                        help="이 폴더 아래 oof_*.csv 를 재귀로 찾는다 (예: ..\\Models)")
    parser.add_argument("--verify", action="store_true", help="검증만 하고 블렌딩은 건너뛴다")
    parser.add_argument("--cv", default="skf5", choices=["skf5", "group5"],
                        help="블렌딩을 잴 분할. 멤버가 이 분할에서 나온 것만 섞는다")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[0.10, 0.15, 0.20, 0.30, 0.40],
                        help="외부 멤버에게 줄 총 가중치를 훑는다")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels, classes = load_labels()
    folds = pd.read_parquet(FOLDS)

    targets: list[tuple[str, Path]] = []
    for spec in args.member:
        name, _, path = spec.partition("=")
        targets.append((name or Path(path).stem, Path(path)))
    if args.scan:
        for path in sorted(args.scan.rglob("oof_*.csv")):
            targets.append((path.stem[4:][:38], path))
    if not targets:
        log("검증할 멤버가 없다. --member 또는 --scan 을 준다.")
        return 1

    log(f"{'멤버':40s} {'행':>6s} {'OOF F1':>8s} {'분할':>8s}  비고")
    log("-" * 100)
    verified = []
    for name, path in targets:
        result = verify(path, folds, labels, classes)
        note = "; ".join(result["notes"]) or "-"
        if result["ok"]:
            log(f"{name:40s} {len(folds):6d} {result['oof_macro_f1']:8.4f} "
                f"{result['fold'] or '?':>8s}  {note}")
            verified.append((name, result))
        else:
            log(f"{name:40s} {'':6s} {'':8s} {'실패':>8s}  {note}")

    if args.verify or not verified:
        return 0

    usable = [(n, r) for n, r in verified if r["fold"] == args.cv]
    if not usable:
        log(f"\n{args.cv} 분할에서 나온 멤버가 없다 — 블렌딩을 건너뛴다.")
        log("다른 분할의 OOF 를 섞으면 합쳐 놓은 숫자만 조용히 거짓말을 한다.")
        return 1

    base, y, error = our_ensemble(args.cv, args.seeds, classes, labels)
    if error:
        log(f"\n우리 앙상블을 못 만들었다: {error}")
        return 1

    base_score = float(macro_f1(y, np.asarray(classes)[base.argmax(1)]))
    log(f"\n우리 GBDT 앙상블 ({args.cv}, seed {args.seeds}, 0.45/0.45/0.10) = {base_score:.4f}")
    log(f"섞을 외부 멤버 {len(usable)}개: {', '.join(n for n, _ in usable)}")

    log(f"\n{'구성':44s} {'macro F1':>9s} {'델타':>9s}   불일치")
    log("-" * 80)
    rows = []
    for name, result in usable:
        disagree = (result["probs"].argmax(1) != base.argmax(1)).mean()
        for w in args.weights:
            score = float(macro_f1(y, np.asarray(classes)[((1 - w) * base + w * result["probs"]).argmax(1)]))
            rows.append((f"+{name} w={w:.2f}", score, score - base_score, disagree))
    if len(usable) > 1:
        pooled = np.mean([r["probs"] for _, r in usable], axis=0)
        disagree = (pooled.argmax(1) != base.argmax(1)).mean()
        for w in args.weights:
            score = float(macro_f1(y, np.asarray(classes)[((1 - w) * base + w * pooled).argmax(1)]))
            rows.append((f"+외부 {len(usable)}종 평균 w={w:.2f}", score, score - base_score, disagree))

    for label, score, delta, disagree in sorted(rows, key=lambda r: -r[1]):
        log(f"{label:44s} {score:9.4f} {delta:+9.4f}   {disagree:5.1%}")

    log(f"\n※ {args.cv} 축이다. 다른 분할의 절대값과 비교하지 말 것 — 두 분할은 같은 모델을")
    log("  다르게 재는 두 자다. 같은 자로 잰 **델타**만 유효하다.")
    log("※ 최고 가중치는 여러 후보 중 최댓값이라 위쪽으로 치우친다. 채택 전에 seed 를 바꿔 확인한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
