#!/usr/bin/env python
r"""OOF 라이브러리에서 Caruana 그리디로 앙상블 멤버를 고른다.

    # 무엇이 라이브러리에 들어가는지만 본다 (학습·선택 없음)
    .\.venv\Scripts\python.exe scripts\greedy_blend.py --cv group5 --list-only

    # 교차적합으로 정직하게 재고, 전체 적합 가중치로 test 를 만든다
    .\.venv\Scripts\python.exe scripts\greedy_blend.py --cv group5 --tag greedy_g5

## 무엇이 다른가

지금은 사람이 멤버 3개를 고르고 `0.45/0.45/0.10` 을 손으로 박는다. `artifacts/oof/` 에는
group5 기준 100개가 넘는 OOF 가 쌓여 있는데 그중 셋만 쓰는 셈이다. 그리디는 그 라이브러리
전체에서 **점수가 가장 오르는 멤버를 하나씩, 복원 허용으로** 담는다. 담긴 횟수가 곧
가중치라 멤버 선택과 가중 최적화가 한 번에 끝난다.

재학습이 없다. 이미 있는 OOF·test 확률 행렬 연산이라 몇 분이면 끝난다.

## 정직한 점수를 어떻게 내는가

**그리디는 라이브러리가 클수록 선택에 쓴 행에 과적합한다.** 100개 중 최고 조합을 고르면
그 조합은 그 fold 들의 우연까지 맞춘다. 그래서 `calibrate_ensemble.py` 와 같은 교차적합을 쓴다.

    fold f 를 뺀 나머지에서 그리디로 가중치를 고른다 → fold f 에서만 점수를 잰다

이렇게 모은 OOF 예측의 macro F1 이 **보고할 값**이다. 전체 OOF 에 한 번 더 맞춘 가중치는
test 변환에만 쓰고 점수로 보고하지 않는다(`--full-fit` 값에 `optimistic` 이라 적어 둔다).

## 라이브러리에 무엇을 넣는가

같은 분할(`--cv`)에서 나온 OOF 만 넣는다. 기본으로 **파생 블렌드는 뺀다** —
`oof_ens*`·`oof_blend*`·`oof_stack*`·`*_raw_blend`·`*_uniform_blend` 는 그 자체가 다른
멤버들의 평균이라, 넣으면 같은 멤버가 두 경로로 중복 계상되고 결과 해석이 어려워진다.
`--include-blends` 로 켤 수 있다.

**fold 검증은 하지 않는다.** `artifacts/oof/` 는 우리가 만든 것이라는 전제다. 외부에서 받은
OOF 는 `scripts/external_members.py --verify` 로 먼저 분할을 확인한다 — 파일명이 group5 인데
실제로는 다른 분할인 사례가 `Models/` 에 9개 있었다.

## DACON 제출

test 예측과 제출 csv 를 로컬에 만들 뿐 업로드하지 않는다.
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

from cancer_hack.ensemble import GreedyEnsembleSelector, weighted_average  # noqa: E402
from cancer_hack.metrics import macro_f1  # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"
RAW = PROJECT_ROOT / "data" / "raw"
FOLDS = PROJECT_ROOT / "data" / "process" / "train_folds.parquet"

#: 그 자체가 다른 멤버의 평균인 파일. 기본으로 제외한다.
DERIVED = re.compile(r"^oof_(ens|blend|stack|xc)|_raw_blend|_uniform_blend|_cal$")


def log(message: str) -> None:
    print(message, flush=True)


def collect(cv: str, include_blends: bool) -> list[tuple[str, Path]]:
    out = []
    for path in sorted((ARTIFACTS / "oof").glob("*.csv")):
        if cv not in path.stem:
            continue
        if not include_blends and DERIVED.search(path.stem):
            continue
        out.append((path.stem[4:] if path.stem.startswith("oof_") else path.stem, path))
    return out


def load_matrix(paths: list[Path], classes: list[str], ids: np.ndarray):
    """(멤버, 행, 클래스) 스택. ID 순서가 어긋난 멤버는 버린다."""
    cols = [f"p_{c}" for c in classes]
    kept, arrays = [], []
    for path in paths:
        frame = pd.read_csv(path)
        if len(frame) != len(ids) or not (frame["ID"].to_numpy() == ids).all():
            continue
        if not all(c in frame.columns for c in cols):
            continue
        values = np.asarray(frame[cols], dtype=float)
        if not np.isfinite(values).all():
            continue
        kept.append(path)
        arrays.append(values)
    return kept, arrays


def test_path_for(oof_path: Path) -> Path:
    stem = oof_path.stem[4:] if oof_path.stem.startswith("oof_") else oof_path.stem
    return ARTIFACTS / "test_predictions" / f"test_{stem}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv", default="group5", choices=["group5", "skf5"])
    parser.add_argument("--fold-column", default=None, help="기본은 fold_<cv>")
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--bag-fraction", type=float, default=0.6,
                        help="라운드마다 후보로 둘 라이브러리 비율. 1.0 이면 bagging 끄기")
    parser.add_argument("--bag-rounds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--include-blends", action="store_true",
                        help="파생 블렌드(ens/blend/stack)도 라이브러리에 넣는다")
    parser.add_argument("--allow-missing-test", action="store_true",
                        help="test 예측이 없는 멤버도 라이브러리에 넣는다 (제출 파일은 못 만든다)")
    parser.add_argument("--list-only", action="store_true", help="라이브러리만 찍고 끝낸다")
    parser.add_argument("--tag", default="greedy", help="산출물 이름")
    parser.add_argument("--top", type=int, default=15, help="보고할 상위 멤버 수")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    fold_column = args.fold_column or f"fold_{args.cv}"

    labels = pd.read_csv(RAW / "train.csv", usecols=["ID", "SUBCLASS"]).set_index("ID")["SUBCLASS"]
    classes = sorted(labels.unique())
    folds = pd.read_parquet(FOLDS)
    ids = folds["ID"].to_numpy()
    y = labels.reindex(ids).astype(str).to_numpy()
    fold_ids = folds[fold_column].to_numpy()

    named = collect(args.cv, args.include_blends)
    log(f"라이브러리 후보 {len(named)}개 ({args.cv}, 파생 블렌드 "
        f"{'포함' if args.include_blends else '제외'})")
    # test 예측이 없는 멤버는 **선택되기 전에** 뺀다. 뒤에서 걸러 내면 그 멤버가 뽑혔을 때
    # 제출 파일을 못 만들고, 남은 것만으로 다시 정규화하면 교차적합 점수와 다른 앙상블이 된다.
    if not args.allow_missing_test:
        before = len(named)
        named = [(n, p) for n, p in named if test_path_for(p).exists()]
        if before != len(named):
            log(f"  test 예측이 없어 제외 {before - len(named)}개")
    kept, arrays = load_matrix([p for _, p in named], classes, ids)
    names = [p.stem[4:] if p.stem.startswith("oof_") else p.stem for p in kept]
    log(f"ID·스키마 검증 통과 {len(kept)}개")
    if args.list_only:
        solo = sorted(((macro_f1(y, np.asarray(classes)[a.argmax(1)]), n)
                       for n, a in zip(names, arrays)), reverse=True)
        for score, name in solo[: args.top]:
            log(f"  {score:.4f}  {name[:80]}")
        return 0
    if len(kept) < 2:
        log("멤버가 2개 미만이라 그리디를 돌릴 수 없다.")
        return 1

    # --- 교차적합: fold 를 뺀 나머지에서 고르고 그 fold 에서만 잰다 ---
    crossfit = np.zeros((len(ids), len(classes)), dtype=np.float64)
    per_fold = []
    for fold in sorted(set(fold_ids.tolist())):
        valid = fold_ids == fold
        train = ~valid
        selector = GreedyEnsembleSelector(
            n_rounds=args.n_rounds, bag_fraction=args.bag_fraction,
            bag_rounds=args.bag_rounds, random_state=args.random_state,
        ).fit([a[train] for a in arrays], y[train], classes)
        crossfit[valid] = weighted_average([a[valid] for a in arrays], selector.weights_)
        score = macro_f1(y[valid], np.asarray(classes)[crossfit[valid].argmax(1)])
        per_fold.append(float(score))
        chosen = int((selector.counts_ > 0).sum())
        log(f"  fold {fold}: 멤버 {chosen}개 선택 · 이 fold 점수 {score:.4f}")

    crossfit_score = float(macro_f1(y, np.asarray(classes)[crossfit.argmax(1)]))
    # 교차적합 OOF 를 남긴다. 없으면 이 앙상블을 **다음 블렌딩의 멤버로 못 쓰고**,
    # 팀 드라이브 규격(`export_for_drive.py`)도 oof 를 요구해 내보내기가 막힌다.
    oof_out = ARTIFACTS / "oof" / f"oof_{args.tag}.csv"
    # `y_true` 를 같이 넣는다 — `export_for_drive.py` 가 팀 규격의 `true_label` 을
    # 여기서 읽는다. 없으면 내보내기가 막힌다.
    pd.DataFrame({"ID": ids,
                  **{f"p_{c}": crossfit[:, j] for j, c in enumerate(classes)},
                  "y_true": y}).to_csv(oof_out, index=False)
    log(f"\nOOF: {oof_out}")
    log(f"교차적합 macro F1 = {crossfit_score:.4f}   ← 보고할 값")
    log(f"  fold 별 {[round(s, 4) for s in per_fold]}")

    # --- 전체 적합: test 변환용 가중치 (점수로 보고하지 않는다) ---
    final = GreedyEnsembleSelector(
        n_rounds=args.n_rounds, bag_fraction=args.bag_fraction,
        bag_rounds=args.bag_rounds, random_state=args.random_state,
    ).fit(arrays, y, classes)
    log(f"전체 적합 macro F1 = {final.train_macro_f1_:.4f}   (낙관적 — 선택·보고 금지)")

    order = np.argsort(-final.weights_)
    log(f"\n선택된 멤버 {(final.counts_ > 0).sum()}개 / 라이브러리 {len(kept)}개 — 상위 {args.top}")
    for index in order[: args.top]:
        if final.counts_[index] == 0:
            break
        solo = macro_f1(y, np.asarray(classes)[arrays[index].argmax(1)])
        log(f"  {final.weights_[index]:.3f}  ({final.counts_[index]:>3d}회)  "
            f"단독 {solo:.4f}  {names[index][:70]}")

    # --- test 예측 ---
    missing = [names[i] for i in range(len(kept)) if final.counts_[i] > 0
               and not test_path_for(kept[i]).exists()]
    if missing:
        log(f"\ntest 예측이 없는 선택 멤버 {len(missing)}개 — 제출 파일을 만들 수 없다:")
        for name in missing[:5]:
            log(f"  {name}")
    else:
        sample = pd.read_csv(RAW / "sample_submission.csv")
        cols = [f"p_{c}" for c in classes]
        test_arrays, weights = [], []
        for i in range(len(kept)):
            if final.counts_[i] == 0:
                continue
            frame = pd.read_csv(test_path_for(kept[i]))
            frame = frame.set_index("ID").reindex(sample["ID"]).reset_index()
            test_arrays.append(np.asarray(frame[cols], dtype=float))
            weights.append(final.weights_[i])
        blended = weighted_average(test_arrays, weights)
        out_dir = ARTIFACTS / "test_predictions"
        pd.DataFrame({"ID": sample["ID"], **{c: blended[:, j] for j, c in enumerate(cols)}}).to_csv(
            out_dir / f"test_{args.tag}.csv", index=False)
        submission = pd.DataFrame({"ID": sample["ID"],
                                   "SUBCLASS": np.asarray(classes)[blended.argmax(1)]})
        sub_path = ARTIFACTS / "submissions" / f"submission_{args.tag}.csv"
        submission.to_csv(sub_path, index=False, encoding="UTF-8-sig")
        log(f"\ntest: {out_dir / f'test_{args.tag}.csv'}")
        log(f"제출 후보: {sub_path}")

    report = {
        "tag": args.tag, "cv": args.cv, "fold_column": fold_column,
        "library_size": len(kept), "n_rounds": args.n_rounds,
        "bag_fraction": args.bag_fraction, "bag_rounds": args.bag_rounds,
        "crossfit_macro_f1": crossfit_score, "crossfit_fold_macro_f1": per_fold,
        "full_fit_macro_f1": {"value": final.train_macro_f1_,
                              "warning": "optimistic_not_for_model_selection"},
        "selected": {names[i]: int(final.counts_[i])
                     for i in range(len(kept)) if final.counts_[i] > 0},
    }
    log_path = ARTIFACTS / "logs" / f"{args.tag}.json"
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"로그: {log_path}")
    log("\n로컬 파일만 만들었다. DACON 업로드는 사람이 직접 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
