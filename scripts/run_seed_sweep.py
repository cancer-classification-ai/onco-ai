#!/usr/bin/env python
"""f16 seed 스윕 — 이미 있는 것은 건너뛰고 없는 것만 학습한다.

    # 무엇이 있고 무엇이 없는지, 예상 시간만 본다
    .\\.venv\\Scripts\\python.exe scripts\\run_seed_sweep.py --dry-run

    # 없는 것만 학습한다 (중간에 끊겨도 다시 돌리면 이어진다)
    .\\.venv\\Scripts\\python.exe scripts\\run_seed_sweep.py

    # 다 끝난 뒤 — 3/4/5 seed × 균등/고정 × skf/sgkf 를 전부 비교한다
    .\\.venv\\Scripts\\python.exe scripts\\run_seed_sweep.py --blend-only

## 무엇을 만드나

기준선 **`Models/Ensemble/v002_seed42_f16_group5_macroF1_0.5165`** 의 설정을 그대로 두고
seed 만 늘린다. 5 seed × 3 모델 × 2 분할 = 30 개의 OOF/test 예측을 채우고, 그 위에서
seed 개수(3/4/5) · 가중(균등 / 0.45·0.45·0.10) · 분할(skf5 / group5) 12 조합을 비교한다.

**바꾸는 축은 seed 하나뿐이다.** 피처(f16) · topk(500) · 가중치(balanced) · fold 파일은
v002 와 같아야 한다. 다른 걸 같이 움직이면 LB 델타에 두 효과가 섞인다.

## 왜 seed 인가

`research/10` §2.1 에서 기존 3-seed OOF 로 재학습 없이 쟀더니 **6개 구성 전부에서
+0.0038**(범위 +0.0025~+0.0046) 이었다. 이 프로젝트에서 방향이 일관되게 양수인 유일한
처치다. 2-seed 는 +0.0019 로 절반이라 3개는 채우는 게 낫다.

덤으로 group5 의 seed 표준편차가 0.0032 로 나왔는데, 그동안 쫓던 블록 델타(0.001~0.006)
한복판이다. **단일 seed 로 잰 개선은 seed 를 바꾸면 사라질 수 있다.**

## 이어 돌리기

파일이 있으면 건너뛴다. 판정은 `artifacts/oof/` 와 `artifacts/test_predictions/` 에
**둘 다** 있고 6,201행 · 26열이 채워져 있는지로 한다 — 중간에 끊긴 실행이 남긴 반쪽짜리
파일을 "있음" 으로 세면 조용히 틀린 앙상블이 나온다.

## DACON 제출

이 스크립트는 제출 파일을 만들지 않는다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
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

ARTIFACTS = PROJECT_ROOT / "artifacts"
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# --- v002 를 재현하는 설정. `--config` 로 고르는 축 말고는 바꾸지 않는다 -----------
#: 기본은 v002 의 피처 구성. `--config f16n` 이면 tag 도 같이 바뀐다(TAG_BY_CONFIG).
CONFIG = "f16"
TAG = "repo16"

#: config -> train_gbdt 의 --tag. 파일명 stem 이 이걸로 갈리므로 섞이면 안 된다.
TAG_BY_CONFIG = {"f16": "repo16", "f16r": "repo16r", "f16n": "repo16n"}
TOPK = 500
N_SPLITS = 5
MODELS = ["xgb", "catboost", "rf"]
#: v002 의 고정 가중. 모델 순서는 MODELS 와 같다.
FIXED_WEIGHTS = [0.45, 0.45, 0.10]
FOLDS_FILE = PROJECT_ROOT / "data" / "process" / "train_folds.parquet"
SEEDS = [42, 7, 2024, 1234, 5678]
CV_KEYS = {"skf5": "skf", "group5": "sgkf"}  # 파일명 조각 -> --cv 값

#: 모델별 대략 학습 시간(초, 분할 하나 기준). `--dry-run` 예상치용.
SECONDS = {"xgb": 700, "catboost": 700, "rf": 250}

#: `train_gbdt.py` 가 붙이는 슬러그. 파라미터를 안 바꾸므로 고정이다.
SLUG = "k500_sp1000m3p2_cm20drishamm83824c_lt64svdnmfl257c23c_gm24shac918bc_sg30sha71bc36"


def log(message: str) -> None:
    print(message, flush=True)


def paths_for(model: str, cv_name: str, seed: int) -> tuple[Path, Path]:
    stem = f"{model}_{TAG}_{CONFIG}_{cv_name}_{SLUG}_s{seed}"
    return (ARTIFACTS / "oof" / f"oof_{stem}.csv",
            ARTIFACTS / "test_predictions" / f"test_{stem}.csv")


def is_complete(model: str, cv_name: str, seed: int) -> bool:
    """반쪽짜리 파일을 '있음' 으로 세지 않는다."""
    oof, test = paths_for(model, cv_name, seed)
    if not (oof.exists() and test.exists()):
        return False
    try:
        o = pd.read_csv(oof)
        t = pd.read_csv(test)
    except Exception:  # noqa: BLE001
        return False
    pc = [c for c in o.columns if c.startswith("p_")]
    return (len(o) == 6201 and len(t) == 2546 and len(pc) == 26
            and bool(o[pc].notna().all().all()))


def inventory() -> tuple[dict, list]:
    have, missing = {}, []
    for cv_name, model, seed in itertools.product(CV_KEYS, MODELS, SEEDS):
        if is_complete(model, cv_name, seed):
            have[(model, cv_name, seed)] = True
        else:
            missing.append((model, cv_name, seed))
    return have, missing


def print_inventory(have: dict, missing: list) -> None:
    log(f"\n{'':7s}{'':9s}" + "  ".join(f"{s:>6}" for s in SEEDS))
    for cv_name in CV_KEYS:
        for model in MODELS:
            cells = ["  있음" if (model, cv_name, s) in have else "  ----" for s in SEEDS]
            log(f"{cv_name:7s}{model:9s}" + "  ".join(f"{c:>6}" for c in cells))
    log(f"\n보유 {len(have)}/{len(SEEDS) * len(MODELS) * len(CV_KEYS)} · 남은 학습 {len(missing)}개")


def train_missing(missing: list, args) -> None:
    """같은 seed·모델의 두 분할은 `--cv all` 한 번으로 묶어 피처 빌드를 아낀다."""
    grouped: dict[tuple[str, int], list[str]] = {}
    for model, cv_name, seed in missing:
        grouped.setdefault((model, seed), []).append(cv_name)

    total = sum(SECONDS[m] * len(cvs) for (m, _), cvs in grouped.items())
    log(f"예상 시간 약 {total / 3600:.1f}h (피처 빌드 제외, 실제로는 더 걸린다)")
    if args.dry_run:
        for (model, seed), cvs in sorted(grouped.items()):
            cv_arg = "all" if len(cvs) == 2 else CV_KEYS[cvs[0]]
            log(f"  --model {model} --cv {cv_arg} --seed {seed}")
        return

    for (model, seed), cvs in sorted(grouped.items()):
        cv_arg = "all" if len(cvs) == 2 else CV_KEYS[cvs[0]]
        command = [
            str(PYTHON), str(PROJECT_ROOT / "scripts" / "train_gbdt.py"),
            "--model", model, "--configs", CONFIG, "--cv", cv_arg,
            "--topk", str(TOPK), "--tag", TAG, "--seed", str(seed), "--no-submission",
        ]
        log(f"\n=== {model} seed {seed} · cv {cv_arg} ===")
        started = time.perf_counter()
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        status = "OK" if result.returncode == 0 else f"실패({result.returncode})"
        log(f"  {status}  {time.perf_counter() - started:.0f}초")


def blend_report(args) -> None:
    """3/4/5 seed × 균등/고정 × skf5/group5 = 12 조합을 OOF 로 비교한다."""
    from sklearn.metrics import f1_score

    labels = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "train.csv",
                         usecols=["ID", "SUBCLASS"]).set_index("ID")["SUBCLASS"]
    classes = sorted(labels.unique())
    cols = [f"p_{c}" for c in classes]

    rows = []
    for cv_name in CV_KEYS:
        for n_seed in (3, 4, 5):
            seeds = SEEDS[:n_seed]
            per_model = {}
            for model in MODELS:
                mats = []
                for seed in seeds:
                    if not is_complete(model, cv_name, seed):
                        break
                    frame = pd.read_csv(paths_for(model, cv_name, seed)[0])
                    y = labels.reindex(frame["ID"]).astype(str).to_numpy()
                    mats.append(np.asarray(frame[cols], dtype=float))
                if len(mats) == len(seeds):
                    per_model[model] = np.mean(mats, axis=0)
            if len(per_model) != len(MODELS):
                log(f"  [건너뜀] {cv_name} · seed {n_seed}개 — 파일이 모자란다")
                continue
            stack = [per_model[m] for m in MODELS]
            for scheme, weights in (("균등", [1 / 3] * 3), ("고정.45/.45/.10", FIXED_WEIGHTS)):
                blended = np.average(stack, axis=0, weights=weights)
                score = f1_score(y, [classes[i] for i in blended.argmax(1)],
                                 labels=classes, average="macro", zero_division=0)
                rows.append({"cv": cv_name, "n_seed": n_seed, "weights": scheme,
                             "macro_f1": float(score)})
                log(f"  {cv_name:7s} seed {n_seed}개  {scheme:16s} macro F1 = {score:.4f}")

    if rows:
        out = ARTIFACTS / "logs" / f"seed_sweep_blend_{CONFIG}.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        best = max(rows, key=lambda r: r["macro_f1"])
        log(f"\n최고: {best['cv']} · seed {best['n_seed']}개 · {best['weights']} "
            f"= {best['macro_f1']:.4f}")
        log("주의: 이 표의 차이는 대부분 페어드 σ(≈0.006) 안이다. CV 로 순위를 못 가린다.")
        log(f"기록: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default=CONFIG, choices=sorted(TAG_BY_CONFIG),
        help="피처 구성. f16 = v002 기준선, f16n = gec 행정규화, f16r = rollup16",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=f"채울 seed 를 고른다 (기본 {SEEDS}). 앞에서부터 잘라 쓰는 게 안전하다 — "
        "blend 가 SEEDS 순서로 3/4/5개를 자른다",
    )
    parser.add_argument(
        "--only-cv", choices=sorted(CV_KEYS), default=None,
        help="한 분할만 채운다. 쓸 수 있는 베이스라인을 먼저 만들 때 group5 만 돌리면 절반이다",
    )
    parser.add_argument("--dry-run", action="store_true", help="계획만 찍고 끝낸다")
    parser.add_argument("--blend-only", action="store_true",
                        help="학습을 건너뛰고 이미 있는 파일로 12 조합만 비교한다")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # 모듈 전역을 갈아끼운다. `paths_for`·`is_complete` 가 이걸 보고 stem 을 만든다 —
    # 여기서 tag 를 같이 안 바꾸면 f16n 파일을 f16 이름으로 찾아 전부 "없음" 이 된다.
    globals()["CONFIG"] = args.config
    globals()["TAG"] = TAG_BY_CONFIG[args.config]
    if args.seeds:
        unknown = [s for s in args.seeds if s not in SEEDS]
        if unknown:
            raise SystemExit(f"SEEDS 에 없는 seed {unknown} — blend 가 못 집는다. {SEEDS} 중에서 고른다")
        globals()["SEEDS"] = [s for s in SEEDS if s in args.seeds]  # 원래 순서 유지
    if args.only_cv:
        globals()["CV_KEYS"] = {args.only_cv: CV_KEYS[args.only_cv]}
    log(f"설정 — config {CONFIG} · tag {TAG} · topk {TOPK} · "
        f"n_splits {N_SPLITS} · weight balanced · folds {FOLDS_FILE.name}")
    log(f"seed {SEEDS} × 모델 {MODELS} × 분할 {list(CV_KEYS)}")

    have, missing = inventory()
    print_inventory(have, missing)

    if not args.blend_only and missing:
        train_missing(missing, args)
        if args.dry_run:
            return
        have, missing = inventory()
        print_inventory(have, missing)

    if missing and not args.blend_only:
        log(f"\n아직 {len(missing)}개가 없다. 다시 돌리면 이어진다.")
    log("\n=== seed 개수 × 가중 × 분할 비교 ===")
    blend_report(args)


if __name__ == "__main__":
    main()
