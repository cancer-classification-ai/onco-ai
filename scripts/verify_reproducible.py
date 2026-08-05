#!/usr/bin/env python
"""이미 만든 모델을 같은 설정으로 다시 돌려 **산출물이 글자 하나까지 같은지** 확인한다.

    python scripts/verify_reproducible.py --run-tag seed3_f16 --models xgb catboost --seeds 42

무엇을 비교하나
---------------
점수만 같은 걸로는 부족하다. 확률 한 자리가 달라도 나중에 블렌딩하면 결과가 갈린다.
그래서 파일 자체를 본다.

    oof_<stem>.csv              sha256 · 확률 최대차 · 라벨 차이
    test_<stem>.csv             sha256 · 확률 최대차 · 라벨 차이
    logs/<stem>.json            model_params · fold별 점수 · OOF

재현이 깨지는 흔한 원인은 seed 가 아니다 — `--gpu-ram-part auto`(실행 시점 GPU 여유로
값이 정해진다)와 `--threads` 변경이다. 이 스크립트는 원본 로그에서 그 값들을 읽어 **그대로**
넘긴다. `docs/reproducibility.md` 참고.

## DACON 제출

파일을 만들 뿐 어디에도 올리지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.paths import PROJECT_ROOT as _ROOT  # noqa: E402,F401


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-tag", required=True, help="검증할 실행 폴더 이름")
    parser.add_argument("--models", nargs="+", default=["xgb", "catboost"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--verify-tag", default=None,
                        help="재실행 결과를 담을 폴더 (기본 <run-tag>_verify)")
    return parser.parse_args()


def find_log(run_dir: Path, model: str, run_tag: str, seed: int) -> Path:
    hits = sorted((run_dir / "logs").glob(f"{model}_{run_tag}_*_s{seed}.json"))
    hits = [h for h in hits if "matrix" not in h.name]
    if len(hits) != 1:
        raise SystemExit(f"{model} seed {seed} 로그가 {len(hits)}개다: {[h.name for h in hits]}")
    return hits[0]


def main() -> int:
    args = parse_args()
    runs = PROJECT_ROOT / "artifacts" / "runs"
    source = runs / args.run_tag
    verify_tag = args.verify_tag or f"{args.run_tag}_verify"
    target = runs / verify_tag
    process = PROJECT_ROOT / f"data/process_{args.run_tag}"

    if not source.exists():
        raise SystemExit(f"{source} 가 없다")
    if not process.exists():
        raise SystemExit(f"{process} 가 없다 — 원본이 쓴 파켓이 있어야 같은 입력이 된다")

    env = dict(os.environ)
    env["ONCO_PROCESS_DIR"] = str(process)     # **같은 파켓**을 읽는다
    env["ONCO_ARTIFACTS_DIR"] = str(target)

    print(f"원본   {source}")
    print(f"재실행 {target}")
    print(f"파켓   {process}  (양쪽 동일)\n")

    rows = []
    for model in args.models:
        for seed in args.seeds:
            log_path = find_log(source, model, args.run_tag, seed)
            original = json.loads(log_path.read_text(encoding="utf-8"))

            # 원본이 쓴 설정을 그대로 재사용한다. 특히 device 는 재현의 핵심 축이다.
            argv = [sys.executable, str(PROJECT_ROOT / "scripts/train_gbdt.py"),
                    "--model", model, "--configs", original["config"],
                    "--cv", original["cv"], "--seed", str(seed),
                    "--topk", str(original["topk"]), "--tag", args.run_tag,
                    "--no-submission",
                    "--gpu-ram-part", str(original["model_params"].get("gpu_ram_part", 0.4))]
            if original.get("device") == "cpu":
                argv += ["--device", "cpu"]

            print(f"[{model} seed {seed}] 재실행…", flush=True)
            done = subprocess.run(argv, capture_output=True, text=True,
                                  encoding="utf-8", cwd=PROJECT_ROOT, env=env)
            if done.returncode != 0:
                print(done.stdout[-2500:]); print(done.stderr[-2500:])
                raise SystemExit(f"{model} seed {seed} 재실행 실패")

            stem = original["stem"]
            rows.append(compare(source, target, stem, original, model, seed))

    print(f"\n{'모델':10s} {'seed':>5s} {'OOF 일치':>9s} {'oof sha':>8s} {'test sha':>9s} "
          f"{'확률 최대차':>12s} {'라벨차':>6s}")
    ok = True
    for row in rows:
        ok &= row["identical"]
        print(f"{row['model']:10s} {row['seed']:>5d} {str(row['score_same']):>9s} "
              f"{str(row['oof_sha_same']):>8s} {str(row['test_sha_same']):>9s} "
              f"{row['max_diff']:>12.2e} {row['label_diff']:>6d}")

    print("\n산출물이 글자 하나까지 같다" if ok else "\n산출물이 다르다 — 위 표를 확인한다")
    return 0 if ok else 1


def compare(source: Path, target: Path, stem: str, original: dict,
            model: str, seed: int) -> dict:
    fresh = json.loads((target / "logs" / f"{stem}.json").read_text(encoding="utf-8"))

    pairs = {
        "oof": (source / "oof" / f"oof_{stem}.csv", target / "oof" / f"oof_{stem}.csv"),
        "test": (source / "test_predictions" / f"test_{stem}.csv",
                 target / "test_predictions" / f"test_{stem}.csv"),
    }
    shas = {k: (sha256(a), sha256(b)) for k, (a, b) in pairs.items()}

    a = pd.read_csv(pairs["oof"][0], encoding="utf-8-sig").sort_values("ID", kind="stable")
    b = pd.read_csv(pairs["oof"][1], encoding="utf-8-sig").sort_values("ID", kind="stable")
    cols = [c for c in a.columns if c.startswith("p_")]
    max_diff = float(np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max())
    label_diff = int((a["y_pred"].to_numpy() != b["y_pred"].to_numpy()).sum())

    score_same = original["oof_macro_f1"] == fresh["oof_macro_f1"]
    folds_same = original["fold_macro_f1"] == fresh["fold_macro_f1"]
    params_same = original["model_params"] == fresh["model_params"]

    if not params_same:
        differing = {k for k in set(original["model_params"]) | set(fresh["model_params"])
                     if original["model_params"].get(k) != fresh["model_params"].get(k)}
        print(f"  [{model} s{seed}] 파라미터가 다르다: {differing}")

    return {
        "model": model, "seed": seed,
        "score_same": score_same and folds_same,
        "oof_sha_same": shas["oof"][0] == shas["oof"][1],
        "test_sha_same": shas["test"][0] == shas["test"][1],
        "max_diff": max_diff, "label_diff": label_diff,
        "identical": (score_same and folds_same and params_same
                      and shas["oof"][0] == shas["oof"][1]
                      and shas["test"][0] == shas["test"][1]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
