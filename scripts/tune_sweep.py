#!/usr/bin/env python
"""f16 세 모델 Optuna 탐색 + 우승 후보 다중 seed 재검증을 한 번에 돌린다.

    # 계획과 예상 시간만 보고 끝낸다 (학습 안 함)
    .\\.venv\\Scripts\\python.exe scripts\\tune_sweep.py --dry-run

    # 실제 실행 — 세 모델을 차례로 탐색하고 상위 후보를 seed 3개로 재검증한다
    .\\.venv\\Scripts\\python.exe scripts\\tune_sweep.py

    # 한 모델만
    .\\.venv\\Scripts\\python.exe scripts\\tune_sweep.py --models rf

## 왜 이 스크립트가 필요한가 — 단일 seed 최고점을 그대로 믿으면 안 된다

`research/10` §2.2 에서 잰 group5 의 **seed 표준편차가 0.0032** 다. 그러면 N 번 뽑기의
최댓값은 실제 개선이 0 이어도 대략 `σ·√(2·ln N)` 만큼 나온다 — 30 trial 이면 0.0084 다.
저장된 f4r study 37 trial 의 group5 최대 이득이 +0.0021 이었으니, **그 우승은 노이즈가
낼 값보다도 작았다**(§6).

그래서 이 스크립트는 두 단계로 간다.

    1) 탐색      — trial 마다 skf·sgkf 를 둘 다 재고 일반화 격차도 남긴다
    2) 재검증    — 상위 K개를 seed 여러 개로 다시 돌려 **평균과 표준편차**를 본다

2단계를 통과하지 못하면 그 우승은 seed 운이다. 채택 기준은 하나다.

    seed 평균이 기준선 seed 평균보다 높고, 그 차이가 seed 표준편차보다 커야 한다.

## 목적함수를 `sgkf` 로 두는 이유

`--cv both` 의 기본 목적은 두 분할 평균인데, 그러면 **skf 를 올리고 sgkf 를 깎는 설정이
상쇄돼 보인다.** 저장된 study 37 trial 중 17개가 정확히 그 모양이었고, `research/08` 이
1순위로 고른 trial 11 은 sgkf 에서 −0.0122 였다. LB 기록(0.3896)을 세운 v002 는
**group5(sgkf)** 로 학습·선택된 모델이다. 축을 맞춘다.

skf 는 계속 **기록**한다. 둘의 벌어짐(`sgkf_minus_skf`)이 쌍둥이 암기 의존도를 보여준다.

## 과적합 억제

`--gap-penalty` 로 목적값에서 `계수 × max(0, 격차 − floor)` 를 뺀다. 같은 CV 면 덜 외우는
쪽을 고른다. floor 기본 0.20 은 CatBoost 실측 격차(0.18)를 통과시키고 XGBoost 실측
격차(0.45)를 벌하는 지점이다.

## DACON 제출

이 스크립트는 제출 파일을 만들지 않는다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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

TUNING_DIR = PROJECT_ROOT / "artifacts" / "tuning"
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

#: 모델별 trial 당 대략 시간(초). f16 실측 fold 시간 × 분할 2개 + 여유.
#: `--dry-run` 이 이 값으로 예상 시간을 낸다. 정확할 필요는 없고 자릿수만 맞으면 된다.
SECONDS_PER_TRIAL = {"xgb": 540, "catboost": 620, "rf": 190}

DEFAULT_MODELS = ["rf", "catboost", "xgb"]  # 싼 것부터 — 중간에 멈춰도 뭔가는 남는다


def log(message: str) -> None:
    print(message, flush=True)


def run_study(model: str, args) -> Path:
    """`tune_optuna.py` 를 서브프로세스로 돌린다.

    같은 프로세스에서 import 해 돌리지 않는 이유는 `Dataset` 이 모델마다 새로 만들어져
    메모리가 겹치기 때문이다. f16 은 `gtype` 26,304열 때문에 한 벌에 2.8GB 쯤 쓴다.
    """
    study = f"{args.config}_{model}_sweep"
    command = [
        str(PYTHON), str(PROJECT_ROOT / "scripts" / "tune_optuna.py"),
        "--model", model,
        "--config", args.config,
        "--cv", "both",
        "--objective", args.objective,
        "--n-trials", str(args.n_trials),
        "--study", study,
        "--topk", str(args.topk),
        "--seed", str(args.seed),
        "--gap-penalty", str(args.gap_penalty),
        "--gap-floor", str(args.gap_floor),
    ]
    log(f"\n{'=' * 100}\n[{model}] 탐색 시작 — {args.n_trials} trial\n  {' '.join(command[1:])}\n{'=' * 100}")
    if args.dry_run:
        return TUNING_DIR / f"{study}_summary.json"

    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        log(f"[{model}] 탐색이 코드 {result.returncode} 로 끝났다 — 이 모델은 건너뛴다")
    log(f"[{model}] 탐색 {time.perf_counter() - started:.0f}초")
    return TUNING_DIR / f"{study}_summary.json"


def top_trials(summary_path: Path, k: int) -> list[dict]:
    """summary json 에서 목적값 상위 K개를 뽑는다. 기준선 trial 은 항상 포함한다."""
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    trials = [t for t in data.get("trials", []) if t.get("value") is not None]
    if not trials:
        return []
    ranked = sorted(trials, key=lambda t: t["value"], reverse=True)[:k]
    baseline = next((t for t in trials if (t.get("user_attrs") or {}).get("note") == "baseline"), None)
    if baseline and all(t["number"] != baseline["number"] for t in ranked):
        ranked.append(baseline)
    return ranked


def revalidate(model: str, trials: list[dict], args) -> list[dict]:
    """상위 후보를 seed 여러 개로 다시 돌린다 — 여기가 이 스크립트의 존재 이유다."""
    import tune_optuna as tu
    from train_gbdt import CONFIGS, Dataset

    log(f"\n[{model}] 재검증 — 후보 {len(trials)}개 × seed {args.revalidate_seeds}")
    data = Dataset(set(CONFIGS[args.config]["blocks"]), n_splits=args.n_splits)

    out = []
    for trial in trials:
        scores: dict[str, list[float]] = {"skf": [], "sgkf": []}
        gaps: list[float] = []
        for seed in args.revalidate_seeds:
            sub = argparse.Namespace(
                model=model, config=args.config, cv_list=["skf", "sgkf"],
                topk=args.topk, n_splits=args.n_splits, seed=seed,
                device="auto", no_track_train=False, objective=args.objective,
                gap_penalty=0.0, gap_floor=args.gap_floor,
            )
            result = tu.evaluate(data, trial["params"], sub)
            for cv in ("skf", "sgkf"):
                scores[cv].append(result[f"{cv}_oof_macro_f1"])
            gap = result.get("sgkf_generalization_gap")
            if gap is not None:
                gaps.append(gap)
        row = {
            "trial": trial["number"],
            "params": trial["params"],
            "is_baseline": (trial.get("user_attrs") or {}).get("note") == "baseline",
            "search_objective": trial["value"],
        }
        for cv in ("skf", "sgkf"):
            row[f"{cv}_mean"] = float(np.mean(scores[cv]))
            row[f"{cv}_sd"] = float(np.std(scores[cv], ddof=1)) if len(scores[cv]) > 1 else 0.0
            row[f"{cv}_seeds"] = scores[cv]
        row["gap_mean"] = float(np.mean(gaps)) if gaps else None
        out.append(row)
        log(f"  trial {row['trial']:>3d}  sgkf {row['sgkf_mean']:.4f} ±{row['sgkf_sd']:.4f}   "
            f"skf {row['skf_mean']:.4f} ±{row['skf_sd']:.4f}   격차 "
            f"{row['gap_mean'] if row['gap_mean'] is not None else float('nan'):+.4f}"
            f"{'   <- 기준선' if row['is_baseline'] else ''}")
    return out


def verdict(rows: list[dict]) -> dict:
    """채택 기준: seed 평균이 기준선보다 seed 표준편차 이상 높아야 한다."""
    baseline = next((r for r in rows if r["is_baseline"]), None)
    if baseline is None:
        return {"adopt": None, "reason": "기준선 trial 이 없어 판정 불가"}
    others = [r for r in rows if not r["is_baseline"]]
    if not others:
        return {"adopt": None, "reason": "비교할 후보가 없다"}
    best = max(others, key=lambda r: r["sgkf_mean"])
    delta = best["sgkf_mean"] - baseline["sgkf_mean"]
    threshold = max(best["sgkf_sd"], baseline["sgkf_sd"])
    adopt = delta > threshold
    return {
        "adopt": adopt,
        "trial": best["trial"],
        "params": best["params"],
        "sgkf_delta": delta,
        "threshold_sd": threshold,
        "reason": (f"seed 평균 델타 {delta:+.4f} "
                   f"{'>' if adopt else '<='} seed 표준편차 {threshold:.4f} — "
                   f"{'채택' if adopt else '기각(seed 운과 구별 불가)'}"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["xgb", "catboost", "rf"])
    parser.add_argument("--config", default="f16", help="기준선 v002 의 피처 구성")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="탐색 중 쓰는 모델 시드")
    parser.add_argument("--objective", default="sgkf",
                        help="sgkf 가 기본이다 — v002 가 group5 로 선택됐다. §목적함수 참고")
    parser.add_argument("--gap-penalty", type=float, default=0.5)
    parser.add_argument("--gap-floor", type=float, default=0.20)
    parser.add_argument("--top-k-revalidate", type=int, default=3)
    parser.add_argument("--revalidate-seeds", type=int, nargs="+", default=[42, 7, 2024])
    parser.add_argument("--skip-revalidate", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="명령과 예상 시간만 찍고 끝낸다")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    search = sum(SECONDS_PER_TRIAL[m] * args.n_trials for m in args.models)
    reval = 0 if args.skip_revalidate else sum(
        SECONDS_PER_TRIAL[m] * (args.top_k_revalidate + 1) * len(args.revalidate_seeds)
        for m in args.models
    )
    log(f"모델 {args.models} · config {args.config} · trial {args.n_trials} · 목적 {args.objective}")
    log(f"과적합 페널티 {args.gap_penalty} (floor {args.gap_floor})")
    log(f"예상 시간 — 탐색 {search/3600:.1f}h + 재검증 {reval/3600:.1f}h = **{(search+reval)/3600:.1f}h**")
    log("※ pruner 가 가망 없는 trial 을 끊으므로 실제로는 이보다 짧다 (기존 study 는 60 중 23 pruned)")

    if args.dry_run:
        for model in args.models:
            run_study(model, args)
        log("\n--dry-run 이라 여기까지. 실제로 돌리려면 --dry-run 을 뺀다.")
        return

    report = {"config": args.config, "objective": args.objective,
              "gap_penalty": args.gap_penalty, "models": {}}
    for model in args.models:
        summary = run_study(model, args)
        rows = [] if args.skip_revalidate else revalidate(
            model, top_trials(summary, args.top_k_revalidate), args)
        entry = {"summary_path": str(summary), "revalidated": rows}
        if rows:
            entry["verdict"] = verdict(rows)
            log(f"[{model}] {entry['verdict']['reason']}")
        report["models"][model] = entry

    path = TUNING_DIR / f"sweep_{args.config}_{args.objective}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n요약: {path}")
    log("채택된 파라미터는 train_gbdt.py --set 으로 다시 돌려 OOF·test 를 뽑는다.")


if __name__ == "__main__":
    main()
