#!/usr/bin/env python
r"""결합 방식을 바꿔 가며 비교한다 — 산술평균 / 랭크평균 / 온도보정 / 클래스별 가중.

    .\.venv\Scripts\python.exe scripts\blend_strategies.py --cv group5

## 왜 필요한가

멤버마다 확률의 **스케일**이 다르다. f16 group5 3-seed 실측:

    모델        평균 최대확률   평균 엔트로피   0 인 셀
    xgb           0.6155        1.2111       0.3%
    catboost      0.4384        1.9111       0.0%
    rf            0.3262        2.2210       9.6%

RandomForest 는 500그루 투표라 확률이 양자화되고(고유값 1만개) 납작하다. 산술평균은
확신이 큰 멤버 쪽으로 기울기 때문에, 이 상태에서 `0.45/0.45/0.10` 을 주면 rf 는 사실상
두 번 눌린다 — 가중치로 한 번, 스케일로 또 한 번.

실제로 Caruana 그리디는 rf 를 **33.5%** 담았다(수동 가중은 0.10). 그게 rf 에 정보가
많아서인지, 스케일 때문에 눌려 있다가 횟수로 보정된 것인지 이 스크립트가 가른다.

## 비교하는 방식

    arith    산술평균 (지금 쓰는 것)
    rank     멤버·클래스마다 표본 간 순위로 바꿔 평균 — 스케일이 통째로 사라진다
    temp     멤버마다 온도 T 를 OOF 에서 찾아 p^(1/T) 로 편 뒤 산술평균
    perclass 클래스마다 멤버 가중을 따로 — 26x3=78 파라미터, 과적합 위험이 크다

## 정직한 점수

`temp`·`perclass` 는 **파라미터를 적합한다.** 적합에 쓴 행에서 재면 낙관적이다. 그래서
전부 교차적합으로 잰다 — fold f 를 뺀 나머지에서 파라미터를 찾고 fold f 에서만 점수를
낸다. `arith`·`rank` 는 적합 파라미터가 없어 교차적합이 무의미하지만 같은 표에 놓기
위해 같은 절차로 잰다.

## DACON 제출

제출 파일을 만들지 않는다. 결합 방식을 고르는 진단 도구다.
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

from cancer_hack.metrics import macro_f1  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
OOF = PROJECT_ROOT / "artifacts" / "oof"
FOLDS = PROJECT_ROOT / "data" / "process" / "train_folds.parquet"
SLUG = "k500_sp1000m3p2_cm20drishamm83824c_lt64svdnmfl257c23c_gm24shac918bc_sg30sha71bc36"
MODELS = ["xgb", "catboost", "rf"]
FIXED_WEIGHTS = np.array([0.45, 0.45, 0.10])


def log(message: str) -> None:
    print(message, flush=True)


def to_rank(prob: np.ndarray) -> np.ndarray:
    """클래스 열마다 표본 간 순위로 바꾼 뒤 행 합을 1 로 맞춘다.

    스케일이 사라지고 **순서만** 남는다. 미보정 출력을 섞을 때의 표준 처방이다.
    """
    ranked = np.empty_like(prob, dtype=np.float64)
    for column in range(prob.shape[1]):
        order = prob[:, column].argsort()
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = np.arange(len(order), dtype=np.float64)
        ranked[:, column] = ranks / max(len(order) - 1, 1)
    totals = ranked.sum(axis=1, keepdims=True)
    return np.divide(ranked, totals, out=np.full_like(ranked, 1 / prob.shape[1]), where=totals > 0)


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    """p^(1/T). T>1 이면 납작해지고 T<1 이면 뾰족해진다."""
    sharpened = np.power(np.clip(prob, 1e-12, None), 1.0 / temperature)
    return sharpened / sharpened.sum(axis=1, keepdims=True)


def fit_temperatures(members: list[np.ndarray], y, classes, grid) -> list[float]:
    """멤버마다 **단독** macro F1 이 가장 높아지는 T. 멤버 간 상호작용은 보지 않는다.

    상호작용까지 맞추면 파라미터가 늘어 과적합한다. 여기 목적은 스케일을 맞추는 것이지
    앙상블을 최적화하는 게 아니다 — 그건 가중치의 몫이다.
    """
    out = []
    for prob in members:
        scores = [macro_f1(y, np.asarray(classes)[apply_temperature(prob, t).argmax(1)]) for t in grid]
        out.append(float(grid[int(np.argmax(scores))]))
    return out


def fit_per_class_weights(members: list[np.ndarray], y, classes, passes: int = 3) -> np.ndarray:
    """클래스별 멤버 가중 (클래스 x 멤버). 좌표하강으로 macro F1 을 직접 올린다."""
    n_members, n_classes = len(members), len(classes)
    weights = np.full((n_classes, n_members), 1.0 / n_members)
    stack = np.stack(members)

    def blended(w):
        return np.einsum("mnc,cm->nc", stack, w)

    best = macro_f1(y, np.asarray(classes)[blended(weights).argmax(1)])
    for _ in range(passes):
        improved = False
        for c in range(n_classes):
            for m in range(n_members):
                for delta in (0.25, -0.25, 0.10, -0.10):
                    trial = weights.copy()
                    trial[c, m] = max(0.0, trial[c, m] + delta)
                    if trial[c].sum() <= 0:
                        continue
                    trial[c] /= trial[c].sum()
                    score = macro_f1(y, np.asarray(classes)[blended(trial).argmax(1)])
                    if score > best + 1e-12:
                        weights, best, improved = trial, score, True
        if not improved:
            break
    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv", default="group5", choices=["group5", "skf5"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    parser.add_argument("--config", default="f16")
    parser.add_argument("--tag", default="repo16")
    parser.add_argument("--tempera-grid", type=float, nargs="+",
                        default=[0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels = pd.read_csv(RAW / "train.csv", usecols=["ID", "SUBCLASS"]).set_index("ID")["SUBCLASS"]
    classes = sorted(labels.unique())
    cols = [f"p_{c}" for c in classes]
    folds = pd.read_parquet(FOLDS)
    ids = folds["ID"].to_numpy()
    y = labels.reindex(ids).astype(str).to_numpy()
    fold_ids = folds[f"fold_{args.cv}"].to_numpy()

    members = []
    for model in MODELS:
        mats = []
        for seed in args.seeds:
            path = OOF / f"oof_{model}_{args.tag}_{args.config}_{args.cv}_{SLUG}_s{seed}.csv"
            if not path.exists():
                log(f"{path.name} 이 없다.")
                return 1
            mats.append(np.asarray(pd.read_csv(path)[cols], dtype=float))
        members.append(np.mean(mats, axis=0))

    log(f"멤버 {MODELS} · {args.config}/{args.cv} · seed {args.seeds}")
    log(f"\n{'모델':10s} {'평균 최대확률':>12s} {'평균 엔트로피':>12s} {'0 인 셀':>8s}")
    for name, prob in zip(MODELS, members):
        ent = -(prob * np.log(np.clip(prob, 1e-12, None))).sum(1)
        log(f"{name:10s} {prob.max(1).mean():12.4f} {ent.mean():12.4f} {(prob < 1e-6).mean():8.1%}")

    strategies = ("arith_uniform", "arith_fixed", "rank_uniform", "rank_fixed",
                  "temp_uniform", "temp_fixed", "perclass")
    crossfit = {s: np.zeros((len(ids), len(classes))) for s in strategies}

    for fold in sorted(set(fold_ids.tolist())):
        valid, train = fold_ids == fold, fold_ids != fold
        uniform = np.full(len(MODELS), 1 / len(MODELS))

        crossfit["arith_uniform"][valid] = np.einsum("mnc,m->nc", np.stack([m[valid] for m in members]), uniform)
        crossfit["arith_fixed"][valid] = np.einsum("mnc,m->nc", np.stack([m[valid] for m in members]), FIXED_WEIGHTS)

        ranked = [to_rank(m)[valid] for m in members]        # 순위는 전체 표본 기준
        crossfit["rank_uniform"][valid] = np.einsum("mnc,m->nc", np.stack(ranked), uniform)
        crossfit["rank_fixed"][valid] = np.einsum("mnc,m->nc", np.stack(ranked), FIXED_WEIGHTS)

        temps = fit_temperatures([m[train] for m in members], y[train], classes, args.tempera_grid)
        tempered = [apply_temperature(m[valid], t) for m, t in zip(members, temps)]
        crossfit["temp_uniform"][valid] = np.einsum("mnc,m->nc", np.stack(tempered), uniform)
        crossfit["temp_fixed"][valid] = np.einsum("mnc,m->nc", np.stack(tempered), FIXED_WEIGHTS)

        per_class = fit_per_class_weights([m[train] for m in members], y[train], classes)
        crossfit["perclass"][valid] = np.einsum("mnc,cm->nc", np.stack([m[valid] for m in members]), per_class)

        log(f"  fold {fold}: 온도 {[round(t, 2) for t in temps]}")

    log(f"\n{'결합 방식':18s} {'교차적합 macro F1':>18s}")
    log("-" * 40)
    results = {}
    for name in strategies:
        score = float(macro_f1(y, np.asarray(classes)[crossfit[name].argmax(1)]))
        results[name] = score
    for name, score in sorted(results.items(), key=lambda r: -r[1]):
        log(f"{name:18s} {score:18.4f}")

    out = PROJECT_ROOT / "artifacts" / "logs" / f"blend_strategies_{args.config}_{args.cv}.json"
    out.write_text(json.dumps({"config": args.config, "cv": args.cv, "seeds": args.seeds,
                               "crossfit_macro_f1": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log(f"\n기록: {out}")
    log("※ 전부 교차적합이다. temp·perclass 는 파라미터를 적합하므로 이 절차가 필수다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
