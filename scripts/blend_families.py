"""4계열(catboost·xgb·rf·mlp) 판에서 **확률을 합치는 방식**을 바꿔 가며 잰다.

`research/14` 는 그리디의 손잡이(라운드·bagging·후보 수·후보 선정 규칙)와 스태킹을
98개 라이브러리 위에서 훑었다. 거기서 안 건드린 축이 하나 남는다 — **합치는 방식 자체**다.
지금까지의 모든 결합은 예외 없이 확률의 산술평균이었다.

이 스크립트가 그 축을 연다.

| 축 | 값 |
|---|---|
| 결합 형태 | 산술 · 기하 · 조화 · 순위 · 중앙값 · 절사평균 |
| 가중치 | 균등 · EXP_040 고정(.45/.45/.10) · MLP 포함 고정 · 학습 |
| 보정 | 클래스별 로짓 바이어스 on/off |
| 계열 집계 | 계열 안에서 먼저 평균 낸 뒤 계열끼리 섞기 |

왜 4계열인가
------------
`artifacts/oof/` 는 xgb 148 · catboost 32 · rf 22 · DL 6 으로 심하게 기울어 있다. 이 위에서
그리디를 돌리면 "어느 계열이 몇 개 있느냐" 가 결과에 섞여 들어와 결합 방식만 따로 볼 수
없다. 계열마다 대표를 하나씩만 세우면 그 축이 사라진다.

점수를 읽는 법
--------------
전부 **교차적합**이다 — fold f 를 뺀 나머지에서 가중치·보정·순위 기준을 정하고 f 에서만
잰다. 그리고 `research/14` 의 교훈대로 **잡음 폭을 먼저 잰다.** 여기서 자의적인 선택은
난수가 아니라 "계열의 대표를 누구로 하느냐" 라서, 같은 전략을 대표만 1위/2위/3위로 바꿔
돌린 폭을 척도로 쓴다. 그 폭보다 작은 차이는 읽지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cancer_hack.ensemble import (  # noqa: E402
    BLEND_FORMS,
    MacroF1Blender,
    combine,
)
from cancer_hack.metrics import macro_f1  # noqa: E402
from cancer_hack.paths import LazyDir, artifacts_dir, process_dir, raw_dir  # noqa: E402
from cancer_hack.provenance import check_fold_fingerprint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = LazyDir(raw_dir)
ARTIFACTS = LazyDir(artifacts_dir)
FOLDS = LazyDir(lambda: process_dir() / "train_folds.parquet")
MEMBERS_FILE = ROOT / "configs" / "family4_members.json"

#: 계열 판정. 순서가 중요하다 — hybrid·set 이 mlp 보다 먼저 걸려야 한다.
FAMILY_RULES = (
    ("hybrid", lambda n: "hybrid" in n),
    ("setenc", lambda n: "set_encoder" in n),
    ("mlp", lambda n: "mlp" in n),
    ("catboost", lambda n: "catboost" in n),
    ("xgb", lambda n: "xgb" in n),
    ("rf", lambda n: "_rf_" in n or n.startswith("oof_rf")),
)

#: 4계열 판의 기본 구성. `mlp` 자리에 DL 계열 전체(mlp·hybrid·setenc)를 후보로 둔다 —
#: 순수 MLP 단독은 0.3872~0.3957 로 낮지만 hybrid 가 0.4388 까지 나온다.
FAMILIES = ("catboost", "xgb", "rf", "mlp")
DL_FAMILIES = ("mlp", "hybrid", "setenc")

#: 가중치 프리셋. 길이는 FAMILIES 와 같다.
WEIGHT_PRESETS = {
    "uniform": (0.25, 0.25, 0.25, 0.25),
    "exp040": (0.45, 0.45, 0.10, 0.00),      # LB 0.4818 을 낸 3모델 구성
    "exp040mlp": (0.40, 0.40, 0.10, 0.10),   # 거기에 MLP 자리를 낸 것
    "cat_heavy": (0.40, 0.25, 0.20, 0.15),
}


def family_of(name: str) -> str | None:
    lowered = name.lower()
    if any(k in lowered for k in ("greedy", "meta", "blend", "calib", "stack")):
        return None
    for family, rule in FAMILY_RULES:
        if rule(lowered):
            return family
    return None


def solo_scores(paths: list[Path], labels: pd.Series, classes: list[str]) -> dict[str, float]:
    scores = {}
    class_array = np.asarray(classes)
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        columns = [c for c in frame.columns if c.startswith("p_")]
        if len(columns) != len(classes) or len(frame) != len(labels):
            continue
        y = labels.reindex(frame["ID"]).astype(str).to_numpy()
        pred = class_array[frame[columns].to_numpy().argmax(axis=1)]
        scores[path.name] = macro_f1(y, pred)
    return scores


def build_members(labels: pd.Series, classes: list[str], depth: int,
                  exclude: tuple[str, ...] = ()) -> dict:
    """계열마다 단독 점수 상위 `depth` 개를 뽑아 얼린다.

    `artifacts/oof/` 는 실험할 때마다 늘어나므로 고르는 시점에 따라 결과가 조용히
    달라진다. 한 번 뽑아 파일로 박아 두고 그 뒤로는 그 파일만 읽는다.
    """

    by_family: dict[str, list[Path]] = {}
    for path in sorted((ARTIFACTS / "oof").glob("*.csv")):
        if any(token in path.name for token in exclude):
            continue
        family = family_of(path.name)
        if family:
            by_family.setdefault(family, []).append(path)

    picks: dict[str, list[str]] = {}
    for family in FAMILIES:
        pool = []
        for source in (DL_FAMILIES if family == "mlp" else (family,)):
            pool.extend(by_family.get(source, []))
        scores = solo_scores(pool, labels, classes)
        # 점수 내림차순, 동점이면 파일 이름 — 어느 기계에서도 같은 순서가 되게.
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        picks[family] = [name for name, _ in ordered[:depth]]
        if len(picks[family]) < depth:
            raise SystemExit(f"{family} 계열에 후보가 {len(picks[family])}개뿐이다 (필요 {depth})")

    solo = solo_scores(
        [ARTIFACTS / "oof" / n for names in picks.values() for n in names], labels, classes
    )
    return {"families": list(FAMILIES), "depth": depth, "exclude": list(exclude),
            "members": picks,
            "solo_macro_f1": {k: round(v, 6) for k, v in solo.items()}}


def load_stack(names: list[str], ids: pd.Index, classes: list[str]) -> list[np.ndarray]:
    arrays = []
    for name in names:
        frame = pd.read_csv(ARTIFACTS / "oof" / name, encoding="utf-8-sig").set_index("ID")
        frame = frame.reindex(ids)
        columns = [f"p_{c}" for c in classes]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise SystemExit(f"{name} 에 클래스 열이 없다: {missing[:3]}")
        arrays.append(frame[columns].to_numpy(dtype=np.float64))
    return arrays


def crossfit(
    arrays: list[np.ndarray],
    y: np.ndarray,
    classes: list[str],
    folds: np.ndarray,
    *,
    form: str,
    weights: tuple[float, ...] | None,
    calibrate: bool,
) -> dict:
    """fold 를 뺀 나머지에서 정하고 그 fold 에서만 잰다.

    `weights=None` 이면 fold 의 train 부분에서 가중치를 **학습**한다. 그 경우에도 평가는
    valid fold 에서만 하므로 낙관이 안 섞인다.
    """

    from cancer_hack.calibration import MacroF1LogitBias

    class_array = np.asarray(classes)
    blended = np.zeros((len(y), len(classes)), dtype=np.float64)
    per_fold, used_weights = [], []

    for fold in sorted(np.unique(folds).tolist()):
        train_mask = folds != fold
        valid_mask = ~train_mask
        train_parts = [a[train_mask] for a in arrays]
        valid_parts = [a[valid_mask] for a in arrays]

        if weights is None:
            blender = MacroF1Blender().fit(train_parts, y[train_mask], classes)
            fold_weights = tuple(float(w) for w in blender.weights_)
        else:
            fold_weights = weights
        used_weights.append(fold_weights)

        train_blend = combine(train_parts, fold_weights, form, reference=train_parts)
        valid_blend = combine(valid_parts, fold_weights, form, reference=train_parts)

        if calibrate:
            calibrator = MacroF1LogitBias().fit(train_blend, y[train_mask], classes)
            valid_blend = calibrator.predict_proba(valid_blend)

        blended[valid_mask] = valid_blend
        per_fold.append(macro_f1(y[valid_mask], class_array[valid_blend.argmax(axis=1)]))

    return {
        "crossfit_macro_f1": macro_f1(y, class_array[blended.argmax(axis=1)]),
        "fold_macro_f1": [round(v, 6) for v in per_fold],
        "fold_std": float(np.std(per_fold)),
        "worst_fold": float(np.min(per_fold)),
        "weights_by_fold": [[round(w, 4) for w in fw] for fw in used_weights],
        "labels": class_array[blended.argmax(axis=1)],
    }


def strategies() -> list[dict]:
    """돌릴 전략 목록. 한 번에 한 축만 움직인다."""

    out: list[dict] = []
    # 축 1 — 결합 형태 (가중치는 균등으로 고정)
    for form in BLEND_FORMS:
        out.append({"name": f"form_{form}", "form": form, "weights": "uniform",
                    "calibrate": False, "axis": "결합 형태"})
    # 축 2 — 가중치 (형태는 산술로 고정)
    for preset in WEIGHT_PRESETS:
        out.append({"name": f"w_{preset}", "form": "mean", "weights": preset,
                    "calibrate": False, "axis": "가중치"})
    out.append({"name": "w_learned", "form": "mean", "weights": None,
                "calibrate": False, "axis": "가중치"})
    # 축 3 — 보정 (형태별로 얹어 본다)
    for form in ("mean", "geometric", "rank"):
        out.append({"name": f"cal_{form}", "form": form, "weights": "uniform",
                    "calibrate": True, "axis": "보정"})
    out.append({"name": "cal_exp040mlp", "form": "mean", "weights": "exp040mlp",
                "calibrate": True, "axis": "보정"})
    out.append({"name": "cal_learned", "form": "mean", "weights": None,
                "calibrate": True, "axis": "보정"})
    # 축 4 — 좋아 보이는 조합
    out.append({"name": "best_geo_exp040mlp", "form": "geometric", "weights": "exp040mlp",
                "calibrate": True, "axis": "조합"})
    out.append({"name": "best_rank_learned", "form": "rank", "weights": None,
                "calibrate": True, "axis": "조합"})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", default="fam4")
    parser.add_argument("--fold-column", default="fold_group5")
    parser.add_argument("--depth", type=int, default=3,
                        help="계열당 후보 수. 1위가 본판, 2·3위는 잡음 폭 측정용")
    parser.add_argument("--rebuild-members", action="store_true",
                        help="후보를 다시 뽑아 후보 파일을 덮어쓴다")
    parser.add_argument("--members-file", type=Path, default=MEMBERS_FILE,
                        help="얼려 둔 후보 파일. 없으면 새로 뽑는다")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="후보를 뽑을 때 이 문자열이 든 파일은 제외 (예: cbopt10)")
    parser.add_argument("--only", help="이 이름의 전략만")
    args = parser.parse_args()

    check_fold_fingerprint(FOLDS)

    labels = pd.read_csv(RAW / "train.csv", usecols=["ID", "SUBCLASS"]).set_index("ID")["SUBCLASS"]
    classes = sorted(labels.unique())
    folds_frame = pd.read_parquet(FOLDS)
    ids = pd.Index(folds_frame["ID"])
    y = labels.reindex(ids).astype(str).to_numpy()
    fold_values = folds_frame[args.fold_column].to_numpy()

    if args.rebuild_members or not args.members_file.exists():
        config = build_members(labels, classes, args.depth, tuple(args.exclude))
        args.members_file.parent.mkdir(parents=True, exist_ok=True)
        args.members_file.write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
        print(f"후보를 새로 뽑아 {args.members_file.name} 에 얼렸다")
    config = json.loads(args.members_file.read_text(encoding="utf-8"))

    print(f"\n계열 대표 (단독 macro F1)")
    for family in FAMILIES:
        for rank, name in enumerate(config["members"][family], start=1):
            mark = "*" if rank == 1 else " "
            print(f"  {mark} {family:9s} {rank}위 {config['solo_macro_f1'][name]:.4f}  {name[:58]}")

    chosen = strategies()
    if args.only:
        chosen = [s for s in chosen if s["name"] == args.only] or chosen

    run = ARTIFACTS / "runs" / args.run_tag
    (run / "logs").mkdir(parents=True, exist_ok=True)

    # 본판 — 계열 1위끼리
    stacks = {r: load_stack([config["members"][f][r] for f in FAMILIES], ids, classes)
              for r in range(config["depth"])}

    results = []
    for strategy in chosen:
        started = time.perf_counter()
        weights = (None if strategy["weights"] is None
                   else WEIGHT_PRESETS[strategy["weights"]])
        outcome = crossfit(stacks[0], y, classes, fold_values, form=strategy["form"],
                           weights=weights, calibrate=strategy["calibrate"])
        row = {k: v for k, v in strategy.items()}
        row.update({k: v for k, v in outcome.items() if k != "labels"})
        row["labels_sha"] = hashlib.sha256(
            "".join(outcome["labels"]).encode("utf-8")).hexdigest()[:16]
        row["seconds"] = round(time.perf_counter() - started, 1)
        results.append(row)
        print(f"  {strategy['name']:20s} {outcome['crossfit_macro_f1']:.4f}  "
              f"({row['seconds']}초)")

    # 잡음 폭 — 같은 전략을 계열 대표만 바꿔 돌린다.
    # **1위 전략을 반드시 포함한다.** 자기 폭을 모르면 자기 점수도 못 읽는다.
    ranked = sorted(results, key=lambda r: -r["crossfit_macro_f1"])
    to_measure = list(dict.fromkeys(
        [ranked[0]["name"]] + [n for n in ("form_mean", "form_geometric", "w_exp040mlp")
                               if any(s["name"] == n for s in chosen)]))
    print("\n잡음 폭 (대표를 1위/2위/3위로)")
    noise = {}
    for name in to_measure:
        strategy = next(s for s in chosen if s["name"] == name)
        weights = (None if strategy["weights"] is None
                   else WEIGHT_PRESETS[strategy["weights"]])
        values = []
        for rank in range(config["depth"]):
            outcome = crossfit(stacks[rank], y, classes, fold_values, form=strategy["form"],
                               weights=weights, calibrate=strategy["calibrate"])
            values.append(round(outcome["crossfit_macro_f1"], 6))
        noise[name] = {"values": values, "spread": round(max(values) - min(values), 6)}
        print(f"  {name:20s} {values}  폭 {noise[name]['spread']:.4f}")

    spread = max(v["spread"] for v in noise.values())
    print(f"\n척도 — 이보다 작은 차이는 읽지 않는다: {spread:.4f}")

    # 형태끼리는 **짝으로** 비교한다. 위의 폭은 멤버 조합이 바뀔 때의 폭이라 두 형태를
    # 각각 그 폭과 견주면 둘 다 "구별 안 됨" 으로 뭉개진다. 같은 멤버 조합에서 A 와 B 를
    # 나란히 재면 멤버 효과가 상쇄되고 형태 차이만 남는다.
    print(f"\n=== 형태끼리 짝 비교 (같은 멤버 조합 {config['depth']}벌) ===")
    paired = {}
    for form in BLEND_FORMS:
        paired[form] = [
            crossfit(stacks[rank], y, classes, fold_values, form=form,
                     weights=WEIGHT_PRESETS["uniform"], calibrate=False)["crossfit_macro_f1"]
            for rank in range(config["depth"])
        ]
    base = paired["mean"]
    print(f"{'형태':12s} " + " ".join(f"{i + 1}벌".rjust(7) for i in range(config["depth"]))
          + f" {'평균차':>8s} {'이긴 횟수':>8s}")
    for form, values in paired.items():
        deltas = [v - b for v, b in zip(values, base)]
        wins = sum(d > 0 for d in deltas)
        print(f"{form:12s} " + " ".join(f"{v:7.4f}" for v in values)
              + f" {np.mean(deltas):+8.4f} {wins:>5d}/{len(deltas)}")
    print("기준은 mean. 이긴 횟수가 전부 차야 형태 차이라고 읽는다.")

    report = {"run_tag": args.run_tag, "fold_column": args.fold_column,
              "members": config, "noise": noise, "noise_floor": spread,
              "paired_forms": {k: [round(v, 6) for v in vals] for k, vals in paired.items()},
              "results": sorted(results, key=lambda r: -r["crossfit_macro_f1"])}
    out = run / "logs" / "blend_families_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'전략':22s} {'축':10s} {'교차적합':>9s} {'fold편차':>8s} {'최저fold':>8s}")
    for row in report["results"]:
        gap = row["crossfit_macro_f1"] - report["results"][0]["crossfit_macro_f1"]
        mark = "  =" if abs(gap) <= spread else "   "
        print(f"{row['name']:22s} {row['axis']:10s} {row['crossfit_macro_f1']:>9.4f} "
              f"{row['fold_std']:>8.4f} {row['worst_fold']:>8.4f}{mark}")
    print(f"\n= 는 1위와 잡음 폭 안에서 구별되지 않는다는 표시")

    # 서로 다른 이름인데 예측이 글자 하나까지 같은 것들 — 축이 겹쳤다는 뜻이라
    # 둘을 별개 결과로 세면 안 된다. 멤버 4개면 median 과 trimmed 가 같은 연산이다.
    same: dict[str, list[str]] = {}
    for row in report["results"]:
        same.setdefault(row["labels_sha"], []).append(row["name"])
    twins = [names for names in same.values() if len(names) > 1]
    if twins:
        print("\n예측이 완전히 같은 전략 (별개로 세지 말 것)")
        for names in twins:
            print(f"  {' = '.join(names)}")

    print(f"\n기록 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
