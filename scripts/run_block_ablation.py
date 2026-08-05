#!/usr/bin/env python
r"""f16 대형 피처 블록의 조건부 기여도를 drop/add 양방향으로 측정한다.

이 스크립트는 ``train_gbdt.py`` 를 수정하지 않고 런타임에만 실험 config 를 등록한다.
따라서 별도로 실행 중인 Optuna 프로세스가 나중에 ``train_gbdt.py`` 를 다시 읽어도 f16
탐색 공간이나 동작은 바뀌지 않는다.

기본 실행은 **계획만 출력**한다. 실제 학습에는 반드시 ``--execute`` 가 필요하다.

    # 현재 상태와 약 1시간짜리 group5 계획만 확인
    .\.venv\Scripts\python.exe scripts\run_block_ablation.py

    # Optuna 종료 후 CatBoost seed42 group5 1차 스크리닝
    .\.venv\Scripts\python.exe scripts\run_block_ablation.py --execute

    # 중단 뒤 같은 명령을 다시 실행하면 완성된 config 는 건너뛴다
    .\.venv\Scripts\python.exe scripts\run_block_ablation.py --execute

    # 학습 없이 현재 로그만 다시 집계
    .\.venv\Scripts\python.exe scripts\run_block_ablation.py --report-only

실험 축
-------
``ba_full`` 은 f16 과 블록·가중치가 같다. ``ba_base`` 는 아래 6개 대형 블록만 뺀
나머지 f16 이다. drop 은 Full 에서 하나를 빼고, add 는 Baseline 에 하나를 더한다.

    gtype, comut, lsvd, lnmf, gmod, csig

OOF Macro F1 외에 fold별 델타, singleton 델타, 클래스별 F1 델타, 차원과 시간도 함께
기록한다. 종합 점수는 탐색 순서를 정하는 보조값일 뿐 채택 기준이 아니다.

DACON 업로드는 하지 않는다. ``train_gbdt`` 에도 ``--no-submission`` 을 강제한다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 경로는 `cancer_hack.paths` 가 정한다 — 값은 쓰는 시점에 정해진다.
from cancer_hack.paths import LAZY_ARTIFACTS as ARTIFACTS  # noqa: E402
TARGET_BLOCKS = ("gtype", "comut", "lsvd", "lnmf", "gmod", "csig")
CV_SLUG = {"skf": "skf5", "sgkf": "group5"}
SECONDS_PER_CONFIG = {"catboost": 270, "xgb": 290, "rf": 100, "lgbm": 320}


def log(message: str) -> None:
    print(message, flush=True)


def experiment_configs(full_blocks: tuple[str, ...]) -> OrderedDict[str, dict[str, Any]]:
    """f16 블록 순서를 보존한 14개 Full/Baseline/drop/add config 를 만든다."""
    missing = [block for block in TARGET_BLOCKS if block not in full_blocks]
    if missing:
        raise ValueError(f"f16 에 대형 블록이 없다: {missing}")
    if len(set(full_blocks)) != len(full_blocks):
        raise ValueError("f16 블록에 중복이 있다")

    baseline = tuple(block for block in full_blocks if block not in TARGET_BLOCKS)
    specs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    specs["ba_full"] = {
        "blocks": tuple(full_blocks),
        "weight": "balanced",
        "desc": "block ablation Full — f16 과 같은 16블록",
    }
    for block in TARGET_BLOCKS:
        specs[f"ba_d_{block}"] = {
            "blocks": tuple(value for value in full_blocks if value != block),
            "weight": "balanced",
            "desc": f"Full - {block}",
        }
    specs["ba_base"] = {
        "blocks": baseline,
        "weight": "balanced",
        "desc": "Full - 6개 대형 블록",
    }
    for block in TARGET_BLOCKS:
        # Full 의 원래 위치에 넣어 열 순서까지 고정한다. CatBoost/GBDT 는 열 순서가
        # 달라져도 학습은 되지만, 같은 블록 조합의 재현성을 위해 원순서를 지킨다.
        blocks = tuple(value for value in full_blocks if value in baseline or value == block)
        specs[f"ba_a_{block}"] = {
            "blocks": blocks,
            "weight": "balanced",
            "desc": f"Baseline + {block}",
        }
    return specs


def phase_names(specs: OrderedDict[str, dict[str, Any]], stage: str) -> list[str]:
    if stage == "drop":
        return ["ba_full", *[f"ba_d_{block}" for block in TARGET_BLOCKS]]
    if stage == "add":
        return ["ba_base", *[f"ba_a_{block}" for block in TARGET_BLOCKS]]
    if stage == "both":
        return list(specs)
    raise ValueError(stage)


def _load_train_gbdt():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("train_gbdt")


def _valid_result(path: Path, *, model: str, config: str, cv: str, seed: int) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "model": model,
        "config": config,
        "cv": cv,
        "seed": seed,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    if "oof_macro_f1" not in payload or "stem" not in payload:
        return None
    stem = payload["stem"]
    if not (ARTIFACTS / "oof" / f"oof_{stem}.csv").exists():
        return None
    if not (ARTIFACTS / "test_predictions" / f"test_{stem}.csv").exists():
        return None
    payload["_log_path"] = str(path)
    return payload


def find_result(*, model: str, tag: str, config: str, cv: str, seed: int) -> dict | None:
    slug = CV_SLUG[cv]
    pattern = f"{model}_{tag}_{config}_{slug}_*_s{seed}.json"
    candidates = sorted((ARTIFACTS / "logs").glob(pattern), reverse=True)
    for path in candidates:
        result = _valid_result(path, model=model, config=config, cv=cv, seed=seed)
        if result is not None:
            return result
    return None


def inventory(names: list[str], args) -> tuple[dict[str, dict], list[str]]:
    complete: dict[str, dict] = {}
    missing: list[str] = []
    for name in names:
        result = find_result(
            model=args.model, tag=args.tag, config=name, cv=args.cv, seed=args.seed
        )
        if result is None:
            missing.append(name)
        else:
            complete[name] = result
    return complete, missing


def _fold_delta(left: dict, right: dict) -> list[float]:
    a = left.get("fold_macro_f1") or []
    b = right.get("fold_macro_f1") or []
    return [float(x - y) for x, y in zip(a, b)] if len(a) == len(b) else []


def _class_delta(left: dict, right: dict) -> dict[str, float]:
    a = left.get("per_class_f1") or {}
    b = right.get("per_class_f1") or {}
    return {label: float(a[label] - b[label]) for label in sorted(set(a) & set(b))}


def build_summary(results: dict[str, dict], *, model: str, tag: str, cv: str, seed: int) -> dict:
    """완성된 로그에서 제거 중요도·추가 이득·종합 점수를 계산한다."""
    full = results.get("ba_full")
    base = results.get("ba_base")
    rows = []
    if full is not None and base is not None:
        for block in TARGET_BLOCKS:
            drop = results.get(f"ba_d_{block}")
            add = results.get(f"ba_a_{block}")
            if drop is None or add is None:
                continue
            removal = float(full["oof_macro_f1"] - drop["oof_macro_f1"])
            addition = float(add["oof_macro_f1"] - base["oof_macro_f1"])
            rows.append(
                {
                    "block": block,
                    "full_f1": float(full["oof_macro_f1"]),
                    "drop_f1": float(drop["oof_macro_f1"]),
                    "removal_importance": removal,
                    "baseline_f1": float(base["oof_macro_f1"]),
                    "add_f1": float(add["oof_macro_f1"]),
                    "addition_gain": addition,
                    "composite": removal + addition,
                    "full_singleton": float(full.get("oof_macro_f1_singleton", 0.0)),
                    "drop_singleton": float(drop.get("oof_macro_f1_singleton", 0.0)),
                    "baseline_singleton": float(base.get("oof_macro_f1_singleton", 0.0)),
                    "add_singleton": float(add.get("oof_macro_f1_singleton", 0.0)),
                    "removal_fold_delta": _fold_delta(full, drop),
                    "addition_fold_delta": _fold_delta(add, base),
                    "removal_class_delta": _class_delta(full, drop),
                    "addition_class_delta": _class_delta(add, base),
                    "drop_n_features": int(drop.get("n_features", 0)),
                    "add_n_features": int(add.get("n_features", 0)),
                    "drop_seconds": float(drop.get("elapsed_seconds", 0.0)),
                    "add_seconds": float(add.get("elapsed_seconds", 0.0)),
                }
            )
        rows.sort(key=lambda row: row["composite"], reverse=True)
    return {
        "model": model,
        "tag": tag,
        "cv": cv,
        "seed": seed,
        "targets": list(TARGET_BLOCKS),
        "full_config": "ba_full",
        "baseline_config": "ba_base",
        "completed_configs": sorted(results),
        "rows": rows,
        "warning": (
            "composite는 우선순위 보조값이다. fold 부호·singleton·클래스별 변화와 "
            "다중 seed 재검증 없이 최종 채택하지 않는다."
        ),
    }


def write_report(results: dict[str, dict], args) -> tuple[Path, Path] | None:
    summary = build_summary(
        results, model=args.model, tag=args.tag, cv=args.cv, seed=args.seed
    )
    if not summary["rows"]:
        log("보고서 보류: Full/Baseline/drop/add 로그가 아직 모두 모이지 않았다.")
        return None
    stem = f"block_ablation_{args.model}_{args.tag}_{CV_SLUG[args.cv]}_s{args.seed}"
    json_path = ARTIFACTS / "logs" / f"{stem}.json"
    csv_path = ARTIFACTS / "logs" / f"{stem}.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = []
    for row in summary["rows"]:
        flat.append(
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            }
        )
    pd.DataFrame(flat).to_csv(csv_path, index=False, encoding="UTF-8-sig")
    log("\n블록 기여도 (OOF Macro F1)")
    log("block      removal      addition     composite")
    for row in summary["rows"]:
        log(
            f"{row['block']:<10} {row['removal_importance']:+.6f}   "
            f"{row['addition_gain']:+.6f}   {row['composite']:+.6f}"
        )
    log(f"JSON: {json_path}")
    log(f"CSV : {csv_path}")
    return json_path, csv_path


def _train_args(tg, args):
    values = [
        "--model", args.model,
        "--configs", "ba_full",  # 아래에서 config 를 직접 넘기므로 표시용
        "--cv", args.cv,
        "--topk", str(args.topk),
        "--seed", str(args.seed),
        "--tag", args.tag,
        "--no-submission",
    ]
    if args.device:
        values.extend(["--device", args.device])
    if args.track_train:
        values.append("--track-train")
    parsed = tg.build_parser().parse_args(values)
    parsed.device = {"gpu": True, "cpu": False, "auto": "auto"}[parsed.device]
    parsed.override = tg._parse_override(parsed.overrides)
    return parsed


def execute(names: list[str], specs: OrderedDict[str, dict[str, Any]], args) -> None:
    tg = _load_train_gbdt()
    # 실행 중인 다른 Python 프로세스의 CONFIGS 와 무관한 현재 프로세스 메모리 안에서만 등록.
    tg.CONFIGS.update(specs)
    train_args = _train_args(tg, args)
    complete, missing = inventory(names, args)
    if args.force:
        missing = list(names)
    if not missing:
        log("모든 config 가 이미 완성돼 있다.")
        write_report(complete, args)
        return

    needed: set[str] = set()
    for name in missing:
        needed.update(specs[name]["blocks"])
    log(f"\n학습 시작: {missing}")
    data = tg.Dataset(needed, n_splits=train_args.n_splits, folds_path=train_args.folds)
    for index, name in enumerate(missing, start=1):
        log(f"\n[{index}/{len(missing)}] {name} · {specs[name]['desc']}")
        started = time.perf_counter()
        tg.run_config(data, config=name, cv=args.cv, args=train_args)
        log(f"{name} 완료 · {time.perf_counter() - started:.0f}초")

    complete, remaining = inventory(names, args)
    if remaining:
        log(f"미완료 config: {remaining} — 같은 명령으로 이어서 실행한다.")
    write_report(complete, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["catboost", "xgb", "rf", "lgbm"], default="catboost")
    parser.add_argument("--cv", choices=["sgkf", "skf"], default="sgkf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="blockab")
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--stage", choices=["drop", "add", "both"], default="both")
    parser.add_argument(
        "--only-configs",
        nargs="+",
        default=None,
        help="stage 안에서 이 config 만 실행한다. 다중 모델 오케스트레이터의 재개용",
    )
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--track-train", action="store_true")
    parser.add_argument("--execute", action="store_true", help="실제 학습. 생략하면 계획만 출력")
    parser.add_argument("--report-only", action="store_true", help="학습 없이 로그만 집계")
    parser.add_argument("--force", action="store_true", help="완성 로그가 있어도 다시 학습")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tg = _load_train_gbdt()
    specs = experiment_configs(tuple(tg.CONFIGS["f16"]["blocks"]))
    names = phase_names(specs, args.stage)
    if args.only_configs:
        unknown = [name for name in args.only_configs if name not in specs]
        if unknown:
            raise SystemExit(f"모르는 block-ablation config: {unknown}")
        requested = set(args.only_configs)
        names = [name for name in names if name in requested]
        if not names:
            raise SystemExit("--stage 범위 안에 --only-configs 대상이 없다")
    complete, missing = inventory(names, args)

    log(
        f"설정: model={args.model} cv={args.cv} seed={args.seed} tag={args.tag} "
        f"stage={args.stage}"
    )
    log(f"완료 {len(complete)}/{len(names)} · 남음 {len(missing)}")
    for name in names:
        mark = "완료" if name in complete else "대기"
        log(f"  [{mark}] {name:<14} {specs[name]['desc']}")
    estimate = sum(SECONDS_PER_CONFIG[args.model] for _ in missing)
    log(f"예상 학습 시간: 약 {estimate / 60:.0f}분 + 최초 피처 준비 시간")

    if args.report_only:
        write_report(complete, args)
        return
    if not args.execute:
        log("계획만 출력했다. 실제 학습은 Optuna 종료 후 --execute 로 시작한다.")
        return
    execute(names, specs, args)


if __name__ == "__main__":
    main()
