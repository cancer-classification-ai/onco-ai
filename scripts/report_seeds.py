#!/usr/bin/env python
"""seed 별로 흩어진 `train_gbdt.py` 로그를 묶어 평균±표준편차를 낸다.

    python scripts/report_seeds.py
    python scripts/report_seeds.py --prefix xgb_cm_
    python scripts/report_seeds.py --metric oof_macro_f1_singleton
    python scripts/report_seeds.py --json artifacts/logs/cm_seed_matrix.json

## 왜 필요한가

`artifacts/logs/*.json` 은 94개가 전부 `_s42` 다. seed 안정성이 한 번도 측정된 적이
없다. 개선이 진짜인지 잡음인지 가르려면 여러 seed 를 돌리고 평균±표준편차로 봐야
하는데, `train_gbdt.py --seed` 는 이미 흐르고 stem 에 `_s{seed}` 가 이미 붙어서
seed 별 로그는 이미 쌓이고 있다. 없는 건 그걸 묶어 보는 도구뿐이다.

읽기 전용이다. `train_gbdt.py` 는 한 줄도 안 건드린다.

## 자체 검사가 하나 딸려 있다

같은 묶음(= seed 뗀 stem) 안에서 `n_features`·`blocks`·`cv`·`config` 가 달라지면
경고한다. 슬러그를 빠뜨려 서로 다른 설정이 같은 stem 에 충돌한 사고
(`sparse_slug`/`comut_slug` 를 낳은 그 버그 부류)를 자동으로 잡아낸다.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 경로는 `cancer_hack.paths` 가 정한다 — 값은 쓰는 시점에 정해진다.
from cancer_hack.paths import LazyDir, artifacts_dir  # noqa: E402

LOGS_DIR = LazyDir(lambda: artifacts_dir() / "logs")

#: 그룹 안에서 상수여야 하는 열 — 다르면 두 설정이 한 stem 에 충돌했다는 뜻이다.
_CONSTANT_KEYS = ("n_features", "blocks", "cv", "config")
_METRIC_KEYS = ("oof_macro_f1", "oof_macro_f1_singleton", "oof_accuracy", "generalization_gap")

_SEED_SUFFIX = re.compile(r"_s\d+$")


def _iter_logs(logs_dir: Path):
    """개별 config 로그만 낸다. 비교표(`*matrix*.json`, 리스트)와 ERROR 로그는 거른다."""
    for path in sorted(logs_dir.glob("*.json")):
        if "matrix" in path.stem or path.stem.endswith("_ERROR"):
            continue
        with open(path, encoding="utf-8") as handle:
            row = json.load(handle)
        if not isinstance(row, dict):
            continue  # 리스트 형태는 비교표다
        if "oof_macro_f1" not in row or "stem" not in row or "seed" not in row:
            continue
        yield path, row


def _group_key(stem: str) -> str:
    return _SEED_SUFFIX.sub("", stem)


def collect(logs_dir: Path, *, prefix: str | None = None) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for path, row in _iter_logs(logs_dir):
        if prefix and not row["stem"].startswith(prefix):
            continue
        groups.setdefault(_group_key(row["stem"]), []).append(row)
    return groups


def _check_constant(key_name: str, group: str, rows: list[dict]) -> list[str]:
    values = {json.dumps(r.get(key_name), sort_keys=True, default=str) for r in rows}
    if len(values) > 1:
        return [
            f"[slug 경고] {group}: {key_name} 이 seed 마다 다르다 ({len(values)}종). "
            "슬러그가 빠져 서로 다른 설정이 한 stem 에 충돌했을 수 있다"
        ]
    return []


def summarize(rows: list[dict]) -> dict:
    seeds = sorted(r["seed"] for r in rows)
    summary: dict = {"n_seeds": len(rows), "seeds": seeds}
    for metric in _METRIC_KEYS:
        values = [r[metric] for r in rows if metric in r]
        if not values:
            continue
        summary[metric] = {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) >= 2 else None,
            "min": min(values),
            "max": max(values),
        }
    first = rows[0]
    summary["n_features"] = first.get("n_features")
    summary["cv"] = first.get("cv")
    summary["config"] = first.get("config")
    summary["blocks"] = first.get("blocks")
    summary["selected_gene_overlap"] = first.get("selected_gene_overlap")
    return summary


def build_report(logs_dir: Path, *, prefix: str | None = None) -> tuple[dict[str, dict], list[str]]:
    groups = collect(logs_dir, prefix=prefix)
    warnings: list[str] = []
    report: dict[str, dict] = {}
    for group, rows in groups.items():
        for key in _CONSTANT_KEYS:
            warnings += _check_constant(key, group, rows)
        report[group] = summarize(rows)
    return report, warnings


def _fmt(entry: dict | None) -> str:
    if not entry:
        return "-"
    sd = f"±{entry['sd']:.4f}" if entry["sd"] is not None else "  n=1 "
    return f"{entry['mean']:.4f}{sd}"


def print_table(report: dict[str, dict], *, metric: str) -> None:
    rows = sorted(
        report.items(), key=lambda kv: kv[1].get(metric, {}).get("mean", -1), reverse=True
    )
    # skf 가 주 지표다 — 먼저 보인다.
    rows.sort(key=lambda kv: kv[1].get("cv") != "skf")
    header = (
        f"{'group':<45}{'cv':<6}{'차원':>7}{'seeds':>7}  "
        f"{metric:<22}선택 안정성"
    )
    print(header)
    print("-" * len(header))
    for group, entry in rows:
        overlap = entry.get("selected_gene_overlap") or {}
        overlap_str = " ".join(f"{k}={v:.2f}" for k, v in overlap.items())
        print(
            f"{group:<45}{str(entry.get('cv')):<6}{entry.get('n_features', '-'):>7}"
            f"{entry['n_seeds']:>7}  {_fmt(entry.get(metric)):<22}{overlap_str}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR)
    parser.add_argument("--prefix", default=None, help="stem 접두사로 거른다 (예: xgb_cm_)")
    parser.add_argument("--metric", default="oof_macro_f1", choices=_METRIC_KEYS)
    parser.add_argument("--json", type=Path, default=None, help="지정하면 리포트를 이 경로에 쓴다")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report, warnings = build_report(args.logs_dir, prefix=args.prefix)
    if not report:
        raise SystemExit(f"{args.logs_dir} 에서 묶을 로그를 못 찾았다 (prefix={args.prefix!r}).")

    print_table(report, metric=args.metric)
    if warnings:
        print()
        for line in warnings:
            print(line)

    print()
    print(
        "채택 기준: mean_new - mean_base > 2 * sd_pooled. seed 하나로는 잡음과 "
        "못 가른다. 최소 seed 3개 이상에서 비교한다."
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\n리포트: {args.json}")


if __name__ == "__main__":
    main()
