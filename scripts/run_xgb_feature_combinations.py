#!/usr/bin/env python
"""Colab용 XGBoost enc3 × gene-evidence 2×4 ablation 실행기.

공통 기반은 ``domain + rollup16`` 이다. enc3 유무 2수준과 evidence
``none / ebovr / ebbnb / both`` 4수준을 조합해 SKF·SGKF를 모두 실행한다.

완료된 case는 JSON 로그와 OOF·test prediction·submission 파일을 함께 검증한 뒤
건너뛴다. 따라서 Colab 런타임이 끊겨도 같은 명령을 다시 실행하면 이어서 진행한다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_gbdt as tg  # noqa: E402
from cancer_hack.validation import SEED as FOLD_SEED  # noqa: E402
from cancer_hack.validation import build_fold_frame, fold_column  # noqa: E402


FACTORIAL_CONFIGS = (
    "r16_base",
    "r16_ebovr",
    "r16_ebbnb",
    "r16_ebboth",
    "f4r",
    "f4r_ebovr",
    "f4r_ebbnb",
    "f4r_ebboth",
)
REFERENCE_CONFIGS = ("f4r", "f4r_gtype", "f4rsig", "full_all")
CANDIDATE_CONFIGS = (
    *REFERENCE_CONFIGS,
    "f4r_gtype_csig",
    "f4r_gtype_ebbnb",
    "f4r_gtype_ebovr",
    "f4r_compact",
    "full_all_ebbnb",
    "full_all_ebovr",
    "f4r_gtype_csig_ebbnb",
    "f4r_compact_ebbnb",
)
SUITES = {
    "factorial": FACTORIAL_CONFIGS,
    "candidates": CANDIDATE_CONFIGS,
    "all": tuple(dict.fromkeys((*FACTORIAL_CONFIGS, *CANDIDATE_CONFIGS))),
}
ANCHOR_CONFIG = {
    "f4r_gtype": "f4r",
    "f4rsig": "f4r",
    "full_all": "f4r",
    "f4r_gtype_csig": "f4r_gtype",
    "f4r_gtype_ebbnb": "f4r_gtype",
    "f4r_gtype_ebovr": "f4r_gtype",
    "f4r_compact": "f4r_gtype_csig",
    "full_all_ebbnb": "full_all",
    "full_all_ebovr": "full_all",
    "f4r_gtype_csig_ebbnb": "f4r_gtype_csig",
    "f4r_compact_ebbnb": "f4r_compact",
}
CVS = ("skf", "sgkf")
EVIDENCE_ORDER = {"none": 0, "ebovr": 1, "ebbnb": 2, "both": 3}

FEATURE_JOBS = (
    ("domain", (), "{split}_domain_features.parquet"),
    (
        "sample",
        ("--include-cell-rollup",),
        "{split}_sample_mutation_features_rollup.parquet",
    ),
    ("enc3", (), "{split}_mutation_encoded.parquet"),
    (
        "gene",
        ("--kind", "mutated"),
        "{split}_gene_mutated_matrix.parquet",
    ),
    ("gene-types", (), "{split}_gene_mutation_type_matrix.parquet"),
    ("sigtokens", (), "{split}_signature_mutation_tokens.parquet"),
    ("parsed-tokens", (), "{split}_parsed_mutation_tokens.parquet"),
    ("parsed", (), "{split}_mutation_parsed_features.parquet"),
    ("burden-extra", (), "{split}_additional_burden_features.parquet"),
    ("amino", (), "{split}_amino_acid_features.parquet"),
)


class Tee:
    """stdout을 Colab 화면과 Drive console log에 동시에 쓴다."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--process-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "auto", "cpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES),
        default="factorial",
        help="factorial=2×4 8개, candidates=reference 4개+후보 8개, all=전체",
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=FOLD_SEED,
        help="train_folds.parquet이 없을 때만 쓰는 분할 시드",
    )
    parser.add_argument("--tag", default="enc3_evidence_2x4")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="완료 산출물이 있어도 16개 case를 다시 실행한다.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def require_inputs(input_dir: Path) -> None:
    required = ("train.csv", "test.csv", "sample_submission.csv")
    missing = [name for name in required if not nonempty(input_dir / name)]
    if missing:
        raise FileNotFoundError(
            f"{input_dir}에서 입력 파일을 찾지 못했다: {missing}. "
            "제출 양식의 실제 파일명은 sample_submission.csv여야 한다."
        )


def ensure_features(input_dir: Path, process_dir: Path) -> None:
    """factorial/full 후보에 필요한 train/test parquet만 누락 시 생성한다."""
    process_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for command, extra, template in FEATURE_JOBS:
            output = process_dir / template.format(split=split)
            if nonempty(output):
                print("CACHE", output)
                continue
            run(
                [
                    sys.executable,
                    "scripts/make_features.py",
                    command,
                    "--split",
                    split,
                    "--input",
                    str(input_dir / f"{split}.csv"),
                    "--output",
                    str(output),
                    *extra,
                ]
            )

    folds_path = process_dir / "train_folds.parquet"
    if nonempty(folds_path):
        print("CACHE", folds_path)
        return

    group_cache = process_dir / "train_group_keys.parquet"
    folds = build_fold_frame(
        input_dir / "train.csv",
        label_column="SUBCLASS",
        n_splits=5,
        seed=FOLD_SEED,
        group_cache_path=group_cache,
    )
    temporary = folds_path.with_suffix(".tmp.parquet")
    folds.to_parquet(temporary, index=False)
    temporary.replace(folds_path)
    atomic_json(
        {
            "source": "train.csv",
            "n_splits": 5,
            "seed": FOLD_SEED,
            "label_column": "SUBCLASS",
            "n_rows": int(len(folds)),
            "n_groups": int(folds["group_key"].nunique()),
            "fold_columns": [
                fold_column(kind, 5) for kind in ("skf", "sgkf")
            ],
        },
        folds_path.with_suffix(".json"),
    )
    print("CREATED", folds_path)


def train_args(
    args: argparse.Namespace, selected_configs: tuple[str, ...]
) -> argparse.Namespace:
    parsed = tg.build_parser().parse_args(
        [
            "--model",
            "xgb",
            "--configs",
            ",".join(selected_configs),
            "--cv",
            "all",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--n-splits",
            str(args.n_splits),
            "--tag",
            args.tag,
            "--submission",
        ]
    )
    parsed.device = {"gpu": True, "cpu": False, "auto": "auto"}[parsed.device]
    parsed.override = tg._parse_override(parsed.overrides)
    return parsed


def artifact_paths(artifacts: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        artifacts / "oof" / f"oof_{stem}.csv",
        artifacts / "test_predictions" / f"test_{stem}.csv",
        artifacts / "submissions" / f"submission_{stem}.csv",
    )


def load_completed(
    artifacts: Path,
    args: argparse.Namespace,
    config: str,
    cv: str,
) -> dict | None:
    pattern = (
        f"xgb_{args.tag}_{config}_{tg.CV_SLUG[cv]}_*_s{args.seed}.json"
    )
    for path in sorted((artifacts / "logs").glob(pattern), reverse=True):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            stem = result["stem"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if (
            result.get("model") != "xgb"
            or result.get("config") != config
            or result.get("cv") != cv
            or result.get("seed") != args.seed
            or result.get("blocks") != list(tg.CONFIGS[config]["blocks"])
        ):
            continue
        if all(nonempty(item) for item in artifact_paths(artifacts, stem)):
            return result
    return None


def evidence_label(config: str) -> str:
    blocks = set(tg.CONFIGS[config]["blocks"])
    if {"ebovr", "ebbnb"} <= blocks:
        return "both"
    if "ebovr" in blocks:
        return "ebovr"
    if "ebbnb" in blocks:
        return "ebbnb"
    return "none"


def comparison_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        config = result["config"]
        blocks = set(result["blocks"])
        rows.append(
            {
                "config": config,
                "cv": result["cv"],
                "group": (
                    "factorial"
                    if config in FACTORIAL_CONFIGS
                    else "reference"
                    if config in REFERENCE_CONFIGS
                    else "candidate"
                ),
                "enc3": "enc3" in blocks,
                "evidence": evidence_label(config),
                "anchor_config": ANCHOR_CONFIG.get(config),
                "n_features": result["n_features"],
                "oof_macro_f1": result["oof_macro_f1"],
                "oof_macro_f1_singleton": result[
                    "oof_macro_f1_singleton"
                ],
                "oof_accuracy": result["oof_accuracy"],
                "elapsed_seconds": result["elapsed_seconds"],
                "device": result["device"],
                "stem": result["stem"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["evidence_order"] = frame["evidence"].map(EVIDENCE_ORDER)
    score_lookup = frame.set_index(["config", "cv"])["oof_macro_f1"]
    frame["delta_vs_anchor"] = [
        (
            score - score_lookup.get((anchor, cv), float("nan"))
            if isinstance(anchor, str)
            else float("nan")
        )
        for anchor, cv, score in zip(
            frame["anchor_config"],
            frame["cv"],
            frame["oof_macro_f1"],
            strict=True,
        )
    ]

    base_lookup = {
        cv: score_lookup.get(("r16_base", cv), float("nan")) for cv in CVS
    }
    frame["delta_vs_r16_base"] = [
        score - base_lookup[cv] for cv, score in zip(
            frame["cv"], frame["oof_macro_f1"], strict=True
        )
    ]
    factorial_none = {
        ("skf", False): "r16_base",
        ("skf", True): "f4r",
        ("sgkf", False): "r16_base",
        ("sgkf", True): "f4r",
    }
    without_enc3 = {
        ("skf", "none"): "r16_base",
        ("skf", "ebovr"): "r16_ebovr",
        ("skf", "ebbnb"): "r16_ebbnb",
        ("skf", "both"): "r16_ebboth",
        ("sgkf", "none"): "r16_base",
        ("sgkf", "ebovr"): "r16_ebovr",
        ("sgkf", "ebbnb"): "r16_ebbnb",
        ("sgkf", "both"): "r16_ebboth",
    }
    frame["delta_evidence_vs_none"] = [
        (
            score
            - score_lookup.get(
                (factorial_none[(cv, enc3)], cv), float("nan")
            )
            if group == "factorial"
            else float("nan")
        )
        for group, cv, enc3, score in zip(
            frame["group"],
            frame["cv"],
            frame["enc3"],
            frame["oof_macro_f1"],
            strict=True,
        )
    ]
    frame["delta_enc3_same_evidence"] = [
        (
            score
            - score_lookup.get(
                (without_enc3[(cv, evidence)], cv), float("nan")
            )
            if group == "factorial" and enc3
            else float("nan")
        )
        for group, cv, evidence, enc3, score in zip(
            frame["group"],
            frame["cv"],
            frame["evidence"],
            frame["enc3"],
            frame["oof_macro_f1"],
            strict=True,
        )
    ]
    return (
        frame.sort_values(["group", "cv", "enc3", "evidence_order", "config"])
        .drop(columns="evidence_order")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.process_dir = args.process_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    require_inputs(args.input_dir)
    if args.n_splits != 5:
        raise ValueError("이 실험과 저장된 fold 계약은 n_splits=5로 고정한다.")
    if args.fold_seed != FOLD_SEED:
        raise ValueError(
            f"기존 비교와 같은 fold를 쓰기 위해 fold_seed={FOLD_SEED}로 고정한다."
        )

    ensure_features(args.input_dir, args.process_dir)
    artifacts = args.output_dir / "artifacts"
    reports = args.output_dir / "reports"
    console_dir = artifacts / "console"
    error_dir = artifacts / "errors"
    for directory in (artifacts, reports, console_dir, error_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tg.RAW_DIR = args.input_dir
    tg.PROC_DIR = args.process_dir
    tg.ARTIFACTS = artifacts
    selected_configs = SUITES[args.suite]
    missing_configs = [name for name in selected_configs if name not in tg.CONFIGS]
    if missing_configs:
        raise RuntimeError(
            f"현재 clone에 요청한 config가 없다: {missing_configs}. 최신 test 브랜치를 pull한다."
        )

    parsed = train_args(args, selected_configs)
    needed = set().union(
        *(set(tg.CONFIGS[name]["blocks"]) for name in selected_configs)
    )
    data = tg.Dataset(needed, n_splits=args.n_splits)
    results: list[dict] = []
    failures: dict[str, str] = {}
    total_cases = len(selected_configs) * len(CVS)

    for index, (config, cv) in enumerate(
        ((config, cv) for config in selected_configs for cv in CVS), start=1
    ):
        case = f"{config}/{cv}"
        cached = None if args.rerun else load_completed(
            artifacts, parsed, config, cv
        )
        if cached is not None:
            cached["run_status"] = "RESUMED"
            results.append(cached)
            print(
                f"[{index:02d}/{total_cases:02d}] RESUME {case:<28} "
                f"F1={cached['oof_macro_f1']:.6f}"
            )
            continue

        console_path = console_dir / f"{args.tag}_{config}_{cv}_s{args.seed}.log"
        try:
            with console_path.open("a", encoding="utf-8") as handle:
                with contextlib.redirect_stdout(Tee(sys.stdout, handle)):
                    print(f"\n[{index:02d}/{total_cases:02d}] START {case}")
                    result = tg.run_config(
                        data, config=config, cv=cv, args=parsed
                    )
                    result["run_status"] = "TRAINED"
                    results.append(result)
                    print(
                        f"[{index:02d}/{total_cases:02d}] DONE  {case} "
                        f"F1={result['oof_macro_f1']:.6f}"
                    )
            tg.write_matrix(parsed)
        except Exception:  # noqa: BLE001
            failures[case] = traceback.format_exc()
            atomic_json(
                {"case": case, "traceback": failures[case]},
                error_dir / f"{args.tag}_{config}_{cv}_s{args.seed}.json",
            )
            print(f"[{index:02d}/{total_cases:02d}] FAILED {case}")
            traceback.print_exc()

    comparison = comparison_frame(results)
    comparison_path = reports / f"comparison_{args.tag}_s{args.seed}.csv"
    html_path = reports / f"comparison_{args.tag}_s{args.seed}.html"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    comparison.to_html(html_path, index=False, float_format=lambda value: f"{value:.6f}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "SUCCESS"
            if len(results) == total_cases and not failures
            else "INCOMPLETE"
        ),
        "suite": args.suite,
        "configs": list(selected_configs),
        "cvs": list(CVS),
        "model": "xgb",
        "seed": args.seed,
        "fold_seed": args.fold_seed,
        "n_splits": args.n_splits,
        "tag": args.tag,
        "input_dir": str(args.input_dir),
        "process_dir": str(args.process_dir),
        "output_dir": str(args.output_dir),
        "completed_cases": len(results),
        "failures": failures,
        "comparison_csv": str(comparison_path),
        "comparison_html": str(html_path),
    }
    atomic_json(manifest, args.output_dir / f"manifest_{args.tag}_s{args.seed}.json")

    print("\n=== 최종 비교표 ===")
    if comparison.empty:
        print("완료된 case가 없다.")
    else:
        columns = [
            "config",
            "cv",
            "group",
            "enc3",
            "evidence",
            "anchor_config",
            "n_features",
            "oof_macro_f1",
            "oof_macro_f1_singleton",
            "delta_vs_anchor",
            "delta_evidence_vs_none",
            "delta_enc3_same_evidence",
            "device",
        ]
        print(comparison[columns].to_string(index=False))
    print("\ncomparison CSV :", comparison_path)
    print("comparison HTML:", html_path)
    print("artifacts      :", artifacts)
    if failures:
        raise SystemExit(
            f"{len(failures)}개 case가 실패했다. 같은 명령을 다시 실행하면 이어서 진행한다."
        )


if __name__ == "__main__":
    main()
