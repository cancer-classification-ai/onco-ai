#!/usr/bin/env python
"""Run the full-feature DL ladder on Kaggle: MLP baseline, then Hybrid.

The two models remain separate experiments and produce independent OOF, log,
test-probability, and submission files.

    python scripts/run_dl_ladder.py --device cuda --seed 42
    python scripts/run_dl_ladder.py --device cuda --seed 42 --dry-run
    python scripts/run_dl_ladder.py --models mlp --device cuda
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.validation import CV_SLUG  # noqa: E402

MODEL_CONFIGS = {
    "mlp": PROJECT_ROOT / "configs/mlp.yaml",
    "set_encoder": PROJECT_ROOT / "configs/gene_set_encoder.yaml",
    "hybrid": PROJECT_ROOT / "configs/hybrid_set_mlp.yaml",
}
FULL_FEATURE_FILES = (
    "train_domain_features.parquet",
    "test_domain_features.parquet",
    "train_sample_mutation_features_rollup.parquet",
    "test_sample_mutation_features_rollup.parquet",
    "train_mutation_parsed_features.parquet",
    "test_mutation_parsed_features.parquet",
    "train_additional_burden_features.parquet",
    "test_additional_burden_features.parquet",
    "train_amino_acid_features.parquet",
    "test_amino_acid_features.parquet",
    "train_folds.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS),
        default=["mlp", "hybrid"],
        help="execution order; default runs the baseline before Hybrid",
    )
    parser.add_argument("--cv", choices=["skf", "sgkf"], default="skf")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tag-prefix",
        default="full",
        help="artifact tag prefix; model and seed are appended automatically",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-submission", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a model only when all expected artifacts already exist",
    )
    return parser.parse_args()


def experiment_tag(model: str, tag_prefix: str, seed: int) -> str:
    return f"{model}_{tag_prefix}_s{seed}"


def artifact_paths(
    model: str,
    tag: str,
    cv: str,
    *,
    include_submission: bool,
) -> list[Path]:
    stem = f"dl_{model}_{tag}_{CV_SLUG[cv]}"
    paths = [
        PROJECT_ROOT / "artifacts/oof" / f"oof_{stem}.csv",
        PROJECT_ROOT / "artifacts/test_predictions" / f"test_{stem}.csv",
        PROJECT_ROOT / "artifacts/logs" / f"{stem}.json",
    ]
    if include_submission:
        paths.append(
            PROJECT_ROOT / "artifacts/submissions" / f"submission_{stem}.csv"
        )
    return paths


def build_train_command(
    model: str,
    *,
    cv: str,
    device: str,
    seed: int,
    tag: str,
    dry_run: bool,
    no_submission: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_dl.py"),
        "--model",
        model,
        "--config",
        str(MODEL_CONFIGS[model]),
        "--cv",
        cv,
        "--device",
        device,
        "--seed",
        str(seed),
        "--tag",
        tag,
    ]
    if dry_run:
        command.append("--dry-run")
    if no_submission:
        command.append("--no-submission")
    return command


def check_inputs(models: list[str], *, need_submission: bool) -> None:
    missing = [
        PROJECT_ROOT / "data/process" / name
        for name in FULL_FEATURE_FILES
        if not (PROJECT_ROOT / "data/process" / name).exists()
    ]
    if any(model in ("set_encoder", "hybrid") for model in models):
        missing.extend(
            path
            for path in (
                PROJECT_ROOT / "data/raw/train.csv",
                PROJECT_ROOT / "data/raw/test.csv",
            )
            if not path.exists()
        )
    if need_submission:
        sample = PROJECT_ROOT / "data/raw/sample_submission.csv"
        if not sample.exists():
            missing.append(sample)
    if missing:
        relative = "\n".join(f"  - {path.relative_to(PROJECT_ROOT)}" for path in missing)
        raise FileNotFoundError(
            "DL ladder inputs are missing:\n"
            f"{relative}\n"
            "Copy raw CSVs and generate full features before training."
        )


def print_summary(models: list[str], tags: dict[str, str], cv: str) -> None:
    print("\nDL ladder summary")
    print(f"{'model':<14}{'OOF Macro F1':>15}{'singleton':>15}{'best epochs':>20}")
    for model in models:
        stem = f"dl_{model}_{tags[model]}_{CV_SLUG[cv]}"
        path = PROJECT_ROOT / "artifacts/logs" / f"{stem}.json"
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        best_epochs = [
            fold.get("best_epoch") for fold in result.get("fold_training", [])
        ]
        print(
            f"{model:<14}{result['oof_macro_f1']:>15.4f}"
            f"{result['oof_macro_f1_singleton']:>15.4f}"
            f"{str(best_epochs):>20}"
        )


def main() -> None:
    args = parse_args()
    models = list(dict.fromkeys(args.models))
    check_inputs(models, need_submission=not args.no_submission and not args.dry_run)
    tags = {
        model: experiment_tag(model, args.tag_prefix, args.seed) for model in models
    }
    for position, model in enumerate(models, start=1):
        expected = artifact_paths(
            model,
            tags[model],
            args.cv,
            include_submission=not args.no_submission,
        )
        if args.skip_existing and not args.dry_run and all(path.exists() for path in expected):
            print(f"[{position}/{len(models)}] {model}: all artifacts exist, skipping")
            continue
        command = build_train_command(
            model,
            cv=args.cv,
            device=args.device,
            seed=args.seed,
            tag=tags[model],
            dry_run=args.dry_run,
            no_submission=args.no_submission,
        )
        print(f"\n[{position}/{len(models)}] {model}", flush=True)
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if args.dry_run:
        print("\nDL ladder dry-run passed; artifacts were not written")
    else:
        print_summary(models, tags, args.cv)


if __name__ == "__main__":
    main()
