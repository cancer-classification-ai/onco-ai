#!/usr/bin/env python
"""Run cumulative gene-rule ablations and the enhanced Hybrid on Kaggle.

    python scripts/run_gene_rule_ladder.py --device cuda --seed 42
    python scripts/run_gene_rule_ladder.py --experiments set_v4_full hybrid_v2 --device cuda
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# 경로는 `cancer_hack.paths` 가 정한다 — 값은 쓰는 시점에 정해진다.
from cancer_hack.paths import artifacts_dir  # noqa: E402
from cancer_hack.validation import CV_SLUG  # noqa: E402
from run_dl_ladder import check_inputs  # noqa: E402

EXPERIMENTS: dict[str, tuple[str, Path]] = {
    "set_v0": ("set_encoder", PROJECT_ROOT / "configs/gene_set_encoder.yaml"),
    "set_v1_count": (
        "set_encoder",
        PROJECT_ROOT / "configs/gene_set_encoder_count.yaml",
    ),
    "set_v2_type": (
        "set_encoder",
        PROJECT_ROOT / "configs/gene_set_encoder_type.yaml",
    ),
    "set_v3_position": (
        "set_encoder",
        PROJECT_ROOT / "configs/gene_set_encoder_position.yaml",
    ),
    "set_v4_full": (
        "set_encoder",
        PROJECT_ROOT / "configs/gene_set_encoder_full.yaml",
    ),
    "hybrid_v2": ("hybrid", PROJECT_ROOT / "configs/hybrid_set_mlp_v2.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--cv", choices=["skf", "sgkf"], default="skf")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-submission", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_command(
    experiment: str,
    *,
    cv: str,
    device: str,
    seed: int,
    dry_run: bool,
    no_submission: bool,
) -> list[str]:
    model, config = EXPERIMENTS[experiment]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_dl.py"),
        "--model",
        model,
        "--config",
        str(config),
        "--cv",
        cv,
        "--device",
        device,
        "--seed",
        str(seed),
        "--tag",
        f"{experiment}_s{seed}",
    ]
    if dry_run:
        command.append("--dry-run")
    if no_submission:
        command.append("--no-submission")
    return command


def main() -> None:
    args = parse_args()
    experiments = list(dict.fromkeys(args.experiments))
    models = [EXPERIMENTS[name][0] for name in experiments]
    check_inputs(models, need_submission=not args.no_submission and not args.dry_run)
    results: list[tuple[str, dict[str, object]]] = []
    for index, experiment in enumerate(experiments, start=1):
        model, _ = EXPERIMENTS[experiment]
        tag = f"{experiment}_s{args.seed}"
        stem = f"dl_{model}_{tag}_{CV_SLUG[args.cv]}"
        expected = [
            artifacts_dir() / "oof" / f"oof_{stem}.csv",
            artifacts_dir() / "test_predictions" / f"test_{stem}.csv",
            artifacts_dir() / "logs" / f"{stem}.json",
        ]
        if not args.no_submission:
            expected.append(
                artifacts_dir() / "submissions"
                / f"submission_{stem}.csv"
            )
        if args.skip_existing and not args.dry_run and all(path.exists() for path in expected):
            print(f"[{index}/{len(experiments)}] {experiment}: skipping existing")
        else:
            command = build_command(
                experiment,
                cv=args.cv,
                device=args.device,
                seed=args.seed,
                dry_run=args.dry_run,
                no_submission=args.no_submission,
            )
            print(f"\n[{index}/{len(experiments)}] {experiment}", flush=True)
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if not args.dry_run:
            with expected[2].open(encoding="utf-8") as handle:
                results.append((experiment, json.load(handle)))

    if args.dry_run:
        print("\nGene-rule ladder dry-run passed")
        return
    print("\nGene-rule ablation summary")
    print(f"{'experiment':<18}{'OOF Macro F1':>15}{'singleton':>15}")
    for experiment, result in results:
        print(
            f"{experiment:<18}{result['oof_macro_f1']:>15.4f}"
            f"{result['oof_macro_f1_singleton']:>15.4f}"
        )


if __name__ == "__main__":
    main()
