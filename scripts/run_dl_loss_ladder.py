#!/usr/bin/env python
"""Run a fold-safe DL loss ladder and write a ranked Kaggle comparison table.

The dense feature recipe stays fixed; only the loss mapping changes.  A-D run
first, then E inherits the best C/D focal setup and adds label smoothing 0.02.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.validation import CV_SLUG  # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"
CONFIG_DIR = ARTIFACTS / "configs" / "dl_loss_ladder"
COMPARISON_DIR = ARTIFACTS / "comparisons"

BASE_EXPERIMENTS: dict[str, dict[str, object]] = {
    "a_weighted_ce": {
        "name": "cross_entropy",
        "class_weight": "balanced",
        "label_smoothing": 0.0,
        "focal_gamma": 0.0,
    },
    "b_ce_smooth002": {
        "name": "cross_entropy",
        "class_weight": "balanced",
        "label_smoothing": 0.02,
        "focal_gamma": 0.0,
    },
    "b_ce_smooth005": {
        "name": "cross_entropy",
        "class_weight": "balanced",
        "label_smoothing": 0.05,
        "focal_gamma": 0.0,
    },
    "c_focal_g1": {
        "name": "focal",
        "class_weight": "none",
        "label_smoothing": 0.0,
        "focal_gamma": 1.0,
    },
    "c_focal_g2": {
        "name": "focal",
        "class_weight": "none",
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
    },
    "d_focal_sqrt_g1": {
        "name": "focal",
        "class_weight": "sqrt_balanced",
        "label_smoothing": 0.0,
        "focal_gamma": 1.0,
    },
    "d_focal_sqrt_g2": {
        "name": "focal",
        "class_weight": "sqrt_balanced",
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
    },
}
FOCAL_CANDIDATES = tuple(
    name for name in BASE_EXPERIMENTS if name.startswith(("c_", "d_"))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "hybrid"], default="mlp")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "configs/mlp_full_features.yaml",
    )
    parser.add_argument(
        "--latent-methods",
        nargs="+",
        choices=["svd", "nmf"],
        default=["svd"],
        help="fixed latent representation used by every loss experiment",
    )
    parser.add_argument("--cv", choices=["skf", "sgkf"], default="skf")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(BASE_EXPERIMENTS),
        default=list(BASE_EXPERIMENTS),
        help="A-D experiments to run; E is added from the best available C/D run",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_base_config(path: Path, model: str) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if config.get("model") != model:
        raise ValueError(
            f"base config model {config.get('model')!r} differs from {model!r}"
        )
    return config


def configure_latent_methods(
    base: dict[str, object],
    methods: list[str],
) -> dict[str, object]:
    """Pin one latent recipe for the whole loss ladder."""
    config = json.loads(json.dumps(base))
    dense = dict(config.get("dense_features", {}))
    latent = dict(dense.get("latent", {}))
    if not bool(latent.get("enabled", False)):
        raise ValueError("loss ladder requires an enabled latent config")
    latent["methods"] = list(dict.fromkeys(methods))
    dense["latent"] = latent
    config["dense_features"] = dense
    return config


def write_experiment_config(
    base: dict[str, object],
    experiment: str,
    loss: dict[str, object],
) -> Path:
    config = json.loads(json.dumps(base))
    training = dict(config.get("training", {}))
    training["loss"] = loss
    training.pop("class_weight", None)
    config["training"] = training
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{experiment}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return path


def artifact_tag(experiment: str, seed: int) -> str:
    return f"loss_{experiment}_s{seed}"


def log_path(model: str, experiment: str, cv: str, seed: int) -> Path:
    stem = f"dl_{model}_{artifact_tag(experiment, seed)}_{CV_SLUG[cv]}"
    return ARTIFACTS / "logs" / f"{stem}.json"


def train_command(
    *,
    model: str,
    config: Path,
    experiment: str,
    cv: str,
    device: str,
    seed: int,
    dry_run: bool,
) -> list[str]:
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
        artifact_tag(experiment, seed),
        "--no-submission",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def run_experiment(
    *,
    base: dict[str, object],
    model: str,
    experiment: str,
    loss: dict[str, object],
    cv: str,
    device: str,
    seed: int,
    dry_run: bool,
    skip_existing: bool,
) -> Path | None:
    output = log_path(model, experiment, cv, seed)
    if skip_existing and output.exists() and not dry_run:
        print(f"[skip] {experiment}: {output}")
        return output
    config = write_experiment_config(base, experiment, loss)
    command = train_command(
        model=model,
        config=config,
        experiment=experiment,
        cv=cv,
        device=device,
        seed=seed,
        dry_run=dry_run,
    )
    print(f"\n[run] {experiment}: {loss}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return None if dry_run else output


def load_result(path: Path, experiment: str) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    support = {str(k): int(v) for k, v in result["support"].items()}
    per_class = {str(k): float(v) for k, v in result["per_class_f1"].items()}
    rare_classes = sorted(support, key=lambda name: (support[name], name))[:5]
    folds = np.asarray(result["fold_macro_f1"], dtype=float)
    return {
        "experiment": experiment,
        "loss": result["loss"]["name"],
        "class_weight": result["loss"]["class_weight"],
        "smoothing": float(result["loss"]["label_smoothing"]),
        "gamma": float(result["loss"]["focal_gamma"]),
        "oof_macro_f1": float(result["oof_macro_f1"]),
        "singleton_macro_f1": float(result["oof_macro_f1_singleton"]),
        "fold_mean": float(folds.mean()),
        "fold_std": float(folds.std()),
        "accuracy": float(result["oof_accuracy"]),
        "rare5_macro_f1": float(np.mean([per_class[name] for name in rare_classes])),
        "rare5_classes": ",".join(rare_classes),
        "elapsed_minutes": float(result["elapsed_seconds"]) / 60.0,
    }


def select_best_focal(rows: list[dict[str, object]]) -> str:
    candidates = [
        row for row in rows if str(row["experiment"]) in FOCAL_CANDIDATES
    ]
    if not candidates:
        raise ValueError("E requires at least one completed C/D focal experiment")
    best = max(
        candidates,
        key=lambda row: (
            float(row["oof_macro_f1"]),
            float(row["singleton_macro_f1"]),
        ),
    )
    return str(best["experiment"])


def comparison_paths(model: str, cv: str, seed: int) -> tuple[Path, Path]:
    stem = f"dl_loss_ladder_{model}_{CV_SLUG[cv]}_s{seed}"
    return COMPARISON_DIR / f"{stem}.csv", COMPARISON_DIR / f"{stem}.md"


def write_comparison(
    rows: list[dict[str, object]],
    *,
    model: str,
    cv: str,
    seed: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(
        ["oof_macro_f1", "singleton_macro_f1"],
        ascending=False,
    )
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    csv_path, markdown_path = comparison_paths(model, cv, seed)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    markdown_path.write_text(
        "\n".join([header, divider, *body]) + "\n",
        encoding="utf-8",
    )
    print(f"\n{frame.to_string(index=False)}")
    print(f"\nwrote {csv_path}")
    print(f"wrote {markdown_path}")
    return frame


def main() -> None:
    args = parse_args()
    base = configure_latent_methods(
        load_base_config(args.base_config, args.model),
        args.latent_methods,
    )
    completed: dict[str, Path] = {}
    for experiment in dict.fromkeys(args.experiments):
        path = run_experiment(
            base=base,
            model=args.model,
            experiment=experiment,
            loss=BASE_EXPERIMENTS[experiment],
            cv=args.cv,
            device=args.device,
            seed=args.seed,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
        if path is not None:
            completed[experiment] = path

    if args.dry_run:
        print("\nloss ladder dry-run passed; metrics table requires full runs")
        return

    rows = [
        load_result(path, experiment)
        for experiment, path in completed.items()
    ]
    available_focal = set(completed).intersection(FOCAL_CANDIDATES)
    if available_focal:
        best_name = select_best_focal(rows)
        e_name = f"e_{best_name}_smooth002"
        e_loss = dict(BASE_EXPERIMENTS[best_name])
        e_loss["label_smoothing"] = 0.02
        e_path = run_experiment(
            base=base,
            model=args.model,
            experiment=e_name,
            loss=e_loss,
            cv=args.cv,
            device=args.device,
            seed=args.seed,
            dry_run=False,
            skip_existing=args.skip_existing,
        )
        assert e_path is not None
        rows.append(load_result(e_path, e_name))

    write_comparison(rows, model=args.model, cv=args.cv, seed=args.seed)


if __name__ == "__main__":
    main()
