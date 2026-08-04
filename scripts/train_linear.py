#!/usr/bin/env python
"""Team-facing fold-safe Logistic Regression training driver.

This runner reuses ``train_gbdt.Dataset`` and ``train_gbdt.run_config`` instead
of maintaining a second copy of the feature assembly code.  It therefore reads
the same ``data/process`` parquet files, uses the same fixed folds, and writes
the same OOF/test/submission artifact schemas as the GBDT runner.

Examples
--------
Prepare data once, then run the production baseline and full feature config::

    python scripts/train_linear.py --configs f4r,full_all --cv skf

Forward any feature-builder option understood by ``train_gbdt.py``::

    python scripts/train_linear.py --configs f4r,f4r_ptok --sparse-topk 500

The regularization strength is selected separately inside each outer fold on
the calibration config (default: ``f4r``), then reused for every requested
config in that fold.  Outer validation is never used for model selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.metrics import macro_f1  # noqa: E402
from cancer_hack.models_linear import SCALERS, create_logistic_model  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/linear.yaml"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"


def log(message: str) -> None:
    print(message, flush=True)


def atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def load_tabular_driver():
    """Load the existing GBDT driver as the shared feature/fold engine."""
    path = PROJECT_ROOT / "scripts/train_gbdt.py"
    spec = importlib.util.spec_from_file_location("linear_tabular_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 or not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("C grid must contain finite positive values")
    return sorted(set(values))


def parse_names(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one config is required")
    return list(dict.fromkeys(values))


class LinearFoldTrainer:
    """Model callback injected into ``train_gbdt.run_config``."""

    def __init__(
        self,
        *,
        calibration_config: str,
        c_grid: list[float],
        inner_valid_fraction: float,
        solver: str,
        scaler: str,
        max_iter: int,
        tol: float,
        seed: int,
        n_splits: int,
    ) -> None:
        self.calibration_config = calibration_config
        self.c_grid = c_grid
        self.inner_valid_fraction = inner_valid_fraction
        self.solver = solver
        self.scaler = scaler
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.n_splits = n_splits
        self.selected_c: dict[str, list[float]] = {}
        self.calibration: dict[str, list[dict]] = {}
        self.current_cv: str | None = None
        self.current_config: str | None = None
        self.fold = 0

    def begin_config(self, *, config: str, cv: str) -> None:
        self.current_config = config
        self.current_cv = cv
        self.fold = 0
        if config != self.calibration_config and len(
            self.selected_c.get(cv, [])
        ) != self.n_splits:
            raise RuntimeError(
                f"{cv} regularization must be calibrated with "
                f"{self.calibration_config!r} before {config!r}"
            )

    def _select_c(self, X: np.ndarray, y: np.ndarray, *, fold: int) -> dict:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.inner_valid_fraction,
            random_state=self.seed + fold,
        )
        inner_train, inner_valid = next(
            splitter.split(np.zeros(len(y)), y)
        )
        inner_weight = compute_sample_weight("balanced", y[inner_train])
        trials = []
        for C in self.c_grid:
            started = time.perf_counter()
            model = create_logistic_model(
                C=C,
                solver=self.solver,
                scaler=self.scaler,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.seed + fold,
            )
            model.fit(X[inner_train], y[inner_train], sample_weight=inner_weight)
            predicted = model.predict(X[inner_valid])
            trials.append(
                {
                    "C": C,
                    "macro_f1": macro_f1(y[inner_valid], predicted),
                    "elapsed_seconds": time.perf_counter() - started,
                    "model": model.describe(),
                }
            )
        selected = max(trials, key=lambda row: (row["macro_f1"], -row["C"]))
        return {
            "outer_fold": fold,
            "inner_train_rows": int(len(inner_train)),
            "inner_valid_rows": int(len(inner_valid)),
            "outer_validation_used": False,
            "selected_C": float(selected["C"]),
            "selected_macro_f1": float(selected["macro_f1"]),
            "trials": trials,
        }

    def __call__(self, args, X, y, sample_weight):
        if self.current_cv is None or self.current_config is None:
            raise RuntimeError("begin_config() must be called before fitting")
        fold = self.fold
        self.fold += 1
        if fold >= self.n_splits:
            raise RuntimeError(f"received more than {self.n_splits} fold fits")

        if self.current_config == self.calibration_config:
            detail = self._select_c(np.asarray(X), np.asarray(y), fold=fold)
            self.selected_c.setdefault(self.current_cv, []).append(
                float(detail["selected_C"])
            )
            self.calibration.setdefault(self.current_cv, []).append(detail)
        else:
            source = self.calibration[self.current_cv][fold]
            detail = {
                "outer_fold": fold,
                "selected_C": float(source["selected_C"]),
                "selection_source": self.calibration_config,
                "selection_cv": self.current_cv,
                "outer_validation_used": False,
            }

        model = create_logistic_model(
            C=detail["selected_C"],
            solver=self.solver,
            scaler=self.scaler,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=args.seed,
        )
        model.fit(X, y, sample_weight=sample_weight)
        model.calibration_ = detail
        return model

    def summary(self, cv: str) -> dict:
        return {
            "config": self.calibration_config,
            "fit_scope": "each_outer_fold_train_only",
            "outer_validation_used": False,
            "c_grid": self.c_grid,
            "inner_valid_fraction": self.inner_valid_fraction,
            "selected_C_per_fold": self.selected_c[cv],
            "folds": self.calibration[cv],
        }


def build_parser(config: dict) -> argparse.ArgumentParser:
    model = config.get("model", {})
    calibration = config.get("calibration", {})
    experiment = config.get("experiment", {})
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--configs",
        type=parse_names,
        default=list(experiment.get("configs", ["f4r", "full_all"])),
    )
    parser.add_argument(
        "--cv",
        choices=["skf", "sgkf", "all"],
        default=experiment.get("cv", "skf"),
    )
    parser.add_argument("--n-splits", type=int, default=int(experiment.get("n_splits", 5)))
    parser.add_argument("--seed", type=int, default=int(experiment.get("seed", 42)))
    parser.add_argument("--tag", default="logreg_v2")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--process-dir", type=Path)
    parser.add_argument(
        "--calibration-config",
        default=calibration.get("config", "f4r"),
    )
    parser.add_argument(
        "--c-grid",
        type=parse_float_list,
        default=[
            float(value)
            for value in calibration.get(
                "c_grid",
                [0.001, 0.003, 0.01, 0.03, 0.1],
            )
        ],
    )
    parser.add_argument(
        "--inner-valid-fraction",
        type=float,
        default=float(calibration.get("inner_valid_fraction", 0.15)),
    )
    parser.add_argument("--solver", default=model.get("solver", "lbfgs"))
    parser.add_argument("--scaler", choices=SCALERS, default=model.get("scaler", "standard"))
    parser.add_argument("--max-iter", type=int, default=int(model.get("max_iter", 1_000)))
    parser.add_argument("--tol", type=float, default=float(model.get("tol", 1e-4)))
    parser.add_argument(
        "--submission",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load and validate all requested feature inputs without training",
    )
    return parser


def driver_args(tg, args, forwarded: list[str]):
    values = [
        *forwarded,
        "--model",
        "logistic",
        "--configs",
        ",".join(args.configs),
        "--cv",
        args.cv,
        "--n-splits",
        str(args.n_splits),
        "--seed",
        str(args.seed),
        "--device",
        "cpu",
        "--tag",
        args.tag,
        "--submission" if args.submission else "--no-submission",
    ]
    parsed = tg.build_parser().parse_args(values)
    parsed.model = "logistic"
    parsed.device = False
    parsed.override = tg._parse_override(parsed.overrides)
    return parsed


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    known, _ = bootstrap.parse_known_args()
    config = load_yaml(known.config_file)
    args, forwarded = build_parser(config).parse_known_args()
    if args.n_splits <= 1:
        raise SystemExit("--n-splits must be at least 2")
    if not 0 < args.inner_valid_fraction < 1:
        raise SystemExit("--inner-valid-fraction must be between 0 and 1")

    tg = load_tabular_driver()
    unknown = sorted(
        set([*args.configs, args.calibration_config]) - set(tg.CONFIGS)
    )
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; available={list(tg.CONFIGS)}")
    if tg.CONFIGS[args.calibration_config]["weight"] != "balanced":
        raise SystemExit("calibration config must use fold-local balanced weight")

    if args.raw_dir is not None:
        tg.RAW_DIR = args.raw_dir.resolve()
    if args.process_dir is not None:
        tg.PROC_DIR = args.process_dir.resolve()
    tg.ARTIFACTS = args.artifacts.resolve()

    requested = list(args.configs)
    execution = [args.calibration_config, *requested]
    execution = list(dict.fromkeys(execution))
    needed = set().union(*(set(tg.CONFIGS[name]["blocks"]) for name in execution))
    log(f"[data] raw={tg.RAW_DIR} process={tg.PROC_DIR}")
    log(f"[configs] requested={requested} calibration={args.calibration_config}")
    data = tg.Dataset(needed, n_splits=args.n_splits)
    if args.dry_run:
        log("[dry-run] shared Dataset and all requested blocks are valid")
        return

    parsed = driver_args(tg, args, forwarded)
    trainer = LinearFoldTrainer(
        calibration_config=args.calibration_config,
        c_grid=args.c_grid,
        inner_valid_fraction=args.inner_valid_fraction,
        solver=args.solver,
        scaler=args.scaler,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        n_splits=args.n_splits,
    )
    cvs = ["skf", "sgkf"] if args.cv == "all" else [args.cv]
    results = []
    started = time.perf_counter()
    for cv in cvs:
        for config_name in execution:
            trainer.begin_config(config=config_name, cv=cv)
            log(f"\n=== logistic · {config_name} · CV {cv} ===")
            result = tg.run_config(
                data,
                config=config_name,
                cv=cv,
                args=parsed,
                fit_model=trainer,
            )
            result["linear_calibration"] = trainer.summary(cv)
            result["runtime_versions"] = {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            }
            atomic_json(
                tg.ARTIFACTS / "logs" / f"{result['stem']}.json",
                result,
            )
            if config_name in requested:
                results.append(result)

    summary = {
        "model": "logistic_regression",
        "requested_configs": requested,
        "calibration_config": args.calibration_config,
        "cv": cvs,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "results": [
            {
                "stem": row["stem"],
                "config": row["config"],
                "cv": row["cv"],
                "n_features_per_fold": row["n_features_per_fold"],
                "oof_macro_f1": row["oof_macro_f1"],
                "oof_macro_f1_singleton": row["oof_macro_f1_singleton"],
                "oof_accuracy": row["oof_accuracy"],
            }
            for row in results
        ],
    }
    summary_path = (
        tg.ARTIFACTS / "logs" / f"logistic_{args.tag}_summary_s{args.seed}.json"
    )
    atomic_json(summary_path, summary)

    log("\n" + "=" * 88)
    log(f"{'config':<14}{'CV':<7}{'features':>12}{'MacroF1':>12}{'singleton':>12}")
    log("-" * 88)
    for row in sorted(results, key=lambda item: -item["oof_macro_f1"]):
        log(
            f"{row['config']:<14}{row['cv']:<7}"
            f"{np.mean(row['n_features_per_fold']):>12,.1f}"
            f"{row['oof_macro_f1']:>12.6f}"
            f"{row['oof_macro_f1_singleton']:>12.6f}"
        )
    log("=" * 88)
    log(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
