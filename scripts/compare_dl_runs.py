#!/usr/bin/env python
"""Compare a Hybrid candidate JSON log with the fixed Hybrid baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASELINE_OOF_MACRO_F1 = 0.42429162605366255
BASELINE_SINGLETON_F1 = 0.41942993257160466
BASELINE_ACCURACY = 0.42670537010159654
BASELINE_FOLDS = [
    0.42490941096916524,
    0.41723778362489977,
    0.42391928905699310,
    0.42610789161461000,
    0.43129868936685710,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="Hybrid baseline JSON. Omit to use the recorded scalar/fold metrics.",
    )
    parser.add_argument("--candidate-log", type=Path, required=True)
    return parser.parse_args()


def load_log(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def baseline_fallback() -> dict[str, object]:
    return {
        "oof_macro_f1": BASELINE_OOF_MACRO_F1,
        "oof_macro_f1_singleton": BASELINE_SINGLETON_F1,
        "oof_accuracy": BASELINE_ACCURACY,
        "fold_macro_f1": BASELINE_FOLDS,
    }


def verdict(oof_delta: float, singleton_delta: float, improved_folds: int) -> str:
    if (
        oof_delta >= 0.005
        and improved_folds >= 3
        and singleton_delta >= -0.003
    ):
        return "ADOPT_CANDIDATE"
    if oof_delta <= 0 or improved_folds <= 2 or singleton_delta < -0.005:
        return "REJECT"
    if 0 < oof_delta < 0.005:
        return "WEAK_POSITIVE"
    return "REJECT"


def compare(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    scalar_keys = (
        "oof_macro_f1",
        "oof_macro_f1_singleton",
        "oof_accuracy",
    )
    missing = [
        f"{side}.{key}"
        for side, log in (("baseline", baseline), ("candidate", candidate))
        for key in scalar_keys
        if key not in log
    ]
    if missing:
        raise KeyError(f"missing metrics: {', '.join(missing)}")

    baseline_folds = np.asarray(baseline.get("fold_macro_f1", []), dtype=float)
    candidate_folds = np.asarray(candidate.get("fold_macro_f1", []), dtype=float)
    if baseline_folds.shape != candidate_folds.shape or baseline_folds.size == 0:
        raise ValueError("baseline/candidate fold_macro_f1 must have equal non-zero length")
    fold_delta = candidate_folds - baseline_folds
    improved = int((fold_delta > 0).sum())
    singleton_delta = float(candidate["oof_macro_f1_singleton"]) - float(
        baseline["oof_macro_f1_singleton"]
    )
    oof_delta = float(candidate["oof_macro_f1"]) - float(
        baseline["oof_macro_f1"]
    )

    result: dict[str, object] = {
        "baseline_oof_macro_f1": float(baseline["oof_macro_f1"]),
        "candidate_oof_macro_f1": float(candidate["oof_macro_f1"]),
        "oof_delta": oof_delta,
        "singleton_delta": singleton_delta,
        "accuracy_delta": float(candidate["oof_accuracy"])
        - float(baseline["oof_accuracy"]),
        "fold_delta": fold_delta.tolist(),
        "improved_folds": improved,
        "worsened_folds": int((fold_delta < 0).sum()),
        "baseline_fold_std": float(np.std(baseline_folds)),
        "candidate_fold_std": float(np.std(candidate_folds)),
        "verdict": verdict(oof_delta, singleton_delta, improved),
    }

    baseline_per_class = baseline.get("per_class_f1")
    candidate_per_class = candidate.get("per_class_f1")
    if isinstance(baseline_per_class, dict) and isinstance(candidate_per_class, dict):
        classes = sorted(set(baseline_per_class) & set(candidate_per_class))
        per_class_delta = {
            label: float(candidate_per_class[label]) - float(baseline_per_class[label])
            for label in classes
        }
        ordered = sorted(per_class_delta.items(), key=lambda item: item[1])
        result["per_class_f1_delta"] = per_class_delta
        result["top_improved_classes"] = list(reversed(ordered[-5:]))
        result["top_worsened_classes"] = ordered[:5]
    return result


def main() -> None:
    args = parse_args()
    baseline = (
        load_log(args.baseline_log)
        if args.baseline_log is not None
        else baseline_fallback()
    )
    result = compare(baseline, load_log(args.candidate_log))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "per_class_f1_delta" not in result:
        print(
            "per-class comparison unavailable: both JSON logs must contain "
            "per_class_f1 mappings"
        )


if __name__ == "__main__":
    main()
