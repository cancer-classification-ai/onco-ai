from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_ladder_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_dl_ladder.py"
    spec = importlib.util.spec_from_file_location("run_dl_ladder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_ladder_command_uses_model_specific_config_and_tag() -> None:
    ladder = _load_ladder_module()
    tag = ladder.experiment_tag("mlp", "full", 42)
    command = ladder.build_train_command(
        "mlp",
        cv="skf",
        device="cuda",
        seed=42,
        tag=tag,
        dry_run=False,
        no_submission=False,
    )

    assert command[0] == sys.executable
    assert command[command.index("--model") + 1] == "mlp"
    assert command[command.index("--config") + 1].endswith("configs/mlp.yaml")
    assert command[command.index("--tag") + 1] == "mlp_full_s42"
    assert "--dry-run" not in command
    assert "--no-submission" not in command


def test_ladder_artifacts_keep_mlp_and_hybrid_separate() -> None:
    ladder = _load_ladder_module()
    mlp = ladder.artifact_paths(
        "mlp", "mlp_full_s42", "skf", include_submission=True
    )
    hybrid = ladder.artifact_paths(
        "hybrid", "hybrid_full_s42", "skf", include_submission=True
    )

    assert len(mlp) == len(hybrid) == 4
    assert not set(mlp).intersection(hybrid)
    assert any("oof_dl_mlp_mlp_full_s42_skf5.csv" in str(path) for path in mlp)
    assert any(
        "submission_dl_hybrid_hybrid_full_s42_skf5.csv" in str(path)
        for path in hybrid
    )
