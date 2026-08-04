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
    assert ladder.DEFAULT_MODELS == ("mlp", "set_encoder", "hybrid")
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


def test_ladder_artifacts_keep_all_models_separate() -> None:
    ladder = _load_ladder_module()
    mlp = ladder.artifact_paths(
        "mlp", "mlp_full_s42", "skf", include_submission=True
    )
    set_encoder = ladder.artifact_paths(
        "set_encoder", "set_encoder_full_s42", "skf", include_submission=True
    )
    hybrid = ladder.artifact_paths(
        "hybrid", "hybrid_full_s42", "skf", include_submission=True
    )

    assert len(mlp) == len(set_encoder) == len(hybrid) == 4
    assert not set(mlp).intersection(set_encoder)
    assert not set(mlp).intersection(hybrid)
    assert not set(set_encoder).intersection(hybrid)
    assert any("oof_dl_mlp_mlp_full_s42_skf5.csv" in str(path) for path in mlp)
    assert any(
        "test_dl_set_encoder_set_encoder_full_s42_skf5.csv" in str(path)
        for path in set_encoder
    )
    assert any(
        "submission_dl_hybrid_hybrid_full_s42_skf5.csv" in str(path)
        for path in hybrid
    )
