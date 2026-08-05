from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_dl_loss_ladder.py"
    spec = importlib.util.spec_from_file_location("run_dl_loss_ladder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ladder_contains_requested_a_to_d_grid() -> None:
    ladder = _load_module()
    assert list(ladder.BASE_EXPERIMENTS) == [
        "a_weighted_ce",
        "b_ce_smooth002",
        "b_ce_smooth005",
        "c_focal_g1",
        "c_focal_g2",
        "d_focal_sqrt_g1",
        "d_focal_sqrt_g2",
    ]


def test_best_focal_uses_oof_then_singleton_tiebreak() -> None:
    ladder = _load_module()
    rows = [
        {
            "experiment": "c_focal_g1",
            "oof_macro_f1": 0.50,
            "singleton_macro_f1": 0.45,
        },
        {
            "experiment": "d_focal_sqrt_g1",
            "oof_macro_f1": 0.50,
            "singleton_macro_f1": 0.47,
        },
    ]
    assert ladder.select_best_focal(rows) == "d_focal_sqrt_g1"


def test_ladder_pins_one_latent_recipe_without_mutating_base() -> None:
    ladder = _load_module()
    base = {
        "model": "mlp",
        "dense_features": {
            "latent": {"enabled": True, "methods": ["svd", "nmf"]}
        },
    }
    configured = ladder.configure_latent_methods(base, ["svd"])
    assert configured["dense_features"]["latent"]["methods"] == ["svd"]
    assert base["dense_features"]["latent"]["methods"] == ["svd", "nmf"]


def test_train_command_uses_no_submission_and_stable_tag(tmp_path) -> None:
    ladder = _load_module()
    command = ladder.train_command(
        model="mlp",
        config=tmp_path / "config.yaml",
        experiment="a_weighted_ce",
        cv="skf",
        device="cuda",
        seed=42,
        dry_run=False,
    )
    assert "--no-submission" in command
    assert command[command.index("--tag") + 1] == "loss_a_weighted_ce_s42"
