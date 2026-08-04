from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_gene_rule_ladder.py"
    spec = importlib.util.spec_from_file_location("run_gene_rule_ladder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gene_rule_ablation_order_is_cumulative() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[1]
    expected = {
        "set_v1_count": {"use_counts", "use_duplicate_stats"},
        "set_v2_type": {
            "use_counts",
            "use_duplicate_stats",
            "use_type_ratios",
        },
        "set_v3_position": {
            "use_counts",
            "use_duplicate_stats",
            "use_type_ratios",
            "use_position_stats",
        },
        "set_v4_full": {
            "use_counts",
            "use_duplicate_stats",
            "use_type_ratios",
            "use_position_stats",
            "use_state_flags",
        },
    }
    for experiment, wanted in expected.items():
        _, config_path = module.EXPERIMENTS[experiment]
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        rules = config["model_params"]["gene_rule_features"]
        active = {
            name
            for name, value in rules.items()
            if name.startswith("use_") and value
        }
        assert active == wanted
        assert config_path.is_relative_to(root)


def test_gene_rule_command_uses_experiment_specific_tag() -> None:
    module = _load_module()
    command = module.build_command(
        "hybrid_v2",
        cv="skf",
        device="cuda",
        seed=2025,
        dry_run=False,
        no_submission=False,
    )

    assert command[0] == sys.executable
    assert command[command.index("--model") + 1] == "hybrid"
    assert command[command.index("--config") + 1].endswith(
        "configs/hybrid_set_mlp_v2.yaml"
    )
    assert command[command.index("--tag") + 1] == "hybrid_v2_s2025"
