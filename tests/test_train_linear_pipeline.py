from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_train_linear():
    path = PROJECT_ROOT / "scripts/train_linear.py"
    spec = importlib.util.spec_from_file_location("test_train_linear_pipeline_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_multiclass(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(150, 6))
    scores = np.column_stack(
        [
            matrix[:, 0] - matrix[:, 1],
            matrix[:, 1] + matrix[:, 2],
            -matrix[:, 0] - matrix[:, 2],
        ]
    )
    labels = np.asarray(["ACC", "BRCA", "DLBC"])[scores.argmax(axis=1)]
    return matrix, labels


def test_calibration_c_is_reused_for_other_configs():
    train_linear = load_train_linear()
    matrix, labels = toy_multiclass()
    weight = compute_sample_weight("balanced", labels)
    trainer = train_linear.LinearFoldTrainer(
        calibration_config="f4r",
        c_grid=[0.01, 0.1],
        inner_valid_fraction=0.2,
        solver="lbfgs",
        scaler="standard",
        max_iter=500,
        tol=1e-4,
        seed=42,
        n_splits=1,
    )
    args = SimpleNamespace(seed=42)

    trainer.begin_config(config="f4r", cv="skf")
    baseline = trainer(args, matrix, labels, weight)
    chosen = baseline.calibration_["selected_C"]

    trainer.begin_config(config="full_all", cv="skf")
    candidate = trainer(args, matrix, labels, weight)
    assert candidate.calibration_["selected_C"] == chosen
    assert candidate.calibration_["selection_source"] == "f4r"
    assert trainer.summary("skf")["outer_validation_used"] is False


def test_tabular_driver_exposes_injected_model_fit():
    train_linear = load_train_linear()
    driver = train_linear.load_tabular_driver()
    assert "fit_model" in driver.run_config.__annotations__ or (
        "fit_model" in __import__("inspect").signature(driver.run_config).parameters
    )


def test_runtime_config_names_are_parsed_once():
    train_linear = load_train_linear()
    assert train_linear.parse_names("f4r,full_all,f4r") == ["f4r", "full_all"]
