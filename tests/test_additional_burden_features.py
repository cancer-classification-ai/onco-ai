"""추가 burden 파생변수의 계산식과 opt-in 스키마 계약."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cancer_hack.features_basic import (
    ADDITIONAL_BURDEN_FEATURE_COLUMNS,
    SAMPLE_FEATURE_COLUMNS,
    make_sample_mutation_features,
)


def _additional_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["s1", "s2"],
            "SUBCLASS": ["A", "B"],
            "TP53": ["Q369* Q369* I368N", "WT"],
            "KRAS": ["K16fs", "WT"],
            "EGFR": ["P11_K12insP", "WT"],
        },
        index=[10, 20],
    )


def test_additional_burden_schema_is_fixed():
    assert ADDITIONAL_BURDEN_FEATURE_COLUMNS == (
        "events_per_mutated_gene",
        "loss_of_function_ratio",
        "indel_ratio",
        "complex_ratio",
        "duplicate_event_count",
        "duplicate_event_ratio",
        "high_multihit_gene_count",
        "max_gene_event_share",
    )
    assert "singleton_gene_count" not in ADDITIONAL_BURDEN_FEATURE_COLUMNS


def test_default_sample_schema_is_unchanged():
    frame = _additional_frame()
    out = make_sample_mutation_features(
        frame, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    assert tuple(out.columns) == SAMPLE_FEATURE_COLUMNS


def test_additional_values_match_definitions():
    frame = _additional_frame()
    out = make_sample_mutation_features(
        frame,
        gene_columns=["TP53", "KRAS", "EGFR"],
        include_additional_burden=True,
    )

    assert list(out.columns[-8:]) == list(ADDITIONAL_BURDEN_FEATURE_COLUMNS)
    assert out.loc[10, "events_per_mutated_gene"] == pytest.approx(5 / 3)
    assert out.loc[10, "loss_of_function_ratio"] == pytest.approx(3 / 5)
    assert out.loc[10, "indel_ratio"] == pytest.approx(1 / 5)
    assert out.loc[10, "complex_ratio"] == 0.0
    assert int(out.loc[10, "duplicate_event_count"]) == 1
    assert out.loc[10, "duplicate_event_ratio"] == pytest.approx(1 / 5)
    assert int(out.loc[10, "high_multihit_gene_count"]) == 1
    assert out.loc[10, "max_gene_event_share"] == pytest.approx(3 / 5)


def test_duplicate_identity_is_gene_aware():
    frame = pd.DataFrame(
        {
            "TP53": ["V600E V600E"],
            "KRAS": ["V600E"],
        }
    )
    out = make_sample_mutation_features(
        frame,
        gene_columns=["TP53", "KRAS"],
        include_additional_burden=True,
    )
    # TP53 내부의 두 V600E 중 하나만 duplicate이고 KRAS V600E는 별개 token이다.
    assert int(out.loc[0, "mutation_event_count"]) == 3
    assert int(out.loc[0, "duplicate_event_count"]) == 1


def test_zero_mutation_row_has_finite_zeros():
    frame = _additional_frame()
    out = make_sample_mutation_features(
        frame,
        gene_columns=["TP53", "KRAS", "EGFR"],
        include_additional_burden=True,
    )
    values = out.loc[20, list(ADDITIONAL_BURDEN_FEATURE_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    assert np.isfinite(values).all()
    assert values.sum() == 0.0


def test_additional_and_cell_rollup_can_be_enabled_together():
    frame = _additional_frame()
    out = make_sample_mutation_features(
        frame,
        gene_columns=["TP53", "KRAS", "EGFR"],
        include_additional_burden=True,
        include_cell_rollup=True,
    )
    assert all(column in out.columns for column in ADDITIONAL_BURDEN_FEATURE_COLUMNS)
    assert "unique_mutation_token_count" in out.columns


def test_additional_burden_cli_writes_parquet(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "train_sample_mutation_features_additional.parquet"
    frame = _additional_frame().reset_index(drop=True)
    frame.to_csv(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/make_features.py"),
            "sample",
            "--split",
            "train",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--include-additional-burden",
            "--overwrite",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    saved = pd.read_parquet(output_path)
    assert "additional_burden: True" in result.stdout
    assert all(
        column in saved.columns for column in ADDITIONAL_BURDEN_FEATURE_COLUMNS
    )
