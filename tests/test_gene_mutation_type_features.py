"""유전자별 변이 유형 wide matrix의 스키마·값·정렬 계약."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cancer_hack.features_basic import (
    GENE_MUTATION_TYPES,
    make_gene_mutation_type_matrix,
)


def test_gene_mutation_type_schema_is_fixed():
    assert GENE_MUTATION_TYPES == (
        "missense",
        "nonsense",
        "frameshift",
        "indel",
        "synonymous",
        "complex",
    )


def test_columns_are_gene_major_and_type_ordered(toy_frame):
    genes = ["TP53", "KRAS"]
    out = make_gene_mutation_type_matrix(toy_frame, gene_columns=genes)
    expected = [
        f"gene_{mutation_type}__{gene}"
        for gene in genes
        for mutation_type in GENE_MUTATION_TYPES
    ]
    assert list(out.columns) == expected
    assert out.shape == (len(toy_frame), len(genes) * len(GENE_MUTATION_TYPES))


def test_each_parser_type_reaches_its_gene_column():
    frame = pd.DataFrame(
        {
            "TP53": [
                "G827R",
                "Q369*",
                "K16fs",
                "P11_K12insP",
                "S622S",
                "312_313QY>HH",
                "WT",
            ]
        }
    )
    out = make_gene_mutation_type_matrix(frame, gene_columns=["TP53"])

    for row_idx, mutation_type in enumerate(GENE_MUTATION_TYPES):
        assert int(out.loc[row_idx, f"gene_{mutation_type}__TP53"]) == 1
        assert int(out.loc[row_idx].sum()) == 1
    assert int(out.loc[6].sum()) == 0


def test_multi_type_and_duplicate_tokens_are_binary():
    frame = pd.DataFrame(
        {
            "TP53": [
                "Q369* I368N",
                "V600E V600E",
            ]
        }
    )
    out = make_gene_mutation_type_matrix(frame, gene_columns=["TP53"])

    assert int(out.loc[0, "gene_nonsense__TP53"]) == 1
    assert int(out.loc[0, "gene_missense__TP53"]) == 1
    assert int(out.loc[0].sum()) == 2
    assert int(out.loc[1, "gene_missense__TP53"]) == 1
    assert int(out.loc[1].sum()) == 1


def test_output_is_int8_and_preserves_index(toy_frame):
    selected = toy_frame.iloc[[2, 0]]
    out = make_gene_mutation_type_matrix(
        selected, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    assert list(out.index) == list(selected.index)
    assert all(dtype == np.dtype("int8") for dtype in out.dtypes)


def test_train_and_test_have_identical_feature_columns(toy_frame):
    genes = ["TP53", "KRAS", "EGFR"]
    train_out = make_gene_mutation_type_matrix(toy_frame, gene_columns=genes)
    test_out = make_gene_mutation_type_matrix(
        toy_frame.drop(columns=["SUBCLASS"]),
        gene_columns=genes,
    )
    assert list(train_out.columns) == list(test_out.columns)


def test_missing_gene_column_raises(toy_frame):
    with pytest.raises(ValueError, match="Missing gene columns"):
        make_gene_mutation_type_matrix(
            toy_frame, gene_columns=["TP53", "NOT_A_GENE"]
        )


def test_gene_types_cli_writes_id_and_wide_features(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "gene_types.parquet"
    pd.DataFrame(
        {
            "ID": ["s1", "s2"],
            "SUBCLASS": ["A", "B"],
            "TP53": ["Q369* I368N", "WT"],
            "KRAS": ["WT", "G12D"],
        }
    ).to_csv(input_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/make_features.py"),
            "gene-types",
            "--split",
            "train",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    saved = pd.read_parquet(output_path)
    assert saved["ID"].tolist() == ["s1", "s2"]
    assert "SUBCLASS" not in saved.columns
    assert int(saved.loc[0, "gene_nonsense__TP53"]) == 1
    assert int(saved.loc[0, "gene_missense__TP53"]) == 1
    assert int(saved.loc[1, "gene_missense__KRAS"]) == 1
