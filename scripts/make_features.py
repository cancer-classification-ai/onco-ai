#!/usr/bin/env python
"""Feature generation entrypoint shared by the team.

Keep feature-specific functions namespaced by feature name so additional
builders can be added without changing existing behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXACT_MUTATION_EMPTY_VALUES = frozenset({"", "WT", "0", "NA", "NAN", "NONE", "."})
EXACT_MUTATION_NO_MUTATION_TOKEN = "SAMPLE__NO_MUTATION"


def row_to_exact_mutation_document(
    row: pd.Series,
    gene_columns: list[str],
) -> str:
    """Convert one sample into a space-delimited Exact Mutation Token document."""
    tokens: list[str] = []

    for gene in gene_columns:
        value = row[gene]
        if pd.isna(value):
            continue

        mutation = str(value).strip()
        if mutation.upper() in EXACT_MUTATION_EMPTY_VALUES:
            continue

        mutation = mutation.replace(" ", "_")
        tokens.append(f"MUT__{gene}__{mutation}")

    if not tokens:
        return EXACT_MUTATION_NO_MUTATION_TOKEN
    return " ".join(tokens)


def build_exact_mutation_documents(
    frame: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    """Build one Exact Mutation Token document for every input sample."""
    missing_columns = sorted(set(gene_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing gene columns: {missing_columns[:10]}")

    return frame.apply(
        row_to_exact_mutation_document,
        axis=1,
        gene_columns=gene_columns,
    )


def make_exact_mutation_token_parquet(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int | str]:
    """Read train CSV and write ID/label/document columns to one Parquet file."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    train = pd.read_csv(input_path)
    required_columns = {"ID", "SUBCLASS"}
    missing_required = sorted(required_columns.difference(train.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    if not train["ID"].is_unique:
        raise ValueError("Train IDs must be unique")

    gene_columns = [
        column for column in train.columns if column not in required_columns
    ]
    if not gene_columns:
        raise ValueError("No gene columns found")

    documents = build_exact_mutation_documents(train, gene_columns)
    output = pd.DataFrame(
        {
            "ID": train["ID"].astype(str),
            "SUBCLASS": train["SUBCLASS"].astype(str),
            "exact_mutation_document": documents,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.parquet")
    output.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sample_count": len(output),
        "gene_count": len(gene_columns),
        "no_mutation_count": int(
            (documents == EXACT_MUTATION_NO_MUTATION_TOKEN).sum()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/train.csv",
        help="Input train CSV containing ID, SUBCLASS, and gene columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/process/train_exact_mutation_tokens.parquet",
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = make_exact_mutation_token_parquet(
        args.input,
        args.output,
        overwrite=args.overwrite,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
