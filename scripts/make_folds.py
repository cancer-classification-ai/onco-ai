"""Fold 할당 스크립트.

train.csv를 읽어 Profile Hash 기반 StratifiedGroupKFold로
각 행에 fold 번호를 부여하고 Parquet로 저장한다.

Usage:
    python scripts/make_folds.py
    python scripts/make_folds.py --n-splits 10 --random-state 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cancer_hack.io import load_parquet, save_parquet
from cancer_hack.validation import assign_fold_column, fold_class_distribution

RAW_TRAIN = Path("data/raw/train.csv")
OUT_PATH = Path("data/interim/train_folds.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Group KFold 생성")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--label-col", type=str, default="SUBCLASS")
    parser.add_argument(
        "--out", type=Path, default=OUT_PATH, help="저장할 Parquet 경로"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train = pd.read_csv(RAW_TRAIN)
    gene_columns = [c for c in train.columns if c not in {"ID", args.label_col}]

    print(f"Train shape : {train.shape}")
    print(f"Gene columns: {len(gene_columns)}")
    print(f"n_splits    : {args.n_splits}")
    print(f"random_state: {args.random_state}")

    train_folds = assign_fold_column(
        train,
        gene_columns=gene_columns,
        label_column=args.label_col,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    dist = fold_class_distribution(
        train_folds,
        label_column=args.label_col,
        fold_column="fold",
    )
    print("\n[Fold × Class distribution (ratio %)]\n")
    print(
        dist["ratio"]
        .unstack(level=args.label_col)
        .round(2)
        .to_string()
    )

    save_parquet(train_folds, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
