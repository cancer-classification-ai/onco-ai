#!/usr/bin/env python
"""팀 공용 피처 생성 CLI.

피처 정의는 전부 `cancer_hack.features_basic` 에 있고 이 스크립트는 껍데기다.
예전에는 Exact Mutation Token 코드가 여기에도 복사돼 있어서 한쪽만 고치면 두 정의가
어긋났다. 그래서 import 로 바꿨다 — 고칠 곳은 패키지 한 군데뿐이다.

    python scripts/make_features.py sample --split train --overwrite
    python scripts/make_features.py sample --split test  --overwrite
    python scripts/make_features.py sample --split train --include-additional-burden --overwrite
    python scripts/make_features.py tokens --split train --overwrite
    python scripts/make_features.py gene   --split train --kind mutated --overwrite
    python scripts/make_features.py gene-types --split train --overwrite
    python scripts/make_features.py gene-types --split test  --overwrite

`sample` 이 복합 변이 처리 전략의 산출물이다. train/test 를 **따로** 돌린다 —
행마다 독립 계산이라 train 통계가 test 로 새지 않는다. fold 안에서 잡아야 하는
`hypermutated_flag`·`burden_quantile_bin` 은 여기서 만들지 않는다. 학습 스크립트가
`BurdenBinner` 를 fold 학습 부분에 fit 해서 붙여야 한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.features_basic import (  # noqa: E402
    GENE_MUTATION_TYPES,
    SAMPLE_FEATURE_COLUMNS,
    make_exact_mutation_token_parquet,
    make_gene_event_count_matrix,
    make_gene_mutation_type_matrix,
    make_gene_mutated_matrix,
    make_sample_mutation_features,
)

RAW_DIR = PROJECT_ROOT / "data/raw"
OUT_DIR = PROJECT_ROOT / "data/process"


def _resolve_output(explicit: Path | None, default_name: str) -> Path:
    return explicit if explicit is not None else OUT_DIR / default_name


def _guard_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_split(split: str, input_path: Path | None) -> tuple[pd.DataFrame, list[str]]:
    path = input_path if input_path is not None else RAW_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없다. 원본 csv 는 git 에 없으니 data/raw 에 먼저 놓는다."
        )
    # 변이 문자열을 그대로 받아야 하므로 dtype=str, 빈 칸도 NaN 으로 바꾸지 않는다.
    frame = pd.read_csv(path, dtype=str, na_filter=False)
    genes = [c for c in frame.columns if c not in ("ID", "SUBCLASS")]
    return frame, genes


def cmd_sample(args: argparse.Namespace) -> dict[str, object]:
    """복합 변이 샘플 단위 피처."""
    suffix = (
        "_additional" if args.include_additional_burden else ""
    )
    output = _resolve_output(
        args.output,
        f"{args.split}_sample_mutation_features{suffix}.parquet",
    )
    _guard_existing(output, args.overwrite)

    frame, genes = _load_split(args.split, args.input)
    features = make_sample_mutation_features(
        frame,
        gene_columns=genes,
        include_cell_rollup=args.include_cell_rollup,
        include_additional_burden=args.include_additional_burden,
    )
    features.insert(0, "ID", frame["ID"].astype(str).to_numpy())
    if "SUBCLASS" in frame.columns:
        features.insert(1, "SUBCLASS", frame["SUBCLASS"].astype(str).to_numpy())

    _write_parquet(features, output)
    return {
        "output_path": str(output),
        "sample_count": len(features),
        "gene_count": len(genes),
        "feature_count": len(features.columns) - (1 + int("SUBCLASS" in frame.columns)),
        "cell_rollup": args.include_cell_rollup,
        "additional_burden": args.include_additional_burden,
    }


def cmd_gene(args: argparse.Namespace) -> dict[str, object]:
    """유전자별 행렬 — gene_mutated__* 또는 gene_event_count__*."""
    output = _resolve_output(
        args.output, f"{args.split}_gene_{args.kind}_matrix.parquet"
    )
    _guard_existing(output, args.overwrite)

    frame, genes = _load_split(args.split, args.input)
    builder = (
        make_gene_mutated_matrix if args.kind == "mutated" else make_gene_event_count_matrix
    )
    matrix = builder(frame, gene_columns=genes)
    matrix.insert(0, "ID", frame["ID"].astype(str).to_numpy())

    _write_parquet(matrix, output)
    return {
        "output_path": str(output),
        "sample_count": len(matrix),
        "column_count": len(matrix.columns) - 1,
    }


def cmd_tokens(args: argparse.Namespace) -> dict[str, object]:
    """Exact Mutation Token 문서 — 셀 안의 변이를 토큰별로 나눠 적는다."""
    output = _resolve_output(
        args.output, f"{args.split}_exact_mutation_tokens.parquet"
    )
    source = args.input if args.input is not None else RAW_DIR / f"{args.split}.csv"
    return make_exact_mutation_token_parquet(
        source, output, overwrite=args.overwrite
    )


def cmd_gene_types(args: argparse.Namespace) -> dict[str, object]:
    """유전자별 6종 변이 유형 존재 여부 행렬."""
    output = _resolve_output(
        args.output, f"{args.split}_gene_mutation_type_matrix.parquet"
    )
    _guard_existing(output, args.overwrite)

    frame, genes = _load_split(args.split, args.input)
    matrix = make_gene_mutation_type_matrix(frame, gene_columns=genes)
    matrix.insert(0, "ID", frame["ID"].astype(str).to_numpy())

    _write_parquet(matrix, output)
    return {
        "output_path": str(output),
        "sample_count": len(matrix),
        "gene_count": len(genes),
        "type_count": len(GENE_MUTATION_TYPES),
        "feature_count": len(matrix.columns) - 1,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--split", choices=["train", "test"], default="train")
        p.add_argument("--input", type=Path, default=None, help="원본 csv 경로 직접 지정")
        p.add_argument("--output", type=Path, default=None, help="출력 parquet 경로")
        p.add_argument("--overwrite", action="store_true")

    p_sample = sub.add_parser("sample", help="복합 변이 샘플 단위 피처")
    add_common(p_sample)
    p_sample.add_argument(
        "--include-cell-rollup",
        action="store_true",
        help="1차 전략표 이름(has_indel·mnv_count·unique_* 등) 17개를 함께 낸다.",
    )
    p_sample.add_argument(
        "--include-additional-burden",
        action="store_true",
        help="추가 burden 비율·중복·고차 multihit 피처 8개를 함께 낸다.",
    )
    p_sample.set_defaults(func=cmd_sample)

    p_gene = sub.add_parser("gene", help="유전자별 행렬")
    add_common(p_gene)
    p_gene.add_argument("--kind", choices=["mutated", "event_count"], default="mutated")
    p_gene.set_defaults(func=cmd_gene)

    p_tokens = sub.add_parser("tokens", help="Exact Mutation Token 문서")
    add_common(p_tokens)
    p_tokens.set_defaults(func=cmd_tokens)

    p_gene_types = sub.add_parser(
        "gene-types",
        help="유전자별 6종 변이 유형 존재 여부 wide matrix",
    )
    add_common(p_gene_types)
    p_gene_types.set_defaults(func=cmd_gene_types)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    summary = args.func(args)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"elapsed_sec: {time.perf_counter() - started:.1f}")
    if args.command == "sample":
        print(f"team_schema_columns: {len(SAMPLE_FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
