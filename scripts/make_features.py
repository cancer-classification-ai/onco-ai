#!/usr/bin/env python
"""팀 공용 피처 생성 CLI.

피처 정의는 전부 `cancer_hack.features_basic` 에 있고 이 스크립트는 껍데기다.
예전에는 Exact Mutation Token 코드가 여기에도 복사돼 있어서 한쪽만 고치면 두 정의가
어긋났다. 그래서 import 로 바꿨다 — 고칠 곳은 패키지 한 군데뿐이다.

    python scripts/make_features.py sample --split train --overwrite
    python scripts/make_features.py sample --split test  --overwrite
    python scripts/make_features.py sample --split train --include-additional-burden --overwrite
    python scripts/make_features.py tokens --split train --overwrite
    python scripts/make_features.py sigtokens --split train --overwrite
    python scripts/make_features.py parsed-tokens --split train --overwrite
    python scripts/make_features.py parsed --split train --overwrite
    python scripts/make_features.py burden-extra --split train --overwrite
    python scripts/make_features.py amino --split train --overwrite
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
    ADDITIONAL_BURDEN_FEATURE_COLUMNS,
    GENE_MUTATION_TYPES,
    MUTATION_STRING_PARSED_COLUMNS,
    SAMPLE_FEATURE_COLUMNS,
    make_exact_mutation_token_parquet,
    make_gene_event_count_matrix,
    make_gene_mutation_type_matrix,
    make_gene_mutated_matrix,
    make_mutation_encoding,
    make_mutation_string_parsed_features,
    make_sample_mutation_features,
    make_unique_mutation_token_parquet,
)
from cancer_hack.features_amino_acid import (  # noqa: E402
    AMINO_ACID_FEATURE_COLUMNS,
    make_amino_acid_features,
)
from cancer_hack.features_domain import (  # noqa: E402
    ALL_DOMAIN_PREFIXES,
    make_domain_features,
)
from cancer_hack.features_sparse import build_parsed_token_documents  # noqa: E402

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
    if args.include_cell_rollup and args.include_additional_burden:
        suffix = "_rollup_additional"
    elif args.include_cell_rollup:
        suffix = "_rollup"
    elif args.include_additional_burden:
        suffix = "_additional"
    else:
        suffix = ""
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


def cmd_enc3(args: argparse.Namespace) -> dict[str, object]:
    """WT/동의/기능성 3단계 유전자 인코딩."""
    output = _resolve_output(
        args.output, f"{args.split}_mutation_encoded.parquet"
    )
    _guard_existing(output, args.overwrite)
    frame, genes = _load_split(args.split, args.input)
    encoded = make_mutation_encoding(frame)
    _write_parquet(encoded, output)
    return {
        "output_path": str(output),
        "sample_count": len(encoded),
        "gene_count": len(genes),
        "column_count": len(encoded.columns) - (1 + int("SUBCLASS" in frame.columns)),
    }


def cmd_domain(args: argparse.Namespace) -> dict[str, object]:
    """도메인 지식 블록 — 드라이버·TMB·유형조성·치환쌍·코돈 역추론.

    `_load_split` 을 쓰지 않는다. 4,386열을 pandas 로 올리면 메모리가 아까워서
    `make_domain_features` 가 `csv.reader` 로 직접 흘려 읽는다.
    """
    output = _resolve_output(args.output, f"{args.split}_domain_features.parquet")
    _guard_existing(output, args.overwrite)

    source = args.input if args.input is not None else RAW_DIR / f"{args.split}.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"{source} 가 없다. 원본 csv 는 git 에 없으니 data/raw 에 먼저 놓는다."
        )
    features = make_domain_features(source, has_label=(args.split == "train"))

    # `"A2_TP53_lof".startswith("A_")` 는 False 라 접두사끼리 겹치지 않는다.
    blocks = {
        prefix: sum(c.startswith(prefix) for c in features.columns)
        for prefix in ALL_DOMAIN_PREFIXES
    }

    _write_parquet(features, output)
    return {
        "output_path": str(output),
        "sample_count": len(features),
        "column_count": len(features.columns),
        "blocks": blocks,
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


def cmd_sigtokens(args: argparse.Namespace) -> dict[str, object]:
    """서명 문서 — 잔기 번호를 지워 이소폼 중복을 접은 TF-IDF 입력."""
    output = _resolve_output(
        args.output, f"{args.split}_signature_mutation_tokens.parquet"
    )
    source = args.input if args.input is not None else RAW_DIR / f"{args.split}.csv"
    return make_unique_mutation_token_parquet(
        source, output, overwrite=args.overwrite
    )


def _write_stateless_block(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    output: Path,
) -> None:
    """행별 독립 피처를 공용 ID/label 계약으로 저장한다."""
    features = features.copy()
    features.insert(0, "ID", frame["ID"].astype(str).to_numpy())
    if "SUBCLASS" in frame.columns:
        features.insert(1, "SUBCLASS", frame["SUBCLASS"].astype(str).to_numpy())
    _write_parquet(features, output)


def cmd_parsed(args: argparse.Namespace) -> dict[str, object]:
    """Mutation 문자열 위치·파싱·유형 구조 19종."""
    output = _resolve_output(
        args.output, f"{args.split}_mutation_parsed_features.parquet"
    )
    _guard_existing(output, args.overwrite)
    frame, genes = _load_split(args.split, args.input)
    features = make_mutation_string_parsed_features(
        frame,
        gene_columns=genes,
        position_bin_size=args.position_bin_size,
    )
    _write_stateless_block(frame, features, output)
    return {
        "output_path": str(output),
        "sample_count": len(features),
        "feature_count": len(MUTATION_STRING_PARSED_COLUMNS),
        "position_bin_size": args.position_bin_size,
    }


def cmd_burden_extra(args: argparse.Namespace) -> dict[str, object]:
    """기존 rollup과 분리한 추가 burden 8종 ablation 블록."""
    output = _resolve_output(
        args.output, f"{args.split}_additional_burden_features.parquet"
    )
    _guard_existing(output, args.overwrite)
    frame, genes = _load_split(args.split, args.input)
    all_features = make_sample_mutation_features(
        frame,
        gene_columns=genes,
        include_additional_burden=True,
    )
    features = all_features[list(ADDITIONAL_BURDEN_FEATURE_COLUMNS)]
    _write_stateless_block(frame, features, output)
    return {
        "output_path": str(output),
        "sample_count": len(features),
        "feature_count": len(ADDITIONAL_BURDEN_FEATURE_COLUMNS),
    }


def cmd_amino(args: argparse.Namespace) -> dict[str, object]:
    """샘플 단위 아미노산 물리화학적 치환 페널티 9종."""
    output = _resolve_output(
        args.output, f"{args.split}_amino_acid_features.parquet"
    )
    _guard_existing(output, args.overwrite)
    frame, genes = _load_split(args.split, args.input)
    features = make_amino_acid_features(frame, gene_columns=genes)
    _write_stateless_block(frame, features, output)
    return {
        "output_path": str(output),
        "sample_count": len(features),
        "feature_count": len(AMINO_ACID_FEATURE_COLUMNS),
    }


def cmd_parsed_tokens(args: argparse.Namespace) -> dict[str, object]:
    """미등록 변이에도 부분 정보를 남기는 일반화 token 문서."""
    output = _resolve_output(
        args.output, f"{args.split}_parsed_mutation_tokens.parquet"
    )
    _guard_existing(output, args.overwrite)
    frame, genes = _load_split(args.split, args.input)
    documents = build_parsed_token_documents(
        frame,
        genes,
        position_bin_size=args.position_bin_size,
    )
    out = pd.DataFrame(
        {
            "ID": frame["ID"].astype(str).to_numpy(),
            "parsed_mutation_document": documents.to_numpy(dtype=object),
        }
    )
    _write_parquet(out, output)
    return {
        "output_path": str(output),
        "sample_count": len(out),
        "position_bin_size": args.position_bin_size,
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

    p_enc3 = sub.add_parser("enc3", help="WT/동의/기능성 3단계 유전자 인코딩")
    add_common(p_enc3)
    p_enc3.set_defaults(func=cmd_enc3)

    p_domain = sub.add_parser("domain", help="도메인 지식 블록 (A/A2/B/C/D/M/N)")
    add_common(p_domain)
    p_domain.set_defaults(func=cmd_domain)

    p_tokens = sub.add_parser("tokens", help="Exact Mutation Token 문서")
    add_common(p_tokens)
    p_tokens.set_defaults(func=cmd_tokens)

    p_gene_types = sub.add_parser(
        "gene-types",
        help="유전자별 6종 변이 유형 존재 여부 wide matrix",
    )
    add_common(p_gene_types)
    p_gene_types.set_defaults(func=cmd_gene_types)

    p_sigtokens = sub.add_parser("sigtokens", help="서명(위치 무시) 변이 문서")
    add_common(p_sigtokens)
    p_sigtokens.set_defaults(func=cmd_sigtokens)

    p_parsed = sub.add_parser("parsed", help="Mutation 문자열 구조 dense 피처 19종")
    add_common(p_parsed)
    p_parsed.add_argument("--position-bin-size", type=int, default=50)
    p_parsed.set_defaults(func=cmd_parsed)

    p_burden_extra = sub.add_parser(
        "burden-extra", help="추가 burden dense 피처 8종"
    )
    add_common(p_burden_extra)
    p_burden_extra.set_defaults(func=cmd_burden_extra)

    p_amino = sub.add_parser("amino", help="아미노산 치환 페널티 dense 피처 9종")
    add_common(p_amino)
    p_amino.set_defaults(func=cmd_amino)

    p_parsed_tokens = sub.add_parser(
        "parsed-tokens", help="미등록 변이 대응 일반화 token 문서"
    )
    add_common(p_parsed_tokens)
    p_parsed_tokens.add_argument("--position-bin-size", type=int, default=50)
    p_parsed_tokens.set_defaults(func=cmd_parsed_tokens)

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
