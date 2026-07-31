#!/usr/bin/env python
"""Feature generation entrypoint shared by the team.

Keep feature-specific functions namespaced by feature name so additional
builders can be added without changing existing behavior.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import pandas as pd

from .parser import (
    _check_columns,
    _classify_token,
    _INDEL_RE,
    _MUTATION_EMPTY,
    _parse_mutation_tokens,
    _row_token_class_counts,
    _SYNONYMOUS_RE,
)

# Mutation encoding 대상에서 제외할 메타 정보 컬럼
_GENE_EXCLUDE = {"ID", "SUBCLASS"}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# parser.py의 _MUTATION_EMPTY와 동일한 값 — 중복 정의 대신 참조
EXACT_MUTATION_EMPTY_VALUES = _MUTATION_EMPTY
EXACT_MUTATION_NO_MUTATION_TOKEN = "SAMPLE__NO_MUTATION"


# 단일 mutation token을 WT / Synonymous / Functional 3단계 값으로 변환
def _encode_single(token: str) -> int:
    if token == "WT":
        return 0
    if _SYNONYMOUS_RE.match(token):
        return 1
    return 2


# 하나의 mutation cell 값을 분석하여 가장 높은 변이 영향도로 인코딩
def encode_mutation(value: str) -> int:
    tokens = value.split()
    if len(tokens) == 1:
        return _encode_single(tokens[0])
    return max(_encode_single(t) for t in tokens)


# 전체 gene column에 mutation encoding을 적용하여 전처리된 데이터셋 생성
def make_mutation_encoding(df: pd.DataFrame) -> pd.DataFrame:
    gene_cols = [c for c in df.columns if c not in _GENE_EXCLUDE]
    meta_cols = [c for c in df.columns if c in _GENE_EXCLUDE]
    encoded = df[gene_cols].map(encode_mutation).astype("int8")
    return pd.concat([df[meta_cols], encoded], axis=1)
    


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


# ---------------------------------------------------------------------------
# 기본 파생 변수 & del 파생변수 처리 정의함수
# ---------------------------------------------------------------------------

# gene_mutated__{gene}: 해당 유전자에 변이가 없으면 0, 변이가 존재하면 1
def compute_gene_mutated(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.DataFrame:
    _check_columns(df, gene_columns)
    return pd.DataFrame(
        {
            f"gene_mutated__{g}": df[g]
            .apply(lambda v: int(bool(_parse_mutation_tokens(v))))
            .astype("int8")
            for g in gene_columns
        },
        index=df.index,
    )


# gene_event_count__{gene}: 해당 유전자에 기록된 mutation token 수
def compute_gene_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.DataFrame:
    _check_columns(df, gene_columns)
    return pd.DataFrame(
        {
            f"gene_event_count__{g}": df[g].apply(
                lambda v: len(_parse_mutation_tokens(v))
            )
            for g in gene_columns
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Per-sample scalar features
# ---------------------------------------------------------------------------

# mutated_gene_count: 변이가 존재하는 고유 유전자 수
def compute_mutated_gene_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(1 for g in gene_columns if _parse_mutation_tokens(row[g])),
        axis=1,
    ).rename("mutated_gene_count")


# mutation_event_count: 공백으로 분리한 전체 mutation token 수
def compute_mutation_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(len(_parse_mutation_tokens(row[g])) for g in gene_columns),
        axis=1,
    ).rename("mutation_event_count")


# synonymous_event_count: 동의 변이 token 수 — 단백질 변화 없음 (예: R895R)
def compute_synonymous_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["synonymous"],
        axis=1,
    ).rename("synonymous_event_count")


# functional_event_count: 단백질 변화를 유발하는 변이 token 수 (missense + nonsense + frameshift + complex + indel)
def compute_functional_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _count(row: pd.Series) -> int:
        counts = _row_token_class_counts(row, gene_columns)
        return counts["missense"] + counts["nonsense"] + counts["frameshift"] + counts["complex"] + counts["indel"]

    return df.apply(_count, axis=1).rename("functional_event_count")


# missense_event_count: 아미노산 치환 변이 token 수 (예: R132H, V600E)
def compute_missense_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["missense"],
        axis=1,
    ).rename("missense_event_count")


# nonsense_event_count: stop codon을 생성하는 변이 token 수 (예: R213*)
def compute_nonsense_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["nonsense"],
        axis=1,
    ).rename("nonsense_event_count")


# frameshift_event_count: 읽기 틀 변화 변이 token 수 (예: K16fs, L1854fs)
def compute_frameshift_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["frameshift"],
        axis=1,
    ).rename("frameshift_event_count")


# complex_event_count: 구조적 복합 변이 token 수 (예: 468_469LG>F*, E746_A750del) - indel 포함
def compute_complex_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _count(row: pd.Series) -> int:
        counts = _row_token_class_counts(row, gene_columns)
        return counts["complex"] + counts["indel"]

    return df.apply(_count, axis=1).rename("complex_event_count")


# multihit_gene_count: 한 유전자에 2개 이상의 변이가 있는 유전자 수
def compute_multihit_gene_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(
            1 for g in gene_columns if len(_parse_mutation_tokens(row[g])) >= 2
        ),
        axis=1,
    ).rename("multihit_gene_count")


# max_events_per_gene: 한 유전자에서 관찰된 최대 mutation token 수
def compute_max_events_per_gene(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _max(row: pd.Series) -> int:
        counts = [len(_parse_mutation_tokens(row[g])) for g in gene_columns]
        return max(counts) if counts else 0

    return df.apply(_max, axis=1).rename("max_events_per_gene")


# no_mutation_flag: 전체 유전자에 변이가 하나도 없으면 1, 아니면 0
def compute_no_mutation_flag(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: int(
            not any(_parse_mutation_tokens(row[g]) for g in gene_columns)
        ),
        axis=1,
    ).rename("no_mutation_flag")


# mutation_gene_ratio: 변이가 존재하는 유전자 수 / 전체 유전자 수
def compute_mutation_gene_ratio(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    n_total = len(gene_columns)
    return df.apply(
        lambda row: (
            sum(1 for g in gene_columns if _parse_mutation_tokens(row[g])) / n_total
            if n_total > 0 else 0.0
        ),
        axis=1,
    ).rename("mutation_gene_ratio")


# multi_mutation_gene_ratio: 복수 변이 유전자 수 / 변이가 존재하는 유전자 수
def compute_multi_mutation_gene_ratio(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _ratio(row: pd.Series) -> float:
        n_mutated = sum(1 for g in gene_columns if _parse_mutation_tokens(row[g]))
        n_multi = sum(1 for g in gene_columns if len(_parse_mutation_tokens(row[g])) >= 2)
        return n_multi / n_mutated if n_mutated > 0 else 0.0

    return df.apply(_ratio, axis=1).rename("multi_mutation_gene_ratio")


# n_unique_mutation_tokens: 중복을 제거한 mutation token 수
def compute_unique_mutation_token_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: len({t for g in gene_columns for t in _parse_mutation_tokens(row[g])}),
        axis=1,
    ).rename("n_unique_mutation_tokens")


# explicit_deletion_event_count: 명시적 del 사건 token 수 (예: R649del, 490del)
def compute_explicit_deletion_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(
            1
            for g in gene_columns
            for t in _parse_mutation_tokens(row[g])
            if _INDEL_RE.search(t)
        ),
        axis=1,
    ).rename("explicit_deletion_event_count")


# explicit_deletion_gene_count: 명시적 deletion이 있는 유전자 수
def compute_explicit_deletion_gene_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(
            1
            for g in gene_columns
            if any(_INDEL_RE.search(t) for t in _parse_mutation_tokens(row[g]))
        ),
        axis=1,
    ).rename("explicit_deletion_gene_count")


# has_explicit_deletion: 명시적 deletion이 하나라도 있으면 1, 아니면 0
def compute_has_explicit_deletion(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: int(
            any(
                _INDEL_RE.search(t)
                for g in gene_columns
                for t in _parse_mutation_tokens(row[g])
            )
        ),
        axis=1,
    ).rename("has_explicit_deletion")


# 단일 행에 대해 모든 burden feature를 한 번의 순회로 계산
def _compute_row_burden(row: pd.Series, gene_columns: list[str]) -> dict:
    mutated_gene_count = 0
    mutation_event_count = 0
    multihit_gene_count = 0
    max_events = 0
    synonymous = 0
    missense = 0
    nonsense = 0
    frameshift = 0
    complex_ = 0
    all_tokens: list[str] = []
    explicit_del_event_count = 0
    explicit_del_gene_count = 0

    for g in gene_columns:
        tokens = _parse_mutation_tokens(row[g])
        n = len(tokens)
        if n > 0:
            mutated_gene_count += 1
        mutation_event_count += n
        if n >= 2:
            multihit_gene_count += 1
        if n > max_events:
            max_events = n

        gene_has_del = False
        for t in tokens:
            all_tokens.append(t)
            cls = _classify_token(t)
            if cls == "synonymous":
                synonymous += 1
            elif cls == "missense":
                missense += 1
            elif cls == "nonsense":
                nonsense += 1
            elif cls == "frameshift":
                frameshift += 1
            else:
                complex_ += 1
            if _INDEL_RE.search(t):
                explicit_del_event_count += 1
                gene_has_del = True
        if gene_has_del:
            explicit_del_gene_count += 1

    n_total = len(gene_columns)
    n_unique = len(set(all_tokens))
    return {
        "mutated_gene_count": mutated_gene_count,
        "mutation_event_count": mutation_event_count,
        "synonymous_event_count": synonymous,
        "functional_event_count": missense + nonsense + frameshift + complex_,
        "missense_event_count": missense,
        "nonsense_event_count": nonsense,
        "frameshift_event_count": frameshift,
        "complex_event_count": complex_,
        "multihit_gene_count": multihit_gene_count,
        "max_events_per_gene": max_events,
        "no_mutation_flag": int(mutated_gene_count == 0),
        "mutation_gene_ratio": mutated_gene_count / n_total if n_total > 0 else 0.0,
        "multi_mutation_gene_ratio": multihit_gene_count / mutated_gene_count if mutated_gene_count > 0 else 0.0,
        "n_unique_mutation_tokens": n_unique,
        "n_repeated_mutation_tokens": len(all_tokens) - n_unique,
        "explicit_deletion_event_count": explicit_del_event_count,
        "explicit_deletion_gene_count": explicit_del_gene_count,
        "has_explicit_deletion": int(explicit_del_event_count > 0),
    }


# 모든 burden feature를 단일 apply 패스로 계산 (개별 함수 15회 호출 대비 ~10x 빠름)
def compute_all_burden_features(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.DataFrame:
    _check_columns(df, gene_columns)
    return pd.DataFrame(
        df.apply(lambda row: _compute_row_burden(row, gene_columns), axis=1).tolist(),
        index=df.index,
    )


# n_repeated_mutation_tokens: 중복 등장 mutation token 수 (전체 token 수 - 고유 token 수)
def compute_repeated_mutation_token_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _count(row: pd.Series) -> int:
        all_tokens = [t for g in gene_columns for t in _parse_mutation_tokens(row[g])]
        return len(all_tokens) - len(set(all_tokens))

    return df.apply(_count, axis=1).rename("n_repeated_mutation_tokens")


# ---------------------------------------------------------------------------


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

    
