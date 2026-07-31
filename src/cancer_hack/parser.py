from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# 변이가 없는 것으로 간주하는 값들
_MUTATION_EMPTY: frozenset[str] = frozenset({"", "WT", "0", "NA", "NAN", "NONE", "."})

# 동의 변이 (예: R895R)
_SYNONYMOUS_RE = re.compile(r"^([A-Z])\d+\1$")

# 복합 변이 (del, ins, dup, splice, _, > 포함)
_COMPLEX_RE = re.compile(r"[_>]|del|ins|dup|splice", re.IGNORECASE)

# 명시적 deletion token (예: R649del, 490del, E746_A750del)
_EXPLICIT_DEL_RE = re.compile(r"del", re.IGNORECASE)

# 프레임시프트 변이 (fs 포함)
_FRAMESHIFT_RE = re.compile(r"fs", re.IGNORECASE)

# 넌센스 변이 (종결 코돈, *로 끝남)
_NONSENSE_RE = re.compile(r"\*$")

# 미스센스 변이 (예: R175H)
_MISSENSE_RE = re.compile(r"^[A-Z]\d+[A-Z]$")

# 하나의 셀에서 유효한 mutation token 목록을 추출
def _parse_mutation_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw or raw.upper() in _MUTATION_EMPTY:
        return []
    return [t for t in raw.split() if t.upper() not in _MUTATION_EMPTY]

# mutation token을 변이 유형으로 분류
def _classify_token(token: str) -> str:
    if _SYNONYMOUS_RE.match(token):
        return "synonymous"
    if _COMPLEX_RE.search(token):
        return "complex"
    if _FRAMESHIFT_RE.search(token):
        return "frameshift"
    if _NONSENSE_RE.search(token):
        return "nonsense"
    if _MISSENSE_RE.match(token):
        return "missense"
    return "complex"

# 지정한 유전자 컬럼이 모두 존재하는지 확인
def _check_columns(df: pd.DataFrame, gene_columns: list[str]) -> None:
    missing = sorted(set(gene_columns).difference(df.columns))
    if missing:
        raise ValueError(f"Missing gene columns: {missing[:10]}")

# 한 샘플(row)의 모든 mutation token을 분류하여 유형별 개수를 계산
def _row_token_class_counts(row: pd.Series, gene_columns: list[str]) -> Counter:
    counts: Counter = Counter()
    for gene in gene_columns:
        for token in _parse_mutation_tokens(row[gene]):
            counts[_classify_token(token)] += 1
    return counts


# ---------------------------------------------------------------------------
# Per-gene binary / count features
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


# functional_event_count: 단백질 변화를 유발하는 변이 token 수 (missense + nonsense + frameshift + complex)
def compute_functional_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _count(row: pd.Series) -> int:
        counts = _row_token_class_counts(row, gene_columns)
        return counts["missense"] + counts["nonsense"] + counts["frameshift"] + counts["complex"]

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


# complex_event_count: 구조적 복합 변이 token 수 (예: 468_469LG>F*, E746_A750del)
def compute_complex_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["complex"],
        axis=1,
    ).rename("complex_event_count")


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
            if _EXPLICIT_DEL_RE.search(t)
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
            if any(_EXPLICIT_DEL_RE.search(t) for t in _parse_mutation_tokens(row[g]))
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
                _EXPLICIT_DEL_RE.search(t)
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
            if _EXPLICIT_DEL_RE.search(t):
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
