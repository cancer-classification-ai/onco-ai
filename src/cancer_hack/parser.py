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

# 삽입/결실 변이 - del로 끝나는 토큰 (예: R649del, 490del, E746_A750del)
_INDEL_RE = re.compile(r"del$", re.IGNORECASE)

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
    if _INDEL_RE.search(token):
        return "indel"
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
