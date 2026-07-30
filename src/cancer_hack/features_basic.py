import re
import pandas as pd

# Synonymous mutation 패턴 확인용 정규식
_SYNONYMOUS_RE = re.compile(r"^([A-Z])\d+\1$")

# Mutation encoding 대상에서 제외할 메타 정보 컬럼
_GENE_EXCLUDE = {"ID", "SUBCLASS"}


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
