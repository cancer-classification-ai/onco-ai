"""희소 피처 생성 — B 전략: 파싱 토큰 → 일반화 조합 토큰 → 희소행렬.

Exact Mutation Token(`MUT__BRAF__V600E`)은 train 미등록 변이에 대해 정보가 0이다.
B 전략은 변이를 구성 요소로 분해해 train에서 학습한 부분 정보를 재사용한다.

    BRAF V600E
    → GENE__BRAF                       # 유전자
    → TYPE__MISSENSE                   # 변이 유형
    → GENE_TYPE__BRAF__MISSENSE        # 유전자 × 유형
    → REF__V                           # 원래 아미노산
    → ALT__E                           # 변경 아미노산
    → AA_CHANGE__V_E                   # 아미노산 치환 패턴
    → POSBIN__600_649                  # 위치 구간 (bin_size=50 기본)
    → GENE_POSBIN__BRAF__600_649       # 유전자 × 위치 구간

파싱 실패(other 유형)는 아래 토큰으로 대체한다.

    TYPE__UNKNOWN
    PARSE_FAILED
    GENE_PARSE_FAILED__BRAF

생성된 토큰 문서(공백 구분 문자열)는 ParsedTokenHasher 로 희소행렬로 변환한다.
CountVectorizer 를 train fold 에만 fit 하므로 어휘는 train에서만 결정되어
fold-safe 하다. 어휘가 보존되므로 get_feature_names_out() 으로 피처 이름을
확인할 수 있어 중요도 분석이 가능하다.

공개 API
--------
    _posbin_label(position, bin_size)         → str
    build_token_list(gene, mutation_token, position_bin_size)  → list[str]
    row_to_parsed_token_document(row, gene_columns, position_bin_size) → str
    build_parsed_token_documents(df, gene_columns, *, position_bin_size) → pd.Series
    ParsedTokenHasher                         — CountVectorizer 래퍼
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from .parser import (
    classify_token,
    extract_token_string_features,
    OTHER,
    split_tokens,
)

_DEFAULT_POSITION_BIN_SIZE = 50
_NO_PARSED_TOKEN_SENTINEL = "SAMPLE__NO_PARSED_TOKEN"


def _posbin_label(position: int, bin_size: int) -> str:
    """위치 숫자를 구간 레이블 문자열로 변환한다.

    >>> _posbin_label(600, 50)
    '600_649'
    >>> _posbin_label(601, 50)
    '600_649'
    >>> _posbin_label(650, 50)
    '650_699'
    >>> _posbin_label(0, 50)
    '0_49'
    """
    lo = (position // bin_size) * bin_size
    return f"{lo}_{lo + bin_size - 1}"


def build_token_list(
    gene: str,
    mutation_token: str,
    position_bin_size: int = _DEFAULT_POSITION_BIN_SIZE,
) -> list[str]:
    """(유전자, 변이 토큰) 한 쌍을 일반화 조합 토큰 리스트로 변환한다.

    항상 포함: ``GENE__<gene>``

    파싱 성공 시 추가:
        TYPE__ · GENE_TYPE__ · REF__ · ALT__ · AA_CHANGE__ · POSBIN__ · GENE_POSBIN__

    파싱 실패(other 유형) 시:
        TYPE__UNKNOWN · PARSE_FAILED · GENE_PARSE_FAILED__<gene>

    REF__ / ALT__ / AA_CHANGE__ / POSBIN__ / GENE_POSBIN__ 은 정규식으로
    해당 정보를 추출하지 못한 경우(빈 문자열 또는 위치 없음) 생략한다.

    >>> build_token_list("BRAF", "V600E")
    ['GENE__BRAF', 'TYPE__MISSENSE', 'GENE_TYPE__BRAF__MISSENSE', 'REF__V', 'ALT__E', 'AA_CHANGE__V_E', 'POSBIN__600_649', 'GENE_POSBIN__BRAF__600_649']
    >>> build_token_list("TP53", "SPLICE")
    ['GENE__TP53', 'TYPE__UNKNOWN', 'PARSE_FAILED', 'GENE_PARSE_FAILED__TP53']
    """
    kind = classify_token(mutation_token)
    out: list[str] = [f"GENE__{gene}"]

    if kind == OTHER:
        out += [
            "TYPE__UNKNOWN",
            "PARSE_FAILED",
            f"GENE_PARSE_FAILED__{gene}",
        ]
        return out

    type_tag = kind.upper()
    out.append(f"TYPE__{type_tag}")
    out.append(f"GENE_TYPE__{gene}__{type_tag}")

    tf = extract_token_string_features(mutation_token)

    if tf.ref_aa:
        out.append(f"REF__{tf.ref_aa}")
    if tf.alt_aa:
        out.append(f"ALT__{tf.alt_aa}")
    if tf.ref_aa and tf.alt_aa:
        out.append(f"AA_CHANGE__{tf.ref_aa}_{tf.alt_aa}")

    if tf.position >= 0:
        pb = _posbin_label(tf.position, position_bin_size)
        out.append(f"POSBIN__{pb}")
        out.append(f"GENE_POSBIN__{gene}__{pb}")

    return out


def row_to_parsed_token_document(
    row: pd.Series,
    gene_columns: list[str],
    position_bin_size: int = _DEFAULT_POSITION_BIN_SIZE,
) -> str:
    """샘플 한 행을 공백 구분 파싱 토큰 문서로 변환한다.

    변이가 없는 샘플(전부 WT)은 ``_NO_PARSED_TOKEN_SENTINEL`` 을 반환한다.

    >>> import pandas as pd
    >>> row = pd.Series({"BRAF": "V600E", "TP53": "WT"})
    >>> doc = row_to_parsed_token_document(row, ["BRAF", "TP53"])
    >>> "GENE__BRAF" in doc
    True
    >>> "GENE__TP53" in doc
    False
    """
    tokens: list[str] = []
    for gene in gene_columns:
        for mt in split_tokens(row[gene]):
            tokens.extend(build_token_list(gene, mt, position_bin_size))
    return " ".join(tokens) if tokens else _NO_PARSED_TOKEN_SENTINEL


def build_parsed_token_documents(
    df: pd.DataFrame,
    gene_columns: list[str],
    *,
    position_bin_size: int = _DEFAULT_POSITION_BIN_SIZE,
) -> pd.Series:
    """DataFrame 전체를 파싱 토큰 문서 Series 로 변환한다.

    반환 Series 의 인덱스는 df 를 그대로 따른다.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"BRAF": ["V600E", "WT"]})
    >>> docs = build_parsed_token_documents(df, ["BRAF"])
    >>> len(docs)
    2
    """
    missing = sorted(set(gene_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing gene columns: {missing[:10]}")
    return df.apply(
        row_to_parsed_token_document,
        axis=1,
        gene_columns=gene_columns,
        position_bin_size=position_bin_size,
    )


class ParsedTokenHasher:
    """파싱 토큰 문서를 희소행렬로 변환한다 (train fold에만 fit — fold-safe).

    CountVectorizer 를 train fold 에만 fit 해 어휘를 확정한다. valid / test 에는
    transform 만 적용하므로 데이터 누수 없이 fold-safe 하다. 어휘가 보존되므로
    get_feature_names_out() 으로 피처 이름을 조회할 수 있다.

    Parameters
    ----------
    min_df : int or float
        최소 문서 빈도. 기본 1 (모든 토큰 유지).
        노이즈 토큰을 제거하려면 2 이상으로 올린다.
    ngram_range : tuple[int, int]
        word n-gram 범위. 기본 (1, 1) — 단일 토큰.
        (1, 2) 로 바꾸면 인접 토큰 쌍도 피처로 추가된다.
    norm : str or None
        행 정규화. None 이면 raw count, ``'l2'`` 이면 단위 벡터.

    Examples
    --------
    >>> import pandas as pd
    >>> docs = pd.Series(["GENE__BRAF TYPE__MISSENSE", "GENE__TP53 TYPE__NONSENSE"])
    >>> hasher = ParsedTokenHasher()
    >>> mat = hasher.fit_transform(docs)
    >>> mat.shape[0]
    2
    >>> "GENE__BRAF" in hasher.get_feature_names_out()
    True
    """

    def __init__(
        self,
        *,
        min_df: int | float = 1,
        ngram_range: tuple[int, int] = (1, 1),
        norm: str | None = None,
    ) -> None:
        self._vec = CountVectorizer(
            analyzer="word",
            lowercase=False,
            min_df=min_df,
            ngram_range=ngram_range,
        )
        self._norm = norm

    def fit(self, documents: pd.Series) -> "ParsedTokenHasher":
        """train fold 문서로 어휘를 구축한다."""
        self._vec.fit(documents)
        return self

    def transform(self, documents: pd.Series) -> sp.csr_matrix:
        """문서 Series 를 희소행렬(n_samples × n_features)로 변환한다."""
        mat = self._vec.transform(documents)
        if self._norm is not None:
            mat = normalize(mat, norm=self._norm)
        return mat

    def fit_transform(self, documents: pd.Series) -> sp.csr_matrix:
        """fit(documents).transform(documents) 의 편의 함수."""
        return self.fit(documents).transform(documents)

    def get_feature_names_out(self) -> np.ndarray:
        """학습된 피처 이름 배열을 반환한다."""
        return self._vec.get_feature_names_out()
