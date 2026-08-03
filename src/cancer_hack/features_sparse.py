"""
변이 문서 -> 희소 TF-IDF 블록. **fold 의 train 문서에서만 fit 한다.**

IDF 는 문서 집합 통계다. train 과 test 를 합쳐 어휘를 만들면 평가 데이터가 학습에
들어간 것이고 대회 규정상 실격이다. 그래서 `validation.Chi2TopKSelector`,
`features_basic.BurdenBinner` 와 같은 계약을 쓴다 — `fit` 과 `transform` 을 갈라 놔서
전체 데이터로 fit 하려면 일부러 해야 한다.

## 왜 서명 문서인가

`features_basic.row_to_unique_mutation_document` 가 내는 `SIG__{gene}__{signature}`
어휘에 train 을 fit 하고 test 를 transform 했을 때(실측):

    min_df   |V|      train 비영 행   test 비영 행   test 평균 nnz
      2      48,000      98.4%          97.4%          25.0
      3      20,913      96.0%          95.5%          16.5
      5       6,358      89.2%          90.3%           8.4

`min_df=3` 이 기본이다. train 과 test 의 비영 비율 차이가 0.5%p 로 가장 작으면서
(min_df=2 는 1.0%p, 5 는 1.1%p) 어휘가 2만 대로 떨어져 chi2 가 감당할 만하다.

같은 표를 `MUT__{gene}__{token}`(원문 문자열)로 만들면 이렇다.

    min_df   |V|      train 비영 행   test 비영 행   test 평균 nnz
      2      15,961      83.7%          72.3%           2.1
      3       2,255      66.5%          60.5%           1.1
      5         430      50.4%          49.2%           0.7

행당 비영 항이 1 근처면 사실상 0 행렬이다. test 가 같은 변이를 전사체마다 다른
좌표로 적기 때문에 train 어휘가 test 를 못 덮는다. 그래서 이 모듈의 1급 입력은
서명 문서이고 원문 문서는 대조군으로만 둔다.

## TF 가 항상 1 이다

`row_to_unique_mutation_document` 가 셀 안에서 서명을 접으므로 한 문서에 같은 토큰이
두 번 못 나온다. 그래서 `sublinear_tf` 나 `binary` 옵션이 결과를 못 바꾼다. 문서 길이
격차(train 행당 41.2 항 / test 132.6 항)는 서명 단계에서 39.9 / 79.8 로 줄고, 남은
2배는 L2 정규화가 흡수한다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from .parser import (
    OTHER,
    classify_token,
    extract_token_string_features,
    split_tokens,
)


#: 기본 TF-IDF 설정. 근거는 모듈 docstring 의 어휘 진단표.
MUTATION_TFIDF_DEFAULTS: dict[str, object] = {
    "min_df": 3,
    # min_df=3 에서는 안 걸린다. min_df 를 낮췄을 때를 위한 안전판이다.
    "max_features": 50_000,
    "use_idf": True,
    "smooth_idf": True,
    "norm": "l2",
}


class MutationTfidfBlock:
    """변이 문서 -> `csr_matrix`. **fold 의 train 문서에서만 fit 한다.**

    토크나이저는 `str.split` 이다. sklearn 기본 `token_pattern` 인 `\\b\\w\\w+\\b` 는
    `SIG__TP53__missense|V>E` 를 `SIG`, `TP`, `missense`, `V`, `E` 로 찢어 버린다.
    `str.split` 은 lambda 와 달리 picklable 이라 모델 저장에도 안전하다.

    >>> block = MutationTfidfBlock(min_df=1).fit(
    ...     ["SIG__A__missense|V>E", "SIG__A__missense|V>E SIG__B__nonsense|Q"]
    ... )
    >>> list(block.feature_names_)
    ['SIG__A__missense|V>E', 'SIG__B__nonsense|Q']
    >>> block.transform(["SIG__Z__missense|X>Y"]).nnz    # 미지 항은 0 행이 된다
    0
    """

    def __init__(self, **params) -> None:
        self.params = {**MUTATION_TFIDF_DEFAULTS, **params}
        self.vectorizer_ = None
        self.feature_names_: np.ndarray | None = None

    def fit(self, documents: Sequence[str]) -> "MutationTfidfBlock":
        # sklearn import 를 함수 안에서 한다 — `models_gbdt` 와 같은 관행이다.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer_ = TfidfVectorizer(
            analyzer=str.split,
            lowercase=False,
            dtype=np.float32,
            **self.params,
        )
        self.vectorizer_.fit(list(documents))
        self.feature_names_ = self.vectorizer_.get_feature_names_out()
        return self

    def transform(self, documents: Sequence[str]) -> sparse.csr_matrix:
        if self.vectorizer_ is None:
            raise RuntimeError("fit() 을 먼저 부른다")
        return self.vectorizer_.transform(list(documents))

    @property
    def idf_(self) -> np.ndarray:
        if self.vectorizer_ is None:
            raise RuntimeError("fit() 을 먼저 부른다")
        return self.vectorizer_.idf_

    def __repr__(self) -> str:
        width = "unfitted" if self.feature_names_ is None else len(self.feature_names_)
        return f"MutationTfidfBlock(|V|={width}, min_df={self.params['min_df']})"


def build_fold_tfidf_block(
    train_documents: Sequence[str],
    test_documents: Sequence[str],
    train_index: np.ndarray,
    y_train_fold: Sequence,
    *,
    prefix: str,
    topk: int | None = 1000,
    **tfidf_params,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """한 fold 분의 TF-IDF 블록 -> (열 이름, train dense, test dense).

    어휘·IDF·chi2 를 **전부 `train_documents[train_index]` 에만** fit 한다.
    반환하는 train 행렬은 valid 행을 포함한 **전체 train 행**이다 — valid 도
    transform 은 받아야 OOF 예측이 나온다. fit 에 안 들어갔을 뿐이다.

    dense 로 돌려주는 이유는 `train_gbdt.run_config` 의 fold 루프가
    `x_train[valid_index]` 같은 팬시 인덱싱과 열 단위 대입을 쓰기 때문이다. top-K 가
    폭을 1,000 열로 묶으므로 6,201 x 1,000 float32 = 25MB 다. 희소로 들고 다닐
    이유가 없다.
    """
    from .validation import Chi2TopKSelector

    train_documents = np.asarray(train_documents, dtype=object)
    test_documents = np.asarray(test_documents, dtype=object)

    block = MutationTfidfBlock(**tfidf_params).fit(train_documents[train_index])
    sparse_train = block.transform(train_documents)
    sparse_test = block.transform(test_documents)

    selector = Chi2TopKSelector(k=topk).fit(sparse_train[train_index], y_train_fold)
    names = [f"{prefix}{name}" for name in block.feature_names_[selector.indices_]]
    return (
        names,
        selector.transform(sparse_train).toarray().astype(np.float32),
        selector.transform(sparse_test).toarray().astype(np.float32),
    )


"""희소 피처 생성 : 파싱 토큰 → 일반화 조합 토큰 → 희소행렬.

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


