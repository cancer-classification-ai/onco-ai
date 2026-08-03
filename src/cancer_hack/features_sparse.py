"""변이 문서 -> 희소 TF-IDF 블록. **fold 의 train 문서에서만 fit 한다.**

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
from scipy import sparse

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
