"""fold 학습 부분에서만 fit 하는지 검증한다.

대회 규정이 콕 집어 금지한 것 — 인코더·스케일러·집계 통계를 test 로 fit 하는 것.
분위수 경계를 잡는 `hypermutated_flag`·`burden_quantile_bin` 과 문서 집합 통계를 쓰는
TF-IDF 의 IDF, 이 둘이 그 위험이 있는 피처다. 나머지는 행마다 독립 계산이라 애초에
샐 곳이 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cancer_hack.features_basic import (
    BurdenBinner,
    make_sample_mutation_features,
)
from cancer_hack.features_sparse import MutationTfidfBlock, build_fold_tfidf_block
from cancer_hack.validation import Chi2TopKSelector


def _burden(values: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"mutation_event_count": values})


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        BurdenBinner().transform(_burden([1, 2, 3]))


def test_threshold_depends_only_on_fit_data():
    train = _burden(list(range(100)))
    other = _burden([10_000] * 100)

    binner = BurdenBinner(hypermutated_quantile=0.95).fit(train)
    threshold = binner.threshold_
    edges = binner.edges_.copy()

    binner.transform(other)  # transform 은 상태를 건드리면 안 된다
    assert binner.threshold_ == threshold
    assert np.array_equal(binner.edges_, edges)


def test_fitting_on_more_data_changes_threshold():
    """경계가 fit 데이터에 실제로 반응해야 의미 있는 검증이다."""
    small = BurdenBinner().fit(_burden(list(range(100))))
    large = BurdenBinner().fit(_burden(list(range(100)) + [10_000] * 100))
    assert small.threshold_ != large.threshold_


def test_valid_rows_can_exceed_train_range():
    """학습 fold 밖 값이 들어와도 터지지 않고 마지막 구간으로 간다."""
    binner = BurdenBinner(n_bins=4).fit(_burden([0, 1, 2, 3, 4, 5, 6, 7]))
    out = binner.transform(_burden([-5, 3, 999]))
    assert int(out["hypermutated_flag"].iloc[-1]) == 1
    assert int(out["burden_quantile_bin"].iloc[0]) == 0
    assert int(out["burden_quantile_bin"].iloc[-1]) == len(binner.edges_)


def test_transform_preserves_input_columns():
    features = _burden([1, 2, 3])
    features["other"] = [9, 9, 9]
    out = BurdenBinner(n_bins=2).fit(features).transform(features)
    assert list(out.columns)[:2] == ["mutation_event_count", "other"]
    assert "hypermutated_flag" in out.columns
    assert "burden_quantile_bin" in out.columns


def test_sample_features_are_row_independent(toy_frame):
    """행을 섞거나 잘라도 같은 샘플의 피처 값이 바뀌지 않아야 한다.

    이게 성립하면 train/test 를 따로 돌려도 누수가 없다.
    """
    genes = ["TP53", "KRAS", "EGFR"]
    full = make_sample_mutation_features(toy_frame, gene_columns=genes)
    subset = make_sample_mutation_features(
        toy_frame.iloc[[2]].reset_index(drop=True), gene_columns=genes
    )
    pd.testing.assert_frame_equal(
        full.iloc[[2]].reset_index(drop=True), subset, check_dtype=True
    )


# --- TF-IDF ------------------------------------------------------------------
# IDF 는 문서 집합 통계다. train 과 test 를 합쳐 어휘를 만들면 규정 위반이라
# `BurdenBinner` 와 같은 강도로 지킨다.
TRAIN_DOCS = [
    "SIG__TP53__missense|V>E SIG__KRAS__nonsense|Q",
    "SIG__TP53__missense|V>E",
    "SIG__KRAS__nonsense|Q",
    "SIG__TP53__missense|V>E SIG__EGFR__frameshift|K",
]
#: train 어휘에 없는 항만 담은 문서. test 행의 5.7% 가 이 상태다.
UNSEEN_DOC = "SIG__ZZZ__missense|A>W SIG__ZZZ__nonsense|W"


def test_tfidf_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        MutationTfidfBlock().transform(TRAIN_DOCS)


def test_tfidf_idf_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        MutationTfidfBlock().idf_


def test_tfidf_vocabulary_depends_only_on_fit_documents():
    """train 과 test 를 합쳐 어휘를 만드는 실수를 잡는 핵심 테스트다."""
    block = MutationTfidfBlock(min_df=1).fit(TRAIN_DOCS)
    vocabulary = list(block.feature_names_)

    block.transform([UNSEEN_DOC] * 20)  # transform 은 상태를 건드리면 안 된다
    assert list(block.feature_names_) == vocabulary
    assert not any("ZZZ" in name for name in block.feature_names_)


def test_tfidf_idf_unchanged_by_transform():
    block = MutationTfidfBlock(min_df=1).fit(TRAIN_DOCS)
    idf = block.idf_.copy()
    block.transform(TRAIN_DOCS + [UNSEEN_DOC])
    assert np.array_equal(block.idf_, idf)


def test_tfidf_unseen_terms_become_zero_rows():
    """미지 항뿐인 문서는 0 행이 되어야 한다 — 예외를 던지면 test 추론이 멈춘다."""
    block = MutationTfidfBlock(min_df=1).fit(TRAIN_DOCS)
    matrix = block.transform([UNSEEN_DOC])
    assert matrix.shape == (1, len(block.feature_names_))
    assert matrix.nnz == 0


def test_tfidf_fit_on_more_documents_changes_idf():
    """IDF 가 fit 데이터에 실제로 반응해야 위 불변 테스트가 의미를 갖는다."""
    small = MutationTfidfBlock(min_df=1).fit(TRAIN_DOCS)
    large = MutationTfidfBlock(min_df=1).fit(TRAIN_DOCS + ["SIG__TP53__missense|V>E"] * 8)
    column = list(small.feature_names_).index("SIG__KRAS__nonsense|Q")
    assert small.idf_[column] != large.idf_[list(large.feature_names_).index(
        "SIG__KRAS__nonsense|Q"
    )]


def test_tfidf_analyzer_keeps_signature_tokens_whole():
    """sklearn 기본 token_pattern 으로 돌아가면 토큰이 다섯 조각으로 찢긴다."""
    block = MutationTfidfBlock(min_df=1).fit(["SIG__TP53__missense|V>E"])
    assert list(block.feature_names_) == ["SIG__TP53__missense|V>E"]


def test_tfidf_min_df_drops_rare_terms():
    block = MutationTfidfBlock(min_df=3).fit(TRAIN_DOCS)
    # V>E 는 4 문서 중 3 개에 나오고 frameshift 는 1 개에만 나온다.
    assert "SIG__TP53__missense|V>E" in block.feature_names_
    assert "SIG__EGFR__frameshift|K" not in block.feature_names_


def test_build_fold_tfidf_block_fits_on_train_index_only():
    """valid 행에만 있는 항이 어휘에 들어오면 fold 누수다."""
    train_documents = [
        "SIG__A__missense|V>E",
        "SIG__A__missense|V>E",
        "SIG__B__nonsense|Q",
        "SIG__B__nonsense|Q",
        "SIG__LEAK__missense|L>K",
        "SIG__LEAK__missense|L>K",
        "SIG__LEAK__missense|L>K",
        "SIG__LEAK__missense|L>K",
    ]
    train_index = np.array([0, 1, 2, 3])
    names, _, _ = build_fold_tfidf_block(
        train_documents,
        ["SIG__A__missense|V>E"],
        train_index,
        ["x", "x", "y", "y"],
        prefix="tfidf__sigtok__",
        topk=None,
        min_df=1,
    )
    assert not any("LEAK" in name for name in names)
    assert "tfidf__sigtok__SIG__A__missense|V>E" in names


def test_build_fold_tfidf_block_returns_every_train_row():
    """valid 행도 transform 은 받아야 OOF 예측이 나온다. fit 에만 안 들어간다."""
    train_documents = ["SIG__A__missense|V>E", "SIG__B__nonsense|Q"] * 4
    names, train_matrix, test_matrix = build_fold_tfidf_block(
        train_documents,
        ["SIG__A__missense|V>E"] * 3,
        np.array([0, 1, 2, 3]),
        ["x", "y", "x", "y"],
        prefix="p__",
        topk=None,
        min_df=1,
    )
    assert train_matrix.shape == (len(train_documents), len(names))
    assert test_matrix.shape == (3, len(names))
    assert train_matrix[4:].any(), "fit 밖 행도 값이 채워져야 한다"


def test_chi2_selector_accepts_sparse_and_matches_dense():
    """dense 와 희소가 같은 열을 골라야 TF-IDF 경로를 믿을 수 있다."""
    dense = np.array([[0, 1, 0], [3, 1, 0], [0, 1, 0], [3, 1, 0]], dtype=np.float64)
    labels = ["a", "b", "a", "b"]
    from_dense = Chi2TopKSelector(k=1).fit(dense, labels)
    from_sparse = Chi2TopKSelector(k=1).fit(sparse.csr_matrix(dense), labels)
    assert np.array_equal(from_dense.indices_, from_sparse.indices_)
    assert sparse.issparse(from_sparse.transform(sparse.csr_matrix(dense)))
    assert from_sparse.transform(sparse.csr_matrix(dense)).shape == (4, 1)


def test_chi2_selector_rejects_negative_sparse():
    negative = sparse.csr_matrix(np.array([[-1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="음수"):
        Chi2TopKSelector(k=1).fit(negative, ["a", "b"])
