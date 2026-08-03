"""fold 학습 부분에서만 fit 하는지 검증한다.

대회 규정이 콕 집어 금지한 것 — 인코더·스케일러·집계 통계를 test 로 fit 하는 것.
분위수 경계를 잡는 `hypermutated_flag`·`burden_quantile_bin` 과 문서 집합 통계를 쓰는
TF-IDF 의 IDF, 이 둘이 그 위험이 있는 피처다. 나머지는 행마다 독립 계산이라 애초에
샐 곳이 없다.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cancer_hack.features_basic import (
    BurdenBinner,
    make_sample_mutation_features,
)
from cancer_hack.features_graph import (
    build_fold_comutation_block,
    select_comutation_pairs,
    transform_comutation_pairs,
)
from cancer_hack.features_latent import (
    MODULE_VALUES,
    build_fold_latent_block,
    build_fold_module_block,
    fit_gene_modules,
    fit_latent_basis,
    transform_gene_modules,
    transform_latent_basis,
)
from cancer_hack.features_signature import (
    build_fold_signature_block,
    select_class_signatures,
    transform_class_signatures,
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


# --- 공변이 쌍 -----------------------------------------------------------
# 유전자 6개(GA..GF), 12행. 행 0-7 이 train_index, 8-11 은 valid 전용이다.
# GA&GB 는 train_index 안에서 클래스 X 를 가르는 진짜 신호(lift>0)다. GC&GD 는
# valid 행에서만 완벽히 클래스 Z 를 가른다 — train_index 로는 절대 안 보여야 한다.
# 행 8 은 GA/GB 도 같이 켜 둬서, valid 행이 transform 은 정상적으로 받는지도 같이 본다.
_COMUT_GENES = ["GA", "GB", "GC", "GD", "GE", "GF"]
_COMUT_MATRIX = np.array(
    [
        [1, 1, 0, 0, 0, 0],  # 0  X
        [1, 1, 0, 0, 0, 0],  # 1  X
        [1, 1, 0, 0, 0, 0],  # 2  X
        [1, 0, 0, 0, 0, 0],  # 3  Y
        [0, 1, 0, 0, 0, 0],  # 4  Y
        [0, 0, 0, 0, 1, 0],  # 5  Y
        [0, 0, 0, 0, 0, 1],  # 6  Y
        [0, 0, 0, 0, 0, 0],  # 7  Y
        [1, 1, 1, 1, 0, 0],  # 8  Z (valid) — GC&GD 누출 + GA&GB 도 켜짐
        [0, 0, 1, 1, 0, 0],  # 9  Z (valid)
        [0, 0, 1, 1, 0, 0],  # 10 Z (valid)
        [0, 0, 1, 1, 0, 0],  # 11 Z (valid)
    ],
    dtype=np.float32,
)
_COMUT_Y = np.array(["X", "X", "X", "Y", "Y", "Y", "Y", "Y", "Z", "Z", "Z", "Z"])
_COMUT_TRAIN_INDEX = np.arange(8)
_COMUT_Y_FOLD = _COMUT_Y[_COMUT_TRAIN_INDEX]
#: 리크 테스트에 쓰는 관대한 임계값 — 필터 자체가 아니라 fold 경계를 시험한다.
_COMUT_KWARGS = dict(
    gene_names=_COMUT_GENES,
    pool="chi2",
    pool_topk=10,
    manual_pairs=(),
    min_support=2,
    min_class_support=1,
    min_purity=0.5,
    min_lift=0.2,
    max_hyper_fraction=1.0,
)


def test_comutation_pair_only_in_valid_rows_is_not_selected():
    """valid 행에만 있는 쌍이 선택되면 fold 누수다."""
    pairs = select_comutation_pairs(
        _COMUT_MATRIX, _COMUT_TRAIN_INDEX, _COMUT_Y_FOLD, **_COMUT_KWARGS
    )
    assert "GA__GB" in pairs.names
    assert "GC__GD" not in pairs.names


def test_comutation_returns_every_train_row():
    """valid 행도 transform 은 받아야 OOF 예측이 나온다. fit 에만 안 들어간다."""
    test_matrix = _COMUT_MATRIX[:3]
    names, train_out, test_out, diagnostics = build_fold_comutation_block(
        _COMUT_MATRIX, test_matrix, _COMUT_TRAIN_INDEX, _COMUT_Y_FOLD, **_COMUT_KWARGS
    )
    assert train_out.shape == (_COMUT_MATRIX.shape[0], len(names))
    assert test_out.shape == (test_matrix.shape[0], len(names))
    assert len(diagnostics) == len(names)
    # 행 8 은 valid 전용인데 GA&GB 가 켜져 있다 — fit 밖 행도 값이 채워져야 한다.
    ga_gb = names.index("comut__mut__GA__GB")
    assert train_out[8, ga_gb] > 0


def test_comutation_selection_unchanged_by_valid_row_content():
    """valid 행(8-11)을 완전히 흔들어도 선택 결과가 그대로여야 한다."""
    disturbed = _COMUT_MATRIX.copy()
    disturbed[8:] = 1.0
    base = select_comutation_pairs(
        _COMUT_MATRIX, _COMUT_TRAIN_INDEX, _COMUT_Y_FOLD, **_COMUT_KWARGS
    )
    shaken = select_comutation_pairs(
        disturbed, _COMUT_TRAIN_INDEX, _COMUT_Y_FOLD, **_COMUT_KWARGS
    )
    assert base.names == shaken.names
    assert base.stats == shaken.stats
    assert base.n_bar == shaken.n_bar
    assert np.array_equal(base.burden_sorted, shaken.burden_sorted)


def test_comutation_selection_responds_to_fit_data():
    """3번이 공허하지 않다는 짝 — train 쪽 신호를 지우면 선택도 바뀌어야 한다."""
    erased = _COMUT_MATRIX.copy()
    ga, gb = _COMUT_GENES.index("GA"), _COMUT_GENES.index("GB")
    erased[:8, [ga, gb]] = 0.0
    base = select_comutation_pairs(
        _COMUT_MATRIX, _COMUT_TRAIN_INDEX, _COMUT_Y_FOLD, **_COMUT_KWARGS
    )
    changed = select_comutation_pairs(
        erased, _COMUT_TRAIN_INDEX, _COMUT_Y_FOLD, **_COMUT_KWARGS
    )
    assert "GA__GB" in base.names
    assert "GA__GB" not in changed.names


def test_select_comutation_pairs_takes_no_test_matrix():
    """test 통계가 선택에 안 닿는다 — 주석이 아니라 시그니처로 보장한다."""
    params = inspect.signature(select_comutation_pairs).parameters
    assert not any("test" in name.lower() for name in params)


def test_comutation_share_value_is_row_independent():
    """행 하나만 넣어도 전체를 넣었을 때와 그 행의 값이 같아야 한다."""
    pairs = select_comutation_pairs(
        _COMUT_MATRIX, _COMUT_TRAIN_INDEX, _COMUT_Y_FOLD, **_COMUT_KWARGS
    )
    full = transform_comutation_pairs(_COMUT_MATRIX, pairs, value="share")
    single = transform_comutation_pairs(_COMUT_MATRIX[[0]], pairs, value="share")
    np.testing.assert_array_equal(full[[0]], single)


def test_comutation_hyper_guard_drops_hypermutator_only_pair():
    """동시 변이가 과변이 샘플에만 몰린 쌍은 가드가 걸러야 한다."""
    genes = ["GA", "GB", "GH", "GI"]
    matrix = np.array(
        [
            [1, 0, 0, 0],  # 0 A
            [0, 1, 0, 0],  # 1 A
            [1, 0, 0, 0],  # 2 B
            [0, 1, 0, 0],  # 3 B
            [1, 0, 0, 0],  # 4 A
            [0, 1, 0, 0],  # 5 A
            [1, 0, 0, 0],  # 6 B
            [0, 1, 0, 0],  # 7 B
            [0, 0, 0, 0],  # 8 A
            [1, 1, 1, 1],  # 9 B — 과변이 행, GH&GI 는 여기서만 동시 변이
        ],
        dtype=np.float32,
    )
    y = np.array(["A", "A", "B", "B", "A", "A", "B", "B", "A", "B"])
    train_index = np.arange(10)
    kwargs = dict(
        gene_names=genes,
        pool="chi2",
        pool_topk=10,
        manual_pairs=(),
        min_support=1,
        min_class_support=1,
        min_purity=0.0,
        min_lift=0.0,
    )
    guarded = select_comutation_pairs(
        matrix, train_index, y, max_hyper_fraction=0.5, **kwargs
    )
    unguarded = select_comutation_pairs(
        matrix, train_index, y, max_hyper_fraction=1.0, **kwargs
    )
    assert "GH__GI" not in guarded.names
    assert "GH__GI" in unguarded.names


def test_comutation_names_are_lexicographically_ordered():
    """A < B 로 고정해야 fold 마다 A&B/B&A 로 갈려 Jaccard 가 가짜로 안 낮아진다."""
    pairs = select_comutation_pairs(
        _COMUT_MATRIX, _COMUT_TRAIN_INDEX, _COMUT_Y_FOLD, **_COMUT_KWARGS
    )
    assert pairs.names
    for name in pairs.names:
        a, b = name.split("__")
        assert a < b


def test_comutation_width_never_exceeds_topk():
    """차원 상한 — topk 는 자동 선별분의 진짜 상한이어야 한다."""
    genes = ["GA", "GB", "GC", "GD"]
    matrix = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    y = np.array(["P", "Q", "P", "Q", "P", "Q"])
    train_index = np.arange(6)
    pairs = select_comutation_pairs(
        matrix,
        train_index,
        y,
        gene_names=genes,
        pool="chi2",
        pool_topk=10,
        manual_pairs=(),
        min_support=1,
        min_class_support=1,
        min_purity=0.0,
        min_lift=0.0,
        max_hyper_fraction=1.0,
        topk=2,
    )
    assert len(pairs.names) <= 2


# --- 유전자 모듈 (잠재 · 하드) ----------------------------------------------
# 유전자 8개, 12행. 행 0-7 이 train_index, 8-11 은 valid 전용이다. GG&GH 는 valid
# 행에서만 함께 켜진다 — fold-train 만 보는 fit 은 그 축을 절대 못 배워야 한다.
# 행 8 은 GA/GB 도 같이 켜 둔다(comut 픽스처와 같은 이유). 안 그러면 valid 행의 변이가
# 전부 지지도 미달 유전자라 투영이 정확히 0 이 나오고, "valid 행도 transform 은 받는다"
# 를 검사할 수가 없다.
_MOD_GENES = ["GA", "GB", "GC", "GD", "GE", "GF", "GG", "GH"]
_MOD_MATRIX = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0, 0],  # 0  X
        [1, 1, 0, 0, 0, 0, 0, 0],  # 1  X
        [1, 1, 1, 0, 0, 0, 0, 0],  # 2  X
        [0, 0, 1, 1, 0, 0, 0, 0],  # 3  Y
        [0, 0, 1, 1, 0, 0, 0, 0],  # 4  Y
        [0, 0, 0, 1, 1, 0, 0, 0],  # 5  Y
        [0, 0, 0, 0, 1, 1, 0, 0],  # 6  Y
        [0, 0, 0, 0, 1, 1, 0, 0],  # 7  Y
        [1, 1, 0, 0, 0, 0, 1, 1],  # 8  Z (valid) — GG&GH 누출 + GA/GB 도 켜짐
        [0, 0, 0, 0, 0, 0, 1, 1],  # 9  Z (valid)
        [0, 0, 0, 0, 0, 0, 1, 1],  # 10 Z (valid)
        [0, 0, 0, 0, 0, 0, 1, 1],  # 11 Z (valid)
    ],
    dtype=np.float32,
)
_MOD_Y = np.array(["X", "X", "X", "Y", "Y", "Y", "Y", "Y", "Z", "Z", "Z", "Z"])
_MOD_TRAIN_INDEX = np.arange(8)
_MOD_Y_FOLD = _MOD_Y[_MOD_TRAIN_INDEX]
_LATENT_KWARGS = dict(gene_names=_MOD_GENES, n_components=3, min_gene_support=1)
_MODULE_KWARGS = dict(
    gene_names=_MOD_GENES, n_modules=3, svd_components=3, min_gene_support=1
)


def test_fit_functions_take_no_test_matrix():
    """누출 없음을 주석이 아니라 시그니처로 보장한다 (comut 과 같은 계약)."""
    for function in (fit_latent_basis, fit_gene_modules):
        parameters = inspect.signature(function).parameters
        assert not [p for p in parameters if "test" in p.lower()]


def test_fit_functions_take_no_label_argument():
    """L3·L4 는 비지도다. `y` 인자가 없으면 지도 사전 필터를 못 끼워 넣는다.

    comut 에는 없는 계약이라 여기서 새로 못박는다 — `build_fold_*_block` 은 계약
    통일성 때문에 `y_train_fold` 를 받지만 진단에만 쓴다.
    """
    for function in (fit_latent_basis, fit_gene_modules):
        parameters = inspect.signature(function).parameters
        assert not [p for p in parameters if p in ("y", "y_train_fold", "labels")]


def test_latent_basis_unchanged_by_valid_row_content():
    disturbed = _MOD_MATRIX.copy()
    disturbed[8:] = 1.0  # valid 행을 전부 흔든다
    a = fit_latent_basis(_MOD_MATRIX, _MOD_TRAIN_INDEX, **_LATENT_KWARGS)
    b = fit_latent_basis(disturbed, _MOD_TRAIN_INDEX, **_LATENT_KWARGS)
    assert np.array_equal(a.components, b.components)
    assert np.array_equal(a.gene_index, b.gene_index)


def test_latent_basis_responds_to_fit_data():
    """위 테스트가 공허하지 않으려면 fit 데이터에는 실제로 반응해야 한다."""
    erased = _MOD_MATRIX.copy()
    erased[:8, :2] = 0.0
    a = fit_latent_basis(_MOD_MATRIX, _MOD_TRAIN_INDEX, **_LATENT_KWARGS)
    b = fit_latent_basis(erased, _MOD_TRAIN_INDEX, **_LATENT_KWARGS)
    assert not np.array_equal(a.components, b.components)


def test_latent_basis_unchanged_by_label_shuffle():
    """라벨을 섞어도 기저가 같아야 비지도 계약이 지켜진 것이다."""
    _, train_a, _, _, basis_a = build_fold_latent_block(
        _MOD_MATRIX, _MOD_MATRIX[:4], _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_LATENT_KWARGS
    )
    _, train_b, _, _, basis_b = build_fold_latent_block(
        _MOD_MATRIX, _MOD_MATRIX[:4], _MOD_TRAIN_INDEX, _MOD_Y_FOLD[::-1], **_LATENT_KWARGS
    )
    assert np.array_equal(basis_a.components, basis_b.components)
    assert np.array_equal(train_a, train_b)


@pytest.mark.parametrize("method", ["svd", "nmf"])
def test_latent_transform_is_row_independent(method):
    """행 하나만 넣은 값 == 전체를 넣었을 때 그 행.

    NMF 가 이 파일에서 제일 안 자명한 지점이다 — `components` 를 고정하고 W 를 푸는
    건 행마다 독립인 문제라서 다른 행이 값에 못 끼어든다. 주장으로 두지 않고 잰다.
    """
    basis = fit_latent_basis(
        _MOD_MATRIX, _MOD_TRAIN_INDEX, method=method, **_LATENT_KWARGS
    )
    full = transform_latent_basis(_MOD_MATRIX, basis)
    for row in (0, 5, 9):
        single = transform_latent_basis(_MOD_MATRIX[[row]], basis)
        assert np.allclose(single[0], full[row], atol=1e-5)


def test_latent_returns_every_train_row():
    """반환하는 train 행렬은 valid 행을 포함한 전체다 (fold 루프가 뒤에서 자른다)."""
    names, train_out, test_out, diagnostics, _ = build_fold_latent_block(
        _MOD_MATRIX, _MOD_MATRIX[:3], _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_LATENT_KWARGS
    )
    assert train_out.shape == (_MOD_MATRIX.shape[0], len(names))
    assert test_out.shape == (3, len(names))
    assert len(diagnostics) == len(names)
    assert np.abs(train_out[8:]).sum() > 0, "valid 행도 transform 은 받아야 한다"


def test_latent_width_is_exactly_n_components():
    """폭이 fold 마다 고정이라야 f4r 과의 델타가 차원 변동에 안 오염된다."""
    for n in (2, 3, 5):
        names, train_out, _, _, _ = build_fold_latent_block(
            _MOD_MATRIX,
            _MOD_MATRIX[:3],
            _MOD_TRAIN_INDEX,
            _MOD_Y_FOLD,
            gene_names=_MOD_GENES,
            n_components=n,
            min_gene_support=1,
        )
        assert len(names) == n == train_out.shape[1]


def test_latent_unseen_gene_pattern_stays_finite():
    """fold-train 에서 빠진 유전자만 켜진 test 행도 유한한 값이어야 한다."""
    basis = fit_latent_basis(
        _MOD_MATRIX,
        _MOD_TRAIN_INDEX,
        gene_names=_MOD_GENES,
        n_components=3,
        min_gene_support=3,
    )
    unseen = np.zeros((1, len(_MOD_GENES)), dtype=np.float32)
    unseen[0, -1] = 1.0
    out = transform_latent_basis(unseen, basis)
    assert np.isfinite(out).all()


def test_l2_row_norm_makes_values_dilation_invariant():
    """**시프트 테스트.** test 의 2.21배 희석을 실행 가능한 단언으로 박는다.

    같은 행에 유전자를 더 켠 '희석본' 을 만들고, `row_norm="l2"` 의 출력이 `"none"`
    보다 덜 움직이는지 본다. 이게 소프트 잠재 쪽 시프트 논증의 전부다. 실측으로도
    확인됐다 — 선두 성분의 |burden 상관| 이 l2 에서 0.458, none 에서 0.995 다
    (`scripts/inspect_latent.py`).
    """
    row = np.zeros((1, len(_MOD_GENES)), dtype=np.float32)
    row[0, :2] = 1.0
    diluted = row.copy()
    diluted[0, 2:6] = 1.0  # 변이 유전자 집합이 균일하게 커진 상황

    moved = {}
    for row_norm in ("l2", "none"):
        basis = fit_latent_basis(
            _MOD_MATRIX, _MOD_TRAIN_INDEX, row_norm=row_norm, **_LATENT_KWARGS
        )
        base = transform_latent_basis(row, basis)
        after = transform_latent_basis(diluted, basis)
        moved[row_norm] = float(np.linalg.norm(after - base))

    assert moved["l2"] < moved["none"], moved


def test_module_membership_covers_every_gene_exactly_once():
    """하드 모듈은 배타적 분할이다 — 지지도를 통과한 유전자는 정확히 한 모듈에."""
    modules = fit_gene_modules(_MOD_MATRIX, _MOD_TRAIN_INDEX, **_MODULE_KWARGS)
    assert modules.labels.size == modules.gene_index.size
    assert modules.module_sizes.sum() == modules.gene_index.size
    assert len(set(modules.gene_index.tolist())) == modules.gene_index.size


def test_module_assignment_unchanged_by_valid_row_content():
    disturbed = _MOD_MATRIX.copy()
    disturbed[8:] = 1.0
    a = fit_gene_modules(_MOD_MATRIX, _MOD_TRAIN_INDEX, **_MODULE_KWARGS)
    b = fit_gene_modules(disturbed, _MOD_TRAIN_INDEX, **_MODULE_KWARGS)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.gene_index, b.gene_index)


def test_share_is_dilation_invariant_but_controls_are_not():
    """대조군이 진짜 대조군인지 잰다 — 이 파일에서 제일 중요한 값 형태 테스트다.

    한 행의 변이 유전자를 균일하게 늘려도 `share`/`enrich` 는 모듈별 몫이 유지돼야
    하고, `fraction`/`any`/`logcount`/`wburden` 은 움직여야 한다. `fraction` 이 요점이다 —
    비율처럼 생겼지만 분모가 모듈 크기라는 **상수**라서 분자만 부푼다.
    """
    modules = fit_gene_modules(_MOD_MATRIX, _MOD_TRAIN_INDEX, **_MODULE_KWARGS)
    # 모듈 0 에 든 유전자와 그 밖 유전자를 같은 비율로 늘린다.
    members = np.where(modules.labels == 0)[0]
    others = np.where(modules.labels != 0)[0]
    if members.size < 2 or others.size < 2:
        pytest.skip("장난감 데이터의 모듈 크기가 이 검사에 부족하다")

    base = np.zeros((1, len(_MOD_GENES)), dtype=np.float32)
    base[0, modules.gene_index[members[:1]]] = 1.0
    base[0, modules.gene_index[others[:1]]] = 1.0
    diluted = base.copy()
    diluted[0, modules.gene_index[members[1:2]]] = 1.0
    diluted[0, modules.gene_index[others[1:2]]] = 1.0

    for value in ("share", "enrich"):
        a = transform_gene_modules(base, modules, value=value)
        b = transform_gene_modules(diluted, modules, value=value)
        assert np.allclose(a, b, atol=1e-6), f"{value} 가 희석에 흔들린다"

    moved = [
        not np.allclose(
            transform_gene_modules(base, modules, value=value),
            transform_gene_modules(diluted, modules, value=value),
        )
        for value in ("fraction", "logcount", "wburden")
    ]
    assert all(moved), "대조군이 희석에 안 움직이면 대조군이 아니다"


def test_module_value_modes_emit_expected_names():
    """7개 값 방식이 전부 동작하고 이름에 방식이 드러나야 한다."""
    for value in MODULE_VALUES:
        names, train_out, test_out, _, _ = build_fold_module_block(
            _MOD_MATRIX,
            _MOD_MATRIX[:3],
            _MOD_TRAIN_INDEX,
            _MOD_Y_FOLD,
            value=value,
            **_MODULE_KWARGS,
        )
        assert all(n.startswith(f"mod_{value}__") for n in names)
        assert train_out.shape == (_MOD_MATRIX.shape[0], len(names))
        assert test_out.shape == (3, len(names))
        assert np.isfinite(train_out).all()


# --- 클래스 서명 (지도) ------------------------------------------------------
# 위 모듈 픽스처를 그대로 쓴다. fold-train(행 0-7)에는 X·Y 두 클래스만 있고 Z 는
# valid 전용이다. 클래스별 변이율은 이렇게 갈린다:
#
#     GA·GB  X 3/3, Y 0/5    GD·GE  Y 3/5, X 0/3    GF  Y 2/5, X 0/3
#     GC     X 1/3, Y 2/5 (대비가 거의 없다)        GG·GH  valid 행에만
#
# 행 2 는 변이 유전자가 3개로 fold-train 상위 분위라 과변이로 잡힌다 — GC 의 X 쪽
# 변이는 그 한 행이 전부라 과변이 가드에 걸린다.
_SIG_KWARGS = dict(gene_names=_MOD_GENES, min_class_support=2, min_lift=0.3)


def test_signature_selection_takes_no_test_matrix():
    """지도 블록이라도 test 를 안 받는 계약은 같다."""
    parameters = inspect.signature(select_class_signatures).parameters
    assert not [p for p in parameters if "test" in p.lower()]


def test_signature_selection_requires_a_label_argument():
    """비지도 블록과 반대 방향의 계약이다 — 여기서는 `y` 가 **필수 인자**여야 한다.

    `fit_latent_basis`·`fit_gene_modules` 는 `y` 를 아예 안 받는 것이 계약이고
    (`test_fit_functions_take_no_label_argument`), 이 블록은 라벨을 본다는 사실이
    시그니처에 드러나야 한다. 지도/비지도가 같은 파일에 섞이면 그 구분이 흐려진다.
    """
    parameters = inspect.signature(select_class_signatures).parameters
    assert "y_train_fold" in parameters
    assert parameters["y_train_fold"].default is inspect.Parameter.empty


def test_signature_unchanged_by_valid_row_content():
    disturbed = _MOD_MATRIX.copy()
    disturbed[8:] = 1.0  # valid 행을 전부 흔든다
    a = select_class_signatures(_MOD_MATRIX, _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_SIG_KWARGS)
    b = select_class_signatures(disturbed, _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_SIG_KWARGS)
    assert [s["genes"] for s in a.stats] == [s["genes"] for s in b.stats]


def test_signature_responds_to_fit_labels():
    """위 테스트가 공허하지 않으려면 라벨에는 실제로 반응해야 한다."""
    a = select_class_signatures(_MOD_MATRIX, _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_SIG_KWARGS)
    b = select_class_signatures(
        _MOD_MATRIX, _MOD_TRAIN_INDEX, _MOD_Y_FOLD[::-1], **_SIG_KWARGS
    )
    assert [s["genes"] for s in a.stats] != [s["genes"] for s in b.stats]


def test_signature_rejects_mismatched_label_length():
    """라벨이 밀린 채로 조용히 학습되는 것보다 멈추는 쪽이 낫다."""
    with pytest.raises(ValueError, match="y_train_fold"):
        select_class_signatures(
            _MOD_MATRIX, _MOD_TRAIN_INDEX, _MOD_Y[:5], **_SIG_KWARGS
        )


def test_signature_picks_the_genes_that_separate_classes():
    """선택 규칙이 실제로 동작하는지 — 이게 안 맞으면 나머지 테스트가 다 공허하다."""
    signatures = select_class_signatures(
        _MOD_MATRIX, _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_SIG_KWARGS
    )
    picked = {s["class"]: set(s["genes"]) for s in signatures.stats}
    assert picked["X"] == {"GA", "GB"}
    assert picked["Y"] == {"GD", "GE", "GF"}
    # valid 전용 유전자는 어느 클래스에도 못 들어간다.
    assert not any({"GG", "GH"} & genes for genes in picked.values())


def test_signature_hyper_guard_drops_hypermutation_driven_genes():
    """과변이 가드가 없으면 서명이 '이 클래스는 변이가 많다' 를 다시 배운다.

    `GB` 는 P 클래스의 과변이 행 하나에서만 켜진다. lift 는 0.33 으로 문턱을 넘지만
    그 대비가 통째로 과변이 행 하나에서 온 것이라 가드가 걸러야 한다. 같은 픽스처의
    `GA` 는 평범한 행에도 있어서 남아야 한다 — 가드가 다 죽이는 게 아니라는 대조다.
    """
    genes = ["GA", "GB", "GC", "GD"]
    matrix = np.array(
        [
            [1, 0, 0, 0],  # P
            [1, 0, 0, 0],  # P
            [1, 1, 1, 0],  # P — 변이 3개, fold-train 상위 분위라 과변이
            [0, 0, 0, 1],  # Q
            [0, 0, 0, 1],  # Q
            [0, 0, 0, 1],  # Q
        ],
        dtype=np.float32,
    )
    y = np.array(["P", "P", "P", "Q", "Q", "Q"])
    index = np.arange(6)
    loose = dict(gene_names=genes, min_class_support=1, min_lift=0.1)
    without = select_class_signatures(
        matrix, index, y, max_hyper_fraction=1.0, **loose
    )
    with_guard = select_class_signatures(
        matrix, index, y, max_hyper_fraction=0.5, **loose
    )
    def p_genes(signatures):
        return {g for s in signatures.stats if s["class"] == "P" for g in s["genes"]}

    assert "GB" in p_genes(without)
    assert "GB" not in p_genes(with_guard)
    assert "GA" in p_genes(with_guard), "가드가 평범한 서명 유전자까지 죽이면 안 된다"


def test_signature_transform_is_row_independent():
    """행 하나만 넣은 값 == 전체를 넣었을 때 그 행."""
    signatures = select_class_signatures(
        _MOD_MATRIX, _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_SIG_KWARGS
    )
    full = transform_class_signatures(_MOD_MATRIX, signatures)
    for row in (0, 5, 9):
        single = transform_class_signatures(_MOD_MATRIX[[row]], signatures)
        assert np.allclose(single[0], full[row], atol=1e-6)


def test_signature_share_is_dilation_invariant_but_controls_are_not():
    """**리뷰가 요청한 핵심 검사.** signature_share 가 변이 부담에 안 흔들려야 한다.

    한 행의 변이 유전자를 서명 안팎으로 같은 비율씩 늘린다. `share` 는 분모가 행 전체의
    변이 유전자 수라 값이 그대로여야 하고, 단순 카운트 계열(`fraction`/`logcount`/
    `wburden`)은 움직여야 한다. test 의 변이 유전자 수가 train 의 2.21배라는 사실을
    실행 가능한 단언으로 박은 것이다.
    """
    signatures = select_class_signatures(
        _MOD_MATRIX, _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_SIG_KWARGS
    )
    inside = [_MOD_GENES.index(g) for g in ("GA", "GB")]  # X 서명
    outside = [_MOD_GENES.index(g) for g in ("GG", "GH")]  # 어느 서명에도 없다

    base = np.zeros((1, len(_MOD_GENES)), dtype=np.float32)
    base[0, [inside[0], outside[0]]] = 1.0
    diluted = base.copy()
    diluted[0, [inside[1], outside[1]]] = 1.0

    for value in ("share", "enrich"):
        a = transform_class_signatures(base, signatures, value=value)
        b = transform_class_signatures(diluted, signatures, value=value)
        assert np.allclose(a, b, atol=1e-6), f"{value} 가 희석에 흔들린다"

    moved = [
        not np.allclose(
            transform_class_signatures(base, signatures, value=value),
            transform_class_signatures(diluted, signatures, value=value),
        )
        for value in ("fraction", "logcount", "wburden")
    ]
    assert all(moved), "대조군이 희석에 안 움직이면 대조군이 아니다"


def test_signature_block_emits_one_column_per_fold_train_class():
    """폭이 fold-train 클래스 수로 고정이라야 f4r 과의 델타가 차원 변동에 안 오염된다."""
    names, train_out, test_out, diagnostics = build_fold_signature_block(
        _MOD_MATRIX, _MOD_MATRIX[:3], _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_SIG_KWARGS
    )
    assert names == ["sig_share__X", "sig_share__Y"]
    assert train_out.shape == (_MOD_MATRIX.shape[0], 2)
    assert test_out.shape == (3, 2)
    assert len(diagnostics) == 2
    assert np.isfinite(train_out).all()
    assert train_out[8:].sum() > 0, "valid 행도 transform 은 받아야 한다"


def test_signature_unseen_gene_pattern_stays_finite():
    """서명 유전자가 하나도 안 켜진 행도, 변이가 아예 없는 행도 유한해야 한다."""
    signatures = select_class_signatures(
        _MOD_MATRIX, _MOD_TRAIN_INDEX, _MOD_Y_FOLD, **_SIG_KWARGS
    )
    empty = np.zeros((2, len(_MOD_GENES)), dtype=np.float32)
    empty[1, -1] = 1.0  # 서명 밖 유전자만 켜진 행
    for value in MODULE_VALUES:
        out = transform_class_signatures(empty, signatures, value=value)
        assert np.isfinite(out).all(), value


# --- stem 슬러그 -----------------------------------------------------------
def _load_train_gbdt():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_gbdt.py"
    spec = importlib.util.spec_from_file_location("train_gbdt_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_gbdt = _load_train_gbdt()


def test_comut_slug_empty_without_pair_block():
    """comut 블록이 없는 config 의 stem 이 디스크의 기존 로그와 바이트 동일해야 한다."""
    assert train_gbdt._comut_slug({}, manual=True) == ""
    assert train_gbdt._comut_slug({}, manual=False) == ""


def test_comut_slug_distinguishes_all_params():
    """11개 파라미터를 하나씩 바꾼 dict 가 전부 다른 슬러그를 내야 한다."""
    base = dict(
        pool="drivers",
        pool_topk=300,
        mode="mutated",
        value="share",
        topk=20,
        min_support=20,
        min_class_support=8,
        min_purity=0.25,
        min_lift=0.15,
        max_hyper_fraction=0.50,
        max_pairs_per_gene=3,
    )
    seen = {train_gbdt._comut_slug(base, manual=True)}
    variants = [
        {**base, "pool": "chi2"},
        {**base, "pool_topk": 100},
        {**base, "mode": "functional"},
        {**base, "value": "and"},
        {**base, "value": "gate"},
        {**base, "topk": 10},
        {**base, "min_support": 25},
        {**base, "min_class_support": 5},
        {**base, "min_purity": 0.3},
        {**base, "min_lift": 0.2},
        {**base, "max_hyper_fraction": 0.6},
        {**base, "max_pairs_per_gene": 2},
    ]
    for variant in variants:
        slug = train_gbdt._comut_slug(variant, manual=True)
        assert slug not in seen, f"충돌: {variant}"
        seen.add(slug)
    manual_off = train_gbdt._comut_slug(base, manual=False)
    assert manual_off not in seen


def test_latent_and_module_slugs_empty_without_their_blocks():
    """잠재·모듈 블록이 없는 config 의 stem 이 디스크의 기존 로그와 바이트 동일해야 한다.

    이 파일에 이미 로그가 129개 쌓여 있어서, 슬러그가 빈 dict 에 `""` 를 안 내면
    과거 실험과 이름이 어긋나 비교가 통째로 끊긴다.
    """
    assert train_gbdt._latent_slug({}) == ""
    assert train_gbdt._module_slug({}) == ""
    assert train_gbdt._signature_slug({}) == ""


def test_latent_slug_distinguishes_all_params():
    base = dict(
        method="svd",
        n_components=64,
        row_norm="l2",
        value="proj",
        mode="mutated",
        gene_weight="none",
        min_gene_support=5,
        random_state=0,
    )
    seen = {train_gbdt._latent_slug(base)}
    variants = [
        {**base, "method": "nmf"},
        {**base, "n_components": 32},
        {**base, "row_norm": "none"},
        {**base, "value": "share"},
        {**base, "mode": "functional"},
        {**base, "gene_weight": "idf"},
        {**base, "min_gene_support": 10},
        {**base, "random_state": 1},
    ]
    for variant in variants:
        slug = train_gbdt._latent_slug(variant)
        assert slug not in seen, f"충돌: {variant}"
        seen.add(slug)


def test_latent_method_is_resolved_per_block():
    """`lnmf` 는 자기 이름으로 방식을 정하고 나머지는 `--latent-method` 를 따른다.

    config 전체에 방식 하나만 정하면 `lsvd`·`lnmf`가 같은 config 에 있을 때 둘 다
    nmf 로 돌아간다 — 실제로 있었던 버그다.
    """
    assert train_gbdt._resolve_latent_methods(["lsvd"], "svd") == ["svd"]
    assert train_gbdt._resolve_latent_methods(["lnmf"], "svd") == ["nmf"]
    assert train_gbdt._resolve_latent_methods(["lsvd", "lnmf"], "svd") == ["svd", "nmf"]
    assert train_gbdt._resolve_latent_methods(["lnmf", "lsvd"], "svd") == ["nmf", "svd"]


def test_latent_slug_is_byte_identical_for_single_block():
    """단일 블록 슬러그가 디스크의 기존 로그 파일명과 문자 그대로 같아야 한다.

    `xgb_lat_f4rl_..._lt64svdl25beb91_...json` · `xgb_lat_f4rn_..._lt64nmfl2bb3575_...json`
    이 이미 쌓여 있다. 잠재 방식을 블록별로 다시 푸는 리팩터 뒤에도 이 문자열이
    그대로 나와야 비교표가 안 끊긴다.
    """
    base = dict(
        n_components=64, row_norm="l2", value="proj", mode="mutated",
        gene_weight="none", min_gene_support=5, random_state=0,
    )
    svd_kwargs = {**base, "method": "".join(dict.fromkeys(
        train_gbdt._resolve_latent_methods(["lsvd"], "svd")
    ))}
    nmf_kwargs = {**base, "method": "".join(dict.fromkeys(
        train_gbdt._resolve_latent_methods(["lnmf"], "svd")
    ))}
    assert train_gbdt._latent_slug(svd_kwargs) == "_lt64svdl25beb91"
    assert train_gbdt._latent_slug(nmf_kwargs) == "_lt64nmfl2bb3575"


def test_latent_slug_distinguishes_combined_methods():
    """svd·nmf·svdnmf 세 슬러그가 서로 달라야 한다 — 조합 config 가 단일 config 를 안 덮는다."""
    base = dict(
        n_components=64, row_norm="l2", value="proj", mode="mutated",
        gene_weight="none", min_gene_support=5, random_state=0,
    )
    slugs = set()
    for blocks in (["lsvd"], ["lnmf"], ["lsvd", "lnmf"]):
        methods = train_gbdt._resolve_latent_methods(blocks, "svd")
        kwargs = {**base, "method": "".join(dict.fromkeys(methods))}
        slug = train_gbdt._latent_slug(kwargs)
        assert slug not in slugs, f"충돌: {blocks} -> {slug}"
        slugs.add(slug)


def test_module_slug_distinguishes_all_params():
    base = dict(
        value="share",
        n_modules=24,
        svd_components=64,
        mode="mutated",
        min_gene_support=5,
        random_state=0,
    )
    seen = {train_gbdt._module_slug(base)}
    for variant in (
        {**base, "value": "fraction"},
        {**base, "n_modules": 12},
        {**base, "svd_components": 32},
        {**base, "mode": "functional"},
        {**base, "min_gene_support": 10},
        {**base, "random_state": 1},
    ):
        slug = train_gbdt._module_slug(variant)
        assert slug not in seen, f"충돌: {variant}"
        seen.add(slug)


def test_signature_slug_distinguishes_all_params_and_never_collides_with_module():
    base = dict(
        value="share",
        topk=30,
        mode="mutated",
        min_class_support=5,
        min_lift=0.05,
        max_hyper_fraction=0.50,
    )
    seen = {train_gbdt._signature_slug(base)}
    for variant in (
        {**base, "value": "fraction"},
        {**base, "topk": 10},
        {**base, "mode": "functional"},
        {**base, "min_class_support": 8},
        {**base, "min_lift": 0.10},
        {**base, "max_hyper_fraction": 0.30},
    ):
        slug = train_gbdt._signature_slug(variant)
        assert slug not in seen, f"충돌: {variant}"
        seen.add(slug)

    # gmod 와 csig 는 접두사로 갈린다 — 두 블록이 한 config 에 같이 들어가도 겹치면
    # 안 된다. 슬러그는 이어 붙으므로 접두사만 다르면 충분하다.
    module = train_gbdt._module_slug(
        dict(
            value="share",
            n_modules=24,
            svd_components=64,
            mode="mutated",
            min_gene_support=5,
            random_state=0,
        )
    )
    assert module.startswith("_gm")
    assert train_gbdt._signature_slug(base).startswith("_sg")
    assert module not in seen


def test_every_config_block_belongs_to_a_known_family():
    """config 이 참조하는 블록이 전부 어느 가족엔가 등록돼 있어야 한다.

    `FOLD_MATRIX_BLOCKS` 에 안 넣은 블록은 `Dataset` 이 dense 주머니에 담고
    `assemble` 이 그대로 붙인다 — 예외도 안 나고 fold 로직도 돌지만 fold 안에서
    만들어야 할 열이 fold 밖에서 만들어진다. 조용히 새는 자리라 여기서 막는다.
    """
    known = (
        set(train_gbdt.BLOCK_SOURCES)
        | set(train_gbdt.SPARSE_SOURCES)
        | {"domain"}
    )
    for config, spec in train_gbdt.CONFIGS.items():
        for block in spec["blocks"]:
            assert block in known, f"{config} 의 {block} 이 소스 표에 없다"
            assert block in train_gbdt.BLOCK_DESC, f"{block} 설명이 없다"


def test_f11_uses_only_cached_blocks():
    """gtype·parsed19·burden8·aa9·ptok 은 test 쪽 parquet 이 없어 즉시 멈춘다.

    앙상블 입력용 config 는 지금 캐시된 것만 써야 `--configs f11` 이 바로 돈다.
    """
    uncached = {"gtype", "parsed19", "burden8", "aa9", "ptok"}
    assert not (set(train_gbdt.CONFIGS["f11"]["blocks"]) & uncached)


def test_f11_covers_every_cached_block():
    """11개가 다 들어 있어야 한다 — 누가 하나 빼면 앙상블 입력 열이 조용히 줄어든다."""
    expected = {
        "domain", "rollup16", "enc3", "gec", "sigtok", "exacttok",
        "comut", "lsvd", "lnmf", "gmod", "csig",
    }
    assert set(train_gbdt.CONFIGS["f11"]["blocks"]) == expected


def test_f16_matches_the_shared_teammate_config():
    """팀원 `repo_allfeat` 재현용이라 블록 구성이 그쪽 config.json 과 같아야 한다.

    스태킹 멤버로 쓰려면 "같은 피처를 우리 fold 로 다시 뽑은 것"이어야 하는데,
    블록이 하나라도 어긋나면 그 전제가 조용히 깨진다. 순서까지 고정한다 — 열 순서가
    바뀌면 CatBoost 의 `rsm`(열 샘플링)이 다른 열을 뽑는다.
    """
    shared = (
        "domain", "rollup", "enc3", "gec", "gtype", "parsed19", "burden8",
        "aa9", "sigtok", "exacttok", "ptok", "comut", "lsvd", "lnmf",
        "gmod", "csig",
    )
    assert train_gbdt.CONFIGS["f16"]["blocks"] == shared


def test_f16_runs_both_latent_methods():
    """팀원 실행은 lsvd·lnmf 동거 버그로 NMF 만 두 번 돌았다(슬러그 lt64nmf...).

    고쳐진 코드에서는 SVD 와 NMF 가 각각 돌아야 한다. 여기가 되돌아가면 f16 이
    재현하려던 16블록이 실제로는 15블록 + 중복 64열이 된다.
    """
    latent = [b for b in train_gbdt.CONFIGS["f16"]["blocks"] if b in train_gbdt.LATENT_BLOCKS]
    assert train_gbdt._resolve_latent_methods(latent, "svd") == ["svd", "nmf"]
