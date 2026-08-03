"""변이 문서(TF-IDF 입력) 계약 — 원문 문서와 서명 문서.

1. 서명 문서는 셀 안에서 이소폼 중복을 접는다. 접힌 결과가 유전자 경계를 넘지
   않는다.
2. 한 문서에 같은 토큰이 두 번 나오지 않는다(TF 가 항상 1). TF-IDF 가 이진
   지시자 x IDF 가 되는 근거이므로 깨지면 문서 길이 시프트가 되살아난다.
3. 토큰 순서가 실행마다 같다. 순서가 흔들리면 parquet 바이트가 매번 달라져
   재현 비교가 안 된다.
4. 원문 문서의 출력은 리팩터링 전후로 한 글자도 안 바뀐다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cancer_hack.features_basic import (
    EXACT_MUTATION_NO_MUTATION_TOKEN,
    SIGNATURE_MUTATION_PREFIX,
    build_exact_mutation_documents,
    build_unique_mutation_documents,
    row_to_exact_mutation_document,
    row_to_unique_mutation_document,
)

GENES = ["TP53", "KRAS", "EGFR"]


def _row(**values) -> pd.Series:
    return pd.Series({gene: values.get(gene, "WT") for gene in GENES})


def test_unique_document_collapses_isoform_duplicates():
    """`M267I M206I` 는 같은 변이를 다른 전사체 좌표로 적은 것이다."""
    document = row_to_unique_mutation_document(_row(TP53="M267I M206I"), GENES)
    assert document == "SIG__TP53__missense|M>I"


def test_unique_document_keeps_distinct_signatures():
    document = row_to_unique_mutation_document(_row(TP53="Q117R K392E K324E"), GENES)
    assert document.split() == ["SIG__TP53__missense|Q>R", "SIG__TP53__missense|K>E"]


def test_unique_document_is_gene_scoped():
    """유전자가 다르면 서명이 같아도 접지 않는다 — 다른 사건이다."""
    document = row_to_unique_mutation_document(_row(TP53="V600E", KRAS="V600E"), GENES)
    assert document.split() == ["SIG__TP53__missense|V>E", "SIG__KRAS__missense|V>E"]


def test_unique_document_folds_stop_codon_notation():
    """train `*` 와 test `X` 가 한 셀에 섞여도 하나로 접힌다."""
    document = row_to_unique_mutation_document(_row(TP53="Q369* Q369X"), GENES)
    assert document == "SIG__TP53__nonsense|Q"


def test_unique_document_term_frequency_is_one():
    """문서에 중복 토큰이 없어야 TF-IDF 가 이진 지시자 x IDF 가 된다."""
    row = _row(TP53="M267I M206I V600E", KRAS="K16fs K437Rfs", EGFR="R649del L454del")
    tokens = row_to_unique_mutation_document(row, GENES).split()
    assert len(tokens) == len(set(tokens))


def test_unique_document_order_is_deterministic():
    """`set` 을 순회하면 순서가 흔들려 parquet 바이트가 매번 달라진다."""
    row = _row(TP53="Q117R K392E V600E R175H", KRAS="G12D G13C")
    first = row_to_unique_mutation_document(row, GENES)
    for _ in range(50):
        assert row_to_unique_mutation_document(row, GENES) == first


def test_unique_document_order_follows_gene_columns():
    row = _row(TP53="V600E", KRAS="Q369*")
    forward = row_to_unique_mutation_document(row, ["TP53", "KRAS"])
    backward = row_to_unique_mutation_document(row, ["KRAS", "TP53"])
    assert forward.split() == backward.split()[::-1]


def test_unique_document_keeps_first_occurrence_order():
    """접힌 뒤에도 원문에서 먼저 나온 서명이 앞에 온다."""
    row = _row(TP53="K392E Q117R K324E")
    assert row_to_unique_mutation_document(row, GENES).split() == [
        "SIG__TP53__missense|K>E",
        "SIG__TP53__missense|Q>R",
    ]


def test_unique_document_uses_the_shared_no_mutation_sentinel():
    """두 문서가 같은 센티넬을 써야 무변이 행이 두 어휘 모두에서 한 열이 된다."""
    row = _row()
    assert row_to_unique_mutation_document(row, GENES) == (
        EXACT_MUTATION_NO_MUTATION_TOKEN
    )
    assert row_to_exact_mutation_document(row, GENES) == (
        EXACT_MUTATION_NO_MUTATION_TOKEN
    )


def test_unique_document_prefix_does_not_collide_with_exact():
    """두 블록을 한 행렬에 이어 붙여도 피처 이름이 안 겹쳐야 한다."""
    row = _row(TP53="V600E")
    assert row_to_unique_mutation_document(row, GENES).startswith(
        SIGNATURE_MUTATION_PREFIX
    )
    assert row_to_exact_mutation_document(row, GENES).startswith("MUT__")


def test_exact_document_is_unchanged_by_the_refactor(toy_frame):
    """공통부를 `_make_document_parquet` 으로 뽑아도 원문 출력은 그대로다."""
    documents = build_exact_mutation_documents(toy_frame, GENES)
    assert list(documents) == [
        "MUT__TP53__E412K MUT__TP53__R1800C MUT__KRAS__S622S MUT__KRAS__G827R"
        " MUT__EGFR__312_313QY>HH",
        "MUT__KRAS__V600E MUT__KRAS__V600E MUT__EGFR__R649del",
        "MUT__TP53__Q369* MUT__TP53__I368N MUT__KRAS__K16fs MUT__EGFR__P11_K12insP",
        EXACT_MUTATION_NO_MUTATION_TOKEN,
    ]


def test_both_builders_return_one_document_per_row(toy_frame):
    exact = build_exact_mutation_documents(toy_frame, GENES)
    unique = build_unique_mutation_documents(toy_frame, GENES)
    assert len(exact) == len(unique) == len(toy_frame)


def test_unique_document_is_never_longer_than_exact(toy_frame):
    """서명 문서는 접기만 하므로 토큰 수가 늘어날 수 없다."""
    exact = build_exact_mutation_documents(toy_frame, GENES)
    unique = build_unique_mutation_documents(toy_frame, GENES)
    for left, right in zip(exact, unique):
        assert len(right.split()) <= len(left.split())


@pytest.mark.parametrize(
    "builder", [build_exact_mutation_documents, build_unique_mutation_documents]
)
def test_missing_gene_column_raises(toy_frame, builder):
    with pytest.raises(ValueError, match="Missing gene columns"):
        builder(toy_frame, ["TP53", "NOPE"])
