"""복합 변이 처리 전략이 표대로 구현됐는지 검증한다.

1차 전략표의 예시 다섯 줄을 그대로 케이스로 박아 뒀다. 파서를 고칠 때 이 표가
깨지면 전략과 구현이 어긋난 것이다.
"""

from __future__ import annotations

import pytest

from cancer_hack.parser import (
    DELETION,
    DELINS,
    INDEL_KINDS,
    INSERTION,
    classify_token,
    parse_cell,
    split_tokens,
)


# 전략표: 원본 값 -> (missense, synonymous, nonsense, frameshift, indel, mnv,
#                    token_count, unique_token_count, duplicate)
STRATEGY_TABLE = [
    ("E412K R1800C", 1, 0, 0, 0, 0, 0, 2, 2, 0),
    ("S622S G827R", 1, 1, 0, 0, 0, 0, 2, 2, 0),
    ("Q369* I368N", 1, 0, 1, 0, 0, 0, 2, 2, 0),
    ("V600E V600E", 1, 0, 0, 0, 0, 0, 2, 1, 1),
    ("312_313QY>HH", 0, 0, 0, 0, 0, 1, 1, 1, 0),
]


@pytest.mark.parametrize(
    "value,missense,synonymous,nonsense,frameshift,indel,mnv,tokens,unique,dup",
    STRATEGY_TABLE,
)
def test_strategy_table_rows(
    value, missense, synonymous, nonsense, frameshift, indel, mnv, tokens, unique, dup
):
    cell = parse_cell(value)
    assert cell.has_missense == missense
    assert cell.has_synonymous == synonymous
    assert cell.has_nonsense == nonsense
    assert cell.has_frameshift == frameshift
    assert cell.has_indel == indel
    assert cell.has_mnv == mnv
    assert cell.mutation_token_count == tokens
    assert cell.unique_mutation_token_count == unique
    assert cell.has_duplicate_token == dup


# 전략표 두 번째 표: 유형별 개수는 중복 토큰을 각각 센다.
@pytest.mark.parametrize(
    "value,missense,synonymous,nonsense",
    [
        ("E412K R1800C", 2, 0, 0),
        ("S622S G827R", 1, 1, 0),
        ("Q369* I368N", 1, 0, 1),
        ("V600E V600E", 2, 0, 0),
    ],
)
def test_event_counts(value, missense, synonymous, nonsense):
    cell = parse_cell(value)
    assert cell.missense_count == missense
    assert cell.synonymous_count == synonymous
    assert cell.nonsense_count == nonsense


def test_duplicate_token_is_collapsed_in_unique_counts():
    cell = parse_cell("V600E V600E")
    assert cell.missense_count == 2
    assert cell.unique_missense_count == 1


def test_wt_and_blank_produce_nothing():
    for value in ["WT", "", "  ", None]:
        cell = parse_cell(value)
        assert cell.mutation_token_count == 0
        assert cell.has_missense == 0
    assert split_tokens("WT") == []


def test_split_is_whitespace_only():
    assert split_tokens("Q369* I368N") == ["Q369*", "I368N"]


# --- test.csv 표기 변형 ------------------------------------------------------
# train 은 정지코돈을 `*`, test 는 `X` 로 적는다. 둘 다 nonsense 여야 한다.
@pytest.mark.parametrize("token", ["Q369*", "Q369X", "R213*", "R213X"])
def test_stop_codon_both_notations(token):
    assert classify_token(token) == "nonsense"


@pytest.mark.parametrize("token", ["K16fs", "L1854fs", "S221Kfs"])
def test_frameshift_notations(token):
    assert classify_token(token) == "frameshift"


@pytest.mark.parametrize(
    "token,kind",
    [
        ("R649del", DELETION),
        ("490del", DELETION),
        ("A630_P649del", DELETION),
        ("P11_K12insP", INSERTION),
        ("R376_A377delinsP", DELINS),
    ],
)
def test_indel_subtypes(token, kind):
    assert classify_token(token) == kind
    assert kind in INDEL_KINDS


def test_range_indels_are_not_counted_as_mnv():
    """`_` 를 쓴다고 다중 잔기 치환이 아니다.

    문자열 포함으로 판정하면 test 의 has_mnv 가 1.30% -> 24.35% 로 뛰는데,
    늘어난 토큰이 전부 범위 표기 indel 이다. 배타 분류를 유지해야 한다.
    """
    for token in ["P11_K12insP", "A630_P649del", "R376_A377delinsP"]:
        cell = parse_cell(token)
        assert cell.has_mnv == 0
        assert cell.has_indel == 1
    assert parse_cell("312_313QY>HH").has_mnv == 1


def test_missense_requires_different_residues():
    assert classify_token("G827R") == "missense"
    assert classify_token("S622S") == "synonymous"
    # 정지코돈 자리의 동의 변이. `other` 로 새면 안 된다.
    assert classify_token("*261*") == "synonymous"


def test_aggregate_names_agree_with_subtypes():
    """`indel_count` 는 세분 유형의 합이라 어긋날 수 없다."""
    cell = parse_cell("R649del P11_K12insP R376_A377delinsP")
    assert cell.deletion_count == 1
    assert cell.insertion_count == 1
    assert cell.delins_count == 1
    assert cell.indel_count == 3
    assert cell.explicit_deletion_count == 2  # del + delins, ins 는 제외
    assert cell.mnv_count == cell.complex_count


def test_functional_count_excludes_synonymous():
    cell = parse_cell("S622S G827R Q369*")
    assert cell.mutation_token_count == 3
    assert cell.synonymous_count == 1
    assert cell.functional_count == 2


def test_unknown_notation_lands_in_other_not_crash():
    cell = parse_cell("M391")
    assert cell.other_count == 1
    assert cell.functional_count == 0
