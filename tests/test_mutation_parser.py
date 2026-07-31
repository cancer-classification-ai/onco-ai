"""복합 변이 처리 전략이 표대로 구현됐는지 검증한다.

1차 전략표의 예시 다섯 줄을 그대로 케이스로 박아 뒀다. 파서를 고칠 때 이 표가
깨지면 전략과 구현이 어긋난 것이다.
"""

from __future__ import annotations

import pytest

import re

from cancer_hack.parser import (
    CELL_FEATURE_COLUMNS,
    DELETION,
    DELINS,
    INDEL_KINDS,
    INSERTION,
    classify_token,
    parse_cell,
    split_tokens,
    token_signature,
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


# --- 서명(위치 무시 중복 판정) ----------------------------------------------
# 잔기 번호를 지우고 유형과 잔기만 남긴다. test 가 같은 변이를 전사체마다 다른
# 좌표로 반복 기재하는 걸 잡으려는 것이다.
SIGNATURE_TABLE = [
    ("V600E", "missense|V>E"),
    ("M267I", "missense|M>I"),
    ("M206I", "missense|M>I"),  # 위와 같은 사건, 다른 전사체 좌표
    ("S622S", "synonymous|S"),
    ("*261*", "synonymous|*"),
    ("Q369*", "nonsense|Q"),
    ("Q369X", "nonsense|Q"),  # train `*` / test `X`
    ("K16fs", "frameshift|K"),  # train 형태 (alt 없음)
    ("K437Rfs", "frameshift|K"),  # test 형태 (alt 있음)
    ("-287fs", "frameshift|-"),
    ("L454del", "deletion|Ldel"),
    ("A630_P649del", "deletion|A_Pdel"),
    ("K538_G539insK", "insertion|K_GinsK"),
    ("R376_A377delinsP", "delins|R_AdelinsP"),
    ("X541delinsX", "delins|XdelinsX"),  # `X` 를 전역 치환하면 안 되는 이유
    ("312_313QY>HH", "complex|QY>HH"),
    ("468_469LG>F*", "complex|LG>F*"),
    ("M391", "other|M"),
]


@pytest.mark.parametrize("token,signature", SIGNATURE_TABLE)
def test_signature_table_rows(token, signature):
    assert token_signature(token) == signature


def test_signature_ignores_residue_position():
    """test 다중 토큰 셀의 94.8% 가 같은 변이를 다른 전사체 좌표로 적은 것이다.

    이게 접히지 않으면 이소폼 중복이 서로 다른 변이로 세어진다.
    """
    assert token_signature("M267I") == token_signature("M206I")


def test_stop_codon_notation_folds_to_one_signature():
    """`*` 와 `X` 가 갈리면 train 과 test 의 nonsense 어휘가 통째로 어긋난다."""
    assert token_signature("Q369*") == token_signature("Q369X")


def test_frameshift_signature_drops_alt_residue():
    """train frameshift 9,911 토큰은 전부 alt 가 없고 test 22,221 개는 alt 가 있다.

    alt 를 서명에 남기면 두 집합의 frameshift 어휘가 겹치지 않아 축이 죽는다.
    """
    assert token_signature("K16fs") == token_signature("K437Rfs")


@pytest.mark.parametrize("token,_", SIGNATURE_TABLE)
def test_signature_never_keeps_a_residue_number(token, _):
    """숫자가 남으면 위치를 지운다는 계약이 깨진 것이다."""
    assert not re.search(r"\d", token_signature(token))


@pytest.mark.parametrize("token,_", SIGNATURE_TABLE)
def test_signature_prefix_is_the_classified_kind(token, _):
    """서명은 분류를 다시 하지 않는다 — `classify_token` 의 결과를 그대로 쓴다."""
    assert token_signature(token).split("|")[0] == classify_token(token)


@pytest.mark.parametrize("token,_", SIGNATURE_TABLE)
def test_signature_kind_hint_matches_autodetect(token, _):
    """`kind` 를 넘겨 재분류를 건너뛰어도 결과가 같아야 한다."""
    assert token_signature(token, classify_token(token)) == token_signature(token)


# 값 -> (duplicate_token_count, duplicate_signature_count)
DUPLICATE_TABLE = [
    ("E412K R1800C", 0, 0),
    ("V600E V600E", 1, 1),  # 문자열까지 같은 중복
    ("M267I M206I", 0, 1),  # 이소폼 중복 — 문자열 기준으로는 안 잡힌다
    ("Q369* Q369X", 0, 1),  # train/test 표기 혼재
    ("K16fs K437Rfs", 0, 1),
    ("Q117R K392E K324E K303E", 0, 2),  # K>E 셋이 하나로 접힌다
    ("WT", 0, 0),
    ("", 0, 0),
]


@pytest.mark.parametrize("value,dup_token,dup_signature", DUPLICATE_TABLE)
def test_duplicate_counts(value, dup_token, dup_signature):
    cell = parse_cell(value)
    assert cell.duplicate_token_count == dup_token
    assert cell.duplicate_signature_count == dup_signature


@pytest.mark.parametrize("value,_,__", DUPLICATE_TABLE)
def test_signature_duplicates_never_undercount_exact(value, _, __):
    """서명 dedup 이 문자열 dedup 보다 거치니 이 부등식은 깨질 수 없다."""
    cell = parse_cell(value)
    assert cell.duplicate_signature_count >= cell.duplicate_token_count


def test_has_duplicate_token_agrees_with_the_count():
    """두 이름이 같은 사실을 말하므로 어긋나면 한쪽이 잘못 채워진 것이다."""
    for value in [v for v, _, _ in DUPLICATE_TABLE]:
        cell = parse_cell(value)
        assert cell.has_duplicate_token == int(cell.duplicate_token_count > 0)


def test_duplicate_counts_are_dataclass_fields():
    """`@property` 로 만들면 `asdict()` 에 안 들어가 피처 컬럼에서 사라진다."""
    assert "duplicate_token_count" in CELL_FEATURE_COLUMNS
    assert "duplicate_signature_count" in CELL_FEATURE_COLUMNS
    assert "duplicate_token_count" in parse_cell("V600E").as_dict()
