"""extract_token_string_features · extract_cell_string_features 통합 테스트.

정규표현식만으로 분해하는 변이 문자열 구조 피처 11종이 예상대로 추출되는지 검증한다.
- 피처 1·2·3·4 : ref_aa, alt_aa, position, log_position
- 피처 5        : has_repeated_position (셀 내 위치 반복 여부)
- 피처 6·7·8   : has_stop, has_frameshift, has_deletion_insertion
- 피처 9·10    : mutation_string_length, numeric_token_count
- 피처 11      : 변이 형식별 count (missense_count, nonsense_count, ...)
"""

from __future__ import annotations

import math

import pytest

from cancer_hack.parser import (
    extract_cell_string_features,
    extract_token_string_features,
)


# ---------------------------------------------------------------------------
# extract_token_string_features — 단일 토큰 9개 구조 피처
# ---------------------------------------------------------------------------

class TestTokenRefAltPosition:
    """피처 1·2·3 — ref_aa, alt_aa, position."""

    @pytest.mark.parametrize(
        "token,ref,alt,pos",
        [
            ("R132H", "R", "H", 132),     # 기본 missense
            ("G12D",  "G", "D", 12),       # 짧은 위치
            ("V600E", "V", "E", 600),      # BRAF V600E
            ("Q369*", "Q", "*", 369),      # stop codon — train 표기
            ("Q369X", "Q", "X", 369),      # stop codon — test 표기
            ("S622S", "S", "S", 622),      # 동의 변이
        ],
    )
    def test_ref_alt_position_missense_and_nonsense(self, token, ref, alt, pos):
        f = extract_token_string_features(token)
        assert f.ref_aa == ref
        assert f.alt_aa == alt
        assert f.position == pos

    def test_ref_aa_empty_when_starts_with_digit(self):
        # 복합 범위 표기 "312_313QY>HH"는 숫자로 시작 → ref_aa 없음
        assert extract_token_string_features("312_313QY>HH").ref_aa == ""

    @pytest.mark.parametrize(
        "token",
        [
            "K16fs",       # frameshift — 소문자로 끝나 alt_aa 없음
            "R649del",     # deletion — 소문자로 끝나 alt_aa 없음
            "A630_P649del",
        ],
    )
    def test_alt_aa_empty_for_non_substitution(self, token):
        assert extract_token_string_features(token).alt_aa == ""

    def test_position_minus_one_when_no_digit(self):
        assert extract_token_string_features("WT").position == -1


class TestTokenLogPosition:
    """피처 4 — log_position = log1p(position)."""

    def test_log_position_matches_log1p(self):
        f = extract_token_string_features("R132H")
        assert f.log_position == pytest.approx(math.log1p(132))

    def test_log_position_zero_when_no_number(self):
        assert extract_token_string_features("WT").log_position == 0.0

    def test_log_position_increases_with_position(self):
        f_low = extract_token_string_features("G12D")
        f_high = extract_token_string_features("V600E")
        assert f_low.log_position < f_high.log_position


class TestTokenHasStop:
    """피처 6 — stop(*, X) 표기 포함 여부."""

    @pytest.mark.parametrize("token", ["Q369*", "R213*", "Q369X", "R213X"])
    def test_stop_true(self, token):
        assert extract_token_string_features(token).has_stop == 1

    @pytest.mark.parametrize("token", ["R132H", "G12D", "K16fs", "E746_A750del"])
    def test_stop_false(self, token):
        assert extract_token_string_features(token).has_stop == 0


class TestTokenHasFrameshift:
    """피처 7 — frameshift(fs) 포함 여부."""

    @pytest.mark.parametrize("token", ["K16fs", "L1854fs", "S221Kfs"])
    def test_frameshift_true(self, token):
        assert extract_token_string_features(token).has_frameshift == 1

    @pytest.mark.parametrize("token", ["R132H", "Q369*", "E746_A750del"])
    def test_frameshift_false(self, token):
        assert extract_token_string_features(token).has_frameshift == 0


class TestTokenHasDeletionInsertion:
    """피처 8 — deletion·insertion(del, ins) 형식 여부."""

    @pytest.mark.parametrize(
        "token",
        [
            "R649del",           # 단순 deletion
            "A630_P649del",      # 범위 deletion
            "P11_K12insP",       # insertion
            "R376_A377delinsP",  # delins
            "E746_A750del",      # 범위 deletion
        ],
    )
    def test_del_ins_true(self, token):
        assert extract_token_string_features(token).has_deletion_insertion == 1

    @pytest.mark.parametrize("token", ["R132H", "Q369*", "K16fs", "S622S"])
    def test_del_ins_false(self, token):
        assert extract_token_string_features(token).has_deletion_insertion == 0


class TestTokenStringLength:
    """피처 9 — mutation_string_length."""

    @pytest.mark.parametrize(
        "token",
        ["R132H", "G12D", "E746_A750del", "R376_A377delinsP"],
    )
    def test_length_equals_len_of_token(self, token):
        assert extract_token_string_features(token).mutation_string_length == len(token)


class TestTokenNumericCount:
    """피처 10 — numeric_token_count (숫자 연속 구간 수)."""

    @pytest.mark.parametrize(
        "token,expected",
        [
            ("R132H", 1),             # 숫자 구간 1개
            ("G12D",  1),
            ("E746_A750del", 2),      # 746·750 두 구간
            ("R376_A377delinsP", 2),  # 376·377 두 구간
            ("312_313QY>HH", 2),      # 312·313 두 구간
        ],
    )
    def test_numeric_count(self, token, expected):
        assert extract_token_string_features(token).numeric_token_count == expected

    def test_numeric_count_zero_when_no_digit(self):
        assert extract_token_string_features("WT").numeric_token_count == 0


# ---------------------------------------------------------------------------
# extract_cell_string_features — 셀 수준 11종 피처
# ---------------------------------------------------------------------------

class TestCellEmptyValues:
    """WT·결측 셀은 모든 피처가 기본값이어야 한다."""

    @pytest.mark.parametrize("value", ["WT", "", None, "0", "NA", "nan"])
    def test_empty_cell_all_defaults(self, value):
        f = extract_cell_string_features(value)
        assert f.ref_aa == ""
        assert f.alt_aa == ""
        assert f.position == -1
        assert f.log_position == 0.0
        assert f.has_repeated_position == 0
        assert f.has_stop == 0
        assert f.has_frameshift == 0
        assert f.has_deletion_insertion == 0
        assert f.mutation_string_length == 0
        assert f.numeric_token_count == 0
        assert f.missense_count == 0


class TestCellFirstTokenFeatures:
    """피처 1-4 — 첫 번째 토큰 기준 추출."""

    def test_single_token(self):
        f = extract_cell_string_features("R132H")
        assert f.ref_aa == "R"
        assert f.alt_aa == "H"
        assert f.position == 132
        assert f.log_position == pytest.approx(math.log1p(132))

    def test_multi_token_uses_first(self):
        # "G827R R132H" — 첫 토큰 G827R 기준이어야 한다
        f = extract_cell_string_features("G827R R132H")
        assert f.ref_aa == "G"
        assert f.position == 827


class TestCellRepeatedPosition:
    """피처 5 — has_repeated_position."""

    def test_same_position_in_two_tokens(self):
        assert extract_cell_string_features("V600E V600K").has_repeated_position == 1

    def test_different_positions_no_repeat(self):
        assert extract_cell_string_features("R132H G827R").has_repeated_position == 0

    def test_single_token_no_repeat(self):
        assert extract_cell_string_features("R132H").has_repeated_position == 0

    def test_three_tokens_two_same_position(self):
        # 위치 132가 두 번 등장 → 반복
        assert extract_cell_string_features("R132H R132S G827R").has_repeated_position == 1


class TestCellHasStop:
    """피처 6 — has_stop (셀 내 어느 토큰이라도)."""

    def test_stop_in_mixed_cell(self):
        assert extract_cell_string_features("Q369* I368N").has_stop == 1

    def test_stop_x_notation(self):
        assert extract_cell_string_features("Q369X").has_stop == 1

    def test_no_stop_all_missense(self):
        assert extract_cell_string_features("R132H G827R").has_stop == 0


class TestCellHasFrameshift:
    """피처 7 — has_frameshift (셀 내 어느 토큰이라도)."""

    def test_frameshift_present(self):
        assert extract_cell_string_features("K16fs R132H").has_frameshift == 1

    def test_frameshift_absent(self):
        assert extract_cell_string_features("R132H G827R").has_frameshift == 0


class TestCellHasDeletionInsertion:
    """피처 8 — has_deletion_insertion (셀 내 어느 토큰이라도)."""

    def test_deletion(self):
        assert extract_cell_string_features("E746_A750del").has_deletion_insertion == 1

    def test_insertion(self):
        assert extract_cell_string_features("P11_K12insP").has_deletion_insertion == 1

    def test_delins(self):
        assert extract_cell_string_features("R376_A377delinsP").has_deletion_insertion == 1

    def test_absent(self):
        assert extract_cell_string_features("R132H Q369*").has_deletion_insertion == 0


class TestCellStringLength:
    """피처 9 — mutation_string_length (모든 토큰 길이 합)."""

    def test_single_token(self):
        assert extract_cell_string_features("R132H").mutation_string_length == 5

    def test_multi_token_sum(self):
        # "R132H"(5) + "G827R"(5) = 10
        assert extract_cell_string_features("R132H G827R").mutation_string_length == 10

    def test_range_token_length(self):
        token = "E746_A750del"
        assert extract_cell_string_features(token).mutation_string_length == len(token)


class TestCellNumericCount:
    """피처 10 — numeric_token_count (모든 숫자 구간 수 합)."""

    def test_single_token_one_number(self):
        assert extract_cell_string_features("R132H").numeric_token_count == 1

    def test_multi_token_sum(self):
        # 각 토큰에 숫자 1개씩 → 합 2
        assert extract_cell_string_features("R132H G827R").numeric_token_count == 2

    def test_range_token_two_numbers(self):
        # "E746_A750del" → 746·750 두 구간
        assert extract_cell_string_features("E746_A750del").numeric_token_count == 2


class TestCellTypeCounts:
    """피처 11 — 변이 형식별 count."""

    def test_single_missense(self):
        f = extract_cell_string_features("R132H")
        assert f.missense_count == 1
        assert f.nonsense_count == 0
        assert f.frameshift_count == 0

    def test_mixed_nonsense_and_missense(self):
        f = extract_cell_string_features("Q369* I368N")
        assert f.nonsense_count == 1
        assert f.missense_count == 1

    def test_synonymous(self):
        f = extract_cell_string_features("S622S G827R")
        assert f.synonymous_count == 1
        assert f.missense_count == 1

    def test_frameshift(self):
        assert extract_cell_string_features("K16fs").frameshift_count == 1

    def test_deletion(self):
        assert extract_cell_string_features("R649del").deletion_count == 1

    def test_insertion(self):
        assert extract_cell_string_features("P11_K12insP").insertion_count == 1

    def test_delins(self):
        assert extract_cell_string_features("R376_A377delinsP").delins_count == 1

    def test_complex(self):
        assert extract_cell_string_features("312_313QY>HH").complex_count == 1

    def test_duplicate_tokens_counted_separately(self):
        # "V600E V600E" — 토큰이 두 개이므로 missense_count == 2
        assert extract_cell_string_features("V600E V600E").missense_count == 2

    def test_all_counts_zero_for_empty(self):
        f = extract_cell_string_features("WT")
        for kind in (
            "missense", "synonymous", "nonsense",
            "frameshift", "deletion", "insertion", "delins", "complex", "other",
        ):
            assert getattr(f, f"{kind}_count") == 0


class TestCellAsDict:
    """as_dict()가 11개 피처 그룹 키를 모두 포함하는지 확인."""

    def test_as_dict_keys(self):
        d = extract_cell_string_features("R132H").as_dict()
        required = [
            "ref_aa", "alt_aa", "position", "log_position",
            "has_repeated_position", "has_stop", "has_frameshift",
            "has_deletion_insertion", "mutation_string_length",
            "numeric_token_count", "missense_count",
        ]
        for key in required:
            assert key in d, f"'{key}' not found in as_dict()"
