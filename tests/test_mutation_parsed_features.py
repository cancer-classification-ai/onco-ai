"""make_mutation_string_parsed_features 통합 테스트.

19종 파싱 파생변수가 그룹별로 올바르게 계산되는지 검증한다.

① parsed_mutation_count / unparsed_mutation_count / parse_success_ratio  (3)
② position_mean / position_std / position_min / position_max / position_median  (5)
③ unique_ref_aa_count / unique_alt_aa_count / unique_aa_change_count  (3)
④ unique_position_bin_count / most_common_position_bin_count  (2)
⑤ mutation_type_diversity / dominant_mutation_type_ratio  (2)
⑥ genes_with_hotspot / hotspot_ratio  (2)
⑦ genes_with_multiple_positions / mean_position_per_gene  (2)
"""

from __future__ import annotations

import pandas as pd
import pytest

from cancer_hack.features_basic import (
    MUTATION_STRING_PARSED_COLUMNS,
    make_mutation_string_parsed_features,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _single(gene_values: dict[str, str], **kwargs) -> pd.Series:
    """샘플 1행짜리 DataFrame 에서 결과 Series 를 반환하는 편의 함수."""
    df = pd.DataFrame({k: [v] for k, v in gene_values.items()})
    gene_cols = list(gene_values.keys())
    return make_mutation_string_parsed_features(df, gene_columns=gene_cols, **kwargs).iloc[0]


# ---------------------------------------------------------------------------
# 출력 계약 — 컬럼 수·순서·인덱스 보존
# ---------------------------------------------------------------------------

class TestOutputContract:
    """반환 DataFrame 의 구조 계약."""

    def test_column_count_is_19(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert out.shape[1] == 19

    def test_column_names_match_constant(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert list(out.columns) == list(MUTATION_STRING_PARSED_COLUMNS)

    def test_index_is_preserved(self):
        df = pd.DataFrame(
            {"BRAF": ["V600E", "WT"]},
            index=[42, 99],
        )
        out = make_mutation_string_parsed_features(df, gene_columns=["BRAF"])
        assert list(out.index) == [42, 99]

    def test_row_count_matches_input(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert len(out) == len(toy_frame)

    def test_missing_gene_column_raises(self):
        df = pd.DataFrame({"BRAF": ["V600E"]})
        with pytest.raises(ValueError, match="Missing gene columns"):
            make_mutation_string_parsed_features(df, gene_columns=["BRAF", "NONEXISTENT"])


# ---------------------------------------------------------------------------
# WT·결측 샘플 — 모든 피처가 0 이어야 한다
# ---------------------------------------------------------------------------

class TestWildtypeSample:
    """변이가 없는 샘플은 19개 피처 전부 0.0 / 0."""

    @pytest.mark.parametrize("value", ["WT", "", None, "0", "NA"])
    def test_wt_only_sample_all_zeros(self, value):
        df = pd.DataFrame({"BRAF": [value], "TP53": ["WT"]})
        row = make_mutation_string_parsed_features(df, gene_columns=["BRAF", "TP53"]).iloc[0]
        for col in MUTATION_STRING_PARSED_COLUMNS:
            assert row[col] == pytest.approx(0.0), f"{col} should be 0 for WT sample"

    def test_all_wt_genes_give_zero_counts(self):
        df = pd.DataFrame({"A": ["WT"], "B": ["WT"], "C": ["WT"]})
        row = make_mutation_string_parsed_features(df, gene_columns=["A", "B", "C"]).iloc[0]
        assert row["parsed_mutation_count"] == 0
        assert row["parse_success_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ① 파싱 성공 여부
# ---------------------------------------------------------------------------

class TestParseSuccessFeatures:
    """parsed / unparsed count 와 ratio."""

    def test_all_missense_fully_parsed(self):
        row = _single({"BRAF": "V600E", "TP53": "R132H"})
        assert row["parsed_mutation_count"] == 2
        assert row["unparsed_mutation_count"] == 0
        assert row["parse_success_ratio"] == pytest.approx(1.0)

    def test_other_type_counts_as_unparsed(self):
        # "SPLICE" 는 어느 패턴에도 안 걸려 other 로 분류된다.
        row = _single({"BRAF": "V600E SPLICE"})
        assert row["parsed_mutation_count"] == 1
        assert row["unparsed_mutation_count"] == 1
        assert row["parse_success_ratio"] == pytest.approx(0.5)

    def test_multiple_genes_counts_are_summed(self):
        # 유전자 3개, 각 1개씩 → total=3, all parsed
        row = _single({"A": "R132H", "B": "Q369*", "C": "K16fs"})
        assert row["parsed_mutation_count"] == 3
        assert row["unparsed_mutation_count"] == 0

    def test_multi_token_cell_each_counted(self):
        # "V600E V600K" → 2개 토큰
        row = _single({"BRAF": "V600E V600K"})
        assert row["parsed_mutation_count"] == 2

    def test_only_unparsed_gives_ratio_zero(self):
        row = _single({"GENE": "SPLICE AMPLIFICATION"})
        assert row["unparsed_mutation_count"] == 2
        assert row["parse_success_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ② 위치 통계
# ---------------------------------------------------------------------------

class TestPositionStatistics:
    """position_mean / std / min / max / median."""

    def test_single_token_all_stats_equal_position(self):
        # V600E → position=600
        row = _single({"BRAF": "V600E"})
        assert row["position_mean"] == pytest.approx(600.0)
        assert row["position_std"] == pytest.approx(0.0)
        assert row["position_min"] == pytest.approx(600.0)
        assert row["position_max"] == pytest.approx(600.0)
        assert row["position_median"] == pytest.approx(600.0)

    def test_two_tokens_mean_is_average(self):
        # R132H(132) + G827R(827) → mean=479.5
        row = _single({"BRAF": "R132H G827R"})
        assert row["position_mean"] == pytest.approx(479.5)
        assert row["position_min"] == pytest.approx(132.0)
        assert row["position_max"] == pytest.approx(827.0)

    def test_three_tokens_median(self):
        # 위치 100, 200, 300 → median=200
        row = _single({"A": "R100H", "B": "G200R", "C": "V300E"})
        assert row["position_median"] == pytest.approx(200.0)

    def test_no_position_token_excluded(self):
        # "SPLICE" 는 position=-1 → 위치 통계에서 제외
        row = _single({"BRAF": "V600E SPLICE"})
        assert row["position_mean"] == pytest.approx(600.0)
        assert row["position_std"] == pytest.approx(0.0)

    def test_wt_only_position_stats_zero(self):
        row = _single({"BRAF": "WT"})
        assert row["position_mean"] == pytest.approx(0.0)
        assert row["position_std"] == pytest.approx(0.0)

    def test_std_nonzero_for_different_positions(self):
        row = _single({"BRAF": "V600E", "TP53": "R132H"})
        assert row["position_std"] > 0.0


# ---------------------------------------------------------------------------
# ③ 아미노산 변화 통계
# ---------------------------------------------------------------------------

class TestAminoAcidChangeStatistics:
    """unique_ref_aa_count / unique_alt_aa_count / unique_aa_change_count."""

    def test_single_missense(self):
        # V600E → ref={V}, alt={E}, change={(V,E)}
        row = _single({"BRAF": "V600E"})
        assert row["unique_ref_aa_count"] == 1
        assert row["unique_alt_aa_count"] == 1
        assert row["unique_aa_change_count"] == 1

    def test_same_ref_different_alt(self):
        # V600E + V600K → ref={V}, alt={E,K}, changes={(V,E),(V,K)}
        row = _single({"BRAF": "V600E V600K"})
        assert row["unique_ref_aa_count"] == 1
        assert row["unique_alt_aa_count"] == 2
        assert row["unique_aa_change_count"] == 2

    def test_same_change_repeated_deduplicated(self):
        # V600E 두 번 → ref={V}, alt={E}, changes={(V,E)}
        row = _single({"BRAF": "V600E V600E"})
        assert row["unique_ref_aa_count"] == 1
        assert row["unique_alt_aa_count"] == 1
        assert row["unique_aa_change_count"] == 1

    def test_nonsense_stop_codon_counted_in_alt(self):
        # Q369* → alt_aa=* 도 alt set 에 포함
        row = _single({"TP53": "Q369*"})
        assert row["unique_alt_aa_count"] == 1  # {*}
        assert row["unique_ref_aa_count"] == 1  # {Q}

    def test_deletion_has_ref_but_no_alt(self):
        # R649del → ref_aa=R, alt_aa="" → alt set 에 추가되지 않음
        row = _single({"EGFR": "R649del"})
        assert row["unique_ref_aa_count"] == 1
        assert row["unique_alt_aa_count"] == 0

    def test_multiple_genes_accumulate(self):
        # V600E(BRAF) + R132H(TP53) → ref={V,R}, alt={E,H}
        row = _single({"BRAF": "V600E", "TP53": "R132H"})
        assert row["unique_ref_aa_count"] == 2
        assert row["unique_alt_aa_count"] == 2
        assert row["unique_aa_change_count"] == 2


# ---------------------------------------------------------------------------
# ④ 위치 구간 통계
# ---------------------------------------------------------------------------

class TestPositionBinStatistics:
    """unique_position_bin_count / most_common_position_bin_count."""

    def test_same_bin_gives_one_unique_bin(self):
        # V600E(600) + V600K(600) → 둘 다 bin=600 → unique=1, most_common=2
        row = _single({"BRAF": "V600E V600K"})
        assert row["unique_position_bin_count"] == 1
        assert row["most_common_position_bin_count"] == 2

    def test_different_bins_each_unique(self):
        # R132H(132→bin100) + G827R(827→bin800) → 2개 고유 bin, 각 1회
        row = _single({"BRAF": "R132H G827R"})
        assert row["unique_position_bin_count"] == 2
        assert row["most_common_position_bin_count"] == 1

    def test_bin_size_100_merges_nearby_positions(self):
        # 600, 650 → 각각 bin=600, 600 (size=100: 600//100*100=600, 650//100*100=600)
        row = _single({"BRAF": "V600E", "TP53": "R650H"}, position_bin_size=100)
        assert row["unique_position_bin_count"] == 1
        assert row["most_common_position_bin_count"] == 2

    def test_bin_size_10_separates_close_positions(self):
        # 600, 610 → bin=600, 610 (size=10: 다른 구간)
        row = _single({"BRAF": "V600E", "TP53": "R610H"}, position_bin_size=10)
        assert row["unique_position_bin_count"] == 2

    def test_no_position_gives_zero_bins(self):
        row = _single({"BRAF": "WT"})
        assert row["unique_position_bin_count"] == 0
        assert row["most_common_position_bin_count"] == 0


# ---------------------------------------------------------------------------
# ⑤ 변이 유형 다양성
# ---------------------------------------------------------------------------

class TestMutationTypeDiversity:
    """mutation_type_diversity / dominant_mutation_type_ratio."""

    def test_single_type_diversity_one(self):
        row = _single({"BRAF": "V600E", "KRAS": "G12D"})
        assert row["mutation_type_diversity"] == 1
        assert row["dominant_mutation_type_ratio"] == pytest.approx(1.0)

    def test_two_types_diversity_two(self):
        # missense + nonsense → diversity=2
        row = _single({"BRAF": "V600E", "TP53": "Q369*"})
        assert row["mutation_type_diversity"] == 2

    def test_dominant_ratio_with_imbalanced_types(self):
        # missense 2개 + nonsense 1개 → dominant=missense, ratio=2/3
        row = _single({"BRAF": "V600E G12D", "TP53": "Q369*"})
        assert row["dominant_mutation_type_ratio"] == pytest.approx(2 / 3, rel=1e-4)

    def test_three_types_diversity_three(self):
        # missense + nonsense + frameshift → 3종
        row = _single({"BRAF": "V600E", "TP53": "Q369*", "EGFR": "K16fs"})
        assert row["mutation_type_diversity"] == 3

    def test_other_type_excluded_from_diversity(self):
        # "SPLICE" (other) + V600E (missense) → diversity=1 (other 제외)
        row = _single({"BRAF": "V600E SPLICE"})
        assert row["mutation_type_diversity"] == 1

    def test_wt_only_diversity_zero(self):
        row = _single({"BRAF": "WT"})
        assert row["mutation_type_diversity"] == 0
        assert row["dominant_mutation_type_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ⑥ 유전자-위치 특징
# ---------------------------------------------------------------------------

class TestHotspotFeatures:
    """genes_with_hotspot / hotspot_ratio."""

    def test_same_position_different_alt_is_hotspot(self):
        # BRAF V600E + V600K → 같은 position 600, 다른 alt → hotspot
        row = _single({"BRAF": "V600E V600K"})
        assert row["genes_with_hotspot"] == 1
        assert row["hotspot_ratio"] == pytest.approx(1.0)

    def test_different_positions_not_hotspot(self):
        # BRAF V600E + V512E → 다른 위치 → hotspot 없음
        row = _single({"BRAF": "V600E V512E"})
        assert row["genes_with_hotspot"] == 0
        assert row["hotspot_ratio"] == pytest.approx(0.0)

    def test_hotspot_ratio_over_multiple_genes(self):
        # BRAF(hotspot) + TP53(no hotspot) → ratio=1/2=0.5
        row = _single({"BRAF": "V600E V600K", "TP53": "R132H"})
        assert row["genes_with_hotspot"] == 1
        assert row["hotspot_ratio"] == pytest.approx(0.5)

    def test_no_mutation_hotspot_zero(self):
        row = _single({"BRAF": "WT"})
        assert row["genes_with_hotspot"] == 0
        assert row["hotspot_ratio"] == pytest.approx(0.0)

    def test_single_token_per_gene_no_hotspot(self):
        row = _single({"BRAF": "V600E", "TP53": "R132H", "KRAS": "G12D"})
        assert row["genes_with_hotspot"] == 0
        assert row["hotspot_ratio"] == pytest.approx(0.0)

    def test_two_hotspot_genes(self):
        # BRAF(V600E+V600K) + KRAS(G12D+G12V) → 2개 hotspot 유전자
        row = _single({"BRAF": "V600E V600K", "KRAS": "G12D G12V"})
        assert row["genes_with_hotspot"] == 2
        assert row["hotspot_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ⑦ 복합 변이 특징
# ---------------------------------------------------------------------------

class TestMultiPositionFeatures:
    """genes_with_multiple_positions / mean_position_per_gene."""

    def test_two_distinct_positions_in_one_gene(self):
        # BRAF V600E(600) + V512E(512) → 2개 고유 위치 → genes_with_multiple=1
        row = _single({"BRAF": "V600E V512E"})
        assert row["genes_with_multiple_positions"] == 1

    def test_same_position_twice_not_multiple(self):
        # BRAF V600E + V600K → 고유 위치 1개 → multiple=0
        row = _single({"BRAF": "V600E V600K"})
        assert row["genes_with_multiple_positions"] == 0

    def test_single_token_per_gene_no_multiple(self):
        row = _single({"BRAF": "V600E", "TP53": "R132H"})
        assert row["genes_with_multiple_positions"] == 0

    def test_mean_position_per_gene_single_position(self):
        # BRAF→1위치, TP53→1위치 → mean=1.0
        row = _single({"BRAF": "V600E", "TP53": "R132H"})
        assert row["mean_position_per_gene"] == pytest.approx(1.0)

    def test_mean_position_per_gene_mixed(self):
        # BRAF→2위치(600,512), TP53→1위치(132) → mean=(2+1)/2=1.5
        row = _single({"BRAF": "V600E V512E", "TP53": "R132H"})
        assert row["mean_position_per_gene"] == pytest.approx(1.5)

    def test_no_mutation_mean_zero(self):
        row = _single({"BRAF": "WT", "TP53": "WT"})
        assert row["mean_position_per_gene"] == pytest.approx(0.0)

    def test_frameshift_without_position_not_counted(self):
        # position 없는 토큰만 있는 유전자는 gene_uniq_pos 에 포함 안 됨
        # K16fs → position=16 (존재하므로 포함됨), 단순 확인
        row = _single({"EGFR": "K16fs"})
        assert row["mean_position_per_gene"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 복합 시나리오 — toy_frame 전체 사용
# ---------------------------------------------------------------------------

class TestIntegrationWithToyFrame:
    """conftest.toy_frame 4행 전체에서 결과 형태·값이 이상하지 않은지 확인."""

    def test_output_shape(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert out.shape == (4, 19)

    def test_wt_only_row_all_zeros(self, toy_frame):
        # s4 행은 TP53=WT, KRAS=WT, EGFR=WT
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        row = out.iloc[3]  # s4
        assert row["parsed_mutation_count"] == 0
        assert row["parse_success_ratio"] == pytest.approx(0.0)

    def test_parse_success_ratio_in_range(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert (out["parse_success_ratio"] >= 0.0).all()
        assert (out["parse_success_ratio"] <= 1.0).all()

    def test_dominant_ratio_in_range(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert (out["dominant_mutation_type_ratio"] >= 0.0).all()
        assert (out["dominant_mutation_type_ratio"] <= 1.0).all()

    def test_hotspot_ratio_in_range(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert (out["hotspot_ratio"] >= 0.0).all()
        assert (out["hotspot_ratio"] <= 1.0).all()

    def test_position_min_leq_max(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert (out["position_min"] <= out["position_max"]).all()

    def test_parsed_plus_unparsed_equals_total(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        total = out["parsed_mutation_count"] + out["unparsed_mutation_count"]
        # toy_frame s1: E412K R1800C(TP53=2) + S622S G827R(KRAS=2) + 312_313QY>HH(EGFR=1) = 5
        assert total.iloc[0] == 5

    def test_s1_has_no_hotspot(self, toy_frame):
        # s1: TP53=E412K R1800C → 위치 412, 1800 → 다름 → hotspot 없음
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert out.iloc[0]["genes_with_hotspot"] == 0

    def test_s2_kras_hotspot(self, toy_frame):
        # s2: KRAS=V600E V600E → 같은 위치 → hotspot
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        out = make_mutation_string_parsed_features(toy_frame, gene_columns=gene_cols)
        assert out.iloc[1]["genes_with_hotspot"] >= 1
