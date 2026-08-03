"""B 전략 희소 조합 토큰 피처 테스트.

build_token_list / row_to_parsed_token_document / build_parsed_token_documents /
ParsedTokenHasher 의 계약을 검증한다.

검증 항목
---------
- _posbin_label       : 위치 구간 레이블 형식
- build_token_list    : 변이 유형별 토큰 집합, 파싱 실패 처리
- row_to_parsed_token_document  : WT·문서 형식·복합 셀
- build_parsed_token_documents  : 인덱스 보존·ValueError
- ParsedTokenHasher   : 출력 형태·stateless 속성·n_features 파라미터
"""

from __future__ import annotations

import pandas as pd
import pytest
import scipy.sparse as sp

from cancer_hack.features_sparse import (
    _NO_PARSED_TOKEN_SENTINEL,
    _posbin_label,
    build_parsed_token_documents,
    build_token_list,
    ParsedTokenHasher,
    row_to_parsed_token_document,
)


# ---------------------------------------------------------------------------
# _posbin_label — 위치 구간 레이블
# ---------------------------------------------------------------------------

class TestPosbinLabel:
    """위치 숫자를 '<lo>_<hi>' 형식으로 변환하는지 확인."""

    @pytest.mark.parametrize(
        "position,bin_size,expected",
        [
            (600, 50, "600_649"),
            (601, 50, "600_649"),   # 같은 bin
            (649, 50, "600_649"),   # bin 경계 끝
            (650, 50, "650_699"),   # 다음 bin
            (0,   50, "0_49"),      # 최소
            (600, 100, "600_699"),  # bin_size=100
            (600, 10,  "600_609"),  # bin_size=10
            (512, 50, "500_549"),   # BRAF V512E 위치
            (369, 50, "350_399"),   # TP53 Q369* 위치
        ],
    )
    def test_label_format(self, position, bin_size, expected):
        assert _posbin_label(position, bin_size) == expected

    def test_adjacent_positions_share_bin(self):
        assert _posbin_label(600, 50) == _posbin_label(601, 50)

    def test_boundary_positions_differ(self):
        assert _posbin_label(649, 50) != _posbin_label(650, 50)


# ---------------------------------------------------------------------------
# build_token_list — 변이 유형별 토큰 집합
# ---------------------------------------------------------------------------

class TestBuildTokenListMissense:
    """BRAF V600E — 8 토큰 전부 포함."""

    def setup_method(self):
        self.tokens = build_token_list("BRAF", "V600E", 50)

    def test_gene_token_always_present(self):
        assert "GENE__BRAF" in self.tokens

    def test_type_token(self):
        assert "TYPE__MISSENSE" in self.tokens

    def test_gene_type_token(self):
        assert "GENE_TYPE__BRAF__MISSENSE" in self.tokens

    def test_ref_token(self):
        assert "REF__V" in self.tokens

    def test_alt_token(self):
        assert "ALT__E" in self.tokens

    def test_aa_change_token(self):
        assert "AA_CHANGE__V_E" in self.tokens

    def test_posbin_token(self):
        assert "POSBIN__600_649" in self.tokens

    def test_gene_posbin_token(self):
        assert "GENE_POSBIN__BRAF__600_649" in self.tokens

    def test_total_token_count(self):
        assert len(self.tokens) == 8


class TestBuildTokenListNonsense:
    """TP53 Q369* — stop codon 은 ALT__* 로 표현."""

    def setup_method(self):
        self.tokens = build_token_list("TP53", "Q369*", 50)

    def test_type_nonsense(self):
        assert "TYPE__NONSENSE" in self.tokens

    def test_ref_q(self):
        assert "REF__Q" in self.tokens

    def test_alt_stop(self):
        assert "ALT__*" in self.tokens

    def test_aa_change_stop(self):
        assert "AA_CHANGE__Q_*" in self.tokens

    def test_posbin(self):
        assert "POSBIN__350_399" in self.tokens

    def test_gene_posbin(self):
        assert "GENE_POSBIN__TP53__350_399" in self.tokens


class TestBuildTokenListNonsenseTestNotation:
    """test 데이터 표기 Q369X — train 의 * 와 같은 nonsense 처리."""

    def test_type_is_nonsense(self):
        tokens = build_token_list("TP53", "Q369X", 50)
        assert "TYPE__NONSENSE" in tokens

    def test_alt_is_x(self):
        tokens = build_token_list("TP53", "Q369X", 50)
        assert "ALT__X" in tokens


class TestBuildTokenListSynonymous:
    """S622S — ref == alt 이므로 AA_CHANGE__S_S 가 생성된다."""

    def setup_method(self):
        self.tokens = build_token_list("KRAS", "S622S", 50)

    def test_type_synonymous(self):
        assert "TYPE__SYNONYMOUS" in self.tokens

    def test_aa_change_same(self):
        assert "AA_CHANGE__S_S" in self.tokens


class TestBuildTokenListDeletion:
    """R649del — alt_aa 없음 → ALT__ · AA_CHANGE__ 토큰 없음."""

    def setup_method(self):
        self.tokens = build_token_list("EGFR", "R649del", 50)

    def test_type_deletion(self):
        assert "TYPE__DELETION" in self.tokens

    def test_ref_present(self):
        assert "REF__R" in self.tokens

    def test_no_alt_token(self):
        assert not any(t.startswith("ALT__") for t in self.tokens)

    def test_no_aa_change_token(self):
        assert not any(t.startswith("AA_CHANGE__") for t in self.tokens)

    def test_posbin_present(self):
        # 649 → 600_649 (bin=50)
        assert "POSBIN__600_649" in self.tokens

    def test_gene_posbin_present(self):
        assert "GENE_POSBIN__EGFR__600_649" in self.tokens


class TestBuildTokenListRangeDeletion:
    """E746_A750del — 복합 범위 deletion."""

    def setup_method(self):
        self.tokens = build_token_list("EGFR", "E746_A750del", 50)

    def test_type_deletion(self):
        assert "TYPE__DELETION" in self.tokens

    def test_no_alt_token(self):
        assert not any(t.startswith("ALT__") for t in self.tokens)


class TestBuildTokenListFrameshift:
    """K16fs — alt_aa 없음 → ALT__ · AA_CHANGE__ 토큰 없음."""

    def setup_method(self):
        self.tokens = build_token_list("EGFR", "K16fs", 50)

    def test_type_frameshift(self):
        assert "TYPE__FRAMESHIFT" in self.tokens

    def test_ref_present(self):
        assert "REF__K" in self.tokens

    def test_no_alt_token(self):
        assert not any(t.startswith("ALT__") for t in self.tokens)

    def test_posbin_is_first_bin(self):
        # 16 → bin 0_49
        assert "POSBIN__0_49" in self.tokens


class TestBuildTokenListDelins:
    """R376_A377delinsP — delins 유형."""

    def test_type_delins(self):
        tokens = build_token_list("EGFR", "R376_A377delinsP", 50)
        assert "TYPE__DELINS" in tokens


class TestBuildTokenListInsertion:
    """P11_K12insP — insertion 유형."""

    def test_type_insertion(self):
        tokens = build_token_list("EGFR", "P11_K12insP", 50)
        assert "TYPE__INSERTION" in tokens


class TestBuildTokenListComplex:
    """312_313QY>HH — complex 유형."""

    def test_type_complex(self):
        tokens = build_token_list("EGFR", "312_313QY>HH", 50)
        assert "TYPE__COMPLEX" in tokens


class TestBuildTokenListParseFailed:
    """파싱 불가 토큰(other 유형) → PARSE_FAILED 토큰 집합."""

    @pytest.mark.parametrize("mutation", ["SPLICE", "AMPLIFICATION", "FUSION"])
    def test_parse_failed_tokens(self, mutation):
        tokens = build_token_list("BRAF", mutation, 50)
        assert "TYPE__UNKNOWN" in tokens
        assert "PARSE_FAILED" in tokens
        assert "GENE_PARSE_FAILED__BRAF" in tokens

    def test_gene_token_still_present_for_failed(self):
        tokens = build_token_list("TP53", "UNKNOWN_MUT", 50)
        assert "GENE__TP53" in tokens

    def test_no_type_missense_for_failed(self):
        tokens = build_token_list("BRAF", "SPLICE", 50)
        assert "TYPE__MISSENSE" not in tokens

    def test_failed_token_count_is_four(self):
        # GENE__ + TYPE__UNKNOWN + PARSE_FAILED + GENE_PARSE_FAILED__
        tokens = build_token_list("BRAF", "SPLICE", 50)
        assert len(tokens) == 4


class TestBuildTokenListPositionBinSize:
    """position_bin_size 파라미터가 POSBIN 레이블에 반영된다."""

    def test_bin_size_100(self):
        tokens = build_token_list("BRAF", "V600E", 100)
        assert "POSBIN__600_699" in tokens
        assert "GENE_POSBIN__BRAF__600_699" in tokens

    def test_bin_size_10(self):
        tokens = build_token_list("BRAF", "V600E", 10)
        assert "POSBIN__600_609" in tokens

    def test_different_bin_sizes_different_posbin(self):
        tokens_50 = build_token_list("BRAF", "V601E", 50)
        tokens_100 = build_token_list("BRAF", "V601E", 100)
        posbin_50 = next(t for t in tokens_50 if t.startswith("POSBIN__"))
        posbin_100 = next(t for t in tokens_100 if t.startswith("POSBIN__"))
        # 601 → bin50=600_649, bin100=600_699
        assert posbin_50 != posbin_100

    def test_nearby_positions_same_bin_50(self):
        tokens_600 = build_token_list("BRAF", "V600E", 50)
        tokens_601 = build_token_list("BRAF", "V601E", 50)
        posbin_600 = next(t for t in tokens_600 if t.startswith("POSBIN__"))
        posbin_601 = next(t for t in tokens_601 if t.startswith("POSBIN__"))
        assert posbin_600 == posbin_601

    def test_bin_boundary_positions_differ(self):
        tokens_649 = build_token_list("BRAF", "V649E", 50)
        tokens_650 = build_token_list("BRAF", "V650E", 50)
        posbin_649 = next(t for t in tokens_649 if t.startswith("POSBIN__"))
        posbin_650 = next(t for t in tokens_650 if t.startswith("POSBIN__"))
        assert posbin_649 != posbin_650


# ---------------------------------------------------------------------------
# row_to_parsed_token_document — 행 단위 문서 생성
# ---------------------------------------------------------------------------

class TestRowToParsedTokenDocument:
    """단일 행 → 공백 구분 토큰 문서."""

    def test_wt_only_returns_sentinel(self):
        row = pd.Series({"BRAF": "WT", "TP53": "WT"})
        doc = row_to_parsed_token_document(row, ["BRAF", "TP53"])
        assert doc == _NO_PARSED_TOKEN_SENTINEL

    def test_empty_string_returns_sentinel(self):
        row = pd.Series({"BRAF": "", "TP53": "0"})
        doc = row_to_parsed_token_document(row, ["BRAF", "TP53"])
        assert doc == _NO_PARSED_TOKEN_SENTINEL

    def test_single_missense_contains_expected_tokens(self):
        row = pd.Series({"BRAF": "V600E", "TP53": "WT"})
        doc = row_to_parsed_token_document(row, ["BRAF", "TP53"])
        assert "GENE__BRAF" in doc
        assert "TYPE__MISSENSE" in doc
        assert "AA_CHANGE__V_E" in doc

    def test_wt_gene_excluded_from_document(self):
        row = pd.Series({"BRAF": "V600E", "TP53": "WT"})
        doc = row_to_parsed_token_document(row, ["BRAF", "TP53"])
        assert "GENE__TP53" not in doc

    def test_multiple_genes_both_in_document(self):
        row = pd.Series({"BRAF": "V600E", "TP53": "R132H"})
        doc = row_to_parsed_token_document(row, ["BRAF", "TP53"])
        assert "GENE__BRAF" in doc
        assert "GENE__TP53" in doc

    def test_multi_token_cell_all_mutations_included(self):
        # "V600E V600K" → 두 변이 모두 포함
        row = pd.Series({"BRAF": "V600E V600K"})
        doc = row_to_parsed_token_document(row, ["BRAF"])
        assert "AA_CHANGE__V_E" in doc
        assert "AA_CHANGE__V_K" in doc

    def test_parse_failed_token_in_document(self):
        row = pd.Series({"BRAF": "SPLICE"})
        doc = row_to_parsed_token_document(row, ["BRAF"])
        assert "PARSE_FAILED" in doc
        assert "GENE_PARSE_FAILED__BRAF" in doc

    def test_document_is_space_separated_string(self):
        row = pd.Series({"BRAF": "V600E"})
        doc = row_to_parsed_token_document(row, ["BRAF"])
        assert isinstance(doc, str)
        for token in doc.split():
            assert "__" in token or token == _NO_PARSED_TOKEN_SENTINEL

    def test_position_bin_size_applied(self):
        row = pd.Series({"BRAF": "V600E"})
        doc_50 = row_to_parsed_token_document(row, ["BRAF"], position_bin_size=50)
        doc_100 = row_to_parsed_token_document(row, ["BRAF"], position_bin_size=100)
        assert "POSBIN__600_649" in doc_50
        assert "POSBIN__600_699" in doc_100


# ---------------------------------------------------------------------------
# build_parsed_token_documents — DataFrame 단위
# ---------------------------------------------------------------------------

class TestBuildParsedTokenDocuments:
    """DataFrame → pd.Series 계약."""

    def test_output_length_matches_input(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        docs = build_parsed_token_documents(toy_frame, gene_cols)
        assert len(docs) == len(toy_frame)

    def test_index_preserved(self):
        df = pd.DataFrame({"BRAF": ["V600E", "WT"]}, index=[10, 20])
        docs = build_parsed_token_documents(df, ["BRAF"])
        assert list(docs.index) == [10, 20]

    def test_missing_gene_column_raises(self):
        df = pd.DataFrame({"BRAF": ["V600E"]})
        with pytest.raises(ValueError, match="Missing gene columns"):
            build_parsed_token_documents(df, ["BRAF", "NONEXISTENT"])

    def test_wt_row_gets_sentinel(self, toy_frame):
        # toy_frame s4 행: TP53=WT, KRAS=WT, EGFR=WT
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        docs = build_parsed_token_documents(toy_frame, gene_cols)
        assert docs.iloc[3] == _NO_PARSED_TOKEN_SENTINEL

    def test_mutated_row_not_sentinel(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        docs = build_parsed_token_documents(toy_frame, gene_cols)
        assert docs.iloc[0] != _NO_PARSED_TOKEN_SENTINEL

    def test_all_documents_are_strings(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        docs = build_parsed_token_documents(toy_frame, gene_cols)
        assert docs.apply(lambda d: isinstance(d, str)).all()

    def test_position_bin_size_kwarg_propagated(self):
        df = pd.DataFrame({"BRAF": ["V600E"]})
        docs_50  = build_parsed_token_documents(df, ["BRAF"], position_bin_size=50)
        docs_100 = build_parsed_token_documents(df, ["BRAF"], position_bin_size=100)
        assert "POSBIN__600_649" in docs_50.iloc[0]
        assert "POSBIN__600_699" in docs_100.iloc[0]


# ---------------------------------------------------------------------------
# ParsedTokenHasher — 희소행렬 변환
# ---------------------------------------------------------------------------

class TestParsedTokenHasher:
    """출력 형태·fit→transform 순서·파라미터 효과."""

    def _docs(self) -> pd.Series:
        return pd.Series([
            "GENE__BRAF TYPE__MISSENSE AA_CHANGE__V_E POSBIN__600_649",
            "GENE__TP53 TYPE__NONSENSE AA_CHANGE__Q_S POSBIN__350_399",
            _NO_PARSED_TOKEN_SENTINEL,
        ])

    def test_output_is_csr_matrix(self):
        mat = ParsedTokenHasher().fit_transform(self._docs())
        assert sp.issparse(mat)

    def test_output_row_count_matches_input(self):
        docs = self._docs()
        mat = ParsedTokenHasher().fit_transform(docs)
        assert mat.shape[0] == len(docs)

    def test_output_col_count_matches_vocabulary(self):
        docs = self._docs()
        hasher = ParsedTokenHasher()
        mat = hasher.fit_transform(docs)
        assert mat.shape[1] == len(hasher.get_feature_names_out())

    def test_fit_builds_nonempty_vocabulary(self):
        hasher = ParsedTokenHasher()
        hasher.fit(self._docs())
        assert len(hasher.get_feature_names_out()) > 0

    def test_fit_returns_self(self):
        hasher = ParsedTokenHasher()
        returned = hasher.fit(self._docs())
        assert returned is hasher

    def test_fit_transform_equals_transform(self):
        hasher = ParsedTokenHasher()
        docs = self._docs()
        mat_ft = hasher.fit_transform(docs)
        mat_t = hasher.transform(docs)
        assert (mat_ft - mat_t).nnz == 0

    def test_same_document_same_row(self):
        doc = "GENE__BRAF TYPE__MISSENSE"
        docs = pd.Series([doc, doc])
        mat = ParsedTokenHasher().fit_transform(docs)
        diff = mat[0] - mat[1]
        assert diff.nnz == 0

    def test_different_documents_different_rows(self):
        docs = pd.Series([
            "GENE__BRAF TYPE__MISSENSE",
            "GENE__TP53 TYPE__NONSENSE",
        ])
        mat = ParsedTokenHasher().fit_transform(docs)
        diff = mat[0] - mat[1]
        assert diff.nnz > 0

    def test_sentinel_row_has_nonzero_entries(self):
        docs = pd.Series([_NO_PARSED_TOKEN_SENTINEL])
        mat = ParsedTokenHasher().fit_transform(docs)
        assert mat.nnz > 0

    def test_all_values_nonnegative(self):
        mat = ParsedTokenHasher().fit_transform(self._docs())
        assert mat.data.min() >= 0

    def test_train_test_transform_consistency(self):
        """fit(train) → transform(test): 같은 토큰은 같은 컬럼에 매핑된다."""
        train_docs = pd.Series(["GENE__BRAF TYPE__MISSENSE"])
        test_docs  = pd.Series(["GENE__BRAF TYPE__MISSENSE"])
        hasher = ParsedTokenHasher()
        mat_train = hasher.fit_transform(train_docs)
        mat_test  = hasher.transform(test_docs)
        diff = mat_train - mat_test
        assert diff.nnz == 0


# ---------------------------------------------------------------------------
# 복합 시나리오 — 전체 파이프라인
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """build_parsed_token_documents → ParsedTokenHasher 파이프라인."""

    def test_pipeline_output_shape(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        docs = build_parsed_token_documents(toy_frame, gene_cols)
        hasher = ParsedTokenHasher()
        mat = hasher.fit_transform(docs)
        assert mat.shape[0] == len(toy_frame)
        assert mat.shape[1] == len(hasher.get_feature_names_out())

    def test_wt_row_sparse_differs_from_mutated_row(self, toy_frame):
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        docs = build_parsed_token_documents(toy_frame, gene_cols)
        mat = ParsedTokenHasher().fit_transform(docs)
        # s4(전부WT) 와 s1(변이 다수) 는 달라야 한다
        diff = mat[0] - mat[3]
        assert diff.nnz > 0

    def test_parse_failed_mutation_propagates_to_sparse(self, toy_frame):
        # toy_frame 에는 312_313QY>HH (complex) 가 있어 GENE_TYPE 이 COMPLEX 가 된다
        gene_cols = [c for c in toy_frame.columns if c not in {"ID", "SUBCLASS"}]
        docs = build_parsed_token_documents(toy_frame, gene_cols)
        assert any("TYPE__COMPLEX" in doc for doc in docs)
