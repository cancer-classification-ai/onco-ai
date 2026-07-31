"""피처 스키마 계약 — 팀 파생변수 목록과 실제 출력이 같은지 본다.

train 과 test 가 같은 컬럼을 같은 순서로 내야 모델에 그대로 들어간다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cancer_hack.features_basic import (
    SAMPLE_FEATURE_COLUMNS,
    _ROLLUP_ANY_COLUMNS,
    _ROLLUP_SUM_COLUMNS,
    make_gene_event_count_matrix,
    make_gene_mutated_matrix,
    make_sample_mutation_features,
)

# 팀 파생변수 목록에 적힌 이름. fold 안에서 잡아야 하는 hypermutated_flag /
# burden_quantile_bin 은 BurdenBinner 담당이라 여기 없고, 비추천인 deletion_ratio 도 뺐다.
TEAM_SCHEMA = (
    "mutated_gene_count",
    "mutation_event_count",
    "synonymous_event_count",
    "functional_event_count",
    "missense_event_count",
    "nonsense_event_count",
    "frameshift_event_count",
    "complex_event_count",
    "multihit_gene_count",
    "max_events_per_gene",
    "no_mutation_flag",
    "log1p_mutated_gene_count",
    "log1p_mutation_event_count",
    "functional_ratio",
    "synonymous_ratio",
    "missense_ratio",
    "nonsense_ratio",
    "frameshift_ratio",
    "multihit_gene_ratio",
    "explicit_deletion_event_count",
    "explicit_deletion_gene_count",
    "has_explicit_deletion",
    "inframe_deletion_count",
    "indel_or_frameshift_count",
    "loss_of_function_count",
)


def test_sample_schema_matches_team_list():
    assert SAMPLE_FEATURE_COLUMNS == TEAM_SCHEMA


def test_default_output_adds_no_extra_columns(toy_frame):
    genes = ["TP53", "KRAS", "EGFR"]
    out = make_sample_mutation_features(toy_frame, gene_columns=genes)
    assert tuple(out.columns) == TEAM_SCHEMA


def test_train_and_test_align(toy_frame):
    """라벨이 없어도 피처 컬럼은 동일해야 한다."""
    genes = ["TP53", "KRAS", "EGFR"]
    train_out = make_sample_mutation_features(toy_frame, gene_columns=genes)
    test_out = make_sample_mutation_features(
        toy_frame.drop(columns=["SUBCLASS"]), gene_columns=genes
    )
    assert list(train_out.columns) == list(test_out.columns)


def test_cell_rollup_is_opt_in(toy_frame):
    genes = ["TP53", "KRAS", "EGFR"]
    base = make_sample_mutation_features(toy_frame, gene_columns=genes)
    rolled = make_sample_mutation_features(
        toy_frame, gene_columns=genes, include_cell_rollup=True
    )
    assert list(rolled.columns)[: len(TEAM_SCHEMA)] == list(base.columns)
    for name in [
        "has_indel",
        "has_mnv",
        "indel_count",
        "mnv_count",
        "has_duplicate_token",
        "unique_mutation_token_count",
    ]:
        assert name in rolled.columns
        assert name not in base.columns
    # 계약 컬럼의 값은 스위치와 무관해야 한다.
    pd.testing.assert_frame_equal(base, rolled[list(TEAM_SCHEMA)])


def test_identities_hold(toy_frame):
    genes = ["TP53", "KRAS", "EGFR"]
    out = make_sample_mutation_features(
        toy_frame, gene_columns=genes, include_cell_rollup=True
    )
    assert (out["functional_event_count"] <= out["mutation_event_count"]).all()
    assert (out["multihit_gene_count"] <= out["mutated_gene_count"]).all()
    assert (out["max_events_per_gene"] <= out["mutation_event_count"]).all()
    assert (
        out["indel_or_frameshift_count"]
        == out["explicit_deletion_event_count"] + out["frameshift_event_count"]
    ).all()
    # fs 를 del 보다 먼저 분류하므로 in-frame 조건이 이미 만족돼 두 값이 같다.
    assert (out["inframe_deletion_count"] == out["explicit_deletion_event_count"]).all()
    assert ((out["no_mutation_flag"] == 1) == (out["mutated_gene_count"] == 0)).all()
    # 1차 표 이름은 세분 유형에서 유도되므로 항상 일치한다.
    assert (out["mnv_count"] == out["complex_event_count"]).all()


def test_ratios_are_bounded(toy_frame):
    genes = ["TP53", "KRAS", "EGFR"]
    out = make_sample_mutation_features(toy_frame, gene_columns=genes)
    ratios = [c for c in out.columns if c.endswith("_ratio")]
    assert ratios
    assert out[ratios].to_numpy().min() >= 0.0
    assert out[ratios].to_numpy().max() <= 1.0


def test_no_mutation_sample_has_zero_ratios():
    frame = pd.DataFrame({"ID": ["s1"], "TP53": ["WT"], "KRAS": ["WT"]})
    out = make_sample_mutation_features(frame, gene_columns=["TP53", "KRAS"])
    assert int(out["no_mutation_flag"].iloc[0]) == 1
    ratios = [c for c in out.columns if c.endswith("_ratio")]
    assert out[ratios].to_numpy().sum() == 0.0


def test_gene_matrices_are_prefixed_and_aligned(toy_frame):
    genes = ["TP53", "KRAS", "EGFR"]
    mutated = make_gene_mutated_matrix(toy_frame, gene_columns=genes)
    counts = make_gene_event_count_matrix(toy_frame, gene_columns=genes)
    assert list(mutated.columns) == [f"gene_mutated__{g}" for g in genes]
    assert list(counts.columns) == [f"gene_event_count__{g}" for g in genes]
    assert mutated.to_numpy().max() <= 1
    # 변이 유무는 토큰 수가 0인지와 같아야 한다.
    assert (mutated.to_numpy() == (counts.to_numpy() > 0)).all()


def test_missing_gene_column_raises(toy_frame):
    with pytest.raises(ValueError, match="Missing gene columns"):
        make_sample_mutation_features(toy_frame, gene_columns=["TP53", "NOPE"])


def test_rollup_width_follows_the_constants(toy_frame):
    """상수만 고치고 소비부를 안 고치면 열이 안 늘어난다. 그걸 잡는다."""
    genes = ["TP53", "KRAS", "EGFR"]
    rolled = make_sample_mutation_features(
        toy_frame, gene_columns=genes, include_cell_rollup=True
    )
    assert len(rolled.columns) == (
        len(TEAM_SCHEMA) + len(_ROLLUP_ANY_COLUMNS) + len(_ROLLUP_SUM_COLUMNS)
    )


def test_rollup_carries_both_duplicate_counts(toy_frame):
    """샘플 단위까지 올라와야 학습 피처가 된다 — 셀 단위 자동 반영과 별개다."""
    genes = ["TP53", "KRAS", "EGFR"]
    rolled = make_sample_mutation_features(
        toy_frame, gene_columns=genes, include_cell_rollup=True
    )
    for name in ("duplicate_token_count", "duplicate_signature_count"):
        assert name in rolled.columns
    # toy_frame 의 KRAS `V600E V600E` 한 건이 두 기준 모두에 잡힌다.
    assert rolled["duplicate_token_count"].sum() == 1
    assert rolled["duplicate_signature_count"].sum() == 1


def test_duplicate_signature_count_stays_out_of_robust_rollup():
    """train 1.30 / test 52.8 로 40배 시프트다. 시프트 내성 블록에 새면 안 된다."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_gbdt import ROBUST_ROLLUP_COLUMNS

    assert "duplicate_signature_count" not in ROBUST_ROLLUP_COLUMNS
    assert "duplicate_token_count" not in ROBUST_ROLLUP_COLUMNS
    # 플래그 버전은 배율이 안정적이라 남아 있어야 한다.
    assert "has_duplicate_token" in ROBUST_ROLLUP_COLUMNS
