"""Train-fold frequency/IDF/rarity 피처의 계산식과 누수 방지 계약."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancer_hack.features_sparse import (
    FREQUENCY_RARITY_FEATURE_COLUMNS,
    TrainFrequencyFeatures,
)


@pytest.fixture
def frequency_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["s1", "s2", "s3", "s4"],
            "SUBCLASS": ["A", "B", "A", "B"],
            "TP53": ["R175H", "R175H", "WT", "WT"],
            "KRAS": ["WT", "G12D", "G12D G12D", "WT"],
            "EGFR": ["WT", "WT", "WT", "WT"],
        },
        index=[10, 20, 30, 40],
    )


def test_gene_token_frequency_idf_and_gene_rarity(frequency_train):
    builder = TrainFrequencyFeatures().fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )

    assert builder.gene_frequency_ == {"TP53": 0.5, "KRAS": 0.5, "EGFR": 0.0}
    assert builder.token_frequency_["MUT__TP53__R175H"] == 0.5
    assert builder.token_frequency_["MUT__KRAS__G12D"] == 0.5
    assert builder.token_document_count_["MUT__KRAS__G12D"] == 2

    expected_idf = np.log(5 / 3) + 1
    assert builder.token_idf_["MUT__TP53__R175H"] == pytest.approx(expected_idf)
    assert builder.gene_rarity_["TP53"] == pytest.approx(-np.log(3 / 5))
    assert builder.gene_rarity_["EGFR"] == pytest.approx(-np.log(1 / 5))


def test_sample_features_include_all_requested_aggregates(frequency_train):
    out = TrainFrequencyFeatures(rare_df_threshold=2).fit_transform(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )

    assert tuple(out.columns) == FREQUENCY_RARITY_FEATURE_COLUMNS
    assert list(out.index) == [10, 20, 30, 40]
    assert int(out.loc[20, "rare_mutation_token_count"]) == 2
    assert out.loc[20, "mean_token_frequency"] == pytest.approx(0.5)
    assert out.loc[20, "mean_token_rarity"] == pytest.approx(np.log(5 / 3) + 1)
    assert out.loc[20, "max_gene_rarity"] == pytest.approx(-np.log(3 / 5))


def test_duplicate_token_counts_once_per_sample(frequency_train):
    builder = TrainFrequencyFeatures().fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    assert builder.token_document_count_["MUT__KRAS__G12D"] == 2

    out = builder.transform(
        frequency_train.loc[[30]], gene_columns=["TP53", "KRAS", "EGFR"]
    )
    assert int(out.iloc[0]["rare_mutation_token_count"]) == 1


def test_unseen_validation_token_and_gene_use_train_statistics(frequency_train):
    builder = TrainFrequencyFeatures(rare_df_threshold=2).fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    validation = pd.DataFrame(
        {
            "TP53": ["NEW1"],
            "KRAS": ["WT"],
            "EGFR": ["E746_A750del"],
        },
        index=[99],
    )
    out = builder.transform(
        validation, gene_columns=["TP53", "KRAS", "EGFR"]
    )

    assert int(out.loc[99, "unseen_token_count"]) == 2
    assert int(out.loc[99, "rare_mutation_token_count"]) == 2
    assert int(out.loc[99, "unseen_mutated_gene_count"]) == 1
    assert out.loc[99, "max_token_rarity"] == pytest.approx(np.log(5) + 1)
    assert out.loc[99, "min_token_frequency"] == 0.0


def test_transform_does_not_change_train_statistics(frequency_train):
    builder = TrainFrequencyFeatures().fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    token_frequency_before = builder.token_frequency_.copy()
    gene_frequency_before = builder.gene_frequency_.copy()

    extreme_validation = pd.DataFrame(
        {
            "TP53": ["X"] * 100,
            "KRAS": ["Y"] * 100,
            "EGFR": ["Z"] * 100,
        }
    )
    builder.transform(
        extreme_validation, gene_columns=["TP53", "KRAS", "EGFR"]
    )

    assert builder.token_frequency_ == token_frequency_before
    assert builder.gene_frequency_ == gene_frequency_before


def test_labels_do_not_affect_statistics(frequency_train):
    changed_labels = frequency_train.copy()
    changed_labels["SUBCLASS"] = "CHANGED"

    original = TrainFrequencyFeatures().fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    changed = TrainFrequencyFeatures().fit(
        changed_labels, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    assert original.gene_frequency_ == changed.gene_frequency_
    assert original.token_frequency_ == changed.token_frequency_
    assert original.token_idf_ == changed.token_idf_


def test_no_mutation_sample_returns_finite_zeros(frequency_train):
    builder = TrainFrequencyFeatures().fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    empty = pd.DataFrame({"TP53": ["WT"], "KRAS": ["WT"], "EGFR": ["WT"]})
    out = builder.transform(empty, gene_columns=["TP53", "KRAS", "EGFR"])

    assert np.isfinite(out.to_numpy(dtype=np.float64)).all()
    assert out.to_numpy(dtype=np.float64).sum() == 0.0


def test_transform_before_fit_and_column_mismatch_raise(frequency_train):
    with pytest.raises(RuntimeError):
        TrainFrequencyFeatures().transform(
            frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
        )

    builder = TrainFrequencyFeatures().fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    with pytest.raises(ValueError, match="match fit"):
        builder.transform(
            frequency_train, gene_columns=["KRAS", "TP53", "EGFR"]
        )


def test_statistics_tables_are_stable_and_shareable(frequency_train):
    builder = TrainFrequencyFeatures().fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    gene_stats = builder.get_gene_statistics()
    token_stats = builder.get_token_statistics()

    assert list(gene_stats.columns) == [
        "gene",
        "mutation_sample_count",
        "mutation_frequency",
        "gene_rarity",
    ]
    assert list(token_stats.columns) == [
        "mutation_token",
        "document_count",
        "frequency",
        "idf",
    ]
    assert token_stats["mutation_token"].is_monotonic_increasing


def test_five_frequency_families_are_distinct_and_backward_compatible():
    frame = pd.DataFrame(
        {
            "TP53": ["R175H", "R175H Q369*", "WT"],
            "KRAS": ["WT", "WT", "R175H"],
        }
    )
    builder = TrainFrequencyFeatures().fit(
        frame, gene_columns=["TP53", "KRAS"]
    )

    assert builder.gene_frequency_["TP53"] == pytest.approx(2 / 3)
    assert builder.aa_change_frequency_["AA__R175H"] == 1.0
    assert builder.gene_aa_change_frequency_["MUT__TP53__R175H"] == pytest.approx(
        2 / 3
    )
    assert builder.gene_aa_change_frequency_["MUT__KRAS__R175H"] == pytest.approx(
        1 / 3
    )
    assert builder.position_frequency_["POS__175"] == 1.0
    assert builder.exact_mutation_frequency_["EXACT__TP53__Q369*|R175H"] == pytest.approx(
        1 / 3
    )

    # token_*은 기존 노트북 호환용 gene_aa_change alias다.
    assert builder.token_frequency_ is builder.gene_aa_change_frequency_
    assert builder.token_document_count_ is builder.gene_aa_change_document_count_


def test_exact_cell_signature_ignores_order_but_preserves_duplicate_count():
    frame = pd.DataFrame(
        {
            "TP53": [
                "R175H Q369*",
                "Q369* R175H",
                "R175H R175H Q369*",
            ]
        }
    )
    builder = TrainFrequencyFeatures().fit(frame, gene_columns=["TP53"])

    assert (
        builder.exact_mutation_document_count_["EXACT__TP53__Q369*|R175H"] == 2
    )
    assert (
        builder.exact_mutation_document_count_[
            "EXACT__TP53__Q369*|R175H|R175H"
        ]
        == 1
    )


def test_position_frequency_uses_numeric_signature_without_expanding_ranges():
    frame = pd.DataFrame(
        {
            "TP53": ["E746_A750del", "WT"],
            "EGFR": ["WT", "746_750QY>HH"],
        }
    )
    builder = TrainFrequencyFeatures().fit(
        frame, gene_columns=["TP53", "EGFR"]
    )

    assert builder.position_document_count_ == {"POS__746_750": 2}
    assert builder.position_frequency_["POS__746_750"] == 1.0


def test_new_frequency_aggregates_are_fixed_and_finite():
    train = pd.DataFrame(
        {
            "TP53": ["R175H", "R175H Q369*", "WT"],
            "KRAS": ["WT", "WT", "G12D"],
        }
    )
    builder = TrainFrequencyFeatures().fit(
        train, gene_columns=["TP53", "KRAS"]
    )
    validation = pd.DataFrame(
        {
            "TP53": ["R175H Q369*", "WT"],
            "KRAS": ["G13D", "WT"],
        },
        index=[7, 8],
    )
    out = builder.transform(
        validation, gene_columns=["TP53", "KRAS"]
    )

    expected_new_columns = (
        "mean_exact_mutation_frequency",
        "min_exact_mutation_frequency",
        "max_exact_mutation_frequency",
        "mean_aa_change_frequency",
        "min_aa_change_frequency",
        "max_aa_change_frequency",
        "max_gene_aa_change_frequency",
        "mean_position_frequency",
        "min_position_frequency",
        "max_position_frequency",
    )
    assert tuple(out.columns[-10:]) == expected_new_columns
    assert out.loc[7, "min_aa_change_frequency"] == 0.0
    assert out.loc[7, "max_aa_change_frequency"] == pytest.approx(2 / 3)
    assert out.loc[8, list(expected_new_columns)].sum() == 0.0
    assert np.isfinite(
        out[list(expected_new_columns)].to_numpy(dtype=np.float64)
    ).all()


def test_frequency_statistics_lists_all_new_families(frequency_train):
    builder = TrainFrequencyFeatures().fit(
        frequency_train, gene_columns=["TP53", "KRAS", "EGFR"]
    )
    stats = builder.get_frequency_statistics()

    assert list(stats.columns) == [
        "family",
        "key",
        "document_count",
        "frequency",
    ]
    assert set(stats["family"]) == {
        "exact_mutation",
        "aa_change",
        "gene_aa_change",
        "position",
    }
