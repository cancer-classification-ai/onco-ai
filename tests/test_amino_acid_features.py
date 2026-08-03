import numpy as np
import pandas as pd
import pytest

from cancer_hack.features_amino_acid import (
    AMINO_ACID_FEATURE_COLUMNS,
    NORMALIZED_AMINO_ACID_PROPERTIES,
    make_amino_acid_features,
    make_gene_amino_acid_penalty_features,
    mutation_substitution_penalty,
    physicochemical_substitution_penalty,
)


@pytest.fixture
def mutation_frame():
    return pd.DataFrame(
        {
            "ID": ["s1", "s2", "s3", "s4"],
            "SUBCLASS": ["A", "A", "B", "B"],
            "TP53": ["V600E V600E", "S622S", "B12E", "WT"],
            "KRAS": ["G12D", "WT", "WT", "WT"],
        },
        index=[10, 20, 30, 40],
    )


def test_all_properties_are_minmax_normalized():
    for property_name in next(iter(NORMALIZED_AMINO_ACID_PROPERTIES.values())):
        values = [
            properties[property_name]
            for properties in NORMALIZED_AMINO_ACID_PROPERTIES.values()
        ]
        assert min(values) == pytest.approx(0.0)
        assert max(values) == pytest.approx(1.0)


def test_same_amino_acid_has_zero_penalty():
    assert physicochemical_substitution_penalty("V", "V") == 0.0
    assert mutation_substitution_penalty("S622S") == 0.0


def test_conservative_change_scores_below_charge_change():
    leucine_to_valine = physicochemical_substitution_penalty("L", "V")
    valine_to_glutamate = physicochemical_substitution_penalty("V", "E")
    aspartate_to_histidine = physicochemical_substitution_penalty("D", "H")

    assert leucine_to_valine < valine_to_glutamate
    assert leucine_to_valine < aspartate_to_histidine


def test_unknown_or_non_substitution_returns_nan():
    assert np.isnan(physicochemical_substitution_penalty("B", "E"))
    assert np.isnan(mutation_substitution_penalty("Q369*"))
    assert np.isnan(mutation_substitution_penalty("not-a-mutation"))


def test_sample_feature_schema_index_and_dtype_are_stable(mutation_frame):
    features = make_amino_acid_features(
        mutation_frame,
        gene_columns=["TP53", "KRAS"],
    )

    assert tuple(features.columns) == AMINO_ACID_FEATURE_COLUMNS
    assert features.index.equals(mutation_frame.index)
    assert all(dtype == np.float32 for dtype in features.dtypes)


def test_duplicate_token_in_same_gene_is_counted_once(mutation_frame):
    features = make_amino_acid_features(
        mutation_frame,
        gene_columns=["TP53", "KRAS"],
    )
    expected_sum = (
        mutation_substitution_penalty("V600E")
        + mutation_substitution_penalty("G12D")
    )

    assert features.loc[10, "scored_missense_count"] == 2
    assert features.loc[10, "aa_substitution_penalty_sum"] == pytest.approx(
        expected_sum
    )


def test_raw_duplicate_mode_keeps_repeated_token(mutation_frame):
    features = make_amino_acid_features(
        mutation_frame,
        gene_columns=["TP53", "KRAS"],
        unique_tokens_per_gene=False,
    )

    assert features.loc[10, "scored_missense_count"] == 3


def test_same_transition_at_different_positions_remains_separate():
    frame = pd.DataFrame({"TP53": ["V600E V100E"]})

    features = make_amino_acid_features(frame, gene_columns=["TP53"])

    assert features.loc[0, "scored_missense_count"] == 2


def test_same_token_in_different_genes_remains_separate():
    frame = pd.DataFrame({"GENE_A": ["V600E"], "GENE_B": ["V600E"]})

    features = make_amino_acid_features(
        frame,
        gene_columns=["GENE_A", "GENE_B"],
    )

    assert features.loc[0, "scored_missense_count"] == 2


def test_synonymous_and_no_missense_samples_have_zero_aggregates(mutation_frame):
    features = make_amino_acid_features(
        mutation_frame,
        gene_columns=["TP53", "KRAS"],
    )

    assert (features.loc[20] == 0).all()
    assert (features.loc[40] == 0).all()


def test_unscorable_missense_is_missing_not_zero(mutation_frame):
    features = make_amino_acid_features(
        mutation_frame,
        gene_columns=["TP53", "KRAS"],
    )

    assert features.loc[30, "scored_missense_count"] == 0
    assert features.loc[30, "aa_substitution_penalty_mean"] == 0
    assert features.loc[30, "aa_penalty_missing_count"] == 1
    assert features.loc[30, "aa_penalty_data_missing"] == 1


def test_threshold_features_follow_configured_cutoffs():
    frame = pd.DataFrame({"TP53": ["L10V V600E"]})
    low = mutation_substitution_penalty("L10V")
    high = mutation_substitution_penalty("V600E")
    midpoint = (low + high) / 2

    features = make_amino_acid_features(
        frame,
        gene_columns=["TP53"],
        conservative_max=midpoint,
        radical_min=midpoint + 0.01,
    )

    assert features.loc[0, "conservative_substitution_count"] == 1
    assert features.loc[0, "radical_substitution_count"] == 1
    assert features.loc[0, "has_radical_substitution"] == 1


def test_gene_features_are_opt_in_and_use_max_penalty(mutation_frame):
    features = make_gene_amino_acid_penalty_features(
        mutation_frame,
        gene_columns=["TP53", "KRAS"],
    )

    assert list(features.columns) == [
        "gene_aa_penalty__TP53",
        "gene_aa_penalty__KRAS",
    ]
    assert features.loc[10, "gene_aa_penalty__TP53"] == pytest.approx(
        mutation_substitution_penalty("V600E")
    )
    assert features.loc[10, "gene_aa_penalty__KRAS"] == pytest.approx(
        mutation_substitution_penalty("G12D")
    )


def test_labels_do_not_affect_features(mutation_frame):
    first = make_amino_acid_features(
        mutation_frame,
        gene_columns=["TP53", "KRAS"],
    )
    relabeled = mutation_frame.copy()
    relabeled["SUBCLASS"] = ["X", "Y", "Z", "W"]
    second = make_amino_acid_features(
        relabeled,
        gene_columns=["TP53", "KRAS"],
    )

    pd.testing.assert_frame_equal(first, second)


def test_invalid_weights_and_thresholds_raise(mutation_frame):
    with pytest.raises(ValueError):
        physicochemical_substitution_penalty(
            "V",
            "E",
            property_weights={"polarity": 1.0},
        )
    with pytest.raises(ValueError):
        make_amino_acid_features(
            mutation_frame,
            gene_columns=["TP53", "KRAS"],
            conservative_max=1.0,
            radical_min=1.0,
        )
