import math

import numpy as np
import pandas as pd
import pytest

from cancer_hack.features_frequency import (
    AA_TRANSITION_FEATURE_COLUMNS,
    N_DIRECTED_MISSENSE_TRANSITIONS,
    TrainAATransitionFeatures,
)


@pytest.fixture
def transition_train():
    return pd.DataFrame(
        {
            "ID": ["s1", "s2", "s3", "s4"],
            "SUBCLASS": ["A", "A", "B", "B"],
            "TP53": ["V600E V600E", "V100E Q369*", "R175H S622S", "WT"],
            "KRAS": ["WT", "WT", "R12H", "G12D"],
        },
        index=[10, 20, 30, 40],
    )


def test_output_schema_is_stable(transition_train):
    features = TrainAATransitionFeatures().fit_transform(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )

    assert tuple(features.columns) == AA_TRANSITION_FEATURE_COLUMNS
    assert features.index.equals(transition_train.index)
    assert all(dtype == np.float32 for dtype in features.dtypes)


def test_fit_counts_unique_directed_transitions_per_sample(transition_train):
    builder = TrainAATransitionFeatures().fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )

    assert builder.transition_document_count_[("V", "E")] == 2
    assert builder.transition_document_count_[("R", "H")] == 1
    assert builder.transition_document_count_[("G", "D")] == 1
    assert builder.total_transition_count_ == 4
    assert ("Q", "*") not in builder.transition_document_count_
    assert ("S", "S") not in builder.transition_document_count_


def test_transform_applies_same_unique_rule_as_fit(transition_train):
    builder = TrainAATransitionFeatures().fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )
    validation = pd.DataFrame(
        {
            "TP53": ["V600E V600E"],
            "KRAS": ["V100E"],
        }
    )

    features = builder.transform(validation, gene_columns=["TP53", "KRAS"])

    assert features.loc[0, "aa_transition_frequency_mean"] == pytest.approx(0.5)
    assert features.loc[0, "aa_transition_frequency_max"] == pytest.approx(0.5)
    assert features.loc[0, "aa_unseen_transition_count"] == 0
    assert features.loc[0, "aa_unseen_transition_ratio"] == 0


def test_frequency_is_normalized_by_train_fold_sample_count(transition_train):
    stats = TrainAATransitionFeatures().fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    ).get_transition_statistics()
    v_to_e = stats[(stats["wt"] == "V") & (stats["mutant"] == "E")].iloc[0]

    assert v_to_e["document_count"] == 2
    assert v_to_e["frequency"] == pytest.approx(0.5)


def test_conditional_rarity_uses_19_possible_substitutions(transition_train):
    builder = TrainAATransitionFeatures(alpha=1.0).fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )

    _, rarity, _, _ = builder._score_transition("V", "E")

    assert rarity == pytest.approx(-math.log((2 + 1) / (2 + 19)))


def test_log_odds_uses_all_380_directed_pairs(transition_train):
    builder = TrainAATransitionFeatures(alpha=1.0).fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )

    _, _, log_odds, _ = builder._score_transition("V", "E")
    denominator = 4 + N_DIRECTED_MISSENSE_TRANSITIONS
    joint = 3 / denominator
    wt_marginal = 21 / denominator
    mutant_marginal = 21 / denominator

    assert N_DIRECTED_MISSENSE_TRANSITIONS == 380
    expected = math.log(joint / (wt_marginal * mutant_marginal))
    assert log_odds == pytest.approx(expected)


def test_unseen_transition_is_counted_and_scores_are_finite(transition_train):
    builder = TrainAATransitionFeatures().fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )
    validation = pd.DataFrame({"TP53": ["L858R"], "KRAS": ["WT"]})

    features = builder.transform(validation, gene_columns=["TP53", "KRAS"])

    assert features.loc[0, "aa_transition_frequency_mean"] == 0
    assert features.loc[0, "aa_unseen_transition_count"] == 1
    assert features.loc[0, "aa_unseen_transition_ratio"] == 1
    assert np.isfinite(features.to_numpy()).all()


def test_sample_without_missense_has_zero_features(transition_train):
    builder = TrainAATransitionFeatures().fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )
    validation = pd.DataFrame({"TP53": ["Q369*"], "KRAS": ["S622S"]})

    features = builder.transform(validation, gene_columns=["TP53", "KRAS"])

    assert (features.iloc[0] == 0).all()
    assert np.isfinite(features.to_numpy()).all()


def test_target_labels_do_not_affect_transition_statistics(transition_train):
    first = TrainAATransitionFeatures().fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )
    shuffled_labels = transition_train.copy()
    shuffled_labels["SUBCLASS"] = ["X", "Y", "Z", "W"]
    second = TrainAATransitionFeatures().fit(
        shuffled_labels,
        gene_columns=["TP53", "KRAS"],
    )

    assert first.transition_document_count_ == second.transition_document_count_
    assert first.wt_transition_count_ == second.wt_transition_count_
    assert first.mutant_transition_count_ == second.mutant_transition_count_


def test_transform_requires_fit_and_matching_gene_columns(transition_train):
    with pytest.raises(RuntimeError):
        TrainAATransitionFeatures().transform(
            transition_train,
            gene_columns=["TP53", "KRAS"],
        )

    builder = TrainAATransitionFeatures().fit(
        transition_train,
        gene_columns=["TP53", "KRAS"],
    )
    with pytest.raises(ValueError):
        builder.transform(transition_train, gene_columns=["KRAS", "TP53"])


def test_alpha_must_be_positive():
    with pytest.raises(ValueError):
        TrainAATransitionFeatures(alpha=0)
