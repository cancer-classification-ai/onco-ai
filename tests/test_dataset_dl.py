from __future__ import annotations

import pandas as pd

from cancer_hack.dataset_dl import (
    GENE_RULE_FEATURE_NAMES,
    MutationSampleDataset,
    MutationTokenizer,
    collate_mutation_samples,
    resolve_domain_prefixes,
    tokenize_frame,
)
from cancer_hack.features_domain import ALL_DOMAIN_PREFIXES, DOMAIN_PREFIXES


def test_tokenize_and_collate_preserve_hierarchy(toy_frame: pd.DataFrame) -> None:
    tokenizer = MutationTokenizer(["TP53", "KRAS", "EGFR"], signature_buckets=32)
    samples = tokenize_frame(toy_frame, tokenizer)
    dataset = MutationSampleDataset(
        samples, toy_frame["ID"], labels=[0, 1, 2, 3]
    )
    batch = collate_mutation_samples(
        [dataset[index] for index in range(len(dataset))]
    )

    assert batch["sample_count"] == 4
    assert batch["gene_to_sample"].tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert len(batch["token_to_gene"]) == 12
    assert batch["labels"].tolist() == [0, 1, 2, 3]
    assert tokenizer.vocab_sizes["gene"] == 4


def test_full_domain_feature_set_includes_m_block() -> None:
    assert resolve_domain_prefixes("default") == DOMAIN_PREFIXES
    assert "M_" not in resolve_domain_prefixes("default")
    assert resolve_domain_prefixes("all") == ALL_DOMAIN_PREFIXES
    assert "M_" in resolve_domain_prefixes("all")


def test_empty_sample_is_retained() -> None:
    frame = pd.DataFrame({"ID": ["empty"], "TP53": ["WT"], "KRAS": [""]})
    tokenizer = MutationTokenizer(["TP53", "KRAS"])
    samples = tokenize_frame(frame, tokenizer)
    batch = collate_mutation_samples(
        [MutationSampleDataset(samples, frame["ID"])[0]]
    )

    assert batch["sample_count"] == 1
    assert batch["gene_id"].numel() == 0
    assert batch["gene_to_sample"].numel() == 0


def test_tokenizer_does_not_treat_operation_words_as_amino_acids() -> None:
    tokenizer = MutationTokenizer(["EGFR"])
    deletion = tokenizer.encode("EGFR", "E746_A750del")
    insertion = tokenizer.encode("EGFR", "P11_K12insP")

    assert deletion.ref_id > 0
    assert deletion.alt_id == 0
    assert insertion.ref_id > 0
    assert insertion.alt_id == insertion.ref_id


def test_multi_amino_acid_frameshift_is_retained_as_unknown_ref() -> None:
    tokenizer = MutationTokenizer(["TP53"])
    token = tokenizer.encode("TP53", "WQ288fs")

    assert token.kind_id > 0
    assert token.position == 288
    assert token.ref_id == 0
    assert token.has_unknown


def test_gene_rule_stats_preserve_count_type_and_position_structure() -> None:
    frame = pd.DataFrame(
        {
            "ID": ["sample"],
            "TP53": ["R132H R132H Q369* R132R"],
        }
    )
    tokenizer = MutationTokenizer(["TP53"])
    samples = tokenize_frame(frame, tokenizer)
    dataset = MutationSampleDataset(samples, frame["ID"])
    batch = collate_mutation_samples([dataset[0]])
    stats = dict(
        zip(GENE_RULE_FEATURE_NAMES, batch["gene_rule_stats"][0].tolist())
    )

    assert batch["gene_rule_stats"].shape == (1, 19)
    assert stats["variant_count"] == 4
    assert stats["unique_variant_count"] == 3
    assert stats["duplicate_count"] == 1
    assert stats["unique_position_count"] == 2
    assert stats["missense_ratio"] == 0.5
    assert stats["synonymous_ratio"] == 0.25
    assert stats["nonsense_ratio"] == 0.25
    assert stats["same_position_ratio"] == 0.5
    assert stats["has_multiple_variants"] == 1
    assert stats["has_mixed_mutation_types"] == 1
    assert stats["has_duplicate_variant"] == 1


def test_gene_rule_stats_are_zero_width_safe_for_wt_samples() -> None:
    frame = pd.DataFrame({"ID": ["sample"], "TP53": ["WT"]})
    tokenizer = MutationTokenizer(["TP53"])
    samples = tokenize_frame(frame, tokenizer)
    dataset = MutationSampleDataset(samples, frame["ID"])
    batch = collate_mutation_samples([dataset[0]])

    assert batch["gene_rule_stats"].shape == (0, len(GENE_RULE_FEATURE_NAMES))
    assert batch["gene_ids"].numel() == 0
