from __future__ import annotations

import pandas as pd

from cancer_hack.dataset_dl import (
    MutationSampleDataset,
    MutationTokenizer,
    collate_mutation_samples,
    tokenize_frame,
)


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
