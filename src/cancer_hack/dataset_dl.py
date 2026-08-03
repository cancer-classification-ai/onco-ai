"""PyTorch datasets for hierarchical mutation-set models.

The raw competition table is wide (one column per gene), but each sample contains
only a small set of mutations.  This module converts it to a sparse hierarchy:

``sample -> mutated gene -> mutation token``.

All token features are deterministic parser outputs.  No labels or test-set
statistics are used while tokenizing, so one tokenized dataset can safely be
reused by every CV fold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features_amino_acid import AMINO_ACID_FEATURE_COLUMNS
from .features_basic import (
    ADDITIONAL_BURDEN_FEATURE_COLUMNS,
    MUTATION_STRING_PARSED_COLUMNS,
)
from .features_domain import DOMAIN_PREFIXES, align_domain_columns
from .parser import (
    ALL_KINDS,
    DELETION,
    DELINS,
    FRAMESHIFT,
    INSERTION,
    MISSENSE,
    NONSENSE,
    OTHER,
    SYNONYMOUS,
    classify_token,
    split_tokens,
    token_signature,
)


ROBUST_ROLLUP_COLUMNS: tuple[str, ...] = (
    "functional_ratio",
    "missense_ratio",
    "synonymous_ratio",
    "nonsense_ratio",
    "frameshift_ratio",
    "multihit_gene_ratio",
    "has_missense",
    "has_synonymous",
    "has_nonsense",
    "has_frameshift",
    "has_duplicate_token",
    "no_mutation_flag",
    "log1p_mutated_gene_count",
    "log1p_mutation_event_count",
)

_POSITION_RE = re.compile(r"\d+")
_SUBSTITUTION_RE = re.compile(r"^([A-Z*])\d+([A-Z*])$")
_RANGE_SUBSTITUTION_RE = re.compile(r"^\d+_\d+([A-Z*]+)>([A-Z*]+)$")
_INSERTED_RE = re.compile(r"(?:DELINS|INS|DUP)([A-Z*]+)$")
_FRAMESHIFT_ALT_RE = re.compile(r"^[A-Z*]+\d+([A-Z*]+)FS$")
_PREFIX_BEFORE_POSITION_RE = re.compile(r"^([^0-9]*)\d+")
_AMINO_ACIDS = "*-ACDEFGHIKLMNPQRSTVWXY"
_AA_TO_ID = {aa: index + 1 for index, aa in enumerate(_AMINO_ACIDS)}
_KIND_TO_ID = {kind: index + 1 for index, kind in enumerate(ALL_KINDS)}

GENE_RULE_FEATURE_NAMES: tuple[str, ...] = (
    "variant_count",
    "unique_variant_count",
    "duplicate_count",
    "unique_position_count",
    "missense_ratio",
    "synonymous_ratio",
    "nonsense_ratio",
    "frameshift_ratio",
    "indel_ratio",
    "unknown_type_ratio",
    "log1p_min_position",
    "log1p_max_position",
    "log1p_position_span",
    "log1p_min_position_distance",
    "same_position_ratio",
    "has_multiple_variants",
    "has_mixed_mutation_types",
    "has_duplicate_variant",
    "has_unknown_token",
)

GENE_RULE_FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "counts": (0, 1, 3),
    "type_ratios": (4, 5, 6, 7, 8, 9),
    "position_stats": (10, 11, 12, 13, 14),
    "duplicate_stats": (2, 17),
    "state_flags": (15, 16, 18),
}


@dataclass(frozen=True)
class MutationToken:
    gene_id: int
    kind_id: int
    ref_id: int
    alt_id: int
    position_id: int
    signature_id: int
    position: int
    raw_key: str
    has_unknown: bool


class MutationTokenizer:
    """Turn mutation strings into fixed categorical fields.

    ``0`` is reserved for unknown/padding in every vocabulary.  Signature IDs
    use a stable hash rather than a fitted vocabulary, avoiding fold leakage and
    making train/test tokenization identical.
    """

    def __init__(
        self,
        gene_names: Sequence[str],
        *,
        position_bin_size: int = 25,
        max_position_bins: int = 512,
        signature_buckets: int = 4096,
    ) -> None:
        if position_bin_size < 1 or max_position_bins < 1 or signature_buckets < 1:
            raise ValueError("tokenizer sizes must be positive")
        self.gene_names = tuple(str(gene) for gene in gene_names)
        self.gene_to_id = {
            gene: index + 1 for index, gene in enumerate(self.gene_names)
        }
        self.position_bin_size = int(position_bin_size)
        self.max_position_bins = int(max_position_bins)
        self.signature_buckets = int(signature_buckets)

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {
            "gene": len(self.gene_names) + 1,
            "kind": len(_KIND_TO_ID) + 1,
            "amino_acid": len(_AA_TO_ID) + 1,
            "position": self.max_position_bins + 1,
            "signature": self.signature_buckets + 1,
        }

    def encode(self, gene: str, token: str) -> MutationToken:
        kind = classify_token(token)
        ref, alt = _amino_acid_endpoints(token)
        ref_id = _AA_TO_ID.get(ref, 0)
        alt_id = _AA_TO_ID.get(alt, 0)
        matched = _POSITION_RE.search(token)
        if matched:
            position = int(matched.group())
            position_id = min(
                position // self.position_bin_size + 1, self.max_position_bins
            )
        else:
            position = 0
            position_id = 0
        signature = token_signature(token, kind)
        digest = hashlib.blake2b(signature.encode("utf-8"), digest_size=8).digest()
        signature_id = int.from_bytes(digest, "little") % self.signature_buckets + 1
        return MutationToken(
            gene_id=self.gene_to_id.get(str(gene), 0),
            kind_id=_KIND_TO_ID.get(kind, 0),
            ref_id=ref_id,
            alt_id=alt_id,
            position_id=position_id,
            signature_id=signature_id,
            position=position,
            raw_key=str(token).strip().upper(),
            has_unknown=(kind == OTHER or not ref),
        )

    def encode_cell(
        self, gene: str, value: object, *, max_tokens: int | None = None
    ) -> list[MutationToken]:
        tokens = split_tokens(value)
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        return [self.encode(gene, token) for token in tokens]


def _amino_acid_endpoints(token: str) -> tuple[str, str]:
    """Extract biological ref/alt symbols without treating ``del/ins/fs`` as AAs."""
    token = token.upper()
    matched = _SUBSTITUTION_RE.match(token)
    if matched:
        return matched.group(1), matched.group(2)
    matched = _RANGE_SUBSTITUTION_RE.match(token)
    if matched:
        return matched.group(1)[0], matched.group(2)[0]
    prefix_match = _PREFIX_BEFORE_POSITION_RE.match(token)
    prefix = prefix_match.group(1) if prefix_match else ""
    ref = prefix if len(prefix) == 1 and prefix in _AA_TO_ID else ""
    matched = _INSERTED_RE.search(token)
    if matched:
        return ref, matched.group(1)[0]
    matched = _FRAMESHIFT_ALT_RE.match(token)
    if matched:
        return ref, matched.group(1)[0]
    return ref, ""


def tokenize_frame(
    frame: pd.DataFrame,
    tokenizer: MutationTokenizer,
    *,
    max_tokens_per_gene: int | None = 32,
) -> list[list[list[MutationToken]]]:
    """Pre-tokenize a raw frame into samples containing mutated genes."""
    missing = [gene for gene in tokenizer.gene_names if gene not in frame.columns]
    if missing:
        raise ValueError(f"raw frame is missing {len(missing)} genes: {missing[:5]}")
    values = frame.loc[:, tokenizer.gene_names].to_numpy(dtype=object)
    samples: list[list[list[MutationToken]]] = []
    for row in values:
        genes: list[list[MutationToken]] = []
        for gene, value in zip(tokenizer.gene_names, row):
            if value is None or str(value).strip().upper() in {"", "WT", "0", "NA", "NAN", "NONE", "."}:
                continue
            encoded = tokenizer.encode_cell(
                gene, value, max_tokens=max_tokens_per_gene
            )
            if encoded:
                genes.append(encoded)
        samples.append(genes)
    return samples


class MutationSampleDataset(Dataset):
    """A view over pre-tokenized mutation samples and optional dense features."""

    def __init__(
        self,
        samples: Sequence[list[list[MutationToken]]],
        ids: Sequence[str],
        *,
        labels: Sequence[int] | None = None,
        dense: np.ndarray | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.samples = samples
        self.ids = np.asarray(ids, dtype=str)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.dense = None if dense is None else np.asarray(dense, dtype=np.float32)
        self.indices = (
            np.arange(len(self.ids), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if len(samples) != len(self.ids):
            raise ValueError("sample and ID counts differ")
        if self.labels is not None and len(self.labels) != len(self.ids):
            raise ValueError("label and ID counts differ")
        if self.dense is not None and len(self.dense) != len(self.ids):
            raise ValueError("dense feature and ID counts differ")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        source = int(self.indices[index])
        return {
            "sample": self.samples[source],
            "id": self.ids[source],
            "label": None if self.labels is None else int(self.labels[source]),
            "dense": None if self.dense is None else self.dense[source],
        }


def collate_mutation_samples(items: Sequence[dict[str, object]]) -> dict[str, object]:
    """Flatten a batch while retaining token->gene and gene->sample indices."""
    fields = {
        "gene_id": [],
        "kind_id": [],
        "ref_id": [],
        "alt_id": [],
        "position_id": [],
        "signature_id": [],
    }
    token_to_gene: list[int] = []
    gene_to_sample: list[int] = []
    gene_ids: list[int] = []
    gene_rule_stats: list[list[float]] = []
    gene_index = 0
    for sample_index, item in enumerate(items):
        for gene_tokens in item["sample"]:
            gene_to_sample.append(sample_index)
            gene_ids.append(gene_tokens[0].gene_id)
            gene_rule_stats.append(_gene_rule_stats(gene_tokens))
            for token in gene_tokens:
                for name in fields:
                    fields[name].append(getattr(token, name))
                token_to_gene.append(gene_index)
            gene_index += 1

    batch: dict[str, object] = {
        name: torch.as_tensor(values, dtype=torch.long)
        for name, values in fields.items()
    }
    batch["token_to_gene"] = torch.as_tensor(token_to_gene, dtype=torch.long)
    batch["gene_to_sample"] = torch.as_tensor(gene_to_sample, dtype=torch.long)
    batch["gene_ids"] = torch.as_tensor(gene_ids, dtype=torch.long)
    batch["gene_rule_stats"] = torch.as_tensor(
        gene_rule_stats,
        dtype=torch.float32,
    ).reshape(-1, len(GENE_RULE_FEATURE_NAMES))
    batch["sample_count"] = len(items)
    batch["ids"] = [str(item["id"]) for item in items]
    labels = [item["label"] for item in items]
    dense = [item["dense"] for item in items]
    batch["labels"] = (
        None
        if labels[0] is None
        else torch.as_tensor(labels, dtype=torch.long)
    )
    batch["dense"] = (
        None
        if dense[0] is None
        else torch.as_tensor(np.stack(dense), dtype=torch.float32)
    )
    return batch


def _gene_rule_stats(tokens: Sequence[MutationToken]) -> list[float]:
    """Return label-free statistics for one mutated gene.

    Count columns remain raw so the model configuration can choose whether to
    apply ``log1p``. Position columns are always ``log1p`` transformed because
    protein lengths are unavailable.
    """
    count = len(tokens)
    raw_keys = [token.raw_key for token in tokens]
    unique_count = len(set(raw_keys))
    duplicate_count = count - unique_count
    positions = [token.position for token in tokens if token.position > 0]
    unique_positions = sorted(set(positions))
    unique_position_count = len(unique_positions)
    kind_ids = [token.kind_id for token in tokens]

    def ratio(*kinds: str) -> float:
        wanted = {_KIND_TO_ID[kind] for kind in kinds}
        return sum(kind_id in wanted for kind_id in kind_ids) / max(count, 1)

    if unique_positions:
        minimum = unique_positions[0]
        maximum = unique_positions[-1]
        span = maximum - minimum
    else:
        minimum = maximum = span = 0
    if len(unique_positions) >= 2:
        minimum_distance = min(
            right - left
            for left, right in zip(unique_positions, unique_positions[1:])
        )
    else:
        minimum_distance = 0
    same_position_ratio = (
        (len(positions) - unique_position_count) / len(positions)
        if positions
        else 0.0
    )
    known_kind_count = len(set(kind_ids))
    has_unknown = any(token.has_unknown for token in tokens)
    return [
        float(count),
        float(unique_count),
        float(duplicate_count),
        float(unique_position_count),
        ratio(MISSENSE),
        ratio(SYNONYMOUS),
        ratio(NONSENSE),
        ratio(FRAMESHIFT),
        ratio(DELETION, INSERTION, DELINS),
        ratio(OTHER),
        float(np.log1p(minimum)),
        float(np.log1p(maximum)),
        float(np.log1p(span)),
        float(np.log1p(minimum_distance)),
        float(same_position_ratio),
        float(count > 1),
        float(known_kind_count > 1),
        float(duplicate_count > 0),
        float(has_unknown),
    ]


@dataclass
class DenseFeatureBundle:
    train_ids: np.ndarray
    test_ids: np.ndarray
    labels: np.ndarray
    train: np.ndarray
    test: np.ndarray
    columns: list[str]
    rollup_train: pd.DataFrame
    rollup_test: pd.DataFrame


def _checked_pair(
    process_dir: Path,
    stem: str,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(process_dir / f"train_{stem}.parquet")
    test = pd.read_parquet(process_dir / f"test_{stem}.parquet")
    for split, frame, ids in (
        ("train", train, train_ids),
        ("test", test, test_ids),
    ):
        if not np.array_equal(frame["ID"].astype(str).to_numpy(), ids):
            raise ValueError(f"{stem}/{split} ID order differs from domain features")
    return train, test


def load_dense_feature_bundle(process_dir: str | Path) -> DenseFeatureBundle:
    """Load domain + rollup14 + parsed19 + burden8 + aa9.

    The remaining two rollup16 columns are fold-dependent and are appended by
    ``train_dl.py`` after fitting :class:`BurdenBinner` on the fold's train rows.
    """
    process_dir = Path(process_dir)
    domain_train = pd.read_parquet(process_dir / "train_domain_features.parquet")
    domain_test = pd.read_parquet(process_dir / "test_domain_features.parquet")
    train_ids = domain_train["ID"].astype(str).to_numpy()
    test_ids = domain_test["ID"].astype(str).to_numpy()
    labels = domain_train["SUBCLASS"].astype(str).to_numpy()

    domain_columns = [
        column for column in domain_train.columns if column.startswith(DOMAIN_PREFIXES)
    ]
    aligned_domain, _, _ = align_domain_columns(domain_test, domain_columns)
    train_parts = [domain_train[domain_columns].to_numpy(np.float32)]
    test_parts = [aligned_domain.to_numpy(np.float32)]
    columns = list(domain_columns)

    rollup_train, rollup_test = _checked_pair(
        process_dir, "sample_mutation_features_rollup", train_ids, test_ids
    )
    blocks = (
        (
            rollup_train,
            rollup_test,
            list(ROBUST_ROLLUP_COLUMNS),
            "rollup16",
        ),
        (*_checked_pair(process_dir, "mutation_parsed_features", train_ids, test_ids),
         list(MUTATION_STRING_PARSED_COLUMNS), "parsed19"),
        (*_checked_pair(process_dir, "additional_burden_features", train_ids, test_ids),
         list(ADDITIONAL_BURDEN_FEATURE_COLUMNS), "burden8"),
        (*_checked_pair(process_dir, "amino_acid_features", train_ids, test_ids),
         list(AMINO_ACID_FEATURE_COLUMNS), "aa9"),
    )
    for train_frame, test_frame, block_columns, prefix in blocks:
        missing_train = set(block_columns) - set(train_frame.columns)
        missing_test = set(block_columns) - set(test_frame.columns)
        if missing_train or missing_test:
            raise ValueError(
                f"{prefix} missing train={sorted(missing_train)[:5]} "
                f"test={sorted(missing_test)[:5]}"
            )
        train_parts.append(train_frame[block_columns].to_numpy(np.float32))
        test_parts.append(test_frame[block_columns].to_numpy(np.float32))
        columns.extend(f"{prefix}__{column}" for column in block_columns)

    return DenseFeatureBundle(
        train_ids=train_ids,
        test_ids=test_ids,
        labels=labels,
        train=np.hstack(train_parts).astype(np.float32, copy=False),
        test=np.hstack(test_parts).astype(np.float32, copy=False),
        columns=columns,
        rollup_train=rollup_train,
        rollup_test=rollup_test,
    )
