"""Train-fold 통계로 만드는 빈도·희귀도 피처.

이 모듈의 피처는 행마다 독립적으로 계산할 수 없다. 반드시 CV의 train fold에서
``fit``하고, 같은 객체로 train/validation/test를 ``transform``해야 한다.
``SUBCLASS``는 어떤 계산에도 사용하지 않는다.

Mutation token은 기존 exact-token 문서와 동일하게
``MUT__{gene}__{mutation}``으로 정의한다. Frequency는 한 샘플 안의 중복을 제거한
document frequency이고, token rarity는 smoothed IDF다.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd

from .parser import split_tokens

TOKEN_PREFIX = "MUT__"
AA_CHANGE_PREFIX = "AA__"
EXACT_MUTATION_PREFIX = "EXACT__"
POSITION_PREFIX = "POS__"
_META_COLUMNS = frozenset({"ID", "SUBCLASS", "fold"})
_POSITION_RE = re.compile(r"\d+")

FREQUENCY_RARITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "rare_mutation_token_count",
    "mean_token_frequency",
    "min_token_frequency",
    "mean_token_rarity",
    "max_token_rarity",
    "mean_mutated_gene_frequency",
    "min_mutated_gene_frequency",
    "mean_gene_rarity",
    "max_gene_rarity",
    "unseen_token_count",
    "unseen_mutated_gene_count",
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


def _resolve_gene_columns(
    frame: pd.DataFrame,
    gene_columns: Sequence[str] | None,
) -> list[str]:
    genes = (
        [column for column in frame.columns if column not in _META_COLUMNS]
        if gene_columns is None
        else list(gene_columns)
    )
    if not genes:
        raise ValueError("No gene columns found")
    if len(genes) != len(set(genes)):
        raise ValueError("gene_columns contains duplicates")
    missing = sorted(set(genes).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing gene columns: {missing[:10]}")
    return genes


def _exact_token(gene: str, mutation: str) -> str:
    return f"{TOKEN_PREFIX}{gene}__{mutation}"


def _aa_change_token(mutation: str) -> str:
    return f"{AA_CHANGE_PREFIX}{mutation}"


def _exact_cell_mutation(gene: str, mutations: Sequence[str]) -> str:
    """셀 안 token 순서만 정규화하고 중복 개수는 보존한 exact signature."""
    return f"{EXACT_MUTATION_PREFIX}{gene}__{'|'.join(sorted(mutations))}"


def _position_token(mutation: str) -> str | None:
    """AA change의 숫자 position signature. 범위는 펼치지 않고 모두 보존한다."""
    positions = _POSITION_RE.findall(mutation)
    if not positions:
        return None
    return f"{POSITION_PREFIX}{'_'.join(positions)}"


def _iter_sample_entities(
    frame: pd.DataFrame,
    gene_columns: Sequence[str],
) -> Iterator[
    tuple[list[str], set[str], set[str], set[str], set[str]]
]:
    """샘플별 gene/exact-cell/AA/gene-AA/position key를 반환한다."""
    values = frame[list(gene_columns)].to_numpy(dtype=object)
    for row in values:
        mutated_genes: list[str] = []
        exact_mutations: set[str] = set()
        aa_changes: set[str] = set()
        gene_aa_changes: set[str] = set()
        positions: set[str] = set()
        for gene, value in zip(gene_columns, row):
            mutations = split_tokens(value)
            if not mutations:
                continue
            mutated_genes.append(gene)
            exact_mutations.add(_exact_cell_mutation(gene, mutations))
            for mutation in mutations:
                aa_changes.add(_aa_change_token(mutation))
                gene_aa_changes.add(_exact_token(gene, mutation))
                position = _position_token(mutation)
                if position is not None:
                    positions.add(position)
        yield (
            mutated_genes,
            exact_mutations,
            aa_changes,
            gene_aa_changes,
            positions,
        )


class TrainFrequencyFeatures:
    """Train-fold 전용 frequency/IDF/rarity transformer.

    저장하는 train 통계

    - ``gene_frequency_``: 유전자별 변이 sample frequency
    - ``exact_mutation_frequency_``: gene + 셀 전체 mutation 조합 frequency
    - ``aa_change_frequency_``: gene 비의존 AA change frequency
    - ``gene_aa_change_frequency_``: gene + AA change frequency
    - ``position_frequency_``: gene 비의존 position signature frequency
    - ``token_frequency_``: ``gene_aa_change_frequency_``의 호환 alias
    - ``token_idf_``: ``log((N + 1) / (df + 1)) + 1``
    - ``gene_rarity_``: 클래스 비의존적 ``-log((count + 1) / (N + 1))``

    ``transform``은 위 lookup을 샘플별 고정 길이 피처로 집계한다. Validation과
    test 자체의 분포는 절대 집계하지 않는다.
    """

    def __init__(self, *, rare_df_threshold: int = 2) -> None:
        if rare_df_threshold < 0:
            raise ValueError("rare_df_threshold must be >= 0")
        self.rare_df_threshold = int(rare_df_threshold)

        self.n_train_samples_: int | None = None
        self.gene_columns_: tuple[str, ...] | None = None
        self.gene_mutation_count_: dict[str, int] | None = None
        self.exact_mutation_document_count_: dict[str, int] | None = None
        self.aa_change_document_count_: dict[str, int] | None = None
        self.gene_aa_change_document_count_: dict[str, int] | None = None
        self.position_document_count_: dict[str, int] | None = None
        self.token_document_count_: dict[str, int] | None = None
        self.gene_frequency_: dict[str, float] | None = None
        self.exact_mutation_frequency_: dict[str, float] | None = None
        self.aa_change_frequency_: dict[str, float] | None = None
        self.gene_aa_change_frequency_: dict[str, float] | None = None
        self.position_frequency_: dict[str, float] | None = None
        self.token_frequency_: dict[str, float] | None = None
        self.token_idf_: dict[str, float] | None = None
        self.gene_rarity_: dict[str, float] | None = None
        self.unseen_token_rarity_: float | None = None

    def fit(
        self,
        train_frame: pd.DataFrame,
        *,
        gene_columns: Sequence[str] | None = None,
    ) -> "TrainFrequencyFeatures":
        """Train 또는 train fold만 받아 lookup 통계를 학습한다."""
        if train_frame.empty:
            raise ValueError("Cannot fit frequency features on an empty frame")
        genes = _resolve_gene_columns(train_frame, gene_columns)
        n_samples = len(train_frame)

        gene_counts: Counter[str] = Counter()
        exact_mutation_counts: Counter[str] = Counter()
        aa_change_counts: Counter[str] = Counter()
        gene_aa_change_counts: Counter[str] = Counter()
        position_counts: Counter[str] = Counter()
        for (
            mutated_genes,
            exact_mutations,
            aa_changes,
            gene_aa_changes,
            positions,
        ) in _iter_sample_entities(train_frame, genes):
            gene_counts.update(mutated_genes)
            exact_mutation_counts.update(exact_mutations)
            aa_change_counts.update(aa_changes)
            gene_aa_change_counts.update(gene_aa_changes)
            position_counts.update(positions)

        self.n_train_samples_ = n_samples
        self.gene_columns_ = tuple(genes)
        self.gene_mutation_count_ = {
            gene: int(gene_counts.get(gene, 0)) for gene in genes
        }
        self.exact_mutation_document_count_ = {
            key: int(count) for key, count in exact_mutation_counts.items()
        }
        self.aa_change_document_count_ = {
            key: int(count) for key, count in aa_change_counts.items()
        }
        self.gene_aa_change_document_count_ = {
            key: int(count) for key, count in gene_aa_change_counts.items()
        }
        self.position_document_count_ = {
            key: int(count) for key, count in position_counts.items()
        }
        # 기존 공개 API는 gene + AA change 의미였으므로 같은 객체를 alias한다.
        self.token_document_count_ = self.gene_aa_change_document_count_
        self.gene_frequency_ = {
            gene: count / n_samples
            for gene, count in self.gene_mutation_count_.items()
        }
        self.exact_mutation_frequency_ = {
            key: count / n_samples
            for key, count in self.exact_mutation_document_count_.items()
        }
        self.aa_change_frequency_ = {
            key: count / n_samples
            for key, count in self.aa_change_document_count_.items()
        }
        self.gene_aa_change_frequency_ = {
            key: count / n_samples
            for key, count in self.gene_aa_change_document_count_.items()
        }
        self.position_frequency_ = {
            key: count / n_samples
            for key, count in self.position_document_count_.items()
        }
        self.token_frequency_ = self.gene_aa_change_frequency_
        self.token_idf_ = {
            token: float(np.log((n_samples + 1) / (count + 1)) + 1.0)
            for token, count in self.token_document_count_.items()
        }
        self.gene_rarity_ = {
            gene: float(-np.log((count + 1) / (n_samples + 1)))
            for gene, count in self.gene_mutation_count_.items()
        }
        # Validation/test에만 나온 token은 train df=0으로 취급한다.
        self.unseen_token_rarity_ = float(np.log(n_samples + 1) + 1.0)
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        gene_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """학습해 둔 train-fold 통계만 lookup해 샘플 피처를 만든다."""
        self._check_fitted()
        genes = self._validate_transform_columns(frame, gene_columns)
        n_rows = len(frame)

        rare_counts = np.zeros(n_rows, dtype=np.int32)
        mean_token_frequency = np.zeros(n_rows, dtype=np.float32)
        min_token_frequency = np.zeros(n_rows, dtype=np.float32)
        mean_token_rarity = np.zeros(n_rows, dtype=np.float32)
        max_token_rarity = np.zeros(n_rows, dtype=np.float32)
        mean_gene_frequency = np.zeros(n_rows, dtype=np.float32)
        min_gene_frequency = np.zeros(n_rows, dtype=np.float32)
        mean_gene_rarity = np.zeros(n_rows, dtype=np.float32)
        max_gene_rarity = np.zeros(n_rows, dtype=np.float32)
        unseen_token_counts = np.zeros(n_rows, dtype=np.int32)
        unseen_gene_counts = np.zeros(n_rows, dtype=np.int32)
        frequency_aggregates = np.zeros((n_rows, 10), dtype=np.float32)

        for row_idx, (
            mutated_genes,
            exact_mutations,
            aa_changes,
            gene_aa_changes,
            positions,
        ) in enumerate(_iter_sample_entities(frame, genes)):
            if gene_aa_changes:
                document_counts = np.fromiter(
                    (
                        self.token_document_count_.get(token, 0)
                        for token in gene_aa_changes
                    ),
                    dtype=np.int64,
                )
                frequencies = document_counts.astype(np.float64) / self.n_train_samples_
                rarities = np.fromiter(
                    (
                        self.token_idf_.get(token, self.unseen_token_rarity_)
                        for token in gene_aa_changes
                    ),
                    dtype=np.float64,
                )
                rare_counts[row_idx] = int(
                    (document_counts <= self.rare_df_threshold).sum()
                )
                unseen_token_counts[row_idx] = int((document_counts == 0).sum())
                mean_token_frequency[row_idx] = frequencies.mean()
                min_token_frequency[row_idx] = frequencies.min()
                mean_token_rarity[row_idx] = rarities.mean()
                max_token_rarity[row_idx] = rarities.max()
                frequency_aggregates[row_idx, 6] = frequencies.max()

            for offset, keys, lookup in (
                (0, exact_mutations, self.exact_mutation_frequency_),
                (3, aa_changes, self.aa_change_frequency_),
                (7, positions, self.position_frequency_),
            ):
                if not keys:
                    continue
                values = np.fromiter(
                    (lookup.get(key, 0.0) for key in keys),
                    dtype=np.float64,
                )
                frequency_aggregates[row_idx, offset] = values.mean()
                frequency_aggregates[row_idx, offset + 1] = values.min()
                frequency_aggregates[row_idx, offset + 2] = values.max()

            if mutated_genes:
                gene_frequencies = np.fromiter(
                    (self.gene_frequency_[gene] for gene in mutated_genes),
                    dtype=np.float64,
                )
                gene_rarities = np.fromiter(
                    (self.gene_rarity_[gene] for gene in mutated_genes),
                    dtype=np.float64,
                )
                gene_counts = np.fromiter(
                    (self.gene_mutation_count_[gene] for gene in mutated_genes),
                    dtype=np.int64,
                )
                mean_gene_frequency[row_idx] = gene_frequencies.mean()
                min_gene_frequency[row_idx] = gene_frequencies.min()
                mean_gene_rarity[row_idx] = gene_rarities.mean()
                max_gene_rarity[row_idx] = gene_rarities.max()
                unseen_gene_counts[row_idx] = int((gene_counts == 0).sum())

        return pd.DataFrame(
            {
                "rare_mutation_token_count": rare_counts,
                "mean_token_frequency": mean_token_frequency,
                "min_token_frequency": min_token_frequency,
                "mean_token_rarity": mean_token_rarity,
                "max_token_rarity": max_token_rarity,
                "mean_mutated_gene_frequency": mean_gene_frequency,
                "min_mutated_gene_frequency": min_gene_frequency,
                "mean_gene_rarity": mean_gene_rarity,
                "max_gene_rarity": max_gene_rarity,
                "unseen_token_count": unseen_token_counts,
                "unseen_mutated_gene_count": unseen_gene_counts,
                "mean_exact_mutation_frequency": frequency_aggregates[:, 0],
                "min_exact_mutation_frequency": frequency_aggregates[:, 1],
                "max_exact_mutation_frequency": frequency_aggregates[:, 2],
                "mean_aa_change_frequency": frequency_aggregates[:, 3],
                "min_aa_change_frequency": frequency_aggregates[:, 4],
                "max_aa_change_frequency": frequency_aggregates[:, 5],
                "max_gene_aa_change_frequency": frequency_aggregates[:, 6],
                "mean_position_frequency": frequency_aggregates[:, 7],
                "min_position_frequency": frequency_aggregates[:, 8],
                "max_position_frequency": frequency_aggregates[:, 9],
            },
            columns=list(FREQUENCY_RARITY_FEATURE_COLUMNS),
            index=frame.index,
        )

    def fit_transform(
        self,
        train_frame: pd.DataFrame,
        *,
        gene_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """``fit(train_frame).transform(train_frame)`` 편의 메서드."""
        return self.fit(
            train_frame, gene_columns=gene_columns
        ).transform(
            train_frame, gene_columns=gene_columns
        )

    def get_gene_statistics(self) -> pd.DataFrame:
        """팀 실험 기록용 유전자별 frequency/rarity 표."""
        self._check_fitted()
        return pd.DataFrame(
            {
                "gene": self.gene_columns_,
                "mutation_sample_count": [
                    self.gene_mutation_count_[gene] for gene in self.gene_columns_
                ],
                "mutation_frequency": [
                    self.gene_frequency_[gene] for gene in self.gene_columns_
                ],
                "gene_rarity": [
                    self.gene_rarity_[gene] for gene in self.gene_columns_
                ],
            }
        )

    def get_token_statistics(self) -> pd.DataFrame:
        """팀 실험 기록용 mutation token별 frequency/IDF 표."""
        self._check_fitted()
        tokens = sorted(self.token_document_count_)
        return pd.DataFrame(
            {
                "mutation_token": tokens,
                "document_count": [
                    self.token_document_count_[token] for token in tokens
                ],
                "frequency": [self.token_frequency_[token] for token in tokens],
                "idf": [self.token_idf_[token] for token in tokens],
            }
        )

    def get_frequency_statistics(self) -> pd.DataFrame:
        """새 frequency family 네 종류를 long 형식으로 반환한다."""
        self._check_fitted()
        frames: list[pd.DataFrame] = []
        for family, counts, frequencies in (
            (
                "exact_mutation",
                self.exact_mutation_document_count_,
                self.exact_mutation_frequency_,
            ),
            ("aa_change", self.aa_change_document_count_, self.aa_change_frequency_),
            (
                "gene_aa_change",
                self.gene_aa_change_document_count_,
                self.gene_aa_change_frequency_,
            ),
            ("position", self.position_document_count_, self.position_frequency_),
        ):
            keys = sorted(counts)
            frames.append(
                pd.DataFrame(
                    {
                        "family": family,
                        "key": keys,
                        "document_count": [counts[key] for key in keys],
                        "frequency": [frequencies[key] for key in keys],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def get_feature_names_out(self) -> np.ndarray:
        """scikit-learn 스타일의 고정 출력 컬럼 목록."""
        return np.asarray(FREQUENCY_RARITY_FEATURE_COLUMNS, dtype=object)

    def _validate_transform_columns(
        self,
        frame: pd.DataFrame,
        gene_columns: Sequence[str] | None,
    ) -> list[str]:
        genes = (
            list(self.gene_columns_)
            if gene_columns is None
            else _resolve_gene_columns(frame, gene_columns)
        )
        missing = sorted(set(genes).difference(frame.columns))
        if missing:
            raise ValueError(f"Missing gene columns: {missing[:10]}")
        if tuple(genes) != self.gene_columns_:
            raise ValueError(
                "gene_columns must match fit() columns and order exactly"
            )
        return genes

    def _check_fitted(self) -> None:
        if self.n_train_samples_ is None:
            raise RuntimeError(
                "TrainFrequencyFeatures.fit()을 train 또는 train fold에서 먼저 "
                "호출해야 합니다."
            )
