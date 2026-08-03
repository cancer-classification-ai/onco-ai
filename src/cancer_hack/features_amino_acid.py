"""아미노산 치환의 물리화학적 변화량을 계산하는 dense 피처.

이 모듈은 train 빈도나 정답 레이블을 사용하지 않는다. 20개 표준 아미노산의
고정된 네 가지 속성을 min-max 정규화한 뒤, 속성 차이의 가중 유클리디안 거리를
계산한다. 따라서 결과는 실제 단백질 구조 손상도나 임상적 병원성이 아니라
치환 전후의 물리화학적 변화 크기를 나타내는 휴리스틱이다.

동일 mutation token이 한 유전자 셀 안에서 반복되면 기본적으로 한 번만 반영한다.
서로 다른 위치나 유전자에서 발생한 같은 AA 전이는 별개의 mutation event로 센다.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .parser import MISSENSE, SYNONYMOUS, classify_token, split_tokens

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PROPERTY_NAMES: tuple[str, ...] = (
    "polarity",
    "hydrophobicity",
    "charge",
    "molecular_weight",
)

# 고정된 교과서적 물성값이다. 런타임에 외부 데이터나 API를 조회하지 않는다.
# Histidine은 생리적 pH에서 부분적으로 양전하를 띠는 성질을 0.1로 근사한다.
AMINO_ACID_PROPERTIES: dict[str, dict[str, float]] = {
    "A": {"polarity": 8.1, "hydrophobicity": 1.8, "charge": 0.0, "molecular_weight": 89.09},
    "R": {"polarity": 10.5, "hydrophobicity": -4.5, "charge": 1.0, "molecular_weight": 174.20},
    "N": {"polarity": 11.6, "hydrophobicity": -3.5, "charge": 0.0, "molecular_weight": 132.12},
    "D": {"polarity": 13.0, "hydrophobicity": -3.5, "charge": -1.0, "molecular_weight": 133.10},
    "C": {"polarity": 5.5, "hydrophobicity": 2.5, "charge": 0.0, "molecular_weight": 121.16},
    "Q": {"polarity": 10.5, "hydrophobicity": -3.5, "charge": 0.0, "molecular_weight": 146.15},
    "E": {"polarity": 12.3, "hydrophobicity": -3.5, "charge": -1.0, "molecular_weight": 147.13},
    "G": {"polarity": 9.0, "hydrophobicity": -0.4, "charge": 0.0, "molecular_weight": 75.07},
    "H": {"polarity": 10.4, "hydrophobicity": -3.2, "charge": 0.1, "molecular_weight": 155.16},
    "I": {"polarity": 5.2, "hydrophobicity": 4.5, "charge": 0.0, "molecular_weight": 131.17},
    "L": {"polarity": 4.9, "hydrophobicity": 3.8, "charge": 0.0, "molecular_weight": 131.17},
    "K": {"polarity": 11.3, "hydrophobicity": -3.9, "charge": 1.0, "molecular_weight": 146.19},
    "M": {"polarity": 5.7, "hydrophobicity": 1.9, "charge": 0.0, "molecular_weight": 149.21},
    "F": {"polarity": 5.2, "hydrophobicity": 2.8, "charge": 0.0, "molecular_weight": 165.19},
    "P": {"polarity": 8.0, "hydrophobicity": -1.6, "charge": 0.0, "molecular_weight": 115.13},
    "S": {"polarity": 9.2, "hydrophobicity": -0.8, "charge": 0.0, "molecular_weight": 105.09},
    "T": {"polarity": 8.6, "hydrophobicity": -0.7, "charge": 0.0, "molecular_weight": 119.12},
    "W": {"polarity": 5.4, "hydrophobicity": -0.9, "charge": 0.0, "molecular_weight": 204.23},
    "Y": {"polarity": 6.2, "hydrophobicity": -1.3, "charge": 0.0, "molecular_weight": 181.19},
    "V": {"polarity": 5.9, "hydrophobicity": 4.2, "charge": 0.0, "molecular_weight": 117.15},
}

DEFAULT_PROPERTY_WEIGHTS: dict[str, float] = {
    "polarity": 1.0,
    "hydrophobicity": 1.0,
    "charge": 2.0,
    "molecular_weight": 1.0,
}

AMINO_ACID_FEATURE_COLUMNS: tuple[str, ...] = (
    "aa_substitution_penalty_sum",
    "aa_substitution_penalty_mean",
    "aa_substitution_penalty_max",
    "scored_missense_count",
    "radical_substitution_count",
    "conservative_substitution_count",
    "has_radical_substitution",
    "aa_penalty_missing_count",
    "aa_penalty_data_missing",
)

_META_COLUMNS = frozenset({"ID", "SUBCLASS", "fold"})
_SUBSTITUTION_RE = re.compile(r"^([A-Z*])\d+([A-Z*])$")


def _normalize_properties() -> dict[str, dict[str, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for property_name in PROPERTY_NAMES:
        values = [
            properties[property_name]
            for properties in AMINO_ACID_PROPERTIES.values()
        ]
        ranges[property_name] = (min(values), max(values))

    normalized: dict[str, dict[str, float]] = {}
    for amino_acid, properties in AMINO_ACID_PROPERTIES.items():
        normalized[amino_acid] = {}
        for property_name in PROPERTY_NAMES:
            minimum, maximum = ranges[property_name]
            normalized[amino_acid][property_name] = (
                properties[property_name] - minimum
            ) / (maximum - minimum)
    return normalized


NORMALIZED_AMINO_ACID_PROPERTIES = _normalize_properties()


def _resolve_weights(
    property_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    weights = (
        dict(DEFAULT_PROPERTY_WEIGHTS)
        if property_weights is None
        else dict(property_weights)
    )
    if set(weights) != set(PROPERTY_NAMES):
        raise ValueError(
            f"property_weights keys must be exactly {list(PROPERTY_NAMES)}"
        )
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("property weights must be finite and >= 0")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one property weight must be > 0")
    return {name: float(weights[name]) for name in PROPERTY_NAMES}


def physicochemical_substitution_penalty(
    ref_amino_acid: str,
    alt_amino_acid: str,
    *,
    property_weights: Mapping[str, float] | None = None,
) -> float:
    """두 표준 아미노산 사이의 정규화된 물성 거리.

    동일 아미노산은 0, 표준 20종 밖의 문자나 잘못된 입력은 ``NaN``을 반환한다.
    전하 역전을 더 크게 반영하기 위해 기본 전하 가중치는 2.0이다.
    """
    ref = str(ref_amino_acid).upper()
    alt = str(alt_amino_acid).upper()
    if ref not in AMINO_ACID_PROPERTIES or alt not in AMINO_ACID_PROPERTIES:
        return float("nan")
    if ref == alt:
        return 0.0

    weights = _resolve_weights(property_weights)
    squared_distance = 0.0
    for property_name in PROPERTY_NAMES:
        difference = (
            NORMALIZED_AMINO_ACID_PROPERTIES[ref][property_name]
            - NORMALIZED_AMINO_ACID_PROPERTIES[alt][property_name]
        )
        squared_distance += weights[property_name] * difference**2
    return float(math.sqrt(squared_distance))


def mutation_substitution_penalty(
    mutation_token: str,
    *,
    property_weights: Mapping[str, float] | None = None,
) -> float:
    """단일 치환 token의 penalty. 치환이 아니거나 파싱 불가하면 ``NaN``."""
    token = str(mutation_token).strip()
    kind = classify_token(token)
    if kind not in {MISSENSE, SYNONYMOUS}:
        return float("nan")
    matched = _SUBSTITUTION_RE.fullmatch(token)
    if matched is None:
        return float("nan")
    return physicochemical_substitution_penalty(
        matched.group(1),
        matched.group(2),
        property_weights=property_weights,
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


def _cell_missense_penalties(
    value: object,
    *,
    property_weights: Mapping[str, float],
    unique_tokens_per_gene: bool,
) -> tuple[list[float], int]:
    tokens = split_tokens(value)
    if unique_tokens_per_gene:
        tokens = list(dict.fromkeys(tokens))

    penalties: list[float] = []
    missing_count = 0
    for token in tokens:
        if classify_token(token) != MISSENSE:
            continue
        penalty = mutation_substitution_penalty(
            token,
            property_weights=property_weights,
        )
        if np.isnan(penalty):
            missing_count += 1
        else:
            penalties.append(penalty)
    return penalties, missing_count


def make_amino_acid_features(
    frame: pd.DataFrame,
    *,
    gene_columns: Sequence[str] | None = None,
    property_weights: Mapping[str, float] | None = None,
    conservative_max: float = 0.5,
    radical_min: float = 1.0,
    unique_tokens_per_gene: bool = True,
) -> pd.DataFrame:
    """샘플별 물리화학적 치환 penalty를 9개 dense 컬럼으로 집계한다.

    이 함수는 행마다 독립적이며 ``fit``이 필요 없다. missense가 없는 샘플의
    집계값은 0이다. missense로 분류됐지만 표준 AA로 점수를 계산할 수 없는 token은
    평균에 0으로 섞지 않고 missing count/flag에 기록한다.
    """
    if not np.isfinite(conservative_max) or conservative_max < 0:
        raise ValueError("conservative_max must be finite and >= 0")
    if not np.isfinite(radical_min) or radical_min <= conservative_max:
        raise ValueError("radical_min must be finite and > conservative_max")

    genes = _resolve_gene_columns(frame, gene_columns)
    weights = _resolve_weights(property_weights)
    output = np.zeros(
        (len(frame), len(AMINO_ACID_FEATURE_COLUMNS)),
        dtype=np.float32,
    )

    values = frame[genes].to_numpy(dtype=object)
    for row_index, row in enumerate(values):
        penalties: list[float] = []
        missing_count = 0
        for value in row:
            cell_penalties, cell_missing = _cell_missense_penalties(
                value,
                property_weights=weights,
                unique_tokens_per_gene=unique_tokens_per_gene,
            )
            penalties.extend(cell_penalties)
            missing_count += cell_missing

        if penalties:
            score_array = np.asarray(penalties, dtype=np.float64)
            radical_count = int(np.count_nonzero(score_array >= radical_min))
            conservative_count = int(
                np.count_nonzero(score_array <= conservative_max)
            )
            output[row_index, :7] = (
                score_array.sum(),
                score_array.mean(),
                score_array.max(),
                len(score_array),
                radical_count,
                conservative_count,
                int(radical_count > 0),
            )
        output[row_index, 7:] = (missing_count, int(missing_count > 0))

    return pd.DataFrame(
        output,
        columns=list(AMINO_ACID_FEATURE_COLUMNS),
        index=frame.index,
    )


def make_gene_amino_acid_penalty_features(
    frame: pd.DataFrame,
    *,
    gene_columns: Sequence[str],
    property_weights: Mapping[str, float] | None = None,
    unique_tokens_per_gene: bool = True,
) -> pd.DataFrame:
    """명시적으로 선택한 유전자별 최대 치환 penalty를 wide matrix로 만든다.

    전체 6,199개 유전자를 자동으로 펼치지 않는다. 모델에서 사용할 패널을 호출부가
    명시해야 컬럼 계약과 메모리 사용량을 팀원·Colab·Kaggle 사이에서 고정할 수 있다.
    """
    genes = _resolve_gene_columns(frame, gene_columns)
    weights = _resolve_weights(property_weights)
    output = np.zeros((len(frame), len(genes)), dtype=np.float32)

    values = frame[genes].to_numpy(dtype=object)
    for row_index, row in enumerate(values):
        for gene_index, value in enumerate(row):
            penalties, _ = _cell_missense_penalties(
                value,
                property_weights=weights,
                unique_tokens_per_gene=unique_tokens_per_gene,
            )
            if penalties:
                output[row_index, gene_index] = max(penalties)

    return pd.DataFrame(
        output,
        columns=[f"gene_aa_penalty__{gene}" for gene in genes],
        index=frame.index,
    )
