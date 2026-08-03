#!/usr/bin/env python
"""팀 공용 피처 정의.

CLI 는 `scripts/make_features.py` 에 있고 이 모듈은 정의만 담는다. 피처 함수는
이름으로 구분해 두어서 새 빌더를 넣어도 기존 동작이 바뀌지 않는다.
"""

from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .parser import (
    _check_columns,
    _classify_token,
    _INDEL_RE,
    _MUTATION_EMPTY,
    _parse_mutation_tokens,
    _row_token_class_counts,
    _SYNONYMOUS_RE,
    CELL_FEATURE_COLUMNS,
    CellMutation,
    parse_cell,
    split_tokens,
)

# Synonymous mutation 패턴 확인용 정규식
_SYNONYMOUS_RE = re.compile(r"^([A-Z])\d+\1$")

# Mutation encoding 대상에서 제외할 메타 정보 컬럼
_GENE_EXCLUDE = {"ID", "SUBCLASS"}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# parser.py의 _MUTATION_EMPTY와 동일한 값 — 중복 정의 대신 참조
EXACT_MUTATION_EMPTY_VALUES = _MUTATION_EMPTY
EXACT_MUTATION_NO_MUTATION_TOKEN = "SAMPLE__NO_MUTATION"


# 단일 mutation token을 WT / Synonymous / Functional 3단계 값으로 변환
def _encode_single(token: str) -> int:
    if token == "WT":
        return 0
    if _SYNONYMOUS_RE.match(token):
        return 1
    return 2


# 하나의 mutation cell 값을 분석하여 가장 높은 변이 영향도로 인코딩
def encode_mutation(value: str) -> int:
    tokens = value.split()
    if len(tokens) == 1:
        return _encode_single(tokens[0])
    return max(_encode_single(t) for t in tokens)


# 전체 gene column에 mutation encoding을 적용하여 전처리된 데이터셋 생성
def make_mutation_encoding(df: pd.DataFrame) -> pd.DataFrame:
    gene_cols = [c for c in df.columns if c not in _GENE_EXCLUDE]
    meta_cols = [c for c in df.columns if c in _GENE_EXCLUDE]
    encoded = df[gene_cols].map(encode_mutation).astype("int8")
    return pd.concat([df[meta_cols], encoded], axis=1)


# ---------------------------------------------------------------- 복합 변이 피처
#
# 한 셀에 변이가 여러 개 들어가는 경우(`'Q369* I368N'`)를 버리지 않고 유형별
# 존재 여부·개수로 펼친다. 토큰 분류 규칙은 `cancer_hack.parser` 에 있다.
#
# 셀 단위(`make_cell_mutation_features`)와 샘플 단위(`make_sample_mutation_features`)
# 두 층을 나눠 둔 이유: 셀 단위 피처를 유전자 4,383개에 그대로 붙이면 컬럼이
# 4,383 x 22 = 96,426개가 되어 그대로는 모델에 못 넣는다. 셀 단위는 특정 유전자만
# 골라 쓸 때(드라이버 패널 등)를 위한 것이고, 전체 유전자를 쓸 때는 샘플 단위로
# 접어서 넣는다.

# 셀 단위 유형 개수 -> 샘플 단위 변수명. 팀 파생변수 목록의 이름을 그대로 쓴다.
_EVENT_COUNT_MAP: dict[str, str] = {
    "synonymous_count": "synonymous_event_count",
    "missense_count": "missense_event_count",
    "nonsense_count": "nonsense_event_count",
    "frameshift_count": "frameshift_event_count",
    "complex_count": "complex_event_count",
}

# 1차 전략표(셀 단위) 이름을 샘플 단위로 올릴 때 쓰는 목록.
# 팀 파생변수 목록에는 없는 이름이라 기본 출력에서는 빠지고,
# `include_cell_rollup=True` 일 때만 붙는다.
_ROLLUP_SUM_COLUMNS: tuple[str, ...] = (
    "unique_mutation_token_count",
    "indel_count",
    "mnv_count",
    "unique_missense_count",
    "unique_synonymous_count",
    "unique_nonsense_count",
    "unique_frameshift_count",
    "unique_indel_count",
    "unique_mnv_count",
    "other_count",
)
_ROLLUP_ANY_COLUMNS: tuple[str, ...] = (
    "has_missense",
    "has_synonymous",
    "has_nonsense",
    "has_frameshift",
    "has_indel",
    "has_mnv",
    "has_duplicate_token",
)


def _parse_cache() -> dict[str, CellMutation]:
    """셀 값 문자열 -> 파싱 결과 캐시.

    train 은 셀 27M개 중 non-WT 가 21.9만개뿐이고 그중 상당수가 같은 문자열이다.
    매번 파싱하지 않고 문자열 단위로 재사용한다.
    """
    return {}


def _parse_cached(value: object, cache: dict[str, CellMutation]) -> CellMutation:
    key = "" if value is None else str(value)
    hit = cache.get(key)
    if hit is None:
        hit = parse_cell(key)
        cache[key] = hit
    return hit


def make_cell_mutation_features(values: pd.Series) -> pd.DataFrame:
    """변이 문자열 Series 하나를 셀 단위 복합 변이 피처 DataFrame으로 편다.

    유전자 컬럼 하나를 그대로 넘기면 된다. 반환 컬럼은
    `cancer_hack.parser.CELL_FEATURE_COLUMNS` 순서를 따르고 인덱스는 입력을 유지한다.

    >>> int(make_cell_mutation_features(pd.Series(["S622S G827R"]))["has_synonymous"][0])
    1
    """
    cache = _parse_cache()
    rows = [_parse_cached(v, cache).as_dict() for v in values]
    return pd.DataFrame(rows, columns=list(CELL_FEATURE_COLUMNS), index=values.index)


SAMPLE_FEATURE_COLUMNS: tuple[str, ...] = (
    # 기본 파생변수
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
    # 비율·변환 변수 (fold 의존 2개는 BurdenBinner 담당)
    "log1p_mutated_gene_count",
    "log1p_mutation_event_count",
    "functional_ratio",
    "synonymous_ratio",
    "missense_ratio",
    "nonsense_ratio",
    "frameshift_ratio",
    "multihit_gene_ratio",
    # del 관련 변수 (비추천인 deletion_ratio 제외)
    "explicit_deletion_event_count",
    "explicit_deletion_gene_count",
    "has_explicit_deletion",
    "inframe_deletion_count",
    "indel_or_frameshift_count",
    "loss_of_function_count",
)


def _resolve_gene_columns(
    df: pd.DataFrame,
    gene_columns: list[str] | None,
) -> list[str]:
    if gene_columns is None:
        gene_columns = [c for c in df.columns if c not in _GENE_EXCLUDE]
    missing = sorted(set(gene_columns).difference(df.columns))
    if missing:
        raise ValueError(f"Missing gene columns: {missing[:10]}")
    return gene_columns


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """분모가 0이면 0을 돌려준다. 변이가 하나도 없는 샘플이 train 94명 있다."""
    return (numerator / denominator.where(denominator > 0, 1)).astype("float32")


def make_sample_mutation_features(
    df: pd.DataFrame,
    *,
    gene_columns: list[str] | None = None,
    include_cell_rollup: bool = False,
) -> pd.DataFrame:
    """샘플(행) 하나를 유전자 전체에 걸친 복합 변이 요약 피처로 접는다.

    컬럼 이름과 정의는 팀 파생변수 목록을 그대로 따른다. 목록에 없는 변수는
    만들지 않는다. fold 안에서 계산해야 하는 `hypermutated_flag`,
    `burden_quantile_bin` 은 여기 없고 `BurdenBinner` 가 담당한다.

    반환 컬럼(25개)

      기본     mutated_gene_count · mutation_event_count · synonymous_event_count
               functional_event_count · missense_event_count · nonsense_event_count
               frameshift_event_count · complex_event_count · multihit_gene_count
               max_events_per_gene · no_mutation_flag
      비율     log1p_mutated_gene_count · log1p_mutation_event_count
               functional_ratio · synonymous_ratio · missense_ratio · nonsense_ratio
               frameshift_ratio · multihit_gene_ratio
      del      explicit_deletion_event_count · explicit_deletion_gene_count
               has_explicit_deletion · inframe_deletion_count
               indel_or_frameshift_count · loss_of_function_count

    `include_cell_rollup=True` 를 주면 1차 전략표(셀 단위) 이름을 샘플 단위로 올린
    17개 컬럼이 뒤에 붙는다 — `has_missense` 계열 플래그, `indel_count`,
    `mnv_count`, `unique_*_count`, `has_duplicate_token`, `other_count`. 팀 목록에
    없는 이름이라 기본값은 False 다. 목록 계약을 깨지 않으면서 1차 표 정보도
    꺼내 쓸 수 있게 스위치로 뒀다.

    `mutation_event_count` 가 1차 표의 `mutation_token_count` 와 같은 값이고,
    `complex_event_count` 가 `mnv_count` 와 같은 값이다. 이름만 팀 목록을 따른다.

    분모 규칙이 두 가지로 갈린다. `functional_ratio` 와 `synonymous_ratio` 는
    전체 변이 사건 수로 나누고, `missense_ratio`·`nonsense_ratio`·`frameshift_ratio`
    는 기능성 변이 수로 나눈다. 목록의 정의 그대로다.

    train/test 사이에서 스케일이 흔들리는 컬럼이 있다. test 는 같은 변이를 여러
    전사체 좌표로 중복 기재하므로 `mutation_event_count` 계열이 부풀고
    (변이 유전자 수 train 35.3 / test 78.1), in-frame indel 은 train 에 3건뿐이라
    `explicit_deletion_*` 은 train 에서 사실상 상수다. CV 로 확인하고 넣는다.

    통계를 train 에서 fit 해 test 에 적용하는 부분이 없다 — 행마다 독립적으로
    계산하므로 대회 규정상 문제가 없다.
    """
    gene_columns = _resolve_gene_columns(df, gene_columns)
    cache = _parse_cache()

    values = df[gene_columns].to_numpy(dtype=object)
    n_rows = len(df)

    event_names = list(_EVENT_COUNT_MAP.values())
    event_sums = np.zeros((n_rows, len(event_names)), dtype=np.int32)
    source_keys = list(_EVENT_COUNT_MAP)

    mutated_gene_count = np.zeros(n_rows, dtype=np.int32)
    mutation_event_count = np.zeros(n_rows, dtype=np.int32)
    functional_event_count = np.zeros(n_rows, dtype=np.int32)
    multihit_gene_count = np.zeros(n_rows, dtype=np.int32)
    max_events_per_gene = np.zeros(n_rows, dtype=np.int32)
    explicit_deletion_event_count = np.zeros(n_rows, dtype=np.int32)
    explicit_deletion_gene_count = np.zeros(n_rows, dtype=np.int32)

    rollup_sums = np.zeros((n_rows, len(_ROLLUP_SUM_COLUMNS)), dtype=np.int32)
    rollup_anys = np.zeros((n_rows, len(_ROLLUP_ANY_COLUMNS)), dtype=np.int8)

    for i in range(n_rows):
        for value in values[i]:
            # WT 가 전체의 99.2% 라 빠른 경로로 먼저 걸러낸다.
            if value is None or value == "WT" or value == "":
                continue
            cell = _parse_cached(value, cache)
            n_tokens = cell.mutation_token_count
            if n_tokens == 0:
                continue

            mutated_gene_count[i] += 1
            mutation_event_count[i] += n_tokens
            functional_event_count[i] += cell.functional_count
            if n_tokens >= 2:
                multihit_gene_count[i] += 1
            if n_tokens > max_events_per_gene[i]:
                max_events_per_gene[i] = n_tokens

            deletions = cell.explicit_deletion_count
            if deletions:
                explicit_deletion_event_count[i] += deletions
                explicit_deletion_gene_count[i] += 1

            for col, key in enumerate(source_keys):
                event_sums[i, col] += getattr(cell, key)

            if include_cell_rollup:
                for col, key in enumerate(_ROLLUP_SUM_COLUMNS):
                    rollup_sums[i, col] += getattr(cell, key)
                for col, key in enumerate(_ROLLUP_ANY_COLUMNS):
                    if getattr(cell, key):
                        rollup_anys[i, col] = 1

    out = pd.DataFrame(event_sums, columns=event_names, index=df.index)
    out["mutated_gene_count"] = mutated_gene_count
    out["mutation_event_count"] = mutation_event_count
    out["functional_event_count"] = functional_event_count
    out["multihit_gene_count"] = multihit_gene_count
    out["max_events_per_gene"] = max_events_per_gene
    out["no_mutation_flag"] = (mutated_gene_count == 0).astype(np.int8)

    out["log1p_mutated_gene_count"] = np.log1p(mutated_gene_count).astype("float32")
    out["log1p_mutation_event_count"] = np.log1p(mutation_event_count).astype("float32")

    out["functional_ratio"] = _safe_ratio(
        out["functional_event_count"], out["mutation_event_count"]
    )
    out["synonymous_ratio"] = _safe_ratio(
        out["synonymous_event_count"], out["mutation_event_count"]
    )
    # 아래 셋만 분모가 기능성 변이 수다. 목록 정의를 그대로 따른 것.
    out["missense_ratio"] = _safe_ratio(
        out["missense_event_count"], out["functional_event_count"]
    )
    out["nonsense_ratio"] = _safe_ratio(
        out["nonsense_event_count"], out["functional_event_count"]
    )
    out["frameshift_ratio"] = _safe_ratio(
        out["frameshift_event_count"], out["functional_event_count"]
    )
    out["multihit_gene_ratio"] = _safe_ratio(
        out["multihit_gene_count"], out["mutated_gene_count"]
    )

    out["explicit_deletion_event_count"] = explicit_deletion_event_count
    out["explicit_deletion_gene_count"] = explicit_deletion_gene_count
    out["has_explicit_deletion"] = (explicit_deletion_event_count > 0).astype(np.int8)
    # `fs` 를 deletion 보다 먼저 분류하므로 deletion 계열에는 frameshift 가 섞이지
    # 않는다. 그래서 in-frame 조건이 이미 만족되어 있고 두 값이 항상 같다.
    out["inframe_deletion_count"] = explicit_deletion_event_count
    out["indel_or_frameshift_count"] = (
        out["explicit_deletion_event_count"] + out["frameshift_event_count"]
    )
    out["loss_of_function_count"] = (
        out["nonsense_event_count"]
        + out["frameshift_event_count"]
        + out["explicit_deletion_event_count"]
    )

    ordered = list(SAMPLE_FEATURE_COLUMNS)
    if include_cell_rollup:
        for col, key in enumerate(_ROLLUP_SUM_COLUMNS):
            out[key] = rollup_sums[:, col]
        for col, key in enumerate(_ROLLUP_ANY_COLUMNS):
            out[key] = rollup_anys[:, col]
        ordered += list(_ROLLUP_ANY_COLUMNS) + list(_ROLLUP_SUM_COLUMNS)
    return out[ordered]


def make_gene_mutated_matrix(
    df: pd.DataFrame,
    *,
    gene_columns: list[str] | None = None,
    prefix: str = "gene_mutated__",
) -> pd.DataFrame:
    """유전자별 변이 유무 행렬 — WT=0, 변이 존재=1.

    >>> int(make_gene_mutated_matrix(pd.DataFrame({"TP53": ["R273H"]}))["gene_mutated__TP53"][0])
    1
    """
    gene_columns = _resolve_gene_columns(df, gene_columns)
    values = df[gene_columns].to_numpy(dtype=object)
    mutated = np.zeros(values.shape, dtype=np.int8)
    cache = _parse_cache()
    for i in range(values.shape[0]):
        for j, value in enumerate(values[i]):
            if value is None or value == "WT" or value == "":
                continue
            if _parse_cached(value, cache).mutation_token_count:
                mutated[i, j] = 1
    return pd.DataFrame(
        mutated,
        columns=[f"{prefix}{gene}" for gene in gene_columns],
        index=df.index,
    )


def make_gene_event_count_matrix(
    df: pd.DataFrame,
    *,
    gene_columns: list[str] | None = None,
    prefix: str = "gene_event_count__",
) -> pd.DataFrame:
    """유전자별 변이 토큰 수 행렬.

    `make_mutation_encoding` 의 3단계 값은 한 셀에 변이가 여러 개면 최대 영향도만
    남기고 개수를 버린다. 이 행렬이 그 잃어버린 개수를 채운다.

    >>> int(make_gene_event_count_matrix(
    ...     pd.DataFrame({"TP53": ["Q369* I368N"]}))["gene_event_count__TP53"][0])
    2
    """
    gene_columns = _resolve_gene_columns(df, gene_columns)
    values = df[gene_columns].to_numpy(dtype=object)
    counts = np.zeros(values.shape, dtype=np.int16)
    cache = _parse_cache()
    for i in range(values.shape[0]):
        for j, value in enumerate(values[i]):
            if value is None or value == "WT" or value == "":
                continue
            counts[i, j] = _parse_cached(value, cache).mutation_token_count
    return pd.DataFrame(
        counts,
        columns=[f"{prefix}{gene}" for gene in gene_columns],
        index=df.index,
    )


GENE_MUTATION_TYPES: tuple[str, ...] = (
    "missense",
    "nonsense",
    "frameshift",
    "indel",
    "synonymous",
    "complex",
)


def make_gene_mutation_type_matrix(
    df: pd.DataFrame,
    *,
    gene_columns: list[str] | None = None,
    prefix: str = "gene_",
) -> pd.DataFrame:
    """유전자별 변이 유형 존재 여부를 0/1 wide matrix로 만든다.

    출력 컬럼은 입력 유전자 순서마다 ``GENE_MUTATION_TYPES`` 순서로 생성한다.
    예를 들어 TP53 뒤에는 ``gene_missense__TP53``부터
    ``gene_complex__TP53``까지 6개 컬럼이 붙는다. 같은 유형의 token이 한 셀에
    여러 개 있어도 존재 여부이므로 값은 1이다.

    이 함수는 각 행을 독립적으로 파싱하며 train 통계나 ``SUBCLASS``를 사용하지
    않는다. 따라서 train/test split별 Parquet을 미리 만들어도 데이터 누수가 없다.

    >>> frame = pd.DataFrame({"TP53": ["Q369* I368N"], "KRAS": ["WT"]})
    >>> out = make_gene_mutation_type_matrix(
    ...     frame, gene_columns=["TP53", "KRAS"]
    ... )
    >>> int(out["gene_nonsense__TP53"].iloc[0])
    1
    >>> int(out["gene_missense__TP53"].iloc[0])
    1
    """
    gene_columns = _resolve_gene_columns(df, gene_columns)
    values = df[gene_columns].to_numpy(dtype=object)
    n_types = len(GENE_MUTATION_TYPES)
    encoded = np.zeros(
        (len(df), len(gene_columns) * n_types),
        dtype=np.int8,
    )
    cache = _parse_cache()

    for row_idx in range(values.shape[0]):
        for gene_idx, value in enumerate(values[row_idx]):
            # 원본의 대부분이 WT이므로 파서 호출 전에 빠르게 건너뛴다.
            if value is None or value == "WT" or value == "":
                continue
            cell = _parse_cached(value, cache)
            if cell.mutation_token_count == 0:
                continue
            offset = gene_idx * n_types
            for type_idx, mutation_type in enumerate(GENE_MUTATION_TYPES):
                encoded[row_idx, offset + type_idx] = getattr(
                    cell, f"has_{mutation_type}"
                )

    columns = [
        f"{prefix}{mutation_type}__{gene}"
        for gene in gene_columns
        for mutation_type in GENE_MUTATION_TYPES
    ]
    return pd.DataFrame(encoded, columns=columns, index=df.index)


class BurdenBinner:
    """`hypermutated_flag` 와 `burden_quantile_bin` — fold 안에서만 fit 한다.

    경계를 전체 데이터에서 잡으면 valid/test 정보가 새고, test 로 잡으면 대회
    규정 위반이다. 그래서 별도 클래스로 떼어 놓고 fold 의 학습 부분에서만
    `fit` 을 부르게 강제한다.

    >>> feats = pd.DataFrame({"mutation_event_count": [0, 1, 2, 3, 100]})
    >>> out = BurdenBinner(n_bins=2, hypermutated_quantile=0.8).fit(feats).transform(feats)
    >>> int(out["hypermutated_flag"].iloc[-1])
    1
    """

    def __init__(
        self,
        *,
        n_bins: int = 10,
        hypermutated_quantile: float = 0.95,
        burden_column: str = "mutation_event_count",
    ) -> None:
        self.n_bins = n_bins
        self.hypermutated_quantile = hypermutated_quantile
        self.burden_column = burden_column
        self.threshold_: float | None = None
        self.edges_: np.ndarray | None = None

    def fit(self, features: pd.DataFrame) -> "BurdenBinner":
        burden = features[self.burden_column].to_numpy(dtype=np.float64)
        self.threshold_ = float(np.quantile(burden, self.hypermutated_quantile))
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        # 분포가 치우쳐 경계가 겹치면 구간 수가 줄어든다. 중복은 접는다.
        self.edges_ = np.unique(np.quantile(burden, quantiles))
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.threshold_ is None or self.edges_ is None:
            raise RuntimeError("fit() before transform()")
        burden = features[self.burden_column].to_numpy(dtype=np.float64)
        out = features.copy()
        out["hypermutated_flag"] = (burden > self.threshold_).astype(np.int8)
        out["burden_quantile_bin"] = np.searchsorted(
            self.edges_, burden, side="right"
        ).astype(np.int8)
        return out


# 비율·변환 변수 10종 — make_sample_mutation_features() 가 낸 8개(stateless)와
# BurdenBinner 가 만드는 2개(stateful)를 정본 순서 하나로 묶는다. 모델 입력 X 의
# 컬럼 순서는 항상 이 상수와 일치한다.
RATIO_TRANSFORM_FEATURE_COLUMNS: tuple[str, ...] = (
    "log1p_mutated_gene_count",
    "log1p_mutation_event_count",
    "functional_ratio",
    "synonymous_ratio",
    "missense_ratio",
    "nonsense_ratio",
    "frameshift_ratio",
    "multihit_gene_ratio",
    "hypermutated_flag",
    "burden_quantile_bin",
)


class RatioTransformFeatures:
    """비율·변환 변수 10종을 모델 입력 `X`로 바로 쓸 수 있게 묶는 fit/transform.

    새 계산식은 없다. 앞 8개는 `make_sample_mutation_features()`가 이미 계산해 둔
    값을 그대로 고르고, 뒤 2개(`hypermutated_flag`, `burden_quantile_bin`)는 내부
    `BurdenBinner`가 만든다. `fit()`은 train(또는 train fold)에서만 부른다.

    >>> feats = pd.DataFrame({
    ...     "log1p_mutated_gene_count": [0.0, 1.0],
    ...     "log1p_mutation_event_count": [0.0, 1.0],
    ...     "functional_ratio": [0.0, 1.0],
    ...     "synonymous_ratio": [0.0, 0.0],
    ...     "missense_ratio": [0.0, 1.0],
    ...     "nonsense_ratio": [0.0, 0.0],
    ...     "frameshift_ratio": [0.0, 0.0],
    ...     "multihit_gene_ratio": [0.0, 0.0],
    ...     "mutation_event_count": [0, 5],
    ... })
    >>> out = RatioTransformFeatures().fit(feats).transform(feats)
    >>> list(out.columns) == list(RATIO_TRANSFORM_FEATURE_COLUMNS)
    True
    """

    def __init__(self, *, burden_binner: BurdenBinner | None = None) -> None:
        self._binner = burden_binner if burden_binner is not None else BurdenBinner()
        self._fitted = False

    def fit(self, train_features: pd.DataFrame) -> "RatioTransformFeatures":
        """`train_features`의 mutation burden 정보만 써서 `BurdenBinner`를 fit한다."""
        burden_column = self._binner.burden_column
        if burden_column not in train_features.columns:
            raise ValueError(f"Missing feature columns: ['{burden_column}']")
        self._binner.fit(train_features)
        self._fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """fit에서 저장한 경계만 써서 정확히 10개 컬럼을 정본 순서로 반환한다."""
        if not self._fitted:
            raise RuntimeError(
                "RatioTransformFeatures.fit() 을 먼저 호출해야 한다 "
                "(train 또는 train fold 에서)."
            )
        required = set(RATIO_TRANSFORM_FEATURE_COLUMNS[:8]) | {self._binner.burden_column}
        missing = sorted(required - set(features.columns))
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        binned = self._binner.transform(features)
        return binned[list(RATIO_TRANSFORM_FEATURE_COLUMNS)].copy()

    def fit_transform(self, train_features: pd.DataFrame) -> pd.DataFrame:
        """`fit(train_features).transform(train_features)`의 편의 함수."""
        return self.fit(train_features).transform(train_features)


def row_to_exact_mutation_document(
    row: pd.Series,
    gene_columns: list[str],
) -> str:
    """Convert one sample into a space-delimited Exact Mutation Token document.

    한 셀에 변이가 여러 개면 **각각 별도 토큰으로 나눈다**. 예전에는 공백을
    밑줄로 바꿔 `MUT__TP53__S622S_G827R` 하나로 합쳤는데, 그러면 같은
    `S622S` 라도 옆에 붙은 변이가 다르면 전부 다른 토큰이 되어 어휘가 폭발하고
    빈도가 1로 흩어진다. `min_df` 를 걸면 통째로 잘려 나간다.

    >>> row_to_exact_mutation_document(pd.Series({"TP53": "S622S G827R"}), ["TP53"])
    'MUT__TP53__S622S MUT__TP53__G827R'
    """
    tokens: list[str] = []

    for gene in gene_columns:
        value = row[gene]
        if pd.isna(value):
            continue

        for mutation in split_tokens(value):
            tokens.append(f"MUT__{gene}__{mutation}")

    if not tokens:
        return EXACT_MUTATION_NO_MUTATION_TOKEN
    return " ".join(tokens)


def build_exact_mutation_documents(
    frame: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    """Build one Exact Mutation Token document for every input sample."""
    missing_columns = sorted(set(gene_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing gene columns: {missing_columns[:10]}")

    return frame.apply(
        row_to_exact_mutation_document,
        axis=1,
        gene_columns=gene_columns,
    )


def make_exact_mutation_token_parquet(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int | str]:
    """Read a raw CSV and write ID/label/document columns to one Parquet file.

    `SUBCLASS` 는 있으면 싣고 없으면 뺀다. test.csv 에는 라벨이 없는데 예전에는
    필수로 요구해서 test 문서를 아예 못 만들었다 — TF-IDF 를 학습만 하고 추론은
    못 하는 상태였다.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    frame = pd.read_csv(input_path)
    if "ID" not in frame.columns:
        raise ValueError("Missing required columns: ['ID']")
    if not frame["ID"].is_unique:
        raise ValueError("IDs must be unique")

    has_label = "SUBCLASS" in frame.columns
    gene_columns = [
        column for column in frame.columns if column not in _GENE_EXCLUDE
    ]
    if not gene_columns:
        raise ValueError("No gene columns found")

    documents = build_exact_mutation_documents(frame, gene_columns)
    columns: dict[str, pd.Series] = {"ID": frame["ID"].astype(str)}
    if has_label:
        columns["SUBCLASS"] = frame["SUBCLASS"].astype(str)
    columns["exact_mutation_document"] = documents
    output = pd.DataFrame(columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.parquet")
    output.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sample_count": len(output),
        "gene_count": len(gene_columns),
        "no_mutation_count": int(
            (documents == EXACT_MUTATION_NO_MUTATION_TOKEN).sum()
        ),
    }


# ---------------------------------------------------------------------------
# 기본 파생 변수 & del 파생변수 처리 정의함수
# ---------------------------------------------------------------------------

# gene_mutated__{gene}: 해당 유전자에 변이가 없으면 0, 변이가 존재하면 1
def compute_gene_mutated(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.DataFrame:
    _check_columns(df, gene_columns)
    return pd.DataFrame(
        {
            f"gene_mutated__{g}": df[g]
            .apply(lambda v: int(bool(_parse_mutation_tokens(v))))
            .astype("int8")
            for g in gene_columns
        },
        index=df.index,
    )


# gene_event_count__{gene}: 해당 유전자에 기록된 mutation token 수
def compute_gene_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.DataFrame:
    _check_columns(df, gene_columns)
    return pd.DataFrame(
        {
            f"gene_event_count__{g}": df[g].apply(
                lambda v: len(_parse_mutation_tokens(v))
            )
            for g in gene_columns
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Per-sample scalar features
# ---------------------------------------------------------------------------

# mutated_gene_count: 변이가 존재하는 고유 유전자 수
def compute_mutated_gene_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(1 for g in gene_columns if _parse_mutation_tokens(row[g])),
        axis=1,
    ).rename("mutated_gene_count")


# mutation_event_count: 공백으로 분리한 전체 mutation token 수
def compute_mutation_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(len(_parse_mutation_tokens(row[g])) for g in gene_columns),
        axis=1,
    ).rename("mutation_event_count")


# synonymous_event_count: 동의 변이 token 수 — 단백질 변화 없음 (예: R895R)
def compute_synonymous_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["synonymous"],
        axis=1,
    ).rename("synonymous_event_count")


# functional_event_count: 단백질 변화를 유발하는 변이 token 수 (missense + nonsense + frameshift + complex + indel)
def compute_functional_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _count(row: pd.Series) -> int:
        counts = _row_token_class_counts(row, gene_columns)
        return counts["missense"] + counts["nonsense"] + counts["frameshift"] + counts["complex"] + counts["indel"]

    return df.apply(_count, axis=1).rename("functional_event_count")


# missense_event_count: 아미노산 치환 변이 token 수 (예: R132H, V600E)
def compute_missense_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["missense"],
        axis=1,
    ).rename("missense_event_count")


# nonsense_event_count: stop codon을 생성하는 변이 token 수 (예: R213*)
def compute_nonsense_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["nonsense"],
        axis=1,
    ).rename("nonsense_event_count")


# frameshift_event_count: 읽기 틀 변화 변이 token 수 (예: K16fs, L1854fs)
def compute_frameshift_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: _row_token_class_counts(row, gene_columns)["frameshift"],
        axis=1,
    ).rename("frameshift_event_count")


# complex_event_count: 구조적 복합 변이 token 수 (예: 468_469LG>F*, E746_A750del) - indel 포함
def compute_complex_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _count(row: pd.Series) -> int:
        counts = _row_token_class_counts(row, gene_columns)
        return counts["complex"] + counts["indel"]

    return df.apply(_count, axis=1).rename("complex_event_count")


# multihit_gene_count: 한 유전자에 2개 이상의 변이가 있는 유전자 수
def compute_multihit_gene_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(
            1 for g in gene_columns if len(_parse_mutation_tokens(row[g])) >= 2
        ),
        axis=1,
    ).rename("multihit_gene_count")


# max_events_per_gene: 한 유전자에서 관찰된 최대 mutation token 수
def compute_max_events_per_gene(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _max(row: pd.Series) -> int:
        counts = [len(_parse_mutation_tokens(row[g])) for g in gene_columns]
        return max(counts) if counts else 0

    return df.apply(_max, axis=1).rename("max_events_per_gene")


# no_mutation_flag: 전체 유전자에 변이가 하나도 없으면 1, 아니면 0
def compute_no_mutation_flag(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: int(
            not any(_parse_mutation_tokens(row[g]) for g in gene_columns)
        ),
        axis=1,
    ).rename("no_mutation_flag")


# mutation_gene_ratio: 변이가 존재하는 유전자 수 / 전체 유전자 수
def compute_mutation_gene_ratio(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    n_total = len(gene_columns)
    return df.apply(
        lambda row: (
            sum(1 for g in gene_columns if _parse_mutation_tokens(row[g])) / n_total
            if n_total > 0 else 0.0
        ),
        axis=1,
    ).rename("mutation_gene_ratio")


# multi_mutation_gene_ratio: 복수 변이 유전자 수 / 변이가 존재하는 유전자 수
def compute_multi_mutation_gene_ratio(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _ratio(row: pd.Series) -> float:
        n_mutated = sum(1 for g in gene_columns if _parse_mutation_tokens(row[g]))
        n_multi = sum(1 for g in gene_columns if len(_parse_mutation_tokens(row[g])) >= 2)
        return n_multi / n_mutated if n_mutated > 0 else 0.0

    return df.apply(_ratio, axis=1).rename("multi_mutation_gene_ratio")


# n_unique_mutation_tokens: 중복을 제거한 mutation token 수
def compute_unique_mutation_token_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: len({t for g in gene_columns for t in _parse_mutation_tokens(row[g])}),
        axis=1,
    ).rename("n_unique_mutation_tokens")


# explicit_deletion_event_count: 명시적 del 사건 token 수 (예: R649del, 490del)
def compute_explicit_deletion_event_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(
            1
            for g in gene_columns
            for t in _parse_mutation_tokens(row[g])
            if _INDEL_RE.search(t)
        ),
        axis=1,
    ).rename("explicit_deletion_event_count")


# explicit_deletion_gene_count: 명시적 deletion이 있는 유전자 수
def compute_explicit_deletion_gene_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: sum(
            1
            for g in gene_columns
            if any(_INDEL_RE.search(t) for t in _parse_mutation_tokens(row[g]))
        ),
        axis=1,
    ).rename("explicit_deletion_gene_count")


# has_explicit_deletion: 명시적 deletion이 하나라도 있으면 1, 아니면 0
def compute_has_explicit_deletion(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)
    return df.apply(
        lambda row: int(
            any(
                _INDEL_RE.search(t)
                for g in gene_columns
                for t in _parse_mutation_tokens(row[g])
            )
        ),
        axis=1,
    ).rename("has_explicit_deletion")


# 단일 행에 대해 모든 burden feature를 한 번의 순회로 계산
def _compute_row_burden(row: pd.Series, gene_columns: list[str]) -> dict:
    mutated_gene_count = 0
    mutation_event_count = 0
    multihit_gene_count = 0
    max_events = 0
    synonymous = 0
    missense = 0
    nonsense = 0
    frameshift = 0
    complex_ = 0
    all_tokens: list[str] = []
    explicit_del_event_count = 0
    explicit_del_gene_count = 0

    for g in gene_columns:
        tokens = _parse_mutation_tokens(row[g])
        n = len(tokens)
        if n > 0:
            mutated_gene_count += 1
        mutation_event_count += n
        if n >= 2:
            multihit_gene_count += 1
        if n > max_events:
            max_events = n

        gene_has_del = False
        for t in tokens:
            all_tokens.append(t)
            cls = _classify_token(t)
            if cls == "synonymous":
                synonymous += 1
            elif cls == "missense":
                missense += 1
            elif cls == "nonsense":
                nonsense += 1
            elif cls == "frameshift":
                frameshift += 1
            else:
                complex_ += 1
            if _INDEL_RE.search(t):
                explicit_del_event_count += 1
                gene_has_del = True
        if gene_has_del:
            explicit_del_gene_count += 1

    n_total = len(gene_columns)
    n_unique = len(set(all_tokens))
    return {
        "mutated_gene_count": mutated_gene_count,
        "mutation_event_count": mutation_event_count,
        "synonymous_event_count": synonymous,
        "functional_event_count": missense + nonsense + frameshift + complex_,
        "missense_event_count": missense,
        "nonsense_event_count": nonsense,
        "frameshift_event_count": frameshift,
        "complex_event_count": complex_,
        "multihit_gene_count": multihit_gene_count,
        "max_events_per_gene": max_events,
        "no_mutation_flag": int(mutated_gene_count == 0),
        "mutation_gene_ratio": mutated_gene_count / n_total if n_total > 0 else 0.0,
        "multi_mutation_gene_ratio": multihit_gene_count / mutated_gene_count if mutated_gene_count > 0 else 0.0,
        "n_unique_mutation_tokens": n_unique,
        "n_repeated_mutation_tokens": len(all_tokens) - n_unique,
        "explicit_deletion_event_count": explicit_del_event_count,
        "explicit_deletion_gene_count": explicit_del_gene_count,
        "has_explicit_deletion": int(explicit_del_event_count > 0),
    }


# 모든 burden feature를 단일 apply 패스로 계산 (개별 함수 15회 호출 대비 ~10x 빠름)
def compute_all_burden_features(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.DataFrame:
    _check_columns(df, gene_columns)
    return pd.DataFrame(
        df.apply(lambda row: _compute_row_burden(row, gene_columns), axis=1).tolist(),
        index=df.index,
    )


# n_repeated_mutation_tokens: 중복 등장 mutation token 수 (전체 token 수 - 고유 token 수)
def compute_repeated_mutation_token_count(
    df: pd.DataFrame,
    gene_columns: list[str],
) -> pd.Series:
    _check_columns(df, gene_columns)

    def _count(row: pd.Series) -> int:
        all_tokens = [t for g in gene_columns for t in _parse_mutation_tokens(row[g])]
        return len(all_tokens) - len(set(all_tokens))

    return df.apply(_count, axis=1).rename("n_repeated_mutation_tokens")


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/train.csv",
        help="Input train CSV containing ID, SUBCLASS, and gene columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/process/train_exact_mutation_tokens.parquet",
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = make_exact_mutation_token_parquet(
        args.input,
        args.output,
        overwrite=args.overwrite,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

    
