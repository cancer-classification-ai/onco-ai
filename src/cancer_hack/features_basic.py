#!/usr/bin/env python
"""팀 공용 피처 정의.

CLI 는 `scripts/make_features.py` 에 있고 이 모듈은 정의만 담는다. 피처 함수는
이름으로 구분해 두어서 새 빌더를 넣어도 기존 동작이 바뀌지 않는다.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import re

from .parser import (
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
EXACT_MUTATION_EMPTY_VALUES = frozenset({"", "WT", "0", "NA", "NAN", "NONE", "."})
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


