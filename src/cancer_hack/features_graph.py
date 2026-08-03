"""공변이(co-mutation) 유전자 쌍 피처 — **fold 의 train 부분에서만** 쌍을 고른다.

두 유전자가 함께 변이된 사건 하나가 단일 유전자보다 강한 신호를 줄 때가 있다
(`IDH1`+`ATRX` -> GBMLGG/LGG, `PTEN`+`CTNNB1` -> UCEC). 문제는 후보가 너무 많다는
것과, test 가 train 보다 변이 유전자 수가 2.21배 많아(중앙값 14 -> 28) 두 유전자가
"생물학적으로" 겹친 건지 "샘플에 변이가 원래 많아서" 겹친 건지 구분해야 한다는 것.

## 후보 풀

문헌 고정 드라이버 92개(`features_domain.DRIVERS`)만 쓴다. chi2 로 고른 풀은 시험해
보면 상위 20쌍 중 15쌍이 `BRAF & 긴 passenger 유전자`(MYH1, PCLO, RYR1, MXRA5...)다 —
공변이가 아니라 "BRAF 변이 x 높은 TMB -> SKCM" 이고, 만들 수 있는 피처 중 시프트에
가장 취약하다. `pool="chi2"` 는 이 축퇴를 일부러 재현하는 대조군으로만 둔다.

## 선택과 변환을 함수로 가른다

`select_comutation_pairs` 는 **test 행렬을 인자로 아예 받지 않는다.** "test 통계가
선택에 안 닿는다"가 주석이 아니라 시그니처로 보장된다 (`Chi2TopKSelector`,
`MutationTfidfBlock` 과 같은 fit/transform 계약이지만 여기서는 한 걸음 더 나간다).

## 값 방식 — 왜 원시 AND 가 기본이 아닌가

원시 0/1 AND 는 test 에서 중앙값 3.1배 부푼다(driver 쌍 536개 실측). 게다가 선택된
쌍의 유전자가 거의 다 `enc3` 의 chi2 top-500 안에 이미 있어서, `max_depth=6` 트리는
`PTEN>0 & CTNNB1>0` 을 두 번의 분기로 표현한다 — 원시 AND 는 이론상 중복이다.
행 자신의 변이 부담으로 나눈 비율은 축 정렬 분기로 표현이 안 된다. 그래서 기본값은
`share`: `n_bar * AND_i / max(n_i, 1)`. 같은 실측에서 중앙값 배율이 1.46배로 준다.

    value   수식                              test/train 중앙 배율
    and     AND_i                             3.12
    share   n_bar * AND_i / max(n_i, 1)        1.46  (기본)
    gate    AND_i * (1 - ECDF_train(n_i))      —     test 가 학습 범위를 넘으면 침묵

## 과변이 가드가 실질을 한다

지지도·클래스별 지지도 필터를 통과해도, "두 유전자가 겹친 이유가 그 샘플에 변이가
3,000개라서"인 쌍이 남는다. `features_basic.BurdenBinner` 를 그대로 재사용해
fold-train 에서 과변이 임계를 잡고, 그 임계를 넘는 샘플에 동시 변이의 절반 이상이
쏠린 쌍을 버린다. 넘기는 값은 `BurdenBinner` 의 정본 열 이름인
`mutation_event_count` 가 아니라 **변이 유전자 수**다 — 공변이 쌍이 그걸로
만들어지고, test 의 전사체 중복 기재로 안 부푸는 축이라서 일부러 그렇게 넘긴다.

## 강제 포함 5쌍

`MANUAL_PAIRS` 는 필터를 통과하지 못해도(특히 fold-train 슬라이스에서 지지도가
`min_support` 밑으로 떨어지는 `BRAF`+`CDKN2A`) fold 와 무관하게 항상 들어간다.
"고정 5쌍 + 자동 선별" 전략의 고정분이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features_basic import BurdenBinner
from .features_domain import DRIVERS
from .validation import Chi2TopKSelector

#: §9 강제 포함 5쌍. 유전자 이름 표기 순서는 무관하다 — 선택 시 프로그램이 사전순으로
#: 다시 정렬한다.
MANUAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("ATRX", "IDH1"),
    ("IDH1", "TP53"),
    ("CTNNB1", "PTEN"),
    ("PIK3CA", "PTEN"),
    ("BRAF", "CDKN2A"),
)


def driver_pool_indices(gene_names: Sequence[str]) -> np.ndarray:
    """`DRIVERS` 92개 중 `gene_names` 에 실제로 있는 것의 열 인덱스.

    문헌 고정 목록이라 fold 와 무관한 상수다 — `features_domain.DRIVERS` 와 같은
    누출 없음 근거를 그대로 물려받는다.
    """
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    return np.array(sorted(name_to_idx[g] for g in DRIVERS if g in name_to_idx), dtype=np.int64)


@dataclass(frozen=True)
class ComutationPairs:
    """`select_comutation_pairs` 가 fold-train 에서 fit 한 결과.

    `gene_a`/`gene_b` 는 전체 유전자 행렬 기준 열 인덱스다. `n_bar`/`burden_sorted` 는
    `share`/`gate` 값 방식이 test 를 변환할 때 쓸 fold-train 통계이고, `stats` 는
    로깅용 진단 정보다(선택을 좌우하지 않는다 — 선택은 이미 끝난 뒤의 기록일 뿐).
    """

    gene_a: np.ndarray
    gene_b: np.ndarray
    names: list[str]
    mode: str
    n_bar: float
    burden_sorted: np.ndarray
    stats: list[dict]


def select_comutation_pairs(
    train_matrix: np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence,
    *,
    gene_names: Sequence[str],
    pool: str = "drivers",
    pool_topk: int = 300,
    mode: str = "mutated",
    min_support: int = 20,
    min_class_support: int = 8,
    min_purity: float = 0.25,
    min_lift: float = 0.15,
    max_hyper_fraction: float = 0.50,
    hyper_quantile: float = 0.95,
    max_pairs_per_gene: int = 3,
    topk: int = 20,
    manual_pairs: Sequence[tuple[str, str]] = MANUAL_PAIRS,
) -> ComutationPairs:
    """공변이 쌍을 고른다. **`train_matrix[train_index]` 만 본다 — test 인자가 없다.**

    순서: 후보 풀(드라이버 92개 + 강제 포함 쌍의 유전자) -> 모든 쌍의 동시 변이 수
    (한 번의 `G.T @ G`) -> 클래스별 지지도 -> 과변이 의존도 가드 -> 추가 정보량(lift)
    -> `score = lift * sqrt(n_AB)` 로 정렬 -> 유전자당 상한을 지키며 상위 `topk` 개
    -> `manual_pairs` 는 이 필터를 전부 건너뛰고 무조건 추가.

    >>> X = np.array([[1,1,0],[1,1,0],[1,0,0],[0,0,1],[0,0,1]], dtype=np.float32)
    >>> pairs = select_comutation_pairs(
    ...     X, np.arange(5), ["A","A","A","B","B"],
    ...     gene_names=["GA","GB","GC"], pool="drivers", manual_pairs=(("GA","GB"),),
    ...     min_support=1, min_class_support=1, min_purity=0.0, min_lift=0.0,
    ... )
    >>> pairs.names
    ['GA__GB']
    """
    if mode not in ("mutated", "functional"):
        raise ValueError(f"알 수 없는 mode: {mode}")

    gene_names = list(gene_names)
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    X = np.asarray(train_matrix)[train_index]
    y_arr = np.asarray(y_train_fold)
    M = (X == 2) if mode == "functional" else (X >= 1)

    burden = M.sum(axis=1).astype(np.float64)
    n_bar = float(burden.mean()) if burden.size else 0.0
    burden_sorted = np.sort(burden)
    empty = ComutationPairs(
        gene_a=np.array([], dtype=np.int64),
        gene_b=np.array([], dtype=np.int64),
        names=[],
        mode=mode,
        n_bar=n_bar,
        burden_sorted=burden_sorted,
        stats=[],
    )

    if pool == "drivers":
        pool_idx = driver_pool_indices(gene_names)
    elif pool == "chi2":
        selector = Chi2TopKSelector(k=pool_topk).fit(M.astype(np.float32), y_arr)
        pool_idx = np.sort(selector.indices_).astype(np.int64)
    else:
        raise ValueError(f"알 수 없는 pool: {pool}")

    # 강제 포함 쌍의 유전자를 풀에 얹는다 — 그래야 아래 벡터화 계산이 그 쌍의
    # 진단치(n_AB·purity·lift·hyper_fraction)도 자동 선별분과 같은 표에서 낸다.
    manual_idx_pairs: list[tuple[int, int]] = []
    manual_gene_idx: list[int] = []
    for a, b in manual_pairs:
        if a != b and a in name_to_idx and b in name_to_idx:
            ia, ib = name_to_idx[a], name_to_idx[b]
            manual_idx_pairs.append((ia, ib))
            manual_gene_idx += [ia, ib]
    if manual_gene_idx:
        pool_idx = np.union1d(pool_idx, np.array(manual_gene_idx, dtype=np.int64))

    if pool_idx.size < 2:
        return empty

    classes = np.unique(y_arr)
    class_index = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_index[v] for v in y_arr])
    Y = np.zeros((len(y_arr), len(classes)), dtype=np.float32)
    Y[np.arange(len(y_arr)), y_idx] = 1.0

    G = M[:, pool_idx].astype(np.float32)
    ii_local, jj_local = np.triu_indices(pool_idx.size, 1)
    if ii_local.size == 0:
        return empty

    # 동시 변이 수 — 한 번의 92x92(대략) 행렬곱.
    C = G.T @ G
    nAB = C[ii_local, jj_local]

    AND = G[:, ii_local] * G[:, jj_local]
    cls_co = AND.T @ Y
    best_idx = cls_co.argmax(axis=1)
    n_best = cls_co[np.arange(cls_co.shape[0]), best_idx]
    purity = n_best / np.maximum(nAB, 1)

    n_g = G.sum(axis=0)
    marg = G.T @ Y
    p_a = marg[ii_local, best_idx] / np.maximum(n_g[ii_local], 1)
    p_b = marg[jj_local, best_idx] / np.maximum(n_g[jj_local], 1)
    lift = purity - np.maximum(p_a, p_b)

    # 과변이 의존도 — 동시 변이의 몇 %가 과변이 샘플(fold-train 상위 분위)에서
    # 나왔는가. `BurdenBinner` 의 정본 열 이름은 `mutation_event_count` 지만
    # 여기서는 **변이 유전자 수**를 그 이름으로 넘긴다 — 공변이 쌍이 그걸로
    # 만들어지고 test 의 전사체 중복 기재로 안 부푸는 축이라 일부러 그렇게 한다.
    binner = BurdenBinner(hypermutated_quantile=hyper_quantile).fit(
        pd.DataFrame({"mutation_event_count": burden})
    )
    hyper = (burden > binner.threshold_).astype(np.float32)
    hyper_frac = (AND.T @ hyper) / np.maximum(nAB, 1)

    ga_global = pool_idx[ii_local]
    gb_global = pool_idx[jj_local]

    forced = np.zeros(ii_local.size, dtype=bool)
    if manual_idx_pairs:
        wanted = {frozenset(p) for p in manual_idx_pairs}
        pair_keys = [frozenset((int(a), int(b))) for a, b in zip(ga_global, gb_global)]
        forced = np.array([key in wanted for key in pair_keys], dtype=bool)

    auto_ok = (
        (nAB >= min_support)
        & (n_best >= min_class_support)
        & (purity >= min_purity)
        & (lift >= min_lift)
        & (hyper_frac <= max_hyper_fraction)
        & ~forced
    )

    score = lift * np.sqrt(np.maximum(nAB, 0.0))
    auto_candidates = np.where(auto_ok)[0]
    order = auto_candidates[np.argsort(-score[auto_candidates], kind="stable")]

    picked = list(np.where(forced)[0])
    gene_use: dict[int, int] = {}
    for idx in order:
        if len(picked) - int(forced.sum()) >= topk:
            break
        a, b = int(ga_global[idx]), int(gb_global[idx])
        if gene_use.get(a, 0) >= max_pairs_per_gene or gene_use.get(b, 0) >= max_pairs_per_gene:
            continue
        picked.append(idx)
        gene_use[a] = gene_use.get(a, 0) + 1
        gene_use[b] = gene_use.get(b, 0) + 1

    if not picked:
        return empty

    picked_arr = np.array(picked, dtype=np.int64)
    gene_a_out = ga_global[picked_arr]
    gene_b_out = gb_global[picked_arr]

    names: list[str] = []
    stats: list[dict] = []
    for k, idx in enumerate(picked_arr):
        a_name, b_name = gene_names[gene_a_out[k]], gene_names[gene_b_out[k]]
        lo, hi = (a_name, b_name) if a_name < b_name else (b_name, a_name)
        name = f"{lo}__{hi}"
        names.append(name)
        stats.append(
            {
                "name": name,
                "gene_a": lo,
                "gene_b": hi,
                "n_ab_train_fold": int(nAB[idx]),
                "best_class": str(classes[best_idx[idx]]),
                "purity": float(purity[idx]),
                "lift": float(lift[idx]),
                "hyper_fraction": float(hyper_frac[idx]),
                "manual": bool(forced[idx]),
            }
        )

    return ComutationPairs(
        gene_a=gene_a_out,
        gene_b=gene_b_out,
        names=names,
        mode=mode,
        n_bar=n_bar,
        burden_sorted=burden_sorted,
        stats=stats,
    )


def transform_comutation_pairs(
    matrix: np.ndarray, pairs: ComutationPairs, *, value: str = "share"
) -> np.ndarray:
    """`select_comutation_pairs` 가 고른 쌍을 행렬로 바꾼다. `y` 도 `fit` 도 없다.

    행 하나만 넣어도 전체를 넣었을 때와 그 행의 값이 같다 — `pairs` 에 담긴 통계
    (`n_bar`/`burden_sorted`) 는 전부 fold-train 에서 이미 고정됐고, 나머지는 그 행
    자신의 값만 본다.
    """
    n_rows = np.asarray(matrix).shape[0]
    if not pairs.names:
        return np.zeros((n_rows, 0), dtype=np.float32)
    if value not in ("and", "share", "gate"):
        raise ValueError(f"알 수 없는 value: {value}")

    X = np.asarray(matrix)
    M = (X == 2) if pairs.mode == "functional" else (X >= 1)
    a = M[:, pairs.gene_a].astype(np.float32)
    b = M[:, pairs.gene_b].astype(np.float32)
    and_ = a * b

    if value == "and":
        return and_.astype(np.float32)

    n_i = M.sum(axis=1).astype(np.float32)
    if value == "share":
        return (pairs.n_bar * and_ / np.maximum(n_i, 1.0)[:, None]).astype(np.float32)

    # value == "gate": fold-train ECDF. test 가 학습 범위를 넘는 부담이면 F_hat -> 1
    # 이라 피처가 오발화 대신 침묵한다 — 안전한 실패 방향.
    denom = max(len(pairs.burden_sorted), 1)
    f_hat = np.searchsorted(pairs.burden_sorted, n_i, side="right") / denom
    return (and_ * (1.0 - f_hat)[:, None]).astype(np.float32)


def build_fold_comutation_block(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence,
    *,
    gene_names: Sequence[str],
    prefix: str = "comut__",
    pool: str = "drivers",
    pool_topk: int = 300,
    mode: str = "mutated",
    value: str = "share",
    topk: int = 20,
    min_support: int = 20,
    min_class_support: int = 8,
    min_purity: float = 0.25,
    min_lift: float = 0.15,
    max_hyper_fraction: float = 0.50,
    hyper_quantile: float = 0.95,
    max_pairs_per_gene: int = 3,
    manual_pairs: Sequence[tuple[str, str]] = MANUAL_PAIRS,
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    """한 fold 분의 공변이 블록 -> (열 이름, train dense, test dense, 진단 목록).

    `features_sparse.build_fold_tfidf_block` 과 같은 계약이다 — 선택은 전부
    `train_matrix[train_index]` 에서만 하고, 반환하는 train 행렬은 valid 행을 포함한
    **전체 train 행**이다. 진단 목록의 `train_rate`/`test_rate`/`rate_ratio` 는 선택이
    끝난 뒤에 계산하는 감사 기록이고 어떤 필터도 이 값을 읽지 않는다.
    """
    pairs = select_comutation_pairs(
        train_matrix,
        train_index,
        y_train_fold,
        gene_names=gene_names,
        pool=pool,
        pool_topk=pool_topk,
        mode=mode,
        min_support=min_support,
        min_class_support=min_class_support,
        min_purity=min_purity,
        min_lift=min_lift,
        max_hyper_fraction=max_hyper_fraction,
        hyper_quantile=hyper_quantile,
        max_pairs_per_gene=max_pairs_per_gene,
        topk=topk,
        manual_pairs=manual_pairs,
    )
    full_prefix = f"{prefix}{mode[:3]}__"
    names = [f"{full_prefix}{n}" for n in pairs.names]
    train_out = transform_comutation_pairs(train_matrix, pairs, value=value)
    test_out = transform_comutation_pairs(test_matrix, pairs, value=value)

    diagnostics: list[dict] = []
    if pairs.names:
        test_arr = np.asarray(test_matrix)
        m_te = (test_arr == 2) if mode == "functional" else (test_arr >= 1)
        n_test = m_te.shape[0]
        n_train_fold = len(train_index)
        test_and = (m_te[:, pairs.gene_a] & m_te[:, pairs.gene_b]).sum(axis=0)
        for stat, name, n_ab_test in zip(pairs.stats, names, test_and):
            train_rate = stat["n_ab_train_fold"] / max(n_train_fold, 1)
            test_rate = float(n_ab_test) / max(n_test, 1)
            diagnostics.append(
                {
                    **stat,
                    "name": name,
                    "n_ab_test": int(n_ab_test),
                    "train_rate": train_rate,
                    "test_rate": test_rate,
                    "rate_ratio": (test_rate / train_rate) if train_rate > 0 else None,
                }
            )

    return names, train_out, test_out, diagnostics
