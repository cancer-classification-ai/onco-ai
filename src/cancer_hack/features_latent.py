"""유전자 모듈 피처 — 개별 유전자를 그룹으로 묶어 **fold 의 train 부분에서만** 집계한다.

`enc3` 는 유전자 4,383개를 각각 따로 본다. 같은 기능 축에 속한 유전자들이 서로 다른
환자에서 변이됐을 때 그 공통점은 chi2 top-500 이 유전자를 하나씩 고르는 방식으로는
안 잡힌다. 묶으면 그 축이 한 열이 된다.

두 가지를 같은 파일에 둔다 — 둘 다 같은 공변이 기하를 쓰고 SVD 코드를 공유한다.

    소프트 (L4)  fit_latent_basis    행을 L2 정규화하고 SVD/NMF 성분에 투영
    하드  (L3)  fit_gene_modules    유전자를 KMeans 로 배타적 모듈에 배정

## 왜 소프트를 같이 두는가

하드 군집은 유전자를 반드시 모듈 하나에만 넣는다. 실제 유전자는 여러 기능에 관여하고,
KMeans 배정은 seed 를 바꾸면 크게 흔들린다 — `scripts/inspect_latent.py` 실측으로
seed 간 ARI 가 **0.296** 이다(`enc3` chi2 의 fold 간 Jaccard 0.719, `comut_auto`
0.448~0.547 이 이 저장소가 받아들인 안정성의 하한이다). 소프트 로딩은 한 유전자가
여러 성분에 기여할 수 있고, 같은 실측에서 fold 간 부분공간 정렬이 **0.829** 였다.

## 값 형태 — 카운트를 안 내는 이유

test 는 변이 유전자가 train 의 **2.21배**다(35.3 -> 78.1). 토큰/유전자 비율은 0.99배로
안정하니 유전자가 *퍼지는* 것이지 토큰이 쌓이는 게 아니다. 그래서 판정 기준이 하나다:

    행의 변이 유전자 집합이 균일하게 커져도 값이 안 변하면 내성 형태다.
    분자와 분모가 같이 부풀어야 한다.

    value       수식                                   시프트 예상   위치
    share       |변이 & 모듈| / |변이 전체|              ~1.0      기본
    enrich      share * (전체 유전자수 / 모듈 크기)       ~1.0      기본
    wshare      희귀도가중 share (분모도 가중)            ~1.0      기본
    any         |변이 & 모듈| > 0                       포화       대조군
    logcount    log1p(|변이 & 모듈|)                    ~1.20x    대조군
    fraction    |변이 & 모듈| / 모듈 크기                ~2.21x    대조군
    wburden     희귀도가중 원시 합                       ~2.21x    대조군

`fraction` 은 비율처럼 보이지만 아니다 — 분모가 모듈 크기라는 **상수**라서 분자만
2.21배 부푼다. 이름에 속기 쉬워서 대조군에 둔 이유를 여기 적어 둔다.

대조군을 지우지 않고 남기는 건 `features_graph` 의 `value="and"` 와 같은 이유다.
대조군이 기본형을 이기면 위 논증이 틀린 것이고, 그건 그것대로 알 값어치가 있다.

## 행 정규화가 소프트 쪽의 시프트 논증 전부다

`row_norm="l2"` 를 빼면 성분 0 이 변이 부담 축에 정렬된다. 실측이 그렇다 — 선두
3성분의 |burden 상관| 최대가 `l2` 에서 **0.458**, `none` 에서 **0.995** 다. 후자는
`features_graph` 가 기록한 실패 모드(chi2 후보 풀이 `BRAF x 긴 passenger 유전자` 로
축퇴해 생물학이 아니라 유전자 길이·TMB 를 학습한 것)를 새 좌표계에서 그대로 재현한
것이다. `row_norm="none"` 은 그 축퇴를 일부러 재현하는 대조군으로만 둔다.

## 선택과 변환을 함수로 가른다

`fit_latent_basis`·`fit_gene_modules` 는 **test 행렬도 `y` 도 인자로 받지 않는다.**
test 를 안 받는 건 `features_graph.select_comutation_pairs` 와 같은 계약이고, `y` 까지
안 받는 건 거기서 한 걸음 더 간 것이다 — L3·L4 는 비지도라서, 나중에 누가 chi2 사전
필터나 지도 성분 랭킹을 조용히 끼워 넣는 걸 시그니처가 구조적으로 막는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import normalize

#: 모듈 집계 값 방식. 앞 3개가 시프트 내성 형태, 뒤 4개가 의도적 노출 대조군이다.
MODULE_VALUES = ("share", "enrich", "wshare", "any", "logcount", "fraction", "wburden")

#: 대조군 값 방식 — 위 docstring 의 배율 표 참고. `train_gbdt.py` 가 로그 경고에 쓴다.
SHIFT_EXPOSED_VALUES = ("any", "logcount", "fraction", "wburden")


# ------------------------------------------------------------------ 공통 유틸
def _binarize(matrix: np.ndarray, mode: str) -> sparse.csr_matrix:
    """enc3 0/1/2 -> 0/1 희소 행렬. `features_graph.py:307` 과 같은 규약이다.

    비WT 셀이 0.81% 뿐이라 희소로 바꾸면 SVD/NMF 가 훨씬 빠르다. 규약을 두 개
    만들지 않으려고 판정식은 `features_graph` 것을 그대로 쓴다.
    """
    if mode not in ("mutated", "functional"):
        raise ValueError(f"알 수 없는 mode: {mode}")
    dense = np.asarray(matrix)
    hit = (dense == 2) if mode == "functional" else (dense >= 1)
    return sparse.csr_matrix(hit.astype(np.float32))


def _gene_idf(fit_matrix: sparse.csr_matrix) -> np.ndarray:
    """fold-train 유전자 희귀도. 흔한 유전자일수록 작다.

    `features_sparse.MutationTfidfBlock` 은 문서 문자열을 받는 물건이라 여기 안 맞는다.
    binary 행렬에서 바로 세는 게 3줄이고 왕복이 없다.
    """
    n_rows = max(fit_matrix.shape[0], 1)
    document_frequency = np.asarray((fit_matrix > 0).sum(axis=0)).ravel()
    return (np.log((1.0 + n_rows) / (1.0 + document_frequency)) + 1.0).astype(np.float32)


def _support_index(fit_matrix: sparse.csr_matrix, min_gene_support: int) -> np.ndarray:
    """fold-train 에서 `min_gene_support` 행 이상 변이된 유전자의 열 인덱스.

    한두 샘플에만 뜨는 passenger 를 분해 전에 털어내는 게 목적이다. fold-train
    통계라 `make_features.py` 가 아니라 여기 있는 게 맞다.
    """
    counts = np.asarray((fit_matrix > 0).sum(axis=0)).ravel()
    return np.where(counts >= max(int(min_gene_support), 1))[0].astype(np.int64)


def _row_l1_share(values: np.ndarray) -> np.ndarray:
    """행별 |값| 합으로 나눈다. 비음수(NMF)면 정확히 혼합 비율이 된다."""
    scale = np.abs(values).sum(axis=1, keepdims=True)
    return (values / np.maximum(scale, 1e-9)).astype(np.float32)


def _apply_row_shape(
    block: sparse.csr_matrix, weight: np.ndarray | None, row_norm: str
) -> sparse.csr_matrix:
    """희귀도 가중 -> 행 정규화. 무상태라 fit 이 없고 누출 경로도 없다.

    `sklearn.preprocessing.normalize` 는 학습하는 게 없다. `weight` 는 fold-train 에서
    이미 고정된 값이라 행마다 같은 상수를 곱할 뿐이다.
    """
    out = block
    if weight is not None:
        out = out.multiply(weight[None, :]).tocsr()
    if row_norm == "l2":
        out = normalize(out, norm="l2", axis=1)
    return out.astype(np.float32)


def aggregate_by_membership(
    binary: sparse.csr_matrix,
    membership: sparse.csr_matrix,
    *,
    group_sizes: np.ndarray,
    gene_weight: np.ndarray,
    n_genes_total: int,
    value: str = "share",
) -> np.ndarray:
    """유전자 그룹별 집계 — 7가지 값 방식의 **단일 구현**.

    `membership` 은 `(n_genes_total, n_groups)` 0/1 희소 행렬이다. 하드 모듈은 행마다
    1이 하나뿐인 분할이고, `features_signature` 의 클래스 서명은 한 유전자가 여러
    클래스에 들어갈 수 있어 행에 1이 여럿이다 — 집계 식은 둘이 같으므로 여기 한 번만 쓴다.

    `share`/`enrich`/`wshare` 의 분모는 **행 전체의 변이 유전자 수**다(그룹 안쪽이
    아니라). 그래야 test 에서 유전자가 균일하게 퍼질 때 분자와 분모가 같이 부푼다.
    """
    if value not in MODULE_VALUES:
        raise ValueError(f"알 수 없는 value: {value} (가능: {MODULE_VALUES})")

    counts = np.asarray((binary @ membership).todense(), dtype=np.float32)

    if value == "any":
        return (counts > 0).astype(np.float32)
    if value == "logcount":
        return np.log1p(counts).astype(np.float32)
    if value == "fraction":
        # 분모가 그룹 크기라는 상수라 분자만 2.21배 부푼다 — 비율처럼 보이는 카운트다.
        return (counts / np.maximum(group_sizes, 1)[None, :]).astype(np.float32)

    if value in ("wshare", "wburden"):
        weighted = np.asarray(
            (binary.multiply(gene_weight[None, :]).tocsr() @ membership).todense(),
            dtype=np.float32,
        )
        if value == "wburden":
            return weighted
        row_weight = np.asarray(
            binary.multiply(gene_weight[None, :]).sum(axis=1), dtype=np.float32
        ).ravel()
        return (weighted / np.maximum(row_weight, 1e-6)[:, None]).astype(np.float32)

    row_total = np.asarray(binary.sum(axis=1), dtype=np.float32).ravel()
    share = counts / np.maximum(row_total, 1.0)[:, None]
    if value == "share":
        return share.astype(np.float32)
    # enrich — 관측 share / 균등 배분 기대 share. 1.0 이 기준선이다.
    expected = np.maximum(group_sizes, 1) / max(n_genes_total, 1)
    return (share / expected[None, :]).astype(np.float32)


# ------------------------------------------------------------------ L4 소프트 잠재
@dataclass(frozen=True)
class LatentBasis:
    """`fit_latent_basis` 가 fold-train 에서 fit 한 기저.

    `components` 는 `(n_components, len(gene_index))` 이고 `gene_index` 는 전체 유전자
    행렬 기준 열 인덱스다. `stats` 는 로깅용 진단이라 변환에 관여하지 않는다.
    """

    method: str
    components: np.ndarray
    gene_index: np.ndarray
    gene_weight: np.ndarray | None
    row_norm: str
    mode: str
    n_components: int
    stats: list[dict]


def fit_latent_basis(
    train_matrix: np.ndarray,
    train_index: np.ndarray,
    *,
    gene_names: Sequence[str],
    method: str = "svd",
    n_components: int = 64,
    mode: str = "mutated",
    row_norm: str = "l2",
    gene_weight: str = "none",
    min_gene_support: int = 5,
    random_state: int = 0,
) -> LatentBasis:
    """잠재 기저를 fit 한다. **`train_matrix[train_index]` 만 본다 — test 도 `y` 도 없다.**

    순서: 이진화 -> fold-train 지지도로 유전자 추림 -> (선택) 희귀도 가중 ->
    행 L2 정규화 -> `TruncatedSVD` 또는 `NMF`.

    >>> X = np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]], dtype=np.float32)
    >>> basis = fit_latent_basis(
    ...     X, np.arange(4), gene_names=["GA","GB","GC","GD"],
    ...     n_components=2, min_gene_support=1,
    ... )
    >>> basis.components.shape
    (2, 4)
    """
    if method not in ("svd", "nmf"):
        raise ValueError(f"알 수 없는 method: {method}")
    if row_norm not in ("l2", "none"):
        raise ValueError(f"알 수 없는 row_norm: {row_norm}")
    if gene_weight not in ("none", "idf"):
        raise ValueError(f"알 수 없는 gene_weight: {gene_weight}")

    binary = _binarize(train_matrix, mode)
    fit_rows = binary[np.asarray(train_index)]
    gene_index = _support_index(fit_rows, min_gene_support)

    if gene_index.size < 2:
        return LatentBasis(
            method=method,
            components=np.zeros((0, 0), dtype=np.float32),
            gene_index=gene_index,
            gene_weight=None,
            row_norm=row_norm,
            mode=mode,
            n_components=0,
            stats=[],
        )

    weight = _gene_idf(fit_rows)[gene_index] if gene_weight == "idf" else None
    fit_block = _apply_row_shape(fit_rows[:, gene_index], weight, row_norm)

    # 성분 수는 열 수보다 작아야 한다 (`TruncatedSVD` 의 요구).
    k = max(int(min(n_components, gene_index.size - 1)), 1)

    if method == "svd":
        estimator = TruncatedSVD(
            n_components=k, algorithm="randomized", random_state=random_state
        )
    else:
        estimator = NMF(
            n_components=k, init="nndsvd", max_iter=200, random_state=random_state
        )
    estimator.fit(fit_block)
    components = np.asarray(estimator.components_, dtype=np.float32)

    stats = [
        {
            "component": int(i),
            "n_genes": int(gene_index.size),
            "top_genes": [
                str(gene_names[gene_index[j]])
                for j in np.argsort(-np.abs(components[i]))[:20]
            ],
        }
        for i in range(components.shape[0])
    ]

    return LatentBasis(
        method=method,
        components=components,
        gene_index=gene_index,
        gene_weight=weight,
        row_norm=row_norm,
        mode=mode,
        n_components=int(components.shape[0]),
        stats=stats,
    )


def transform_latent_basis(
    matrix: np.ndarray, basis: LatentBasis, *, value: str = "proj"
) -> np.ndarray:
    """행을 잠재 좌표로 옮긴다. `fit` 도 `y` 도 없다.

    행 하나만 넣어도 전체를 넣었을 때와 그 행의 값이 같다 — `basis` 에 담긴 건 전부
    fold-train 에서 고정된 값이고, 나머지는 그 행 자신만 본다. NMF 도 마찬가지다:
    `components` 가 고정된 상태에서 W 를 푸는 건 행마다 독립인 문제라 다른 행이 값에
    못 끼어든다 (`tests/test_fold_fit_only.py` 가 svd·nmf 둘 다 검증한다).
    """
    n_rows = np.asarray(matrix).shape[0]
    if basis.n_components == 0:
        return np.zeros((n_rows, 0), dtype=np.float32)
    if value not in ("proj", "share"):
        raise ValueError(f"알 수 없는 value: {value}")

    binary = _binarize(matrix, basis.mode)
    block = _apply_row_shape(
        binary[:, basis.gene_index], basis.gene_weight, basis.row_norm
    )

    if basis.method == "svd":
        projected = np.asarray(block @ basis.components.T, dtype=np.float32)
    else:
        projected = _nmf_transform(block, basis.components)

    if value == "share":
        return _row_l1_share(projected)
    return projected.astype(np.float32)


def _nmf_transform(
    block: sparse.csr_matrix, components: np.ndarray, *, n_iter: int = 200
) -> np.ndarray:
    """고정된 `components` 에 대해 W >= 0 을 푼다 (곱셈 갱신).

    행마다 독립인 문제라 한 행만 넣어도 값이 같다. `sklearn.decomposition.NMF.transform`
    과 같은 계산이지만, 추정기 객체 대신 `components` 배열만 들고 다니면 `LatentBasis`
    가 그대로 직렬화·비교 가능해진다.
    """
    n_rows = block.shape[0]
    k = components.shape[0]
    H = components.astype(np.float64)
    W = np.full((n_rows, k), 1.0 / max(k, 1), dtype=np.float64)
    HHt = H @ H.T
    XHt = np.asarray(block @ H.T, dtype=np.float64)
    for _ in range(n_iter):
        W *= XHt / np.maximum(W @ HHt, 1e-9)
    return W.astype(np.float32)


def build_fold_latent_block(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence,
    *,
    gene_names: Sequence[str],
    prefix: str = "lat__",
    value: str = "proj",
    method: str = "svd",
    n_components: int = 64,
    mode: str = "mutated",
    row_norm: str = "l2",
    gene_weight: str = "none",
    min_gene_support: int = 5,
    random_state: int = 0,
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict], LatentBasis]:
    """한 fold 분의 잠재 블록 -> (열 이름, train dense, test dense, 진단, 기저).

    `features_graph.build_fold_comutation_block` 과 같은 계약이다 — fit 은 전부
    `train_matrix[train_index]` 에서만 하고, 반환하는 train 행렬은 valid 행을 포함한
    **전체 train 행**이다.

    comut 과 달리 fit 결과(`LatentBasis`)를 5번째로 돌려준다. 호출부가 fold 간
    `subspace_alignment` 를 재려면 기저가 필요한데, 안 돌려주면 호출부가 `fit` 을 한 번
    더 부르게 되고 그게 이 블록에서 제일 비싼 연산이다(NMF fold 당 30~60초).

    `y_train_fold` 는 계약 통일성 때문에 받지만 **진단에만 쓴다.** 기저 자체는 라벨을
    안 본다 — `fit_latent_basis` 에 `y` 인자가 아예 없다.
    """
    basis = fit_latent_basis(
        train_matrix,
        train_index,
        gene_names=gene_names,
        method=method,
        n_components=n_components,
        mode=mode,
        row_norm=row_norm,
        gene_weight=gene_weight,
        min_gene_support=min_gene_support,
        random_state=random_state,
    )
    names = [f"{prefix}{method}__c{i:03d}" for i in range(basis.n_components)]
    train_out = transform_latent_basis(train_matrix, basis, value=value)
    test_out = transform_latent_basis(test_matrix, basis, value=value)
    diagnostics = _latent_diagnostics(
        basis, train_matrix, train_index, y_train_fold, names, train_out, test_out
    )
    return names, train_out, test_out, diagnostics, basis


def _latent_diagnostics(
    basis: LatentBasis,
    train_matrix: np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence,
    names: list[str],
    train_out: np.ndarray,
    test_out: np.ndarray,
) -> list[dict]:
    """선택이 끝난 뒤에만 계산하는 감사 기록. 어떤 필터도 이 값을 읽지 않는다.

    `burden_corr` 가 1 에 가까운 성분은 변이 부담(TMB) 축이다 — `features_graph` 가
    기록한 축퇴를 새 좌표계에서 재현한 것이라 그 자체로 기각 근거가 된다.
    `ratio` 는 시프트 노출도이고 설계 목표는 1.0 이다.
    """
    if not names:
        return []

    fit_rows = np.asarray(train_index)
    burden = np.asarray(_binarize(train_matrix, basis.mode).sum(axis=1)).ravel()
    burden = burden[fit_rows].astype(np.float64)
    fold_out = train_out[fit_rows]
    test_arr = np.asarray(test_out)
    y_arr = np.asarray(y_train_fold)
    classes = np.unique(y_arr)

    diagnostics = []
    for i, name in enumerate(names):
        column = fold_out[:, i].astype(np.float64)
        varying = column.std() > 1e-12
        burden_corr = (
            float(np.corrcoef(column, burden)[0, 1])
            if varying and burden.std() > 1e-12
            else 0.0
        )
        train_mean_abs = float(np.abs(column).mean())
        test_mean_abs = float(np.abs(test_arr[:, i]).mean())
        # 클래스 분리도 — 클래스별 평균의 분산 / 전체 분산. `y` 를 쓰는 유일한 자리이고
        # 기저 fit 이 끝난 뒤라 선택에 못 닿는다.
        separation = (
            float(
                np.array([column[y_arr == c].mean() for c in classes]).var()
                / column.var()
            )
            if varying
            else 0.0
        )
        diagnostics.append(
            {
                "name": name,
                "component": i,
                "burden_corr": burden_corr,
                "train_mean_abs": train_mean_abs,
                "test_mean_abs": test_mean_abs,
                "ratio": (test_mean_abs / train_mean_abs) if train_mean_abs > 0 else None,
                "class_separation": separation,
                "top_genes": basis.stats[i]["top_genes"] if i < len(basis.stats) else [],
            }
        )
    return diagnostics


def subspace_alignment(a: LatentBasis, b: LatentBasis) -> float:
    """두 fold 기저의 주각 코사인 평균. 1 에 가까우면 같은 부분공간이다.

    열 이름이 매 fold `lat__svd__c000` 으로 고정이라 `train_gbdt._selection_overlap` 의
    Jaccard 는 항상 1.000 을 낸다 — 안정성을 모르는 대상에 대해 거짓말하는 숫자다.
    이름이 고정된 기저의 안정성은 이쪽으로 잰다.
    """
    if a.n_components == 0 or b.n_components == 0:
        return 0.0
    # 두 fold 의 기저가 서로 다른 유전자 집합에 살 수 있다. 공통 축으로 올려서 비교한다.
    width = max(int(a.gene_index.max()), int(b.gene_index.max())) + 1
    lifted = []
    for basis in (a, b):
        full = np.zeros((basis.n_components, width), dtype=np.float64)
        full[:, basis.gene_index] = basis.components
        lifted.append(normalize(full, norm="l2", axis=1))
    k = min(lifted[0].shape[0], lifted[1].shape[0])
    singular = np.linalg.svd(lifted[0][:k] @ lifted[1][:k].T, compute_uv=False)
    return float(np.clip(singular, 0.0, 1.0).mean())


# ------------------------------------------------------------------ L3 하드 모듈
@dataclass(frozen=True)
class GeneModules:
    """`fit_gene_modules` 가 fold-train 에서 fit 한 배타적 유전자 분할.

    `labels[j]` 는 `gene_index[j]` 유전자가 속한 모듈 번호다. 지지도 미달로 빠진
    유전자는 `gene_index` 에 아예 없다 — 어느 모듈에도 안 들어간다.
    """

    labels: np.ndarray
    gene_index: np.ndarray
    module_sizes: np.ndarray
    gene_weight: np.ndarray
    distances: np.ndarray
    n_modules: int
    mode: str
    n_genes_total: int
    stats: list[dict]


def fit_gene_modules(
    train_matrix: np.ndarray,
    train_index: np.ndarray,
    *,
    gene_names: Sequence[str],
    n_modules: int = 24,
    svd_components: int = 64,
    mode: str = "mutated",
    min_gene_support: int = 5,
    random_state: int = 0,
) -> GeneModules:
    """유전자를 공변이 프로파일로 군집한다. **test 도 `y` 도 인자로 받지 않는다.**

    순서: 이진화 -> 지지도로 유전자 추림 -> **유전자를 행으로** 전치 -> L2 정규화 ->
    SVD 축소 -> **다시** L2 정규화 -> KMeans.

    두 번 정규화하는 게 요점이다. 전치 직후 정규화를 빼면 군집이 공변이 패턴이 아니라
    변이 빈도를 따라 생기고, SVD 뒤에 다시 안 하면 성분 1의 크기(=빈도)가 거리를
    지배한다. `features_graph` 가 기록한 `BRAF x passenger` 축퇴와 같은 함정이다.
    """
    binary = _binarize(train_matrix, mode)
    fit_rows = binary[np.asarray(train_index)]
    gene_index = _support_index(fit_rows, min_gene_support)
    n_genes_total = binary.shape[1]
    idf = _gene_idf(fit_rows)

    if gene_index.size < 2:
        return GeneModules(
            labels=np.zeros(0, dtype=np.int64),
            gene_index=gene_index,
            module_sizes=np.zeros(0, dtype=np.int64),
            gene_weight=idf,
            distances=np.zeros(0, dtype=np.float32),
            n_modules=0,
            mode=mode,
            n_genes_total=n_genes_total,
            stats=[],
        )

    k = int(min(n_modules, gene_index.size))
    profiles = normalize(fit_rows[:, gene_index].T.tocsr(), norm="l2", axis=1)

    n_svd = int(min(svd_components, min(profiles.shape) - 1))
    if n_svd >= 1:
        reduced = TruncatedSVD(
            n_components=n_svd, algorithm="randomized", random_state=random_state
        ).fit_transform(profiles)
        embedding = normalize(reduced, norm="l2", axis=1)
    else:
        embedding = np.asarray(profiles.todense(), dtype=np.float32)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(embedding)
    labels = kmeans.labels_.astype(np.int64)
    distances = np.linalg.norm(
        embedding - kmeans.cluster_centers_[labels], axis=1
    ).astype(np.float32)
    module_sizes = np.bincount(labels, minlength=k).astype(np.int64)

    stats = []
    for m in range(k):
        members = np.where(labels == m)[0]
        order = members[np.argsort(distances[members])]
        stats.append(
            {
                "module": int(m),
                "size": int(module_sizes[m]),
                "top_genes": [str(gene_names[gene_index[j]]) for j in order[:20]],
            }
        )

    return GeneModules(
        labels=labels,
        gene_index=gene_index,
        module_sizes=module_sizes,
        gene_weight=idf,
        distances=distances,
        n_modules=k,
        mode=mode,
        n_genes_total=n_genes_total,
        stats=stats,
    )


def transform_gene_modules(
    matrix: np.ndarray, modules: GeneModules, *, value: str = "share"
) -> np.ndarray:
    """모듈별 집계. `fit` 도 `y` 도 없고, 행 하나만 넣어도 그 행의 값이 같다."""
    n_rows = np.asarray(matrix).shape[0]
    if modules.n_modules == 0:
        return np.zeros((n_rows, 0), dtype=np.float32)

    binary = _binarize(matrix, modules.mode)
    # 배타적 배정을 전체 유전자 축의 멤버십 행렬로 편다 — 지지도 미달로 빠진 유전자는
    # 행이 통째로 0 이라 어느 모듈에도 안 센다.
    membership = sparse.csr_matrix(
        (
            np.ones(modules.labels.size, dtype=np.float32),
            (modules.gene_index, modules.labels),
        ),
        shape=(modules.n_genes_total, modules.n_modules),
    )
    return aggregate_by_membership(
        binary,
        membership,
        group_sizes=modules.module_sizes,
        gene_weight=modules.gene_weight,
        n_genes_total=modules.n_genes_total,
        value=value,
    )


def build_fold_module_block(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence,
    *,
    gene_names: Sequence[str],
    prefix: str = "mod_",
    value: str = "share",
    n_modules: int = 24,
    svd_components: int = 64,
    mode: str = "mutated",
    min_gene_support: int = 5,
    random_state: int = 0,
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict], GeneModules]:
    """한 fold 분의 하드 모듈 블록 -> (열 이름, train dense, test dense, 진단, 모듈).

    `build_fold_latent_block` 과 같은 계약이다 — 5번째로 fit 결과를 돌려주는 것도
    같은 이유다(호출부가 모듈맵 CSV 를 쓰려고 KMeans 를 두 번 돌리지 않게).
    `y_train_fold` 는 진단에만 쓴다.
    """
    modules = fit_gene_modules(
        train_matrix,
        train_index,
        gene_names=gene_names,
        n_modules=n_modules,
        svd_components=svd_components,
        mode=mode,
        min_gene_support=min_gene_support,
        random_state=random_state,
    )
    names = [f"{prefix}{value}__m{m:03d}" for m in range(modules.n_modules)]
    train_out = transform_gene_modules(train_matrix, modules, value=value)
    test_out = transform_gene_modules(test_matrix, modules, value=value)

    diagnostics: list[dict] = []
    if names:
        fold_out = train_out[np.asarray(train_index)]
        test_arr = np.asarray(test_out)
        for m, name in enumerate(names):
            column = fold_out[:, m].astype(np.float64)
            train_mean = float(np.abs(column).mean())
            test_mean = float(np.abs(test_arr[:, m]).mean())
            diagnostics.append(
                {
                    "name": name,
                    "module": m,
                    "size": int(modules.module_sizes[m]),
                    "train_mean": train_mean,
                    "test_mean": test_mean,
                    "ratio": (test_mean / train_mean) if train_mean > 0 else None,
                    "train_nonzero_rate": float((column > 0).mean()),
                    "test_nonzero_rate": float((test_arr[:, m] > 0).mean()),
                    "top_genes": modules.stats[m]["top_genes"],
                }
            )
    return names, train_out, test_out, diagnostics, modules


def module_membership_frame(modules: GeneModules, gene_names: Sequence[str]):
    """모듈맵 CSV 용 프레임. `train_gbdt.py` 가 fold 별로 쓴다."""
    import pandas as pd

    return pd.DataFrame(
        {
            "module": modules.labels,
            "gene": [str(gene_names[i]) for i in modules.gene_index],
            "distance_to_centroid": modules.distances,
            "module_size": modules.module_sizes[modules.labels],
        }
    ).sort_values(["module", "distance_to_centroid"], ignore_index=True)
