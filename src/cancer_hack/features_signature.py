"""클래스 서명 피처 — 유전자를 **fold-train 의 라벨로** 클래스별 집합에 묶는다.

`features_latent` 의 두 블록은 비지도라 `fit` 시그니처에 `y` 가 아예 없다. 이쪽은
반대로 `y` 가 필수 인자다. 그 차이가 이 파일이 따로 있는 이유의 전부다 — 지도 블록은
누출 경로가 하나 더 있고, 그 경로를 여기 한 곳에 가둬 둔다.

## 무엇을 만드는가

클래스 c 마다 fold-train 에서 "c 에서 유난히 자주 변이되는 유전자" 를 골라 집합
`S_c` 를 만든다. 그러면 26개 클래스가 26열이 되고, 한 행의 c 번째 값은

    signature_share(c) = |변이 유전자 & S_c| / |변이 유전자 전체|

다. 분모가 **행 전체의 변이 유전자 수**인 게 요점이다 — test 는 변이 유전자가 train 의
2.21배(35.3 -> 78.1)라 단순 카운트는 그대로 부푼다. 분자·분모가 같이 부푸는 형태여야
값이 안 움직인다. 집계식은 `features_latent.aggregate_by_membership` 한 곳에만 있고
7가지 값 방식(share/enrich/wshare + 대조군 4종)의 정의와 시프트 배율 표도 거기 있다.

## 선택 규칙 — `features_graph` 의 쌍 선별과 같은 규약

    지지도    n_mut(g, c) >= min_class_support     클래스 안에서 몇 행이나 변이됐나
    lift      rate_in(c) - rate_out(c) >= min_lift  나머지 클래스 대비 얼마나 더 흔한가
    과변이    hyper_fraction <= max_hyper_fraction  그 변이가 과변이 샘플에 몰렸나
    점수      lift * sqrt(n_mut(g, c))              흔하면서 대비가 큰 순

덧셈 lift 를 쓰는 것도, 점수를 `lift * sqrt(n)` 으로 두는 것도 `select_comutation_pairs`
와 같다. 규약을 두 개 만들면 두 블록의 파라미터를 나란히 못 읽는다.

과변이 가드가 있어야 하는 이유는 `share` 분모만으로 안 되기 때문이다. 분모는 **행**
쪽 부담을 지우지만, 선택은 **유전자** 쪽에서 일어난다 — 어떤 클래스의 과변이 샘플
몇 개가 passenger 유전자를 잔뜩 켜 두면 그 유전자들의 `rate_in` 이 올라가서 서명이
"이 클래스는 변이가 많다" 를 다시 배운다. `features_graph` 가 `BRAF x 긴 passenger`
로 겪은 축퇴와 같은 것이다.

## 누출 — 이 블록에서 제일 조심할 자리

fold-train 행렬과 fold-train 라벨만 본다. `select_class_signatures` 는 test 행렬을
인자로 **받지 않고**, 반환한 멤버십은 상수라 `transform` 은 행마다 독립이다. 여기까지는
chi2 top-K 선택(`Chi2TopKSelector`)이나 comut 쌍 선별과 정확히 같은 계약이다.

다만 한 가지가 다르고, 그게 이 블록을 두 CV 로 재야 하는 이유다. **프로파일이 같은
쌍둥이 행**이 train 6,201행 중 1,016행(16.4%)이고, 그중 **842행**은 자기 그룹이 skf 의
fold 경계를 가로지른다(그런 그룹이 364개). 그 842행은 valid 로 갈 때 자기와 변이
프로파일이 똑같은 행의 라벨이 이미 서명에 반영된 상태다. 유전자 단위 chi2 보다
이쪽이 셀 여지가 크다 — chi2 는 유전자를 고르고 말지만 서명은 라벨을 26열에 직접
인코딩하기 때문이다. sgkf(Profile Group CV)는 그 경로를 막는다.

**실측은 그 예상과 반대 방향이었다.** f4r 대비 델타(시드 42/7/2024):

    skf    +0.0054 / +0.0041 / +0.0054   평균 +0.0050
    sgkf   +0.0179 / +0.0079 / +0.0209   평균 +0.0156

쌍둥이 경로를 막은 쪽이 오히려 3배 크다. 즉 이 블록의 이득은 쌍둥이 라벨 되읽기가
아니다. 어디서 오는지는 클래스별 F1 이 말해 준다 — sgkf 기준 **DLBC +0.26(n=38),
ACC +0.12(n=72)** 이고 나머지 24개 클래스의 변화를 다 더해도 그 둘에 못 미친다.
macro F1 은 26개 클래스를 같은 무게로 세므로 이 둘만으로 +0.0147 이 설명된다.

기전도 그대로 읽힌다. `enc3` 의 chi2 top-500 은 유전자를 **전 클래스 통합**으로
줄세우므로 38행짜리 DLBC 를 가리키는 유전자는 상위에 못 올라온다. 서명은 클래스마다
자기 몫(최대 `topk`개)을 따로 갖는다 — 희소 클래스가 처음으로 자기 열을 얻는다.
실제로 fold 0 의 DLBC 서명은 BTG2·BTG1·PIM1·KMT2D·MYC·B2M 로, 문헌이 아는 DLBCL
드라이버를 **라벨만 보고** 되찾았다.

그래도 판정은 sgkf 로 본다. 이유가 바뀌었을 뿐이다 — 누출이 무서워서가 아니라,
라벨을 보는 블록에서 두 CV 가 갈릴 때 보수적인 쪽을 기준으로 삼는 게 규율이기
때문이다. skf 평균 +0.0050 은 저장소가 쓰는 paired sigma(0.0055~0.0079) 아래라
그 지표만으로는 기권이 맞다. 두 숫자를 다 남긴다.

남은 위험 두 가지는 숫자로 적어 둔다. (1) 그룹 키는 **완전 일치** 프로파일만 묶으므로
한 유전자만 다른 근사 쌍둥이는 sgkf 도 못 막는다. (2) ACC 열은 test/train 평균비가
1.43 이다(26열 중앙값 0.82). `share` 가 행 쪽 부담은 지워도 그 클래스의 서명 유전자가
test 에서 더 자주 켜지는 것까지는 못 막는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .features_latent import _binarize, _gene_idf, aggregate_by_membership

#: 과변이 판정 분위. `features_graph.select_comutation_pairs` 와 같은 값이다.
HYPER_QUANTILE = 0.95


@dataclass(frozen=True)
class ClassSignatures:
    """`select_class_signatures` 가 fold-train 에서 고른 클래스별 유전자 집합.

    `membership` 은 `(n_genes_total, n_classes)` 0/1 이고 한 유전자가 여러 클래스에
    속할 수 있다(하드 모듈의 배타적 분할과 대비되는 지점). `stats` 는 클래스별 진단이라
    변환에 관여하지 않는다.
    """

    classes: np.ndarray
    membership: sparse.csr_matrix
    sizes: np.ndarray
    gene_weight: np.ndarray
    mode: str
    n_genes_total: int
    stats: list[dict]


def select_class_signatures(
    train_matrix: np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence,
    *,
    gene_names: Sequence[str],
    topk: int = 30,
    mode: str = "mutated",
    min_class_support: int = 5,
    min_lift: float = 0.05,
    max_hyper_fraction: float = 0.50,
) -> ClassSignatures:
    """클래스별 서명 유전자를 고른다. **test 행렬을 인자로 받지 않는다.**

    `y_train_fold` 는 `train_index` 와 같은 길이여야 한다 — `train_gbdt.py` 가
    `data.y[train_index]` 를 넘기는 것과 같은 규약이고, 길이가 어긋나면 라벨이 밀린
    채로 조용히 학습되므로 여기서 막는다.

    >>> X = np.array([[1,0,0,0],[1,0,0,0],[0,0,1,0],[0,0,1,0]], dtype=np.float32)
    >>> sig = select_class_signatures(
    ...     X, np.arange(4), ["A","A","B","B"], gene_names=["GA","GB","GC","GD"],
    ...     min_class_support=2, min_lift=0.5,
    ... )
    >>> [s["genes"] for s in sig.stats]
    [['GA'], ['GC']]
    """
    if mode not in ("mutated", "functional"):
        raise ValueError(f"알 수 없는 mode: {mode}")

    fit_rows = np.asarray(train_index)
    y_arr = np.asarray(y_train_fold)
    if y_arr.shape[0] != fit_rows.shape[0]:
        raise ValueError(
            f"y_train_fold 길이 {y_arr.shape[0]} 가 train_index {fit_rows.shape[0]} 와 "
            "다르다. fold 의 train 부분 라벨만 넘긴다 (data.y[train_index])"
        )

    binary = _binarize(train_matrix, mode)
    n_genes_total = binary.shape[1]
    fold_rows = binary[fit_rows]
    idf = _gene_idf(fold_rows)
    classes = np.unique(y_arr)

    # 클래스별 변이 행 수 — 한 번의 (n_genes x n_rows) @ (n_rows x n_classes) 곱.
    indicator = np.zeros((y_arr.shape[0], classes.size), dtype=np.float32)
    for i, c in enumerate(classes):
        indicator[y_arr == c, i] = 1.0
    # 희소 @ 조밀이라 결과는 이미 ndarray 다 (`todense()` 를 부르면 터진다).
    per_class = np.asarray(fold_rows.T @ indicator, dtype=np.float32)
    class_sizes = indicator.sum(axis=0)
    total = per_class.sum(axis=1, keepdims=True)

    rate_in = per_class / np.maximum(class_sizes, 1.0)[None, :]
    rate_out = (total - per_class) / np.maximum(
        y_arr.shape[0] - class_sizes, 1.0
    )[None, :]
    lift = rate_in - rate_out

    # 과변이 의존도 — 클래스 안 변이 행 중 과변이 샘플의 몫. 분위 경계도 fold-train
    # 에서만 잡는다. 변이 **유전자 수**로 재는 건 comut 과 같은 이유다(test 의 전사체
    # 중복 기재로 안 부푸는 축).
    burden = np.asarray(fold_rows.sum(axis=1), dtype=np.float32).ravel()
    threshold = float(np.quantile(burden, HYPER_QUANTILE)) if burden.size else 0.0
    hyper = (burden > threshold).astype(np.float32)
    hyper_hits = np.asarray(
        fold_rows.T @ (indicator * hyper[:, None]), dtype=np.float32
    )
    hyper_fraction = hyper_hits / np.maximum(per_class, 1.0)

    ok = (
        (per_class >= min_class_support)
        & (lift >= min_lift)
        & (hyper_fraction <= max_hyper_fraction)
    )
    score = lift * np.sqrt(np.maximum(per_class, 0.0))

    rows: list[int] = []
    cols: list[int] = []
    stats: list[dict] = []
    for i, c in enumerate(classes):
        candidates = np.where(ok[:, i])[0]
        order = candidates[np.argsort(-score[candidates, i], kind="stable")][
            : max(int(topk), 0)
        ]
        rows.extend(int(g) for g in order)
        cols.extend([i] * order.size)
        stats.append(
            {
                "class": str(c),
                "n_train_rows": int(class_sizes[i]),
                "size": int(order.size),
                "genes": [str(gene_names[g]) for g in order],
                "lift": [float(lift[g, i]) for g in order],
                "class_support": [int(per_class[g, i]) for g in order],
                "hyper_fraction": [float(hyper_fraction[g, i]) for g in order],
            }
        )

    membership = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_genes_total, classes.size),
    )
    sizes = np.asarray(membership.sum(axis=0), dtype=np.int64).ravel()

    return ClassSignatures(
        classes=classes,
        membership=membership,
        sizes=sizes,
        gene_weight=idf,
        mode=mode,
        n_genes_total=n_genes_total,
        stats=stats,
    )


def transform_class_signatures(
    matrix: np.ndarray, signatures: ClassSignatures, *, value: str = "share"
) -> np.ndarray:
    """클래스별 집계. `fit` 도 `y` 도 없고, 행 하나만 넣어도 그 행의 값이 같다."""
    n_rows = np.asarray(matrix).shape[0]
    if signatures.classes.size == 0:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return aggregate_by_membership(
        _binarize(matrix, signatures.mode),
        signatures.membership,
        group_sizes=signatures.sizes,
        gene_weight=signatures.gene_weight,
        n_genes_total=signatures.n_genes_total,
        value=value,
    )


def build_fold_signature_block(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence,
    *,
    gene_names: Sequence[str],
    prefix: str = "sig_",
    value: str = "share",
    topk: int = 30,
    mode: str = "mutated",
    min_class_support: int = 5,
    min_lift: float = 0.05,
    max_hyper_fraction: float = 0.50,
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    """한 fold 분의 서명 블록 -> (열 이름, train dense, test dense, 진단 목록).

    `features_latent.build_fold_module_block` 과 같은 계약이다 — 선택은 전부
    `train_matrix[train_index]` 와 `y_train_fold` 에서만 하고, 반환하는 train 행렬은
    valid 행을 포함한 **전체 train 행**이다(fold 루프가 뒤에서 자른다).

    fit 결과를 5번째로 돌려주지 않는 건 잠재·모듈 블록과 다른 점이다. 저쪽은 호출부가
    부분공간 정렬이나 모듈맵 CSV 를 만들려고 기저가 필요했는데, 서명은 고른 유전자
    목록이 이미 진단에 문자열로 들어가 있어서 호출부가 객체를 들고 있을 이유가 없다.
    """
    signatures = select_class_signatures(
        train_matrix,
        train_index,
        y_train_fold,
        gene_names=gene_names,
        topk=topk,
        mode=mode,
        min_class_support=min_class_support,
        min_lift=min_lift,
        max_hyper_fraction=max_hyper_fraction,
    )
    names = [f"{prefix}{value}__{c}" for c in signatures.classes]
    train_out = transform_class_signatures(train_matrix, signatures, value=value)
    test_out = transform_class_signatures(test_matrix, signatures, value=value)

    fold_out = train_out[np.asarray(train_index)]
    test_arr = np.asarray(test_out)
    diagnostics = []
    for i, (name, stat) in enumerate(zip(names, signatures.stats)):
        column = fold_out[:, i].astype(np.float64)
        train_mean = float(np.abs(column).mean())
        test_mean = float(np.abs(test_arr[:, i]).mean())
        diagnostics.append(
            {
                "name": name,
                **stat,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "ratio": (test_mean / train_mean) if train_mean > 0 else None,
                "train_nonzero_rate": float((column > 0).mean()),
                "test_nonzero_rate": float((test_arr[:, i] > 0).mean()),
            }
        )
    return names, train_out, test_out, diagnostics
