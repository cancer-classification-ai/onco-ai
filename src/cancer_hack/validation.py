"""교차검증 fold 생성과 fold 안에서만 fit 하는 선택기.

대회 규정이 콕 집어 금지한 것 — 인코더·스케일러·집계 통계를 평가 데이터로 fit 하는 것.
그래서 fold 내부에서만 fit 해야 하는 물건을 여기 모아 두고 `fit`/`transform` 을 갈라
놨다. 실수하려면 일부러 해야 한다.

## 왜 그룹 CV 가 필요한가

이 데이터에는 변이 프로파일이 **완전히 같은** 샘플들이 있다. 단순 `StratifiedKFold` 를
쓰면 같은 프로파일이 train 과 valid 로 갈라져 사실상 정답을 외운 채 맞히게 된다.
`StratifiedGroupKFold` 로 같은 프로파일을 한 fold 에 묶으면 그 누수가 사라진다.

두 방식을 **둘 다 기록**하는 것이 이 저장소의 관행이다. 어느 쪽이 진실에 가까운지는
리더보드가 알려 주고, 그 전까지는 두 숫자의 간격 자체가 정보다.

## 그룹 키는 왜 단순 해시인가

유전자 이진 프로파일의 Jaccard 유사도 0.999 이상을 union-find 로 묶는 방식도 있고
`analysis/exp_fe_blocks_xgb.py` 에 구현돼 있다. 실측해 보니 **결과가 완전히 같았다** —
Jaccard 로 찾은 422쌍이 전부 이미 문자열이 동일한 중복이었고, 최종 분할이 5,636그룹 /
다중 멤버 451개 / 최대 94명으로 두 방식이 일치했다. union-find 는 6,201² 밀집 행렬을
세 개 만들어 피크 1.9GB 를 쓰는데 기여가 정확히 0이다. 그래서 기본은 해시다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SEED = 42
META_COLUMNS = ("ID", "SUBCLASS")
CVKind = Literal["skf", "sgkf"]

#: fold 파일·아티팩트 이름에 쓰는 약칭. 기존 `oof_..._group5_...csv` 규약을 잇는다.
CV_SLUG: dict[str, str] = {"skf": "skf5", "sgkf": "group5"}


# ---------------------------------------------------------------- 그룹 키
def mutation_profile_group_keys(
    frame: pd.DataFrame,
    gene_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """변이 프로파일이 완전히 같은 행에 같은 정수를 준다.

    `frame` 은 원본 csv 를 `dtype=str, na_filter=False` 로 읽은 것이어야 한다.

    `DataFrame.apply(axis=1)` 로 join 하면 실측 80초가 걸린다. numpy object 배열을
    직접 도는 쪽이 8.6초다.

    >>> f = pd.DataFrame({"A": ["WT", "WT", "R1H"], "B": ["Q2*", "Q2*", "WT"]})
    >>> keys = mutation_profile_group_keys(f)
    >>> keys[0] == keys[1], keys[0] == keys[2]
    (np.True_, np.False_)
    """
    if gene_columns is None:
        gene_columns = [c for c in frame.columns if c not in META_COLUMNS]
    values = frame[list(gene_columns)].to_numpy(dtype=object)
    # \x1f (unit separator) — 변이 문자열에 절대 나오지 않는 구분자
    joined = np.array(["\x1f".join(row) for row in values], dtype=object)
    codes, _ = pd.factorize(joined)
    return codes.astype(np.int64)


def build_group_keys(
    raw_csv_path: str | Path,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """원본 csv 에서 `ID, group_key` 프레임을 만든다. 캐시가 있으면 읽는다."""
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_parquet(cache_path)

    frame = pd.read_csv(raw_csv_path, dtype=str, na_filter=False)
    out = pd.DataFrame(
        {
            "ID": frame["ID"].astype(str),
            "group_key": mutation_profile_group_keys(frame),
        }
    )
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def twin_group_keys_jaccard(
    gene_binary: np.ndarray,
    base_keys: np.ndarray,
    *,
    threshold: float = 0.999,
    min_genes: int = 3,
    block: int = 1024,
) -> np.ndarray:
    """Jaccard 유사도로 근사 쌍둥이까지 묶는다. **기본 경로가 아니다.**

    현재 데이터에서는 `mutation_profile_group_keys` 와 결과가 같아 쓸 이유가 없다.
    파서나 전처리가 바뀌어 근사 중복이 생기면 다시 재 볼 값어치가 있어 남겨 둔다.

    원본(`analysis/exp_fe_blocks_xgb.py`)은 밀집 행렬 세 개를 동시에 들고 있어
    피크 1.9GB 를 썼다. 여기서는 `block` 행씩 잘라 계산한다.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 같은 base_key 끼리 먼저 묶는다 (문자열 완전 일치)
    order = pd.Series(base_keys)
    for _, positions in order.groupby(order).groups.items():
        positions = list(positions)
        for other in positions[1:]:
            union(int(positions[0]), int(other))

    counts = gene_binary.sum(axis=1)
    candidates = np.where(counts >= min_genes)[0]
    subset = gene_binary[candidates].astype(np.float32)
    sizes = counts[candidates].astype(np.float32)

    for start in range(0, len(candidates), block):
        stop = min(start + block, len(candidates))
        inter = subset[start:stop] @ subset.T
        union_size = sizes[start:stop, None] + sizes[None, :] - inter
        jaccard = np.divide(
            inter, union_size, out=np.zeros_like(inter), where=union_size > 0
        )
        for local, global_row in enumerate(range(start, stop)):
            hits = np.where(jaccard[local] >= threshold)[0]
            for hit in hits:
                if hit != global_row:
                    union(int(candidates[global_row]), int(candidates[hit]))

    groups = np.arange(len(gene_binary), dtype=np.int64)
    for node in list(parent):
        groups[node] = find(node)
    return groups


# ---------------------------------------------------------------- fold
def make_splitter(kind: CVKind, n_splits: int = 5, seed: int = SEED):
    """`skf` 는 StratifiedKFold, `sgkf` 는 StratifiedGroupKFold."""
    if kind == "skf":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if kind == "sgkf":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"알 수 없는 CV 방식: {kind!r} (skf 또는 sgkf)")


def check_all_classes_present(
    y: np.ndarray, train_index: np.ndarray, classes: Sequence[str]
) -> None:
    """fold 의 train 부분에 26개 클래스가 다 있어야 한다.

    빠지면 그 fold 모델의 `classes_` 가 짧아지고, 확률 열이 밀려서 예외 없이 조용히
    오염된다. 소수 클래스(DLBC 38명)가 있는 데이터라 실제로 일어날 수 있다.
    """
    missing = sorted(set(map(str, classes)) - set(map(str, np.unique(y[train_index]))))
    if missing:
        raise ValueError(f"fold train 에 없는 클래스: {missing}")


def check_no_group_leakage(
    train_index: np.ndarray, valid_index: np.ndarray, groups: np.ndarray
) -> None:
    """같은 그룹이 train 과 valid 에 동시에 있으면 안 된다."""
    shared = np.intersect1d(groups[train_index], groups[valid_index])
    if shared.size:
        raise ValueError(
            f"train/valid 에 걸친 그룹 {shared.size}개 (예: {shared[:5].tolist()})"
        )


def iter_folds(
    y: Sequence,
    *,
    kind: CVKind,
    n_splits: int = 5,
    seed: int = SEED,
    groups: np.ndarray | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """`(fold_index, train_index, valid_index)` 를 낸다.

    fold 마다 클래스 누락과 그룹 누수를 강제로 검사한다. 호출부가 잊어버릴 수 없게
    여기서 본다.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if kind == "sgkf" and groups is None:
        raise ValueError("sgkf 에는 groups 가 필요하다")

    splitter = make_splitter(kind, n_splits=n_splits, seed=seed)
    placeholder = np.zeros(len(y))
    splits = (
        splitter.split(placeholder, y, groups=groups)
        if kind == "sgkf"
        else splitter.split(placeholder, y)
    )
    for fold, (train_index, valid_index) in enumerate(splits):
        check_all_classes_present(y, train_index, classes)
        if groups is not None and kind == "sgkf":
            check_no_group_leakage(train_index, valid_index, groups)
        yield fold, train_index, valid_index


def fold_assignment(
    y: Sequence,
    *,
    kind: CVKind,
    n_splits: int = 5,
    seed: int = SEED,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """행마다 속한 valid fold 번호. 저장해 두면 모든 모델이 같은 fold 를 쓴다."""
    assignment = np.full(len(y), -1, dtype=np.int8)
    for fold, _, valid_index in iter_folds(
        y, kind=kind, n_splits=n_splits, seed=seed, groups=groups
    ):
        assignment[valid_index] = fold
    if (assignment < 0).any():
        raise RuntimeError("어느 fold 에도 안 들어간 행이 있다")
    return assignment


# ---------------------------------------------------------------- 선택기
class Chi2TopKSelector:
    """chi2 상위 K개 열을 고른다. **fold 의 train 부분에서만 fit 한다.**

    `features_basic.BurdenBinner` 와 같은 계약이다 — fit 과 transform 을 갈라 놔서
    전체 데이터로 고르는 실수를 구조적으로 막는다.

    `k=None` 이면 전부 통과시킨다. 덕분에 "top-K 구성"과 "전체 구성"이 분기 없이
    같은 코드 경로를 탄다.

    chi2 는 음수를 못 받는다. 유전자 3단계 인코딩(0/1/2)도 L2 정규화된 TF-IDF 도
    조건을 만족한다.

    희소 입력(`csr_matrix`)을 densify 하지 않고 그대로 넘긴다. TF-IDF 블록은
    6,201 x 20,912 를 내는데 dense 로 바꾸면 메모리가 아깝고, 그전에
    `np.asarray(csr)` 이 0-d object 배열을 만들어 `X.min()` 에서 터진다.
    출력 타입은 입력을 따라간다 — 희소를 넣으면 희소가 나온다.

    >>> X = np.array([[0, 1], [2, 1], [0, 1], [2, 1]])
    >>> sel = Chi2TopKSelector(k=1).fit(X, ["a", "b", "a", "b"])
    >>> sel.transform(X).shape
    (4, 1)
    """

    def __init__(self, k: int | None = 500) -> None:
        if k is not None and k <= 0:
            raise ValueError(f"k 는 양수여야 한다: {k}")
        self.k = k
        self.indices_: np.ndarray | None = None
        self.scores_: np.ndarray | None = None

    def fit(self, X, y) -> "Chi2TopKSelector":
        X = X if issparse(X) else np.asarray(X)
        if X.shape[0] and X.shape[1] and X.min() < 0:
            raise ValueError("chi2 는 음수 피처를 받지 않는다")
        if self.k is None:
            self.indices_ = np.arange(X.shape[1])
            self.scores_ = None
            return self

        scores, _ = chi2(X, np.asarray(y))
        # fold 안에서 전부 0인 열은 NaN 을 낸다. train 전체에도 무변이 유전자가 154개다.
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        k = min(self.k, X.shape[1])
        # stable 정렬 — 동점일 때 순서가 매번 같아야 재현된다.
        self.indices_ = np.sort(np.argsort(-scores, kind="stable")[:k])
        self.scores_ = scores
        return self

    def transform(self, X):
        if self.indices_ is None:
            raise RuntimeError("fit() 을 먼저 부른다")
        if issparse(X):
            return X[:, self.indices_]
        return np.asarray(X)[:, self.indices_]
