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

## 두 갈래 API — 어느 쪽을 쓰나

프로파일 해시의 **정의는 `make_profile_hash` 하나**다(PR#16). 아래
`mutation_profile_group_keys` 는 그 해시를 정수 코드로 접어 주는 얇은 껍데기이고,
학습 파이프라인은 정수 코드 쪽을 쓴다 — `train_gbdt` 의 fold 루프가 0-based 정수
fold 번호를 쓰고 `group_size_inverse_weight` 가 그룹 크기를 세기 때문이다.

fold 생성도 두 갈래다.

    make_stratified_kfold / make_profile_group_kfold   (index 쌍을 내는 제너레이터)
    make_splitter / iter_folds / fold_assignment       (fold 번호 배열을 내는 쪽)

앞쪽은 노트북에서 fold 를 눈으로 확인할 때, 뒤쪽은 `train_folds.parquet` 에
`fold_skf5`·`fold_group5` 두 열을 함께 저장할 때 쓴다. 뒤쪽만 skf 를 만들 수 있고
0-based 라 학습 스크립트가 그대로 인덱싱한다.

## fold 파일은 누가 만드나

`build_fold_frame` 이 그 두 열을 한 프레임으로 묶고, **파일로 쓰는 건
`scripts/make_folds.py` 하나뿐이다.** `scripts/train_gbdt.py` 는 읽기만 한다.
예전에는 학습 스크립트가 파일이 없으면 제 손으로 만들어 저장했는데, 그러면 같은
이름의 파일이 두 경로에서 나오고 `artifacts/oof/` 의 예측이 어느 분할에서
나왔는지 사후에 확인할 방법이 없어진다. fold 방식을 바꿀 일이 생기면
`make_folds.py` 만 다시 돌린다.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

#: fold 분할 시드. 모델 시드와 별개다.
#: **바꾸면 `artifacts/oof/` 의 예측 100여 개가 전부 무효가 된다** — fold 경계가 달라져
#: 예전 OOF 와 새 OOF 를 섞는 순간 교차적합 보정이 valid fold 를 보게 된다.
SEED = 42

#: seed 앙상블에 쓰는 모델 시드. **이 셋을 계속 쓴다.**
#:
#: 지금까지의 3-seed 산출물(`*_seed3_*`, EXP_041 제출본 등)이 전부 이 값으로 나왔다.
#: 여기를 바꾸면 그 기록들과 비교가 끊기고, 무엇보다 코드만 받은 사람이 우리와 다른
#: 숫자를 얻는다. 시드는 취향이 아니라 **재현의 계약**이라 한 곳에 박아 두고 공유한다.
#:
#: 늘릴 때는 뒤에 덧붙인다 — 앞의 셋을 유지해야 기존 OOF 를 그대로 재사용할 수 있다.
SEED_ENSEMBLE: tuple[int, ...] = (42, 7, 2024)

#: seed 분산을 재는 스윕용. 앞 셋은 `SEED_ENSEMBLE` 과 같아야 재사용이 된다.
SEED_SWEEP: tuple[int, ...] = SEED_ENSEMBLE + (1234, 5678)

META_COLUMNS = ("ID", "SUBCLASS")
CVKind = Literal["skf", "sgkf"]
FoldIndex = tuple[pd.Index, pd.Index]

#: fold 파일·아티팩트 이름에 쓰는 약칭. 기존 `oof_..._group5_...csv` 규약을 잇는다.
#: **5-fold 전용이다.** 다른 분할 수로 학습할 때는 `cv_slug(kind, n_splits)` 를 쓴다 —
#: 이 사전을 그대로 쓰면 10-fold 산출물이 `group5` 라는 이름을 달고 나온다.
CV_SLUG: dict[str, str] = {"skf": "skf5", "sgkf": "group5"}

#: fold 파일에서 fold 번호가 아닌 열. 순서가 파일의 앞 두 열 순서다.
FOLD_META_COLUMNS: tuple[str, ...] = ("ID", "group_key")


# ---------------------------------------------------------------- 그룹 키
def make_profile_hash(df: pd.DataFrame, gene_columns: list[str]) -> pd.Series:
    """유전자 변이 벡터를 해시로 변환해 동일 변이 프로파일을 식별한다.

    동일한 유전자 변이 벡터를 가진 행은 같은 해시값을 가진다.
    StratifiedGroupKFold의 groups 인자로 직접 사용할 수 있다.
    """
    return (
        pd.util.hash_pandas_object(df[gene_columns], index=False)
        .astype(str)
        .rename("profile_hash")
        .reset_index(drop=True)
    )


def mutation_profile_group_keys(
    frame: pd.DataFrame,
    gene_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """변이 프로파일이 완전히 같은 행에 같은 정수를 준다.

    해시 자체는 `make_profile_hash` 가 만든다 — 정의를 두 벌 두지 않는다. 여기서는
    그 해시를 **등장 순서대로 정수 코드**로 접기만 한다. 학습 파이프라인이 정수를
    요구해서다(`fold_assignment` 의 0-based fold 번호, `group_size_inverse_weight`
    의 `np.bincount`).

    `frame` 은 원본 csv 를 `dtype=str, na_filter=False` 로 읽은 것이어야 한다.

    >>> f = pd.DataFrame({"A": ["WT", "WT", "R1H"], "B": ["Q2*", "Q2*", "WT"]})
    >>> keys = mutation_profile_group_keys(f)
    >>> keys[0] == keys[1], keys[0] == keys[2]
    (np.True_, np.False_)
    """
    if gene_columns is None:
        gene_columns = [c for c in frame.columns if c not in META_COLUMNS]
    hashed = make_profile_hash(frame, list(gene_columns))
    codes, _ = pd.factorize(hashed)
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


# ------------------------------------------------- fold (PR#16 · 제너레이터)
# 아래 네 함수는 PR#16 원문 그대로다. index 쌍을 내는 제너레이터 계열이라
# 노트북에서 fold 를 눈으로 확인할 때 쓴다. fold 번호는 1-based 이고 group CV
# 하나만 만든다 — 학습 스크립트는 아래 `fold_assignment`(0-based, skf+sgkf)를
# 쓴다. 두 갈래가 같은 프로파일 해시를 보므로 그룹 경계는 서로 같다.
def make_stratified_kfold(
    df: pd.DataFrame,
    gene_columns: list[str],
    label_column: str = "SUBCLASS",
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[FoldIndex]:
    """일반 StratifiedKFold를 생성한다.

    샘플 단위 예측 성능 측정에 사용한다.
    Profile-Grouped CV와 점수 차이를 비교해 데이터 누수 규모를 확인한다.

    Yields:
        (train_index, valid_index) — DataFrame의 정수 위치 인덱스 쌍
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = df[gene_columns].reset_index(drop=True)
    y = df[label_column].reset_index(drop=True)

    for train_pos, valid_pos in skf.split(X, y):
        yield X.index[train_pos], X.index[valid_pos]


def make_profile_group_kfold(
    df: pd.DataFrame,
    gene_columns: list[str],
    label_column: str = "SUBCLASS",
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[FoldIndex]:
    """Profile Hash 기반 StratifiedGroupKFold를 생성한다.

    동일 변이 벡터(profile_hash)를 가진 행들을 항상 같은 Fold에 배치한다.
    Train Fold와 Validation Fold 사이에 동일 변이 벡터가 섞이지 않아
    CV 점수의 낙관적 편향을 방지한다.

    모든-WT 그룹(94행)처럼 매우 큰 그룹이 존재하므로
    Fold별 클래스 분포가 완벽히 균등하지 않을 수 있다.
    반드시 fold_class_distribution()으로 실제 분포를 확인해야 한다.

    Yields:
        (train_index, valid_index) — DataFrame의 정수 위치 인덱스 쌍
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    X = df[gene_columns].reset_index(drop=True)
    y = df[label_column].reset_index(drop=True)
    groups = make_profile_hash(df, gene_columns)

    for train_pos, valid_pos in sgkf.split(X, y, groups=groups):
        yield X.index[train_pos], X.index[valid_pos]


def assign_fold_column(
    df: pd.DataFrame,
    gene_columns: list[str],
    label_column: str = "SUBCLASS",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """각 행에 fold 번호(1-based)를 부여한 DataFrame을 반환한다.

    반환된 DataFrame은 원본과 같은 행 순서를 유지하며
    'fold' 컬럼(int, 1~n_splits)이 추가된다.
    make_folds.py 스크립트에서 Parquet로 저장하는 용도로 사용한다.
    """
    result = df.reset_index(drop=True).copy()
    result["fold"] = -1

    for fold_num, (_, valid_pos) in enumerate(
        make_profile_group_kfold(
            df,
            gene_columns=gene_columns,
            label_column=label_column,
            n_splits=n_splits,
            random_state=random_state,
        ),
        start=1,
    ):
        result.loc[valid_pos, "fold"] = fold_num

    return result


def fold_class_distribution(
    df: pd.DataFrame,
    label_column: str = "SUBCLASS",
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Fold별 클래스 분포를 비율(%)로 반환한다.

    assign_fold_column() 이후 반드시 호출해 Fold 균형을 확인한다.
    반환값: MultiIndex(fold, SUBCLASS) → count, ratio(%) DataFrame
    """
    counts = (
        df.groupby([fold_column, label_column])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["ratio"] = counts.groupby(fold_column)["count"].transform(
        lambda s: s / s.sum() * 100
    )
    return counts.set_index([fold_column, label_column])


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


def fold_column(kind: CVKind, n_splits: int = 5) -> str:
    """fold 파일의 열 이름을 만든다.

    `CV_SLUG` 는 아티팩트 이름 규약이라 5-fold 가 문자열에 박혀 있다. 열 이름은
    실제 분할 수를 따라가야 `--n-splits 10` 으로 만든 파일이 5-fold 인 척하지
    않는다. 기본값에서는 두 규약이 같은 문자열을 낸다.

    >>> fold_column("skf"), fold_column("sgkf")
    ('fold_skf5', 'fold_group5')
    >>> fold_column("sgkf", 10)
    'fold_group10'
    """
    if kind not in CV_SLUG:
        raise ValueError(f"알 수 없는 CV 방식: {kind!r} (skf 또는 sgkf)")
    return f"fold_{'skf' if kind == 'skf' else 'group'}{n_splits}"


def cv_slug(kind: CVKind, n_splits: int = 5) -> str:
    """산출물 이름에 들어갈 약칭. **분할 수를 반영한다.**

    `CV_SLUG` 사전은 5-fold 문자열이 박혀 있어서, `--n-splits 10` 으로 학습해도
    `oof_..._group5_....csv` 라는 이름이 나온다. 그러면 10-fold 예측이 5-fold
    라이브러리에 조용히 섞인다 — `greedy_blend.py` 는 이름으로 후보를 모으므로
    그 순간부터 fold 경계가 어긋난 채로 결합이 돌아간다. 파일만 봐서는 못 알아챈다.

    기본값(5)에서는 기존 규약과 **같은 문자열**을 낸다. 지금까지의 산출물 이름은
    한 글자도 안 바뀐다.

    >>> cv_slug("skf"), cv_slug("sgkf")
    ('skf5', 'group5')
    >>> cv_slug("sgkf", 10), cv_slug("skf", 10)
    ('group10', 'skf10')
    """
    if kind not in CV_SLUG:
        raise ValueError(f"알 수 없는 CV 방식: {kind!r} (skf 또는 sgkf)")
    return f"{'skf' if kind == 'skf' else 'group'}{n_splits}"


def build_fold_frame(
    raw_csv_path: str | Path,
    *,
    label_column: str = "SUBCLASS",
    n_splits: int = 5,
    seed: int = SEED,
    group_cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """원본 csv -> `ID · group_key · fold_skf{n} · fold_group{n}` 프레임.

    **fold 를 만드는 정의는 여기 하나다.** 파일로 떨어뜨리는 건 `make_folds.py`
    뿐이고 `train_gbdt.py` 는 그 파일을 읽기만 한다. 학습 스크립트가 자기 fold 를
    따로 만들면 "이 OOF 가 어느 분할에서 나왔나"를 아무도 답할 수 없게 된다.

    행 순서는 원본 csv 그대로다. 피처 parquet 들이 전부 같은 순서라 소비하는 쪽은
    ID 배열만 대조하면 된다.

    `sgkf` 의 groups 로 `build_group_keys` 의 **정수 코드**를 넘긴다.
    `make_profile_group_kfold` 는 같은 해시를 문자열로 넘기는데,
    `StratifiedGroupKFold` 가 내부에서 `np.unique` 로 그룹을 정렬하므로 문자열과
    정수는 순서가 달라 분할이 갈린다. `artifacts/oof/` 에 쌓인 예측이 전부 정수
    코드 쪽에서 나왔으므로 이 경로를 정본으로 둔다.
    """
    groups = build_group_keys(raw_csv_path, cache_path=group_cache_path)
    labels = pd.read_csv(
        raw_csv_path, usecols=["ID", label_column], dtype=str, na_filter=False
    )

    ids = labels["ID"].astype(str).to_numpy()
    if not np.array_equal(groups["ID"].astype(str).to_numpy(), ids):
        raise ValueError(
            "그룹 키의 ID 순서가 원본 csv 와 다르다. "
            f"캐시({group_cache_path})가 오래된 것일 수 있다."
        )

    y = labels[label_column].to_numpy()
    frame = pd.DataFrame({"ID": ids, "group_key": groups["group_key"].to_numpy()})
    for kind in ("skf", "sgkf"):
        frame[fold_column(kind, n_splits)] = fold_assignment(
            y,
            kind=kind,
            n_splits=n_splits,
            seed=seed,
            groups=frame["group_key"].to_numpy(),
        )
    return frame


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
