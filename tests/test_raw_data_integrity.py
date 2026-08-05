"""원본 csv 가 우리가 아는 그 파일인지.

`data/raw/` 는 git 에 없다(용량·규정). 각자 대회 페이지에서 받아 놓기 때문에, 누가 다른
버전을 받아 놨거나 실수로 덮어썼을 때 **파이프라인은 조용히 돌고 점수만 이상해진다.**
그 상태로 뽑은 OOF 를 팀 블렌드에 넣으면 원인을 못 찾는다.

원본이 없는 환경(CI·클론 직후)에서는 자연히 skip 된다.
"""

from __future__ import annotations

import pandas as pd

from conftest import requires_raw

N_TRAIN = 6201
N_TEST = 2546
N_GENES = 4384
N_CLASSES = 26


def _read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(requires_raw(name), **kwargs)


def test_train_shape():
    train = _read("train.csv", dtype=str, na_filter=False, nrows=5)
    assert train.shape[1] == N_GENES + 2, f"열 {train.shape[1]}, 기대 {N_GENES + 2}"
    assert list(train.columns[:2]) == ["ID", "SUBCLASS"]
    assert len(_read("train.csv", usecols=["ID"])) == N_TRAIN


def test_test_shape():
    test = _read("test.csv", dtype=str, na_filter=False, nrows=5)
    assert test.shape[1] == N_GENES + 1, f"열 {test.shape[1]}, 기대 {N_GENES + 1}"
    assert test.columns[0] == "ID"
    assert len(_read("test.csv", usecols=["ID"])) == N_TEST


def test_gene_columns_match_between_train_and_test():
    """유전자 열이 어긋나면 피처 정렬이 조용히 깨진다 — 순서까지 같아야 한다."""
    train = _read("train.csv", dtype=str, na_filter=False, nrows=1)
    test = _read("test.csv", dtype=str, na_filter=False, nrows=1)
    assert list(train.columns[2:]) == list(test.columns[1:])


def test_ids_are_unique_and_disjoint():
    train = _read("train.csv", usecols=["ID"], dtype=str)
    test = _read("test.csv", usecols=["ID"], dtype=str)
    assert train["ID"].is_unique and test["ID"].is_unique
    assert not (set(train["ID"]) & set(test["ID"])), "train 과 test 에 같은 ID 가 있다"


def test_class_count_and_extremes():
    """26클래스 · 최소 DLBC 38행 · 최대 BRCA 786행.

    분포가 바뀌면 `balanced` 샘플 가중과 fold 층화가 달라져 예전 점수와 비교가 끊긴다.
    """
    counts = _read("train.csv", usecols=["SUBCLASS"])["SUBCLASS"].value_counts()
    assert len(counts) == N_CLASSES
    assert counts.min() == 38 and counts.idxmin() == "DLBC"
    assert counts.max() == 786 and counts.idxmax() == "BRCA"


def test_sample_submission_matches_test():
    """제출은 ID 로 병합하지만, 애초에 두 파일의 ID 집합이 같아야 한다."""
    sample = _read("sample_submission.csv", dtype=str)
    test = _read("test.csv", usecols=["ID"], dtype=str)
    assert list(sample.columns) == ["ID", "SUBCLASS"]
    assert len(sample) == N_TEST
    assert set(sample["ID"]) == set(test["ID"])


def test_wt_is_the_dominant_value():
    """정상은 `WT` 다. 이 표기가 바뀌면 모든 파서와 인코더가 조용히 틀린다."""
    train = _read("train.csv", dtype=str, na_filter=False, nrows=200)
    values = train[train.columns[2:]].to_numpy().ravel()
    assert (values == "WT").mean() > 0.9, "앞 200행에서 WT 비율이 90% 미만이다"
