"""`artifacts/submissions/` 의 제출 csv 가 전부 규격을 지키는지.

DACON 은 형식이 어긋난 파일을 그냥 거절한다. 하루 4회 제한이 걸린 자원이라 형식 문제로
한 번을 태우면 그날 실험 하나가 통째로 날아간다. 그래서 **만들어 둔 것을 전부** 훑는다.

원본 csv 나 제출 파일이 없는 환경에서는 자연히 skip 된다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import requires_raw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT_ROOT / "artifacts" / "submissions"

N_TEST = 2546
N_CLASSES = 26


def _submissions() -> list[Path]:
    if not SUBMISSIONS.exists():
        return []
    return sorted(SUBMISSIONS.glob("*.csv"))


def _sample() -> pd.DataFrame:
    return pd.read_csv(requires_raw("sample_submission.csv"), dtype=str)


@pytest.mark.parametrize("path", _submissions(), ids=lambda p: p.name)
def test_submission_matches_the_required_schema(path: Path):
    sample = _sample()
    frame = pd.read_csv(path, dtype=str)

    assert list(frame.columns) == ["ID", "SUBCLASS"], f"{path.name} 컬럼"
    assert len(frame) == N_TEST, f"{path.name} 행 수 {len(frame)}"
    # 순서 가정에 기대지 않고 만들지만, 결과물은 sample 과 같은 순서여야 한다.
    assert (frame["ID"].to_numpy() == sample["ID"].to_numpy()).all(), f"{path.name} ID 순서"
    assert frame["SUBCLASS"].notna().all(), f"{path.name} 빈 라벨"
    assert (frame["SUBCLASS"].str.strip() != "").all(), f"{path.name} 공백 라벨"


@pytest.mark.parametrize("path", _submissions(), ids=lambda p: p.name)
def test_submission_labels_are_known_classes(path: Path):
    """train 에 없는 라벨이 들어가면 채점에서 전부 오답이 된다."""
    train = pd.read_csv(requires_raw("train.csv"), usecols=["SUBCLASS"])
    known = set(train["SUBCLASS"].unique())
    assert len(known) == N_CLASSES

    labels = set(pd.read_csv(path, dtype=str)["SUBCLASS"])
    unknown = labels - known
    assert not unknown, f"{path.name} 에 train 에 없는 라벨: {sorted(unknown)[:5]}"


@pytest.mark.parametrize("path", _submissions(), ids=lambda p: p.name)
def test_submission_is_not_single_class(path: Path):
    """한 클래스로 몰린 파일은 파이프라인이 어딘가 끊긴 것이다 — macro F1 이 0.04 가 된다."""
    labels = pd.read_csv(path, dtype=str)["SUBCLASS"]
    assert labels.nunique() >= 10, f"{path.name} 클래스가 {labels.nunique()}종뿐이다"


def test_there_is_at_least_one_submission():
    if not _submissions():
        pytest.skip("artifacts/submissions 에 csv 가 없다")


def test_encoding_is_utf8_sig():
    """대회 베이스라인이 `UTF-8-sig` 를 쓴다. BOM 이 없으면 일부 도구가 첫 컬럼을 깬다."""
    paths = _submissions()
    if not paths:
        pytest.skip("제출 파일 없음")
    missing = [p.name for p in paths if not p.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert not missing, f"BOM 없는 제출 파일: {missing[:5]}"
