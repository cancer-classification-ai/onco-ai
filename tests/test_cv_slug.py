"""산출물 이름이 분할 수를 반영하는지.

`--n-splits 10` 으로 학습했는데 파일 이름이 `group5` 로 붙던 결함을 막는다.
그러면 10-fold 예측이 5-fold 라이브러리에 조용히 섞이고, `greedy_blend.py` 는
이름으로 후보를 모으므로 그 순간부터 fold 경계가 어긋난 채 결합이 돈다.
로그 안에는 `n_splits: 10` 이 제대로 적혀 있어서 **파일 이름만 봐서는 못 알아챈다.**
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cancer_hack.validation import CV_SLUG, cv_slug, fold_column

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_default_matches_the_old_convention():
    """지금까지의 산출물 이름이 한 글자도 안 바뀌어야 한다."""

    for kind, slug in CV_SLUG.items():
        assert cv_slug(kind) == slug


@pytest.mark.parametrize("kind,n_splits,expected", [
    ("skf", 5, "skf5"),
    ("sgkf", 5, "group5"),
    ("skf", 10, "skf10"),
    ("sgkf", 10, "group10"),
    ("sgkf", 3, "group3"),
])
def test_slug_follows_n_splits(kind, n_splits, expected):
    assert cv_slug(kind, n_splits) == expected


def test_slug_and_fold_column_agree():
    """이름 약칭과 fold 열 이름이 같은 분할을 가리켜야 한다."""

    for kind in ("skf", "sgkf"):
        for n_splits in (5, 10):
            assert fold_column(kind, n_splits) == f"fold_{cv_slug(kind, n_splits)}"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="알 수 없는 CV 방식"):
        cv_slug("loo", 5)


@pytest.mark.parametrize("script", ["train_gbdt.py", "train_dl.py"])
def test_trainers_do_not_hardcode_the_slug(script):
    """학습 스크립트가 `CV_SLUG[...]` 로 이름을 짓고 있으면 안 된다.

    사전을 직접 인덱싱하면 분할 수가 무시된다. `cv_slug(kind, n_splits)` 를 써야 한다.
    """

    source = (SCRIPTS / script).read_text(encoding="utf-8")
    offenders = re.findall(r"CV_SLUG\[[^\]]+\]", source)
    assert not offenders, (
        f"{script} 가 CV_SLUG 를 직접 인덱싱한다: {offenders}. "
        "cv_slug(kind, n_splits) 로 바꾼다 — 안 그러면 10-fold 산출물이 group5 이름을 단다.")


@pytest.mark.parametrize("script", ["train_gbdt.py", "train_dl.py"])
def test_trainers_pass_n_splits_to_the_slug(script):
    """`cv_slug(...)` 를 부르되 분할 수를 같이 넘겨야 한다."""

    source = (SCRIPTS / script).read_text(encoding="utf-8")
    calls = re.findall(r"cv_slug\(([^)]*)\)", source)
    calls = [c for c in calls if c.strip()]          # 정의부·import 는 제외
    assert calls, f"{script} 가 cv_slug 를 안 쓴다"
    for call in calls:
        assert "n_splits" in call, (
            f"{script} 의 cv_slug({call}) 에 n_splits 가 없다 — 5-fold 로 굳는다")
