"""Caruana 그리디 선택의 계약.

이 알고리즘의 위험은 하나다 — **선택에 쓴 행에서 잰 점수는 낙관적이다.** 라이브러리가
91개면 그중 최고 조합은 그 fold 들의 우연까지 맞춘다. 그래서 `train_macro_f1_` 을
보고값으로 쓰면 안 되고, 정직한 값은 교차적합으로만 나온다
(`scripts/greedy_blend.py` 가 그렇게 한다).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cancer_hack.ensemble import GreedyEnsembleSelector, weighted_average  # noqa: E402
from cancer_hack.metrics import macro_f1  # noqa: E402

_spec = importlib.util.spec_from_file_location("greedy_blend_mod", PROJECT_ROOT / "scripts" / "greedy_blend.py")
gb = importlib.util.module_from_spec(_spec)
sys.modules["greedy_blend_mod"] = gb
_spec.loader.exec_module(gb)


@pytest.fixture
def library():
    """약한 멤버 5개 + 쓸모없는 멤버 4개. 단독으로는 아무도 완벽하지 않다."""
    n, k = 600, 6
    y = np.asarray([f"c{i % k}" for i in range(n)])
    classes = [f"c{i}" for i in range(k)]
    true = np.asarray([int(c[1]) for c in y])

    def useful(seed):
        rng = np.random.default_rng(seed)
        p = rng.random((n, k)) * 0.5
        p[np.arange(n), true] += 0.42
        return p / p.sum(1, keepdims=True)

    def useless(seed):
        rng = np.random.default_rng(1000 + seed)
        p = rng.random((n, k)) + 1e-6
        return p / p.sum(1, keepdims=True)

    members = [useful(s) for s in range(5)] + [useless(s) for s in range(4)]
    return members, y, classes


def test_beats_the_best_single_member(library):
    """앙상블이 단독 최고보다 나쁘면 그리디를 쓸 이유가 없다."""
    members, y, classes = library
    best_solo = max(macro_f1(y, np.asarray(classes)[m.argmax(1)]) for m in members)
    selector = GreedyEnsembleSelector(n_rounds=20, random_state=0).fit(members, y, classes)
    assert selector.train_macro_f1_ >= best_solo


def test_useless_members_get_no_weight(library):
    """정보가 없는 멤버가 가중치를 받으면 선택이 노이즈를 따라간 것이다."""
    members, y, classes = library
    selector = GreedyEnsembleSelector(n_rounds=20, random_state=0).fit(members, y, classes)
    assert selector.weights_[5:].sum() < 0.15


def test_weights_are_counts_normalised(library):
    """담긴 횟수가 곧 가중치다 — 이 규약이 깨지면 test 변환이 OOF 와 달라진다."""
    members, y, classes = library
    selector = GreedyEnsembleSelector(n_rounds=17, random_state=0).fit(members, y, classes)
    assert selector.counts_.sum() > 0
    assert np.allclose(selector.weights_, selector.counts_ / selector.counts_.sum())
    assert np.isclose(selector.weights_.sum(), 1.0)


def test_replacement_is_allowed(library):
    """복원 없이 담으면 가중치가 전부 같아진다 — Caruana 의 핵심이 사라진다."""
    members, y, classes = library
    selector = GreedyEnsembleSelector(n_rounds=25, random_state=0).fit(members, y, classes)
    assert selector.counts_.max() > 1


def test_predict_proba_matches_the_learned_weights(library):
    members, y, classes = library
    selector = GreedyEnsembleSelector(n_rounds=12, random_state=0).fit(members, y, classes)
    expected = weighted_average(members, selector.weights_)
    assert np.allclose(selector.predict_proba(members), expected)


def test_bagging_changes_the_selection(library):
    """bag_fraction<1 이면 라운드마다 후보가 달라져 한 멤버로 쏠리는 걸 막는다."""
    members, y, classes = library
    full = GreedyEnsembleSelector(n_rounds=20, bag_fraction=1.0, random_state=0).fit(members, y, classes)
    bagged = GreedyEnsembleSelector(n_rounds=20, bag_fraction=0.5, bag_rounds=4,
                                    random_state=0).fit(members, y, classes)
    assert int((bagged.counts_ > 0).sum()) >= int((full.counts_ > 0).sum())


def test_rejects_mismatched_shapes(library):
    members, y, classes = library
    with pytest.raises(ValueError):
        GreedyEnsembleSelector(n_rounds=3).fit([members[0], members[1][:10]], y, classes)


def test_rejects_labels_outside_classes(library):
    members, y, classes = library
    # `y` 는 고정폭 유니코드 배열(<U2)이라 긴 문자열을 넣으면 잘린다. object 로 바꿔 넣는다.
    bad = y.astype(object)
    bad[0] = "없는클래스"
    with pytest.raises(ValueError, match="없는클래스"):
        GreedyEnsembleSelector(n_rounds=3).fit(members, bad, classes)


@pytest.mark.parametrize("kwargs", [
    {"n_rounds": 0}, {"bag_fraction": 0.0}, {"bag_fraction": 1.5}, {"bag_rounds": 0},
])
def test_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        GreedyEnsembleSelector(**kwargs)


# --- 러너 스크립트 계약 -------------------------------------------------------

def test_derived_blends_are_excluded_by_default():
    """블렌드의 블렌드를 라이브러리에 넣으면 같은 멤버가 두 경로로 중복 계상된다."""
    for stem in ("oof_ens16_w455510_group5", "oof_blendA_w455510", "oof_stackFinal4",
                 "oof_xc_group5", "oof_x_group5_raw_blend", "oof_x_group5_uniform_blend",
                 "oof_f4r_cal"):
        assert gb.DERIVED.search(stem), f"{stem} 은 파생 블렌드로 걸러져야 한다"


def test_plain_members_are_not_excluded():
    for stem in ("oof_xgb_repo16_f16_group5_k500_s42", "oof_teamcat_group5",
                 "oof_rf_repo16n_f16n_group5_k500_s7"):
        assert not gb.DERIVED.search(stem), f"{stem} 은 라이브러리에 남아야 한다"


def test_test_path_mirrors_the_oof_name():
    """선택된 멤버의 test 예측을 못 찾으면 제출 파일을 못 만든다."""
    path = gb.test_path_for(Path("artifacts/oof/oof_xgb_repo16_f16_group5_k500_s42.csv"))
    assert path.name == "test_xgb_repo16_f16_group5_k500_s42.csv"
    assert path.parent.name == "test_predictions"
