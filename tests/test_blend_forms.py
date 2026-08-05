"""확률을 합치는 방식들 — 성질과 규정 준수를 지킨다.

`weighted_average` 말고도 기하·조화·순위·중앙값이 생겼다. 이 파일이 지키는 것은 셋이다.

1. **어떤 형태든 확률을 낸다** — 비음수이고 행 합이 1. 뒤에 로짓 보정이 붙으므로
   이게 깨지면 보정 쪽에서 이상한 값이 나온다.
2. **`mean` 은 예전과 같다** — 기존 제출본이 전부 이 경로로 만들어졌다.
3. **`rank` 는 행 독립이다** — test 행끼리 줄을 세우면 대회 규정("test 한 행만 따로 넣어도
   같은 결과가 나오는가")을 어긴다. reference 를 주는 형태가 그걸 막는지 본다.
"""

from __future__ import annotations

import numpy as np
import pytest

from cancer_hack.ensemble import (
    BLEND_FORMS,
    combine,
    power_mean,
    rank_average,
    trimmed_mean,
    weighted_average,
)


@pytest.fixture
def members():
    rng = np.random.default_rng(11)
    return [rng.dirichlet(np.ones(5), size=40) for _ in range(4)]


@pytest.fixture
def weights():
    return [0.4, 0.3, 0.2, 0.1]


@pytest.mark.parametrize("form", BLEND_FORMS)
def test_every_form_returns_probabilities(members, weights, form):
    mixed = combine(members, weights, form)
    assert mixed.shape == (40, 5)
    assert (mixed >= 0).all()
    assert np.allclose(mixed.sum(axis=1), 1.0)
    assert np.isfinite(mixed).all()


def test_mean_is_unchanged(members, weights):
    """기존 제출본이 전부 이 경로다. 값이 달라지면 재현이 깨진다."""

    assert np.array_equal(combine(members, weights, "mean"),
                          weighted_average(members, weights))


def test_power_mean_one_equals_arithmetic(members, weights):
    assert np.allclose(power_mean(members, weights, 1.0),
                       weighted_average(members, weights))


@pytest.mark.parametrize("form", ["mean", "geometric", "harmonic"])
def test_single_member_is_identity(members, form):
    """멤버가 하나면 그 멤버 그대로여야 한다."""

    mixed = combine([members[0]], [1.0], form)
    assert np.allclose(mixed, members[0] / members[0].sum(axis=1, keepdims=True))


def test_geometric_punishes_a_confident_dissenter():
    """산술은 확신한 한 명에게 끌려가고 기하는 안 그렇다 — 그게 이 형태를 쓰는 이유다."""

    # 확신한 한 명이 0번을 찍고, 나머지 둘은 0번에 낮은 값을 주며 1번을 민다.
    loud = np.array([[0.97, 0.01, 0.01, 0.01]])
    quiet_a = np.array([[0.02, 0.50, 0.28, 0.20]])
    quiet_b = np.array([[0.02, 0.48, 0.30, 0.20]])
    stack = [loud, quiet_a, quiet_b]
    even = [1 / 3] * 3

    assert weighted_average(stack, even).argmax() == 0     # 확신한 쪽이 이긴다
    assert power_mean(stack, even, 0.0).argmax() == 1      # 다수가 이긴다


def test_zero_probability_does_not_break_geometric():
    """0 이 들어와도 log 가 터지면 안 된다."""

    a = np.array([[0.0, 1.0]])
    b = np.array([[0.5, 0.5]])
    mixed = power_mean([a, b], [0.5, 0.5], 0.0)
    assert np.isfinite(mixed).all()
    assert np.allclose(mixed.sum(axis=1), 1.0)


def test_trimmed_drops_the_extremes():
    """가운데 둘만 남는지 — 멤버 4개에 trim=1 이면 최대·최소가 빠진다."""

    stack = [
        np.array([[0.90, 0.10]]),
        np.array([[0.60, 0.40]]),
        np.array([[0.40, 0.60]]),
        np.array([[0.10, 0.90]]),
    ]
    mixed = trimmed_mean(stack, trim=1)
    assert np.allclose(mixed, [[0.5, 0.5]])


def test_median_and_trimmed_coincide_for_four_members(members):
    """멤버가 4개면 둘이 같은 연산이다. 표에서 별개 결과로 세면 안 된다."""

    assert np.array_equal(combine(members, [0.25] * 4, "median"),
                          combine(members, [0.25] * 4, "trimmed"))


def test_trim_cannot_eat_everything(members):
    with pytest.raises(ValueError):
        trimmed_mean(members, trim=2)


def test_rank_with_reference_is_row_independent(members, weights):
    """대회 규정의 판정 기준 — test 한 행만 넣어도 같은 값이 나와야 한다."""

    rng = np.random.default_rng(5)
    reference = [rng.dirichlet(np.ones(5), size=200) for _ in range(4)]

    whole = rank_average(members, weights, reference=reference)
    alone = rank_average([m[7:8] for m in members], weights, reference=reference)
    assert np.allclose(whole[7], alone[0])


def test_rank_without_reference_is_not_row_independent(members, weights):
    """그래서 reference 없이 test 에 쓰면 안 된다 — 이 성질을 못 박아 둔다."""

    whole = rank_average(members, weights)
    alone = rank_average([m[7:8] for m in members], weights)
    assert not np.allclose(whole[7], alone[0])


def test_rank_reference_shape_is_checked(members, weights):
    with pytest.raises(ValueError):
        rank_average(members, weights, reference=members[:2])


def test_unknown_form_is_rejected(members, weights):
    with pytest.raises(ValueError, match="모르는 결합 형태"):
        combine(members, weights, "softmax")


@pytest.mark.parametrize("form", ["mean", "geometric", "harmonic", "rank"])
def test_bad_weights_are_rejected(members, form):
    with pytest.raises(ValueError):
        combine(members, [0.5, 0.5, -0.5, 0.5], form)
    with pytest.raises(ValueError):
        combine(members, [0.25, 0.25, 0.5], form)


class TestGreedyForm:
    """그리디가 기하평균으로도 담을 수 있어야 한다.

    합치는 방식이 바뀌면 **어느 멤버를 담는 게 이득인지도 바뀐다.** 그래서 선택 루프
    안에서도 같은 형태를 써야 하고, 그렇지 않으면 고를 때와 합칠 때의 목적함수가
    달라진다. 아래가 그 계약을 지킨다.
    """

    @staticmethod
    def _library(seed: int = 7, n_members: int = 8, n_rows: int = 400, n_classes: int = 4):
        """멤버마다 **다른 행에서 틀리고** 확신의 세기도 다른 라이브러리.

        두 조건이 다 필요하다. 전부 같은 정답 방향을 가리키면 몇 개만 평균 내도 만점이
        나오는데, 그러면 모든 후보가 동점이라 그리디는 그냥 첫 번째를 담는다 — 형태를
        바꿔도 결과가 같아져서 아무것도 검증하지 못한다. 그리고 뾰족한 멤버와 뭉툭한
        멤버가 섞여 있어야 기하평균이 산술평균과 다르게 움직인다. 실제 라이브러리가
        그렇다 — DL 은 뾰족하고 RF 는 뭉툭하며, 서로 다른 행에서 틀린다.
        """

        rng = np.random.default_rng(seed)
        classes = [chr(65 + i) for i in range(n_classes)]
        target = rng.integers(0, n_classes, size=n_rows)
        members = []
        for index in range(n_members):
            wrong = rng.random(n_rows) < (0.30 + 0.20 * (index % 3) / 2)
            shifted = (target + rng.integers(1, n_classes, size=n_rows)) % n_classes
            belief = np.where(wrong, shifted, target)
            temperature = 0.30 if index % 2 else 1.8
            raw = 0.5 * np.eye(n_classes)[belief] + 0.5 * rng.dirichlet(
                np.ones(n_classes), size=n_rows)
            sharp = raw ** (1.0 / temperature)
            members.append(sharp / sharp.sum(axis=1, keepdims=True))
        return members, np.asarray(classes)[target], classes

    def test_mean_is_the_default(self):
        """기본값이 바뀌면 기존 제출본이 재현되지 않는다."""

        from cancer_hack.ensemble import GreedyEnsembleSelector

        assert GreedyEnsembleSelector().form == "mean"

    def test_geometric_runs_and_picks_members(self):
        from cancer_hack.ensemble import GreedyEnsembleSelector

        members, y, classes = self._library()
        fitted = GreedyEnsembleSelector(n_rounds=8, random_state=0,
                                        form="geometric").fit(members, y, classes)
        assert fitted.counts_.sum() > 0
        assert np.isclose(fitted.weights_.sum(), 1.0)
        assert 0.0 <= fitted.train_macro_f1_ <= 1.0

    def test_geometric_predicts_with_geometric(self):
        """`predict_proba` 가 산술로 되돌아가면 선택과 합치기가 어긋난다."""

        from cancer_hack.ensemble import GreedyEnsembleSelector

        members, y, classes = self._library()
        fitted = GreedyEnsembleSelector(n_rounds=6, random_state=0,
                                        form="geometric").fit(members, y, classes)
        assert np.allclose(fitted.predict_proba(members),
                           power_mean(members, fitted.weights_, 0.0))

    def test_two_forms_can_disagree(self):
        """같은 라이브러리·같은 난수인데 형태만 다르면 다른 선택이 나올 수 있어야 한다.

        둘이 항상 같다면 `form` 인자가 실제로는 아무 일도 안 하고 있다는 뜻이다.
        """

        from cancer_hack.ensemble import GreedyEnsembleSelector

        members, y, classes = self._library(seed=3)
        kwargs = dict(n_rounds=12, bag_fraction=0.6, bag_rounds=3, random_state=0)
        mean = GreedyEnsembleSelector(**kwargs, form="mean").fit(members, y, classes)
        geo = GreedyEnsembleSelector(**kwargs, form="geometric").fit(members, y, classes)
        assert not np.array_equal(mean.counts_, geo.counts_)

    def test_unknown_form_is_rejected(self):
        from cancer_hack.ensemble import GreedyEnsembleSelector

        with pytest.raises(ValueError, match="form 은"):
            GreedyEnsembleSelector(form="rank")

    def test_geometric_is_deterministic(self):
        """같은 난수면 두 번 돌려 같아야 한다 — 재현이 이 프로젝트의 전제다."""

        from cancer_hack.ensemble import GreedyEnsembleSelector

        members, y, classes = self._library()
        kwargs = dict(n_rounds=10, bag_fraction=0.6, bag_rounds=3,
                      random_state=42, form="geometric")
        first = GreedyEnsembleSelector(**kwargs).fit(members, y, classes)
        second = GreedyEnsembleSelector(**kwargs).fit(members, y, classes)
        assert np.array_equal(first.counts_, second.counts_)


def test_crossfit_blend_defaults_to_mean():
    """`form` 을 안 주던 기존 호출이 같은 값을 내야 한다."""

    from cancer_hack.ensemble import crossfit_calibrated_blend

    rng = np.random.default_rng(3)
    classes = ["A", "B", "C"]
    arrays = [rng.dirichlet(np.ones(3), size=60) for _ in range(2)]
    y = np.asarray(classes)[rng.integers(0, 3, size=60)]
    folds = np.tile([0, 1, 2], 20)

    a = crossfit_calibrated_blend(arrays, y, classes, folds, [0.6, 0.4])
    b = crossfit_calibrated_blend(arrays, y, classes, folds, [0.6, 0.4], form="mean")
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
