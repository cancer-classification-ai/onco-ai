"""RF/ExtraTrees 공통 wrapper 계약 — `cancer_hack.models_rf`.

`models_gbdt.BaseGBDT` 를 상속하지 않지만 같은 수준의 불변조건(선언은 spec
`docs/specs/random_forest_stacking_model.md` §9.1, tickets 문서 Ticket 1a)을
지킨다: fit 전 호출 방어, 확률 shape/유한값/[0,1]/행합 검증, canonical class
order 정렬, sample_weight 검증, 잘못된 kind/param 에 대한 명확한 오류.

전부 합성 데이터로만 돈다 — 원본 대회 데이터나 개인 경로에 의존하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from cancer_hack.models_rf import ForestModel, create_model, registered_models

N_CLASSES = 6
CLASS_ORDER = [f"C{i:02d}" for i in range(N_CLASSES)]


def make_data(n_samples=180, n_features=12, random_state=0):
    X, yi = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    y = np.array([CLASS_ORDER[i] for i in yi])
    return X.astype(np.float32), y


# ---------------------------------------------------------------- factory


@pytest.mark.parametrize("kind", ["rf", "random_forest", "et", "extra_trees"])
def test_create_model_accepts_known_aliases(kind):
    model = create_model(kind, n_estimators=10, random_state=42)
    assert isinstance(model, ForestModel)


def test_registered_models_lists_rf_and_et():
    names = registered_models()
    assert "rf" in names
    assert "et" in names


def test_unknown_model_kind_raises_clear_error():
    with pytest.raises(ValueError, match="알 수 없는 모델"):
        create_model("xgb", n_estimators=10)


def test_unsupported_parameter_raises_clear_error():
    with pytest.raises(ValueError):
        create_model("rf", not_a_real_sklearn_param=123)


# ---------------------------------------------------------------- fit/predict_proba


@pytest.mark.parametrize("kind", ["rf", "et"])
def test_fit_predict_proba_shape(kind):
    X, y = make_data()
    model = create_model(kind, n_estimators=20, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), N_CLASSES)


@pytest.mark.parametrize("kind", ["rf", "et"])
def test_predict_proba_rows_sum_to_one_and_finite(kind):
    X, y = make_data()
    model = create_model(kind, n_estimators=20, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1).all()


@pytest.mark.parametrize("kind", ["rf", "et"])
def test_classes_match_canonical_alphabetical_order(kind):
    X, y = make_data()
    model = create_model(kind, n_estimators=20, random_state=42)
    model.fit(X, y)
    assert list(model.classes_) == sorted(set(y.tolist()))


def test_feature_importances_shape_matches_n_features():
    X, y = make_data(n_features=15)
    model = create_model("rf", n_estimators=20, random_state=42)
    model.fit(X, y)
    assert model.feature_importances_.shape == (15,)


# ---------------------------------------------------------------- fit 전 방어


@pytest.mark.parametrize("kind", ["rf", "et"])
def test_predict_proba_before_fit_raises(kind):
    model = create_model(kind, n_estimators=10, random_state=42)
    X, _ = make_data()
    with pytest.raises(RuntimeError, match="학습되지 않았다"):
        model.predict_proba(X)


def test_feature_importances_before_fit_raises():
    model = create_model("rf", n_estimators=10, random_state=42)
    with pytest.raises(RuntimeError, match="학습되지 않았다"):
        _ = model.feature_importances_


# ---------------------------------------------------------------- 결정론성


def test_same_seed_is_deterministic():
    X, y = make_data()
    proba_a = create_model("rf", n_estimators=30, random_state=7).fit(X, y).predict_proba(X)
    proba_b = create_model("rf", n_estimators=30, random_state=7).fit(X, y).predict_proba(X)
    assert np.array_equal(proba_a, proba_b)


def test_different_seed_changes_result():
    X, y = make_data()
    proba_a = create_model("rf", n_estimators=30, random_state=1).fit(X, y).predict_proba(X)
    proba_b = create_model("rf", n_estimators=30, random_state=2).fit(X, y).predict_proba(X)
    assert not np.array_equal(proba_a, proba_b)


# ---------------------------------------------------------------- sample_weight


def test_sample_weight_is_passed_to_estimator_and_changes_result():
    X, y = make_data()
    weight = np.ones(len(y))
    weight[y == CLASS_ORDER[0]] = 5.0

    unweighted = create_model("rf", n_estimators=30, random_state=42).fit(X, y).predict_proba(X)
    weighted = (
        create_model("rf", n_estimators=30, random_state=42)
        .fit(X, y, sample_weight=weight)
        .predict_proba(X)
    )
    assert not np.array_equal(unweighted, weighted)


def test_sample_weight_length_mismatch_raises():
    X, y = make_data()
    model = create_model("rf", n_estimators=10, random_state=42)
    with pytest.raises(ValueError, match="길이"):
        model.fit(X, y, sample_weight=np.ones(len(y) - 1))


def test_sample_weight_with_nan_raises():
    X, y = make_data()
    weight = np.ones(len(y))
    weight[0] = np.nan
    model = create_model("rf", n_estimators=10, random_state=42)
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X, y, sample_weight=weight)


def test_sample_weight_with_inf_raises():
    X, y = make_data()
    weight = np.ones(len(y))
    weight[0] = np.inf
    model = create_model("rf", n_estimators=10, random_state=42)
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X, y, sample_weight=weight)


def test_sample_weight_with_negative_raises():
    X, y = make_data()
    weight = np.ones(len(y))
    weight[0] = -1.0
    model = create_model("rf", n_estimators=10, random_state=42)
    with pytest.raises(ValueError, match="음수"):
        model.fit(X, y, sample_weight=weight)


# ---------------------------------------------------------------- class_order 검증


def test_class_order_rejects_unexpected_label():
    X, y = make_data()
    model = create_model("rf", n_estimators=10, random_state=42, class_order=CLASS_ORDER)
    y_bad = y.copy()
    y_bad[0] = "NOT_A_CLASS"
    with pytest.raises(ValueError, match="canonical class order 밖"):
        model.fit(X, y_bad, sample_weight=None)


def test_class_order_rejects_missing_train_partition_class():
    X, y = make_data()
    # canonical order 에 훈련 데이터에 없는 클래스를 하나 추가한다.
    extended_order = CLASS_ORDER + ["ZZZZ"]
    model = create_model("rf", n_estimators=10, random_state=42, class_order=extended_order)
    with pytest.raises(ValueError, match="train partition 에 canonical 클래스가 없다"):
        model.fit(X, y)


def test_class_order_accepts_when_all_present():
    X, y = make_data()
    model = create_model("rf", n_estimators=10, random_state=42, class_order=CLASS_ORDER)
    model.fit(X, y)
    assert list(model.classes_) == CLASS_ORDER


def test_no_class_order_still_uses_alphabetical_sklearn_default():
    X, y = make_data()
    model = create_model("rf", n_estimators=10, random_state=42)
    model.fit(X, y)
    assert list(model.classes_) == sorted(set(y.tolist()))


# ---------------------------------------------------------------- test 데이터 격리


def test_predict_proba_does_not_mutate_or_require_fit_data():
    """`predict_proba` 는 새 X 를 받을 뿐 fit 에 쓰인 데이터를 다시 참조하지 않는다.

    test 행렬이 fit 경로로 흘러들어가지 않는다는 걸 구조적으로 보여준다 — fit 이
    끝난 뒤 별도로 생성한(= fit 이 존재를 몰랐던) 배열에 대해서도 정상 동작해야
    한다.
    """
    X_train, y_train = make_data(n_samples=120, random_state=3)
    X_other, _ = make_data(n_samples=40, random_state=99)
    model = create_model("rf", n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_other)
    assert proba.shape == (40, N_CLASSES)


# ---------------------------------------------------------------- n_jobs


def test_default_n_jobs_is_positive_int():
    from cancer_hack.models_rf import default_n_jobs

    assert isinstance(default_n_jobs(), int)
    assert default_n_jobs() >= 1


def test_explicit_n_jobs_is_respected():
    model = create_model("rf", n_estimators=10, random_state=42, n_jobs=1)
    assert model.n_jobs == 1
    assert model.estimator_.n_jobs == 1
