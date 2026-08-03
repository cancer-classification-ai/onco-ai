"""RandomForest 래퍼 계약 — 앙상블 3번째 멤버가 나머지 둘과 같은 약속을 지켜야 한다.

`train_gbdt.py` 의 fold 루프는 백엔드를 구별하지 않는다. `predict_proba` 의 열 순서가
`classes_` 와 어긋나면 예외 없이 조용히 틀린 라벨이 나오고, `describe()["device"]` 가
거짓말하면 실험 기록만으로 복기가 안 된다. RF 는 GBDT 가 아니라서 그 약속을 우연히
지킬 이유가 없다 — 그래서 못 박는다.
"""

from __future__ import annotations

import numpy as np
import pytest

from cancer_hack.models_gbdt import RFModel, create_model, registered_models


def _toy_data(n_classes: int = 26, n_samples: int = 260, n_features: int = 12):
    rng = np.random.RandomState(0)
    X = rng.rand(n_samples, n_features).astype(np.float32)
    yi = rng.randint(0, n_classes, size=n_samples)
    y = np.array([f"C{i:02d}" for i in yi])
    return X, y, n_classes


def test_rf_is_registered():
    assert "rf" in registered_models()
    assert isinstance(create_model("rf"), RFModel)


def test_rf_default_params_has_no_max_depth():
    """`--set max_depth=None` 이 안 먹힌다 — sklearn 기본(무제한)에 맡긴다."""
    assert "max_depth" not in RFModel.default_params()


def test_rf_default_params_has_no_class_weight():
    """파이프라인이 balanced sample_weight 를 준다. 이중 가중은 실측에서 성능을 떨어뜨렸다."""
    assert "class_weight" not in RFModel.default_params()


def test_rf_aliases_are_empty():
    """별칭을 나중에 누가 추가하면 `--set` 이 모델마다 다른 뜻이 된다."""
    assert RFModel.aliases == {}


def test_rf_reports_cpu_even_when_gpu_available(monkeypatch):
    import cancer_hack.models_gbdt as models_gbdt

    monkeypatch.setattr(models_gbdt, "gpu_available", lambda: True)
    model = create_model("rf", use_gpu="auto")
    X, y, _ = _toy_data()
    model.fit(X, y)
    assert model.describe()["device"] == "cpu"


def test_rf_forces_cpu_when_gpu_requested():
    model = create_model("rf", use_gpu=True)
    X, y, _ = _toy_data()
    with pytest.warns(UserWarning, match="CPU 전용"):
        model.fit(X, y)
    assert model.describe()["device"] == "cpu"


def test_rf_probability_contract():
    X, y, n_classes = _toy_data()
    model = create_model("rf", n_estimators=20)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(model.predict(X), model.classes_[proba.argmax(axis=1)])
    assert list(model.classes_) == sorted(set(y))


def test_rf_accepts_sparse_input():
    from scipy import sparse

    X, y, _ = _toy_data()
    model = create_model("rf", n_estimators=20)
    model.fit(sparse.csr_matrix(X), y)
    proba = model.predict_proba(sparse.csr_matrix(X))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_rf_ignores_eval_set_with_warning():
    X, y, _ = _toy_data()
    model = create_model("rf", n_estimators=10)
    with pytest.warns(UserWarning, match="eval_set"):
        model.fit(X, y, eval_set=(X, y), early_stopping_rounds=10)
