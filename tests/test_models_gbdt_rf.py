"""GBDT 학습 경로에서 사용하는 RandomForest 래퍼 계약.

`train_gbdt.py`의 fold 루프는 백엔드를 구별하지 않는다. `predict_proba`의 열 순서가
`classes_`와 어긋나면 조용히 틀린 라벨이 나오고, `describe()["device"]`가
거짓말하면 실험 기록만으로 복기가 안 된다. 별도 RF 파이프라인의
`cancer_hack.models_rf` 계약과 구분해 이 파일에서 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from cancer_hack.models_gbdt import RFModel, create_model, registered_models


def _toy_data(n_classes: int = 26, n_samples: int = 260, n_features: int = 12):
    rng = np.random.RandomState(0)
    x = rng.rand(n_samples, n_features).astype(np.float32)
    yi = rng.randint(0, n_classes, size=n_samples)
    y = np.array([f"C{i:02d}" for i in yi])
    return x, y, n_classes


def test_rf_is_registered():
    assert "rf" in registered_models()
    assert isinstance(create_model("rf"), RFModel)


def test_rf_default_params_has_no_max_depth():
    """`--set max_depth=None`이 안 먹히므로 sklearn 기본(무제한)에 맡긴다."""
    assert "max_depth" not in RFModel.default_params()


def test_rf_default_params_has_no_class_weight():
    """파이프라인이 balanced sample_weight를 주므로 이중 가중하지 않는다."""
    assert "class_weight" not in RFModel.default_params()


def test_rf_aliases_are_empty():
    """별칭을 추가해 `--set`이 모델마다 다른 뜻이 되는 일을 막는다."""
    assert RFModel.aliases == {}


def test_rf_reports_cpu_even_when_gpu_available(monkeypatch):
    import cancer_hack.models_gbdt as models_gbdt

    monkeypatch.setattr(models_gbdt, "gpu_available", lambda: True)
    model = create_model("rf", use_gpu="auto")
    x, y, _ = _toy_data()
    model.fit(x, y)
    assert model.describe()["device"] == "cpu"


def test_rf_forces_cpu_when_gpu_requested():
    model = create_model("rf", use_gpu=True)
    x, y, _ = _toy_data()
    with pytest.warns(UserWarning, match="CPU 전용"):
        model.fit(x, y)
    assert model.describe()["device"] == "cpu"


def test_rf_probability_contract():
    x, y, n_classes = _toy_data()
    model = create_model("rf", n_estimators=20)
    model.fit(x, y)

    proba = model.predict_proba(x)
    assert proba.shape == (x.shape[0], n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(model.predict(x), model.classes_[proba.argmax(axis=1)])
    assert list(model.classes_) == sorted(set(y))


def test_rf_accepts_sparse_input():
    from scipy import sparse

    x, y, _ = _toy_data()
    model = create_model("rf", n_estimators=20)
    model.fit(sparse.csr_matrix(x), y)
    proba = model.predict_proba(sparse.csr_matrix(x))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_rf_ignores_eval_set_with_warning():
    x, y, _ = _toy_data()
    model = create_model("rf", n_estimators=10)
    with pytest.warns(UserWarning, match="eval_set"):
        model.fit(x, y, eval_set=(x, y), early_stopping_rounds=10)
