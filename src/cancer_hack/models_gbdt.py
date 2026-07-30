"""GBDT 모델 래퍼 — XGBoost / LightGBM / CatBoost를 하나의 인터페이스로 쓴다.

    from cancer_hack.models_gbdt import create_model, balanced_sample_weight

    model = create_model("lgbm", n_estimators=800)
    model.fit(X_tr, y_tr, sample_weight=balanced_sample_weight(y_tr))
    proba = model.predict_proba(X_va)      # (n, n_classes), 열 순서 = model.classes_

백엔드가 바뀌어도 다음이 항상 같다.

    - fit / predict_proba / predict 시그니처
    - predict_proba 열 순서가 classes_ 와 일치
    - 문자열 라벨 그대로 입력 가능
    - scipy 희소행렬 입력 지원

공통 파라미터 이름은 백엔드 이름으로 자동 변환된다 (`aliases`).
예: n_estimators -> CatBoost 의 iterations, max_depth -> depth,
colsample_bytree -> rsm, reg_lambda -> l2_leaf_reg.

규정: eval_set 에 test 를 넣으면 평가 데이터를 학습에 쓰는 것이라 수상에서 제외된다.
early stopping 이 필요하면 train 을 다시 쪼개서 만든다.
"""

from __future__ import annotations

import functools
import importlib.util
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder

__all__ = [
    "BaseGBDT",
    "XGBModel",
    "LGBMModel",
    "CatBoostModel",
    "create_model",
    "create_models",
    "available_models",
    "registered_models",
    "balanced_sample_weight",
    "gpu_available",
]

SEED = 42


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """CUDA 사용 가능 여부. 실패해도 각 백엔드가 CPU 로 되돌린다."""
    try:
        import xgboost as xgb

        return bool(xgb.build_info().get("USE_CUDA", False))
    except Exception:
        return False


def balanced_sample_weight(y: Sequence) -> np.ndarray:
    """클래스 빈도 역수 가중치. **폴드 내부의 train 부분에서만 호출한다.**"""
    from sklearn.utils.class_weight import compute_sample_weight

    return compute_sample_weight("balanced", y)


class BaseGBDT(ABC):
    """GBDT 백엔드 공통 인터페이스."""

    name: str = "base"

    #: 공통 이름 -> 백엔드 고유 이름. 호출부는 항상 공통 이름을 쓰면 된다.
    aliases: dict[str, str] = {}

    def __init__(self, use_gpu: bool | str = "auto", random_state: int = SEED, **params: Any):
        self.random_state = random_state
        self.use_gpu = self._default_gpu() if use_gpu == "auto" else bool(use_gpu)
        merged = {**self.default_params(), **{k: v for k, v in params.items() if v is not None}}
        self.params = self._normalize(merged)
        self.estimator_: Any = None
        self.label_encoder_: LabelEncoder | None = None
        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self.best_iteration_: int | None = None

    # --- 백엔드가 구현 ---
    @classmethod
    @abstractmethod
    def default_params(cls) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @abstractmethod
    def _build(self) -> Any: ...

    @abstractmethod
    def _fit_backend(self, est, X, yi, sample_weight, eval_set, early_stopping_rounds, verbose): ...

    def _default_gpu(self) -> bool:
        return gpu_available()

    @classmethod
    def _normalize(cls, params: dict[str, Any]) -> dict[str, Any]:
        """공통 파라미터 이름을 백엔드 이름으로 바꾼다. 나중 값이 이긴다."""
        out: dict[str, Any] = {}
        for key, value in params.items():
            out[cls.aliases.get(key, key)] = value
        return out

    # --- 공통 ---
    def fit(
        self,
        X,
        y: Sequence,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[Any, Sequence] | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
    ) -> "BaseGBDT":
        if not self.is_available():
            raise ImportError(
                f"'{self.name}' 백엔드가 설치돼 있지 않다. pip install -r requirements.txt"
            )

        self.label_encoder_ = LabelEncoder()
        yi = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        eval_enc = None
        if eval_set is not None:
            Xv, yv = eval_set
            unseen = set(np.unique(np.asarray(yv))) - set(self.classes_)
            if unseen:
                raise ValueError(f"eval_set 에 train 에 없는 라벨이 있다: {sorted(unseen)}")
            eval_enc = (Xv, self.label_encoder_.transform(np.asarray(yv)))

        if early_stopping_rounds is not None and eval_enc is None:
            warnings.warn("eval_set 이 없어 early_stopping_rounds 를 무시한다.", stacklevel=2)
            early_stopping_rounds = None

        self.estimator_ = self._build()
        self._fit_backend(
            self.estimator_, X, yi, sample_weight, eval_enc, early_stopping_rounds, verbose
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        p = np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != self.n_classes_:
            raise RuntimeError(f"predict_proba 형상 오류: {p.shape}, 기대 (n, {self.n_classes_})")
        return p

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.estimator_.feature_importances_, dtype=np.float64)

    @property
    def tag(self) -> str:
        """OOF·로그 파일명에 쓰는 식별자."""
        return f"{self.name}_s{self.random_state}"

    def describe(self) -> dict[str, Any]:
        """실험 기록용 메타데이터."""
        return {
            "name": self.name,
            "tag": self.tag,
            "device": "gpu" if self.use_gpu else "cpu",
            "random_state": self.random_state,
            "fitted": self.estimator_ is not None,
            "n_classes": self.n_classes_,
            "params": dict(self.params),
        }

    def clone_with(self, **overrides: Any) -> "BaseGBDT":
        """같은 설정의 **미학습** 인스턴스. 폴드마다 새로 만들 때 쓴다."""
        return type(self)(
            use_gpu=self.use_gpu,
            random_state=self.random_state,
            **{**self.params, **overrides},
        )

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseGBDT":
        import joblib

        return joblib.load(Path(path))

    def _check_fitted(self) -> None:
        if self.estimator_ is None:
            raise RuntimeError(f"{type(self).__name__} 가 학습되지 않았다. fit() 을 먼저 호출한다.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {'gpu' if self.use_gpu else 'cpu'})"


class XGBModel(BaseGBDT):
    name = "xgb"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "objective": "multi:softprob",
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.4,
            "reg_lambda": 5.0,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("xgboost") is not None

    def _build(self):
        from xgboost import XGBClassifier

        p = {k: v for k, v in self.params.items() if k != "num_class"}
        p["device"] = "cuda" if self.use_gpu else "cpu"
        p["random_state"] = self.random_state
        return XGBClassifier(**p)

    def _fit_backend(self, est, X, yi, sample_weight, eval_set, early_stopping_rounds, verbose):
        if early_stopping_rounds is not None:
            est.set_params(early_stopping_rounds=early_stopping_rounds)
        kw: dict[str, Any] = {"verbose": verbose}
        if eval_set is not None:
            kw["eval_set"] = [eval_set]
        est.fit(X, yi, sample_weight=sample_weight, **kw)
        self.best_iteration_ = getattr(est, "best_iteration", None)


class LGBMModel(BaseGBDT):
    """pip 기본 휠은 CPU 전용 빌드라 use_gpu='auto' 면 CPU 를 쓴다."""

    name = "lgbm"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "objective": "multiclass",
            "n_estimators": 800,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.4,
            "reg_lambda": 5.0,
            "n_jobs": -1,
            "verbosity": -1,
        }

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("lightgbm") is not None

    def _default_gpu(self) -> bool:
        return False

    def _build(self):
        from lightgbm import LGBMClassifier

        p = dict(self.params)
        p["random_state"] = self.random_state
        p["num_class"] = self.n_classes_
        if self.use_gpu:
            p["device_type"] = "gpu"
        return LGBMClassifier(**p)

    def _fit_backend(self, est, X, yi, sample_weight, eval_set, early_stopping_rounds, verbose):
        import lightgbm as lgb

        callbacks = [lgb.log_evaluation(period=1 if verbose else 0)]
        kw: dict[str, Any] = {}
        if eval_set is not None:
            kw["eval_set"] = [eval_set]
            kw["eval_metric"] = "multi_logloss"
            if early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=verbose))
        try:
            est.fit(X, yi, sample_weight=sample_weight, callbacks=callbacks, **kw)
        except Exception as exc:
            if not self.use_gpu:
                raise
            warnings.warn(f"LightGBM GPU 실패 -> CPU 로 되돌린다: {exc}", stacklevel=2)
            self.use_gpu = False
            self.estimator_ = est = self._build()
            est.fit(X, yi, sample_weight=sample_weight, callbacks=callbacks, **kw)
        self.best_iteration_ = getattr(est, "best_iteration_", None)


class CatBoostModel(BaseGBDT):
    name = "catboost"
    aliases = {
        "n_estimators": "iterations",
        "max_depth": "depth",
        "colsample_bytree": "rsm",
        "reg_lambda": "l2_leaf_reg",
        "min_child_samples": "min_data_in_leaf",
        "random_state": "random_seed",
    }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "rsm": 0.4,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "allow_writing_files": False,
        }

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("catboost") is not None

    def _build(self):
        from catboost import CatBoostClassifier

        p = dict(self.params)
        p["random_seed"] = self.random_state
        p["task_type"] = "GPU" if self.use_gpu else "CPU"
        if self.use_gpu:
            p.pop("rsm", None)  # GPU 모드는 rsm 미지원
        return CatBoostClassifier(**p)

    def _fit_backend(self, est, X, yi, sample_weight, eval_set, early_stopping_rounds, verbose):
        kw: dict[str, Any] = {"verbose": verbose}
        if eval_set is not None:
            kw["eval_set"] = eval_set
            if early_stopping_rounds is not None:
                kw["early_stopping_rounds"] = early_stopping_rounds
        try:
            est.fit(X, yi, sample_weight=sample_weight, **kw)
        except Exception as exc:
            if not self.use_gpu:
                raise
            warnings.warn(f"CatBoost GPU 실패 -> CPU 로 되돌린다: {exc}", stacklevel=2)
            self.use_gpu = False
            self.estimator_ = est = self._build()
            est.fit(X, yi, sample_weight=sample_weight, **kw)
        self.best_iteration_ = est.get_best_iteration() if eval_set is not None else None


_REGISTRY: dict[str, type[BaseGBDT]] = {
    "xgb": XGBModel,
    "xgboost": XGBModel,
    "lgbm": LGBMModel,
    "lightgbm": LGBMModel,
    "cat": CatBoostModel,
    "catboost": CatBoostModel,
}


def registered_models() -> list[str]:
    """등록된 이름 전부(별칭 포함)."""
    return sorted(_REGISTRY)


def available_models() -> list[str]:
    """설치돼 실제로 쓸 수 있는 백엔드의 대표 이름."""
    out: list[str] = []
    for cls in _REGISTRY.values():
        if cls.name not in out and cls.is_available():
            out.append(cls.name)
    return out


def create_model(
    name: str, *, use_gpu: bool | str = "auto", random_state: int = SEED, **params: Any
) -> BaseGBDT:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"알 수 없는 모델 {name!r}. 사용 가능: {registered_models()}")
    return _REGISTRY[key](use_gpu=use_gpu, random_state=random_state, **params)


def create_models(names: Iterable[str] | None = None, **common: Any) -> list[BaseGBDT]:
    """앙상블용 — 여러 백엔드를 한 번에 만든다. names 생략 시 설치된 전부."""
    return [create_model(n, **common) for n in (names if names is not None else available_models())]
