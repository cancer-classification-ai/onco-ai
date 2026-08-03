"""Random Forest / ExtraTrees 공통 wrapper — `models_gbdt.BaseGBDT` 와 별개 계약.

    from cancer_hack.models_rf import create_model

    model = create_model("rf", n_estimators=500, class_weight="balanced_subsample")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)   # (n, n_classes), 열 순서 = model.classes_

RF 와 ExtraTrees 는 둘 다 이미 sklearn 표준 인터페이스
(`fit`/`predict_proba`/`classes_`/`feature_importances_`)를 그대로 만족한다.
`BaseGBDT`(`models_gbdt.py`)가 흡수해야 했던 문제 — 백엔드마다 다른 `fit`
시그니처, `eval_set`, `label_encoder_` — 가 여기엔 없다. 그래서 `BaseGBDT` 를
상속하지 않는다. 대신 이 모듈은 RF/ET 가 **공유해야 하는 불변조건**만 한
클래스(`ForestModel`)에 모은다: fit 전 호출 방어, 확률 shape·유한값·[0,1]·
행합 검증, canonical class order 정렬, sample_weight 검증, 잘못된 종류/
파라미터에 대한 명확한 오류. RF/ET 를 각각 서브클래스로 만들지 않는 이유도
같다 — 둘의 차이는 어떤 sklearn 클래스를 감싸는지뿐이고, 나머지 로직을
복붙하면 그 로직이 언젠가 갈라진다.

규정 준수: eval_set/early stopping 은 이 wrapper 에 아예 없다(RF 는 반복
학습이 아니라 test 로 반복수를 고를 여지 자체가 없다). class_weight 외
sample_weight 를 중복 적용할지는 호출부(`scripts/train_rf.py`)의 결정이다 —
이 wrapper 는 받은 sample_weight 를 검증해 그대로 전달할 뿐이다.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

__all__ = [
    "ForestModel",
    "create_model",
    "registered_models",
    "default_n_jobs",
    "SEED",
]

SEED = 42

_REGISTRY: dict[str, type] = {
    "rf": RandomForestClassifier,
    "random_forest": RandomForestClassifier,
    "randomforest": RandomForestClassifier,
    "et": ExtraTreesClassifier,
    "extra_trees": ExtraTreesClassifier,
    "extratrees": ExtraTreesClassifier,
}

#: 별칭 -> 대표 짧은 이름. 파일명/로그 태그에 쓴다.
_CANONICAL_NAME = {
    RandomForestClassifier: "rf",
    ExtraTreesClassifier: "et",
}


def registered_models() -> list[str]:
    """등록된 이름 전부(별칭 포함)."""
    return sorted(_REGISTRY)


def default_n_jobs() -> int:
    """논리 CPU 수 - 1(최소 1). 코어 수를 못 읽으면 1(직렬)로 되돌린다.

    전체 코어를 다 쓰면(`-1`) 같은 머신에서 fold 루프 바깥의 다른 작업(로깅,
    다음 fold 준비)이 밀려 8코어 16GB 환경(spec §11)에서 스왑 위험이 커진다.
    한 코어를 비워 두는 보수적 기본값이고, 호출부가 `n_jobs=` 를 명시하면
    이 기본값은 쓰이지 않는다.
    """
    count = os.cpu_count()
    if not count:
        return 1
    return max(1, count - 1)


class ForestModel:
    """`RandomForestClassifier`/`ExtraTreesClassifier` 공통 래퍼.

    책임 범위(spec §9.1): 모델 생성, 파라미터 검증, fit 상태 관리, canonical
    class order 정렬, 누락/예상외 클래스 방어, sample_weight 전달, seed/n_jobs
    설정. 피처 조립·fold 분할·확률 파일 저장은 `scripts/train_rf.py` 의 몫이다.

    `class_order` 를 주면 `classes_` 는 그 순서를 그대로 반환하고,
    `predict_proba` 의 열도 **실제로** 그 순서에 맞춰 재배열된다 — sklearn
    estimator 는 항상 `np.unique(y)` 순서(사전식)로 `classes_`/확률 열을 내는데,
    `class_order` 가 사전식이 아니면(예: 팀이 별도로 고정한 순서) 이 둘이
    어긋난다. `classes_` 속성만 바꾸고 확률 배열은 그대로 두면 열 이름과 실제
    값이 어긋난 채로 조용히 저장된다 — 그래서 fit 시점에 `estimator.classes_`
    위치를 `class_order` 위치로 옮기는 인덱스(`_reorder_index_`)를 만들어 두고
    `predict_proba` 에서 매번 적용한다. `class_order` 를 안 주면 재정렬하지
    않고 sklearn 순서를 그대로 쓴다.
    """

    def __init__(
        self,
        kind: str,
        *,
        class_order: Sequence[str] | None = None,
        random_state: int = SEED,
        n_jobs: int | None = None,
        **params: Any,
    ) -> None:
        key = kind.strip().lower()
        if key not in _REGISTRY:
            raise ValueError(f"알 수 없는 모델 {kind!r}. 사용 가능: {registered_models()}")

        estimator_cls = _REGISTRY[key]
        self.kind = key
        self.name = _CANONICAL_NAME[estimator_cls]
        self.class_order = list(class_order) if class_order is not None else None
        self.random_state = random_state
        self.n_jobs = default_n_jobs() if n_jobs is None else n_jobs

        try:
            self.estimator_ = estimator_cls(
                random_state=random_state, n_jobs=self.n_jobs, **params
            )
        except TypeError as exc:
            raise ValueError(f"{self.name} 에 지원하지 않는 파라미터: {exc}") from exc

        self.classes_: np.ndarray | None = None
        self._reorder_index_: np.ndarray | None = None
        self._fitted = False

    # -- fit ---------------------------------------------------------------
    def fit(
        self, X, y: Sequence, sample_weight: np.ndarray | None = None
    ) -> "ForestModel":
        y = np.asarray(y).astype(str)

        if self.class_order is not None:
            observed = set(y.tolist())
            canonical = set(self.class_order)
            unexpected = sorted(observed - canonical)
            if unexpected:
                raise ValueError(
                    f"train 라벨에 canonical class order 밖의 값이 있다: {unexpected}"
                )
            missing = sorted(canonical - observed)
            if missing:
                raise ValueError(
                    f"train partition 에 canonical 클래스가 없다: {missing} "
                    "— 26개(또는 지정한 전체) 확률 계약을 지킬 수 없다"
                )

        weight = self._validate_sample_weight(sample_weight, n=len(y))

        self.estimator_.fit(X, y, sample_weight=weight)
        estimator_classes = np.asarray(self.estimator_.classes_).astype(str)

        if self.class_order is None:
            self.classes_ = estimator_classes
            self._reorder_index_ = None
        else:
            # 위에서 이미 관측 라벨 집합 == canonical 집합을 확인했으므로 이건
            # "일어나면 안 되는" 상태에 대한 방어일 뿐이다(정상 경로에서는 항상 같다).
            if set(estimator_classes.tolist()) != set(self.class_order):
                raise RuntimeError(
                    "학습 후 estimator.classes_ 집합이 canonical class order 집합과 "
                    f"다르다: {sorted(estimator_classes.tolist())} vs "
                    f"{sorted(self.class_order)}"
                )
            position = {c: i for i, c in enumerate(estimator_classes)}
            self._reorder_index_ = np.array(
                [position[c] for c in self.class_order], dtype=np.int64
            )
            self.classes_ = np.array(self.class_order)

        self._fitted = True
        return self

    @staticmethod
    def _validate_sample_weight(
        sample_weight: np.ndarray | None, *, n: int
    ) -> np.ndarray | None:
        if sample_weight is None:
            return None
        weight = np.asarray(sample_weight, dtype=np.float64)
        if len(weight) != n:
            raise ValueError(f"sample_weight 길이 {len(weight)} 가 y 길이 {n} 와 다르다")
        if not np.all(np.isfinite(weight)):
            raise ValueError("sample_weight 에 NaN/Inf 가 있다")
        if np.any(weight < 0):
            raise ValueError("sample_weight 에 음수가 있다")
        return weight

    # -- predict -------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        proba = np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)
        if self._reorder_index_ is not None:
            proba = proba[:, self._reorder_index_]
        n_classes = len(self.classes_)
        if proba.ndim != 2 or proba.shape[1] != n_classes:
            raise RuntimeError(
                f"predict_proba 형상 오류: {proba.shape}, 기대 (n, {n_classes})"
            )
        if not np.all(np.isfinite(proba)):
            raise RuntimeError("predict_proba 결과에 NaN/Inf 가 있다")
        if (proba < 0).any() or (proba > 1).any():
            raise RuntimeError("predict_proba 결과가 [0,1] 범위를 벗어난다")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("predict_proba 행별 확률 합이 1 이 아니다")
        return proba

    @property
    def feature_importances_(self) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.estimator_.feature_importances_, dtype=np.float64)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} 모델이 학습되지 않았다. fit() 을 먼저 호출한다.")

    def __repr__(self) -> str:
        return f"ForestModel({self.name!r}, fitted={self._fitted})"


def create_model(kind: str, **kwargs: Any) -> ForestModel:
    return ForestModel(kind, **kwargs)
