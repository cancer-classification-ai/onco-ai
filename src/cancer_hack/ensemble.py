"""OOF에서만 가중치를 학습하는 확률 앙상블."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score


def _probability_stack(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    if not probabilities:
        raise ValueError("앙상블할 확률 배열이 없다")
    arrays = [np.asarray(values, dtype=np.float64) for values in probabilities]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[1] < 2 or any(values.shape != shape for values in arrays):
        raise ValueError("모든 확률 배열은 같은 (n, classes) 형상이어야 한다")
    stack = np.stack(arrays, axis=0)
    if not np.isfinite(stack).all() or (stack < 0).any():
        raise ValueError("확률에 음수·NaN·무한대가 있다")
    totals = stack.sum(axis=2, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("합이 0인 확률 행이 있다")
    return stack / totals


def weighted_average(
    probabilities: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    """비음수·합 1 가중치로 모델 확률을 섞는다."""

    stack = _probability_stack(probabilities)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(stack),):
        raise ValueError(f"weights 형상 {weights.shape}, 기대 {(len(stack),)}")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights 는 유한한 비음수이고 합이 양수여야 한다")
    weights = weights / weights.sum()
    return np.tensordot(weights, stack, axes=(0, 0))


class MacroF1Blender:
    """Macro F1을 기준으로 비음수 확률 혼합 가중치를 찾는다.

    평가용 점수는 이 클래스의 `train_macro_f1_`가 아니라 별도 held-out fold에서 계산해야
    한다. 최종 test 변환용 가중치는 전체 OOF에 다시 맞춘다.
    """

    def __init__(
        self,
        *,
        steps: Sequence[float] = (0.25, 0.10, 0.05, 0.02, 0.01),
        max_passes: int = 5,
        tolerance: float = 1e-12,
    ) -> None:
        if not steps or any(step <= 0 or step > 1 for step in steps):
            raise ValueError("steps 는 0보다 크고 1 이하여야 한다")
        if max_passes < 1:
            raise ValueError("max_passes 는 1 이상이어야 한다")
        self.steps = tuple(float(step) for step in steps)
        self.max_passes = int(max_passes)
        self.tolerance = float(tolerance)

    def fit(
        self,
        probabilities: Sequence[np.ndarray],
        y_true: Sequence,
        classes: Sequence[str],
    ) -> "MacroF1Blender":
        stack = _probability_stack(probabilities)
        n_models, n_rows, n_classes = stack.shape
        classes = np.asarray(classes, dtype=str)
        if classes.shape != (n_classes,) or len(set(classes.tolist())) != n_classes:
            raise ValueError("classes 가 확률 열과 맞지 않거나 중복됐다")
        y_true = np.asarray(y_true, dtype=str)
        if y_true.shape != (n_rows,):
            raise ValueError(f"y_true 형상 {y_true.shape}, 기대 {(n_rows,)}")
        lookup = {label: index for index, label in enumerate(classes)}
        unknown = sorted(set(y_true.tolist()) - set(lookup))
        if unknown:
            raise ValueError(f"classes 에 없는 y_true: {unknown[:5]}")
        target = np.asarray([lookup[label] for label in y_true], dtype=np.int64)
        labels = np.arange(n_classes)

        def score(weights: np.ndarray) -> float:
            mixed = np.tensordot(weights, stack, axes=(0, 0))
            return float(
                f1_score(
                    target,
                    mixed.argmax(axis=1),
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )

        starts = [np.full(n_models, 1.0 / n_models)]
        starts.extend(np.eye(n_models, dtype=np.float64))
        weights = starts[0]
        best_score = score(weights)
        for candidate in starts[1:]:
            candidate_score = score(candidate)
            if candidate_score > best_score + self.tolerance:
                weights = candidate.copy()
                best_score = candidate_score

        history = [{"step": None, "score": best_score, "weights": weights.tolist()}]
        for step in self.steps:
            for _ in range(self.max_passes):
                improved = False
                for donor in range(n_models):
                    for receiver in range(n_models):
                        if donor == receiver or weights[donor] <= 0:
                            continue
                        amount = min(step, float(weights[donor]))
                        candidate = weights.copy()
                        candidate[donor] -= amount
                        candidate[receiver] += amount
                        candidate_score = score(candidate)
                        if candidate_score > best_score + self.tolerance:
                            weights = candidate
                            best_score = candidate_score
                            improved = True
                history.append(
                    {"step": step, "score": best_score, "weights": weights.tolist()}
                )
                if not improved:
                    break

        self.classes_ = classes
        self.weights_ = weights / weights.sum()
        self.train_macro_f1_ = score(self.weights_)
        self.history_ = history
        return self

    def predict_proba(self, probabilities: Sequence[np.ndarray]) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit() 을 먼저 호출해야 한다")
        return weighted_average(probabilities, self.weights_)

    def predict(self, probabilities: Sequence[np.ndarray]) -> np.ndarray:
        mixed = self.predict_proba(probabilities)
        return self.classes_[mixed.argmax(axis=1)]
