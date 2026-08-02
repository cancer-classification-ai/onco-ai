"""OOF 라벨로 학습하는 다중 클래스 결정 로짓 보정.

확률 보정(calibration)이라는 파일명을 따르지만 목적은 확률의 신뢰도 추정이 아니라 Macro
F1 의 클래스별 결정 경계를 옮기는 것이다. 같은 OOF로 오프셋을 학습하고 그 점수를 그대로
보고하면 낙관 편향이 생기므로, 성능 평가는 반드시 바깥 코드에서 교차적합으로 수행한다.
최종 test 변환용 오프셋만 전체 OOF에 다시 맞춘다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score


def _validate_probabilities(proba: np.ndarray) -> np.ndarray:
    values = np.asarray(proba, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"확률은 (n, class>=2) 2차원이어야 한다: {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("확률에 음수·NaN·무한대가 있다")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("합이 0인 확률 행이 있다")
    return values / totals


def apply_logit_bias(
    proba: np.ndarray,
    bias: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """확률에 클래스별 logit 오프셋을 더하고 다시 정규화한다."""

    values = _validate_probabilities(proba)
    offsets = np.asarray(bias, dtype=np.float64)
    if offsets.shape != (values.shape[1],):
        raise ValueError(f"bias 형상 {offsets.shape}, 기대 {(values.shape[1],)}")
    if not np.isfinite(offsets).all():
        raise ValueError("bias 에 NaN·무한대가 있다")
    logits = np.log(np.clip(values, epsilon, 1.0)) + offsets
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


class MacroF1LogitBias:
    """결정 경계를 옮기는 클래스별 오프셋을 좌표 탐색으로 학습한다.

    비연속인 Macro F1 목적함수에 미분 최적화를 억지로 쓰지 않고, 큰 간격에서 작은
    간격으로 내려가는 결정론적 좌표 탐색을 쓴다. 오프셋 절댓값을 제한해 소수 클래스 몇
    건에 맞춘 과도한 이동을 막는다.
    """

    def __init__(
        self,
        *,
        steps: Sequence[float] = (0.5, 0.2, 0.1, 0.05, 0.02),
        max_passes: int = 4,
        max_abs_bias: float = 2.0,
        tolerance: float = 1e-12,
    ) -> None:
        if not steps or any(step <= 0 for step in steps):
            raise ValueError("steps 는 양수여야 한다")
        if max_passes < 1 or max_abs_bias <= 0:
            raise ValueError("max_passes/max_abs_bias 가 유효하지 않다")
        self.steps = tuple(float(step) for step in steps)
        self.max_passes = int(max_passes)
        self.max_abs_bias = float(max_abs_bias)
        self.tolerance = float(tolerance)

    def fit(
        self,
        proba: np.ndarray,
        y_true: Sequence,
        classes: Sequence[str],
    ) -> "MacroF1LogitBias":
        values = _validate_probabilities(proba)
        classes = np.asarray(classes, dtype=str)
        if classes.shape != (values.shape[1],) or len(set(classes.tolist())) != len(classes):
            raise ValueError("classes 가 확률 열과 맞지 않거나 중복됐다")
        y_true = np.asarray(y_true, dtype=str)
        if y_true.shape != (len(values),):
            raise ValueError(f"y_true 형상 {y_true.shape}, 기대 {(len(values),)}")
        lookup = {label: index for index, label in enumerate(classes)}
        unknown = sorted(set(y_true.tolist()) - set(lookup))
        if unknown:
            raise ValueError(f"classes 에 없는 y_true: {unknown[:5]}")
        target = np.asarray([lookup[label] for label in y_true], dtype=np.int64)
        labels = np.arange(len(classes))
        base_logits = np.log(np.clip(values, 1e-12, 1.0))
        bias = np.zeros(len(classes), dtype=np.float64)

        def score(candidate: np.ndarray) -> float:
            prediction = (base_logits + candidate).argmax(axis=1)
            return float(
                f1_score(
                    target,
                    prediction,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )

        best_score = score(bias)
        history = [{"step": None, "score": best_score}]
        for step in self.steps:
            for _ in range(self.max_passes):
                improved = False
                for class_index in range(len(classes)):
                    current_value = bias[class_index]
                    best_value = current_value
                    local_score = best_score
                    for delta in (-2 * step, -step, step, 2 * step):
                        value = float(
                            np.clip(current_value + delta, -self.max_abs_bias, self.max_abs_bias)
                        )
                        if value == current_value:
                            continue
                        candidate = bias.copy()
                        candidate[class_index] = value
                        candidate_score = score(candidate)
                        if candidate_score > local_score + self.tolerance:
                            local_score = candidate_score
                            best_value = value
                    if best_value != current_value:
                        bias[class_index] = best_value
                        best_score = local_score
                        improved = True
                history.append({"step": step, "score": best_score})
                if not improved:
                    break

        # 모든 클래스에 같은 상수를 더하는 것은 softmax/argmax 에 영향을 주지 않는다.
        # 평균 0으로 고정해 저장값을 재현 가능하고 해석 가능하게 만든다.
        bias -= bias.mean()
        self.classes_ = classes
        self.bias_ = bias
        self.train_macro_f1_ = score(bias)
        self.history_ = history
        return self

    def predict_proba(self, proba: np.ndarray) -> np.ndarray:
        if not hasattr(self, "bias_"):
            raise RuntimeError("fit() 을 먼저 호출해야 한다")
        return apply_logit_bias(proba, self.bias_)

    def predict(self, proba: np.ndarray) -> np.ndarray:
        adjusted = self.predict_proba(proba)
        return self.classes_[adjusted.argmax(axis=1)]

    def bias_by_class(self) -> dict[str, float]:
        if not hasattr(self, "bias_"):
            raise RuntimeError("fit() 을 먼저 호출해야 한다")
        return {
            str(label): float(value) for label, value in zip(self.classes_, self.bias_)
        }
