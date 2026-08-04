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


def crossfit_calibrated_blend(
    oof_arrays: Sequence[np.ndarray],
    y_true: np.ndarray,
    classes: Sequence[str],
    fold_values: np.ndarray,
    weights: Sequence[float],
):
    """고정 가중 블렌드에 로짓 보정을 **교차적합**으로 얹는다.

    fold 를 뺀 나머지에서 클래스별 바이어스를 찾고 그 fold 에만 적용한다. 전체 OOF 에
    한 번에 맞추면 fit 과 evaluate 가 같은 행을 보게 돼 점수가 부풀고, 그 값으로는
    구성을 고를 수 없다(실측으로 0.5210 -> 0.5438 만큼 부푼다).

    `(raw_blend, calibrated, fold_results)` 를 낸다. 앞의 둘은 OOF 행 순서 그대로다.

    `scripts/calibrate_ensemble.py` 와 재현 노트북이 **같은 이 함수를 부른다.**
    로직이 두 벌이면 노트북이 제출본을 재현하지 못하는 순간이 오는데, 그때는 대회
    심사에서 코드와 제출물이 어긋난 것으로 보인다.
    """
    from .calibration import MacroF1LogitBias  # 순환 import 회피
    from .metrics import macro_f1

    oof_arrays = [np.asarray(values, dtype=np.float64) for values in oof_arrays]
    weights = np.asarray(weights, dtype=np.float64)
    class_array = np.asarray(classes)

    raw_blend = np.zeros_like(oof_arrays[0])
    calibrated = np.zeros_like(oof_arrays[0])
    fold_results = []

    for fold in sorted(np.unique(fold_values).tolist()):
        train_mask = fold_values != fold
        valid_mask = ~train_mask

        train_blend = weighted_average([v[train_mask] for v in oof_arrays], weights)
        calibrator = MacroF1LogitBias().fit(train_blend, y_true[train_mask], classes)

        valid_blend = weighted_average([v[valid_mask] for v in oof_arrays], weights)
        valid_adjusted = calibrator.predict_proba(valid_blend)
        raw_blend[valid_mask] = valid_blend
        calibrated[valid_mask] = valid_adjusted

        fold_results.append(
            {
                "fold": int(fold),
                "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "weights": [float(w) for w in weights],
                "bias": calibrator.bias_by_class(),
                "raw_blend_macro_f1": macro_f1(
                    y_true[valid_mask], class_array[valid_blend.argmax(axis=1)]
                ),
                "calibrated_macro_f1": macro_f1(
                    y_true[valid_mask], class_array[valid_adjusted.argmax(axis=1)]
                ),
            }
        )

    return raw_blend, calibrated, fold_results


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


class GreedyEnsembleSelector:
    """Caruana 방식 그리디 선택 — 라이브러리에서 멤버를 **복원 허용**으로 하나씩 담는다.

    매 라운드마다 "지금 담긴 것들의 평균에 이 멤버를 하나 더 넣으면 macro F1 이 가장
    오르는가" 를 보고 하나를 담는다. 같은 멤버를 여러 번 담을 수 있고, **담긴 횟수가 곧
    가중치**가 된다. 그래서 멤버 선택과 가중 최적화가 한 번에 끝난다.

    `MacroF1Blender` 와 무엇이 다른가
    ---------------------------------
    `MacroF1Blender` 는 멤버 수만큼의 연속 가중치를 좌표하강으로 찾는다. 멤버가 3~4개일
    때는 그걸로 충분하지만, 라이브러리가 100개를 넘으면 파라미터가 100개가 되어 OOF 에
    과적합한다. 그리디는 라운드 수(`n_rounds`)가 곧 복잡도 상한이라 멤버가 몇 개든
    파라미터가 그만큼 늘지 않는다 — Caruana 가 복원을 허용한 이유도 같다.

    과적합 방지
    -----------
    이 클래스가 내는 `train_macro_f1_` 은 **선택에 쓴 바로 그 행에서 잰 값이라 낙관적이다.**
    라이브러리가 클수록 심해진다. 정직한 점수는 반드시 held-out fold 에서 따로 재야 한다
    (`scripts/greedy_blend.py` 가 교차적합으로 그렇게 한다).

    `bag_fraction` 은 Caruana 의 bagging 이다. 라운드마다 라이브러리의 일부만 후보로 두면
    "우연히 이 fold 에서 좋아 보이는 멤버" 가 매번 뽑히는 걸 막는다. 1.0 이면 끄는 것.
    """

    def __init__(
        self,
        *,
        n_rounds: int = 30,
        bag_fraction: float = 1.0,
        bag_rounds: int = 1,
        random_state: int = 0,
        init_top_k: int = 1,
    ) -> None:
        if n_rounds < 1:
            raise ValueError("n_rounds 는 1 이상이어야 한다")
        if not 0 < bag_fraction <= 1:
            raise ValueError("bag_fraction 은 0 초과 1 이하여야 한다")
        if bag_rounds < 1:
            raise ValueError("bag_rounds 는 1 이상이어야 한다")
        if init_top_k < 0:
            raise ValueError("init_top_k 는 0 이상이어야 한다")
        self.n_rounds = int(n_rounds)
        self.bag_fraction = float(bag_fraction)
        self.bag_rounds = int(bag_rounds)
        self.random_state = int(random_state)
        self.init_top_k = int(init_top_k)

    def fit(
        self,
        probabilities: Sequence[np.ndarray],
        y_true: Sequence,
        classes: Sequence[str],
    ) -> "GreedyEnsembleSelector":
        stack = _probability_stack(probabilities)
        classes = np.asarray(classes, dtype=str)
        y_true = np.asarray(y_true, dtype=str)
        if y_true.shape != (stack.shape[1],):
            raise ValueError(f"y_true 형상 {y_true.shape}, 기대 {(stack.shape[1],)}")
        if stack.shape[2] != len(classes):
            raise ValueError("확률 열 수와 classes 길이가 다르다")

        lookup = {label: index for index, label in enumerate(classes)}
        unknown = sorted(set(y_true.tolist()) - set(lookup))
        if unknown:
            raise ValueError(f"classes 에 없는 y_true: {unknown[:5]}")
        target = np.asarray([lookup[label] for label in y_true], dtype=np.int64)
        labels = np.arange(len(classes))

        def score_of(total: np.ndarray, count: int) -> float:
            return float(
                f1_score(target, (total / count).argmax(axis=1),
                         labels=labels, average="macro", zero_division=0)
            )

        n_members = len(stack)
        rng = np.random.default_rng(self.random_state)
        counts = np.zeros(n_members, dtype=np.int64)

        # 라운드마다 후보를 새로 뽑으므로 bag 별로 독립 실행한 뒤 횟수를 합친다.
        for _ in range(self.bag_rounds):
            bag_counts = np.zeros(n_members, dtype=np.int64)
            total = np.zeros(stack.shape[1:], dtype=np.float64)
            picked = 0

            # 시작점 — 단독 최고 몇 개를 먼저 담는다. 빈 상태에서 시작하면 1라운드가
            # 사실상 "단독 최고 고르기" 라 같은 일을 두 번 하게 된다.
            if self.init_top_k:
                solo = np.asarray([score_of(stack[i], 1) for i in range(n_members)])
                for index in np.argsort(-solo)[: self.init_top_k]:
                    total += stack[index]
                    bag_counts[index] += 1
                    picked += 1

            history: list[dict] = []
            for _round in range(self.n_rounds):
                if self.bag_fraction >= 1.0:
                    candidates = np.arange(n_members)
                else:
                    size = max(1, int(round(n_members * self.bag_fraction)))
                    candidates = rng.choice(n_members, size=size, replace=False)

                best_index, best_score = -1, -np.inf
                for index in candidates:
                    candidate = score_of(total + stack[index], picked + 1)
                    if candidate > best_score + 1e-12:
                        best_index, best_score = int(index), candidate
                if best_index < 0:
                    break
                total += stack[best_index]
                bag_counts[best_index] += 1
                picked += 1
                history.append({"round": _round, "picked": best_index, "macro_f1": best_score})

            counts += bag_counts
            self.history_ = history

        self.counts_ = counts
        self.weights_ = counts / counts.sum()
        self.train_macro_f1_ = score_of(
            np.tensordot(self.weights_, stack, axes=(0, 0)), 1
        )
        return self

    def predict_proba(self, probabilities: Sequence[np.ndarray]) -> np.ndarray:
        return weighted_average(probabilities, self.weights_)

    def predict(self, probabilities: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("클래스 이름이 필요하다 — predict_proba 의 argmax 를 쓴다")
