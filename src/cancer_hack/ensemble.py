"""OOF에서만 가중치를 학습하는 확률 앙상블."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score

#: 0 확률에서 log·역수가 터지는 걸 막는 바닥값. 기하·조화평균에서만 쓴다.
_EPS = 1e-12


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


def _validated_weights(weights: Sequence[float], n_members: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_members,):
        raise ValueError(f"weights 형상 {weights.shape}, 기대 {(n_members,)}")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights 는 유한한 비음수이고 합이 양수여야 한다")
    return weights / weights.sum()


def weighted_average(
    probabilities: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    """비음수·합 1 가중치로 모델 확률을 섞는다."""

    stack = _probability_stack(probabilities)
    return np.tensordot(_validated_weights(weights, len(stack)), stack, axes=(0, 0))


def power_mean(
    probabilities: Sequence[np.ndarray], weights: Sequence[float], power: float = 1.0
) -> np.ndarray:
    """가중 거듭제곱평균. `power=1` 산술 · `0` 기하 · `-1` 조화.

    산술평균은 **한 멤버가 확신하면 그대로 끌려간다.** 0.99 를 낸 멤버 하나가 0.1 을 낸
    셋을 이긴다. 지수를 내리면 그 반대가 된다 — 기하평균은 한 멤버가 0 에 가까우면
    전체를 0 으로 끌어내려서, 모든 멤버가 동의해야 높은 값이 남는다.

    어느 쪽이 맞는지는 멤버의 성격에 달렸다. 서로 다른 것을 잘 맞히는 멤버끼리는
    산술이 낫고(한쪽이 아는 걸 살린다), 같은 것을 보며 잡음만 다른 멤버끼리는
    기하가 낫다(잡음이 상쇄된다). 그래서 재 봐야 한다.
    """

    stack = _probability_stack(probabilities)
    weights = _validated_weights(weights, len(stack))
    if not np.isfinite(power):
        raise ValueError("power 는 유한해야 한다")

    if abs(power) < 1e-12:  # 기하평균 — log 합
        mixed = np.exp(np.tensordot(weights, np.log(np.maximum(stack, _EPS)), axes=(0, 0)))
    else:
        mixed = np.tensordot(weights, np.maximum(stack, _EPS) ** power, axes=(0, 0)) ** (1.0 / power)

    totals = mixed.sum(axis=1, keepdims=True)
    return mixed / np.maximum(totals, _EPS)


def rank_average(
    probabilities: Sequence[np.ndarray],
    weights: Sequence[float],
    reference: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """클래스 열마다 **순위(백분위)**로 바꾼 뒤 섞는다.

    멤버마다 확률의 눈금이 다르다. DL 은 뾰족하고 RF 는 뭉툭한데, 산술평균은 그 눈금을
    그대로 받아서 뾰족한 쪽에 끌려간다. 순위로 바꾸면 눈금이 사라지고 **순서만** 남는다.

    열 안에서 순위를 매기는 게 핵심이다 — 클래스 c 를 놓고 행을 줄 세운다. 그러면
    38행짜리 DLBC 열도 786행짜리 BRCA 열과 같은 (0, 1] 범위로 펴진다. macro F1 이
    26클래스를 동등하게 세므로 이 펴짐이 그대로 이득이 될 수 있다.
    대신 "얼마나 확신하는가" 는 통째로 버려진다.

    규정 — `reference` 를 반드시 train 쪽으로 준다
    ---------------------------------------------
    `reference=None` 이면 **자기 자신 안에서** 순위를 매긴다. 이걸 test 에 쓰면
    test 행끼리 줄을 세우게 되고, 그러면 "test 한 행만 따로 넣어도 같은 결과가 나오는가"
    를 못 지킨다 — 대회 규정이 콕 집어 든 "test 자신의 통계값으로 처리" 에 해당한다.

    그래서 실제 경로에서는 train(또는 fold 의 train 부분) 확률을 `reference` 로 넘긴다.
    그러면 각 행이 **미리 정해진 함수**를 통과할 뿐이라 행 하나만 넣어도 같은 값이 나오고,
    fit 은 train 에서만 일어난다. `None` 은 분석용이다.
    """

    stack = _probability_stack(probabilities)
    weights = _validated_weights(weights, len(stack))
    ref = stack if reference is None else _probability_stack(reference)
    if len(ref) != len(stack) or ref.shape[2] != stack.shape[2]:
        raise ValueError("reference 는 같은 멤버 수·클래스 수여야 한다")

    ranked = np.empty_like(stack)
    n_ref = ref.shape[1]
    for member in range(len(stack)):
        for column in range(stack.shape[2]):
            grid = np.sort(ref[member, :, column])
            ranked[member, :, column] = (
                np.searchsorted(grid, stack[member, :, column], side="right") / n_ref
            )

    mixed = np.tensordot(weights, np.maximum(ranked, _EPS), axes=(0, 0))
    return mixed / mixed.sum(axis=1, keepdims=True)


def trimmed_mean(
    probabilities: Sequence[np.ndarray], trim: int = 1
) -> np.ndarray:
    """양 끝 `trim` 개를 버리고 평균한다 (가중치 없음).

    멤버 하나가 망가져도 결과가 안 무너지게 하는 쪽이다. `trim=0` 이면 그냥 평균이고,
    멤버가 홀수이고 양쪽을 최대로 자르면 중앙값이 된다.
    """

    stack = _probability_stack(probabilities)
    n_members = len(stack)
    if trim < 0 or 2 * trim >= n_members:
        raise ValueError(f"trim 은 0 이상 {(n_members - 1) // 2} 이하여야 한다")
    if trim == 0:
        mixed = stack.mean(axis=0)
    else:
        ordered = np.sort(stack, axis=0)
        mixed = ordered[trim : n_members - trim].mean(axis=0)
    return mixed / mixed.sum(axis=1, keepdims=True)


#: `combine()` 이 받는 결합 형태. 스크립트·노트북이 문자열로 고를 수 있게 열어 둔다.
BLEND_FORMS = ("mean", "geometric", "harmonic", "rank", "median", "trimmed")


def combine(
    probabilities: Sequence[np.ndarray],
    weights: Sequence[float],
    form: str = "mean",
    reference: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """이름으로 결합 형태를 고른다. 기본값 `mean` 이 기존 동작이다.

    `reference` 는 `rank` 에서만 쓰인다 — 순위를 매길 기준 분포이고 train 쪽을 준다.
    """

    if form == "mean":
        return weighted_average(probabilities, weights)
    if form == "geometric":
        return power_mean(probabilities, weights, power=0.0)
    if form == "harmonic":
        return power_mean(probabilities, weights, power=-1.0)
    if form == "rank":
        return rank_average(probabilities, weights, reference=reference)
    if form == "median":
        return trimmed_mean(probabilities, trim=(len(probabilities) - 1) // 2)
    if form == "trimmed":
        return trimmed_mean(probabilities, trim=1)
    raise ValueError(f"모르는 결합 형태 {form!r} — {BLEND_FORMS} 중 하나여야 한다")


def crossfit_calibrated_blend(
    oof_arrays: Sequence[np.ndarray],
    y_true: np.ndarray,
    classes: Sequence[str],
    fold_values: np.ndarray,
    weights: Sequence[float],
    form: str = "mean",
):
    """고정 가중 블렌드에 로짓 보정을 **교차적합**으로 얹는다.

    fold 를 뺀 나머지에서 클래스별 바이어스를 찾고 그 fold 에만 적용한다. 전체 OOF 에
    한 번에 맞추면 fit 과 evaluate 가 같은 행을 보게 돼 점수가 부풀고, 그 값으로는
    구성을 고를 수 없다(실측으로 0.5210 -> 0.5438 만큼 부푼다).

    `(raw_blend, calibrated, fold_results)` 를 낸다. 앞의 둘은 OOF 행 순서 그대로다.

    `form` 은 확률을 합치는 방식이다(`BLEND_FORMS`). 기본 `"mean"` 이 기존 동작이라
    이 인자를 안 주던 호출은 그대로 같은 값을 낸다.

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

        # `rank` 는 fold 의 train 부분을 기준 분포로 삼는다. valid 를 valid 안에서
        # 줄 세우면 test 에 그대로 못 옮기는 방식이 되고, 점수도 낙관적으로 나온다.
        train_parts = [v[train_mask] for v in oof_arrays]
        train_blend = combine(train_parts, weights, form, reference=train_parts)
        calibrator = MacroF1LogitBias().fit(train_blend, y_true[train_mask], classes)

        valid_blend = combine(
            [v[valid_mask] for v in oof_arrays], weights, form, reference=train_parts
        )
        valid_adjusted = calibrator.predict_proba(valid_blend)
        raw_blend[valid_mask] = valid_blend
        calibrated[valid_mask] = valid_adjusted

        fold_results.append(
            {
                "fold": int(fold),
                "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "form": form,
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

    `form`
    ------
    담긴 멤버들을 무엇으로 합치느냐다. 기본 `"mean"` 은 확률의 산술평균이고,
    `"geometric"` 은 로그 확률의 합이다(= 기하평균). **선택 과정에도 같이 적용된다** —
    합치는 방식이 바뀌면 어느 멤버를 담는 게 이득인지도 바뀌므로, 합칠 때만 기하로
    하고 고를 때는 산술로 하면 서로 다른 목적함수를 최적화하게 된다.

    로그 합의 argmax 는 기하평균의 argmax 와 같아서(지수는 단조) 선택 루프는 그대로다.
    """

    def __init__(
        self,
        *,
        n_rounds: int = 30,
        bag_fraction: float = 1.0,
        bag_rounds: int = 1,
        random_state: int = 0,
        init_top_k: int = 1,
        form: str = "mean",
    ) -> None:
        if n_rounds < 1:
            raise ValueError("n_rounds 는 1 이상이어야 한다")
        if not 0 < bag_fraction <= 1:
            raise ValueError("bag_fraction 은 0 초과 1 이하여야 한다")
        if bag_rounds < 1:
            raise ValueError("bag_rounds 는 1 이상이어야 한다")
        if init_top_k < 0:
            raise ValueError("init_top_k 는 0 이상이어야 한다")
        if form not in ("mean", "geometric"):
            raise ValueError(f"form 은 'mean' 또는 'geometric' 이어야 한다 (받은 값 {form!r})")
        self.form = form
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

        # 기하평균이면 로그 공간에서 더한다. 누적합의 argmax 가 곧 기하평균의 argmax 라
        # 아래 선택 루프는 한 줄도 안 바뀐다.
        work = np.log(np.maximum(stack, _EPS)) if self.form == "geometric" else stack

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
            total = np.zeros(work.shape[1:], dtype=np.float64)
            picked = 0

            # 시작점 — 단독 최고 몇 개를 먼저 담는다. 빈 상태에서 시작하면 1라운드가
            # 사실상 "단독 최고 고르기" 라 같은 일을 두 번 하게 된다.
            if self.init_top_k:
                solo = np.asarray([score_of(work[i], 1) for i in range(n_members)])
                for index in np.argsort(-solo)[: self.init_top_k]:
                    total += work[index]
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
                    candidate = score_of(total + work[index], picked + 1)
                    if candidate > best_score + 1e-12:
                        best_index, best_score = int(index), candidate
                if best_index < 0:
                    break
                total += work[best_index]
                bag_counts[best_index] += 1
                picked += 1
                history.append({"round": _round, "picked": best_index, "macro_f1": best_score})

            counts += bag_counts
            self.history_ = history

        self.counts_ = counts
        self.weights_ = counts / counts.sum()
        self.train_macro_f1_ = score_of(
            np.tensordot(self.weights_, work, axes=(0, 0)), 1
        )
        return self

    def predict_proba(self, probabilities: Sequence[np.ndarray]) -> np.ndarray:
        return combine(probabilities, self.weights_, self.form)

    def predict(self, probabilities: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("클래스 이름이 필요하다 — predict_proba 의 argmax 를 쓴다")
