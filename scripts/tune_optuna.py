#!/usr/bin/env python
"""f4r 하이퍼파라미터 Optuna 탐색 — 파이프라인은 그대로 두고 파라미터만 흔든다.

    # 1) 재현 검증 — 기준선 파라미터로 f4r 실측값이 그대로 나오는지 본다 (필수 선행)
    .\\.venv\\Scripts\\python.exe scripts\\tune_optuna.py --verify --cv both

    # 2) 탐색 — skf 와 sgkf 를 둘 다 재고 평균으로 고른다
    .\\.venv\\Scripts\\python.exe scripts\\tune_optuna.py --cv both --n-trials 60

    # 3) 최고 설정을 train_gbdt.py 로 재실행 (OOF·test·제출 파일까지)
    .\\.venv\\Scripts\\python.exe scripts\\tune_optuna.py --show-best

## 무엇이 같고 무엇이 다른가

`train_gbdt.run_config` 와 **피처·fold·fold 내 fit 순서가 같다.** `Dataset` 을 그대로
import 해서 쓰므로 열 순서까지 동일하다 — `colsample_bytree` 가 열 순서에 반응하기
때문에 이건 재현의 전제조건이다.

다른 건 셋뿐이고 전부 점수에 영향이 없다.

    - test `predict_proba` 를 하지 않는다 (탐색에는 안 쓴다). trial 당 ~10초를 아낀다.
    - OOF/test/제출 csv 를 쓰지 않는다. 최종 산출물은 `train_gbdt.py` 로 다시 뽑는다.
    - fold 마다 `trial.report()` 를 호출해 가망 없는 trial 을 중간에 끊는다.

`--verify` 가 이 주장을 실제로 검증한다. 기준선 파라미터로 돌려 fold 점수 5개가
`artifacts/logs/xgb_v2_f4r_skf5_k500_s42.json` 과 소수 4자리까지 같은지 본다.
여기서 어긋나면 탐색 결과는 전부 무효다.

## 규정 준수

`train_gbdt.py` 와 같다. chi2 선택·BurdenBinner·balanced 가중치는 fold 의 train
부분에서만 fit 하고, test 는 이 스크립트에서 아예 건드리지 않는다. `eval_set` 도
쓰지 않는다 — 탐색 대상에 `n_estimators` 를 넣어 반복수를 CV 로 고르게 한다.

## `--cv both` — 두 분할을 같이 재는 이유

`--cv both` 는 trial 마다 skf5 와 sgkf5 를 둘 다 돌린다. trial 당 시간이 두 배가 되지만
얻는 게 있다. 두 분할은 **같은 모델을 다르게 재는 두 자**이지 난이도가 다른 두 과제가
아니다. sgkf 가 늘 0.006 쯤 높은데 그 이득은 전부 쌍둥이 행에서 나온다 — 변이 프로파일이
같은 행을 한 fold 로 묶어 모델이 파트너 라벨을 외우지 못하게 막기 때문이다.

그래서 **sgkf 가 높다는 건 그 설정이 더 좋다는 뜻이 아니다.** 두 숫자에서 읽을 것은
절대 높이가 아니라 **둘의 벌어짐**이다. sgkf−skf 가 크면 그 설정은 쌍둥이 암기에
그만큼 더 기대고 있다는 뜻이고, LB 에는 그 보호막이 없다. `sgkf_minus_skf` 를
trial 마다 기록하는 이유다.

`--objective` 로 무엇을 최대화할지 고른다.

    mean   두 분할 평균 (`--cv both` 의 기본값). 양쪽이 같이 올라야 이긴다
    skf    주 지표 하나만. `research/05` 가 결정 기준으로 삼는 값
    sgkf   sgkf 하나만
    min    둘 중 낮은 쪽. skf 가 늘 낮아 사실상 skf 와 같아진다

## 주의: CV 를 올린다고 LB 가 오르지 않았다

`research/05_f4r_spec_and_next_steps.md` §2 기준으로 CV 가 오른 구성 세 개가 LB 에서
전부 −2σ 넘게 떨어졌다. 그래서 이 스크립트는 CV 점수만 남기지 않고 trial 마다
**일반화 격차(train − valid)** 를 같이 기록한다. `--show-best` 는 CV 최고와 함께
"격차가 기준선보다 작으면서 CV 도 높은" 후보를 따로 뽑아 준다.

## DACON 제출

이 스크립트는 파일조차 만들지 않는다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 이 파일의 문서·로그에는 cp949 에 없는 글자가 있다(em dash, U+2212 빼기표).
# Windows 기본 스트림은 cp949 라 `--help` 조차 UnicodeEncodeError 로 죽고, 탐색
# 중간 출력에서 죽으면 몇 시간짜리 실행을 통째로 잃는다. 파일로 리다이렉트할 때도
# 같은 인코딩이 걸려서 로그가 깨졌다. 아무것도 찍기 전에 UTF-8 로 고정한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # 파이프·캡처 등 reconfigure 가 없는 경우
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import optuna  # noqa: E402

from cancer_hack.features_basic import BurdenBinner  # noqa: E402
from cancer_hack.metrics import macro_f1  # noqa: E402
from cancer_hack.models_gbdt import create_model, gpu_available, resolve_sample_weight  # noqa: E402
from cancer_hack.validation import (  # noqa: E402
    Chi2TopKSelector,
    check_all_classes_present,
    fold_column,
)
from train_gbdt import (  # noqa: E402
    BURDEN_COLUMNS,
    CONFIGS,
    GENE_BLOCKS,
    LATENT_BLOCKS,
    MODEL_PARAMS,
    MODULE_BLOCKS,
    PAIR_BLOCKS,
    SPARSE_BLOCKS,
    Dataset,
    log,
)

ARTIFACTS = PROJECT_ROOT / "artifacts"
TUNING_DIR = ARTIFACTS / "tuning"

#: `artifacts/logs/xgb_v2_f4r_{skf5,group5}_k500_s42.json` 의 실측값. `--verify` 의 정답지다.
BASELINE_CV = {
    "skf": {
        "fold_macro_f1": [0.4433, 0.4659, 0.4536, 0.4616, 0.4526],
        "oof_macro_f1": 0.4563,
    },
    "sgkf": {
        "fold_macro_f1": [0.4861, 0.4918, 0.4654, 0.4628, 0.4586],
        "oof_macro_f1": 0.4786,
    },
}
BASELINE_N_FEATURES = 1055

#: 탐색 대상이 아닌 고정값. 기준선과 같아야 한다.
FIXED_PARAMS = {"objective": "multi:softprob", "tree_method": "hist"}


def suggest_params(trial: optuna.Trial) -> dict:
    """탐색 공간. 기준선을 안쪽에 품도록 잡았다.

    `reg_alpha`·`gamma` 는 기준선이 0 이라 log 스케일을 쓰지 않는다. log 로 잡으면
    하한이 0 을 못 담아서 `enqueue_trial` 로 기준선을 그대로 넣을 수가 없다.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
    }


#: 기준선 f4r 을 탐색 공간의 좌표로 옮긴 것. `MODEL_PARAMS["xgb"]` 에 없는 축은
#: XGBoost 기본값(0/1)을 적는다. trial 0 으로 넣어 "기준선보다 나은가" 를 같은
#: 자로 재게 만든다.
BASELINE_POINT = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "colsample_bylevel": 1.0,
}


# ---------------------------------------------------------------- CV
def cross_validate(
    data: Dataset,
    params: dict,
    *,
    config: str,
    cv: str,
    topk: int,
    n_splits: int,
    seed: int,
    use_gpu,
    track_train: bool = True,
    trial: optuna.Trial | None = None,
    verbose: bool = False,
    prune_state: dict | None = None,
) -> dict:
    """f4r 의 fold 루프. `train_gbdt.run_config` 와 학습 경로가 같다.

    다른 점은 test 를 안 만지고 파일을 안 쓴다는 것뿐이다. `trial` 을 주면 fold 마다
    중간값을 보고해 pruner 가 끊을 수 있게 한다.

    `prune_state` 는 `--cv both` 에서 두 분할의 fold 를 **하나의 연속된 step 수열**로
    잇는다. 분할마다 step 0 부터 다시 세면 pruner 가 skf fold 0 과 sgkf fold 0 을 같은
    칸에 놓고 비교해 버린다. 두 분할은 높이가 다르니(sgkf 가 늘 위다) 그 비교는 무의미하다.
    """
    spec = CONFIGS[config]
    # 이 튜너는 dense + GENE_BLOCKS 만 만든다. fold 안에서 새로 fit 하는 블록
    # (TF-IDF·공변이 쌍·잠재·모듈)은 `assemble` 이 건너뛰고 여기서도 안 붙이므로,
    # 막지 않으면 f4rl 을 튜닝하고 f4r 숫자를 돌려받는다 — 조용히 틀린 답이 나온다.
    unsupported = [
        b
        for b in spec["blocks"]
        if b in SPARSE_BLOCKS or b in PAIR_BLOCKS or b in LATENT_BLOCKS or b in MODULE_BLOCKS
    ]
    if unsupported:
        raise ValueError(
            f"config {config} 의 블록 {unsupported} 은 이 튜너가 만들지 않는다. "
            "그대로 돌리면 그 블록이 빠진 채 학습돼 다른 config 의 점수가 나온다. "
            "scripts/train_gbdt.py 로 돌리거나, cross_validate 에 fold 스탠자를 옮겨 온다."
        )
    gene_blocks = [b for b in spec["blocks"] if b in GENE_BLOCKS]
    names, dense_train, _ = data.assemble(config)
    burden_positions = [i for i, n in enumerate(names) if n in BURDEN_COLUMNS]
    fold_ids = data.folds[fold_column(cv, n_splits)].to_numpy()
    group_keys = data.folds["group_key"].to_numpy()

    oof = np.zeros((len(data.y), len(data.classes)), dtype=np.float64)
    fold_scores: list[float] = []
    fold_train_scores: list[float] = []
    devices: list[str] = []
    n_features = len(names)
    started = time.perf_counter()

    for fold in range(n_splits):
        valid_index = np.where(fold_ids == fold)[0]
        train_index = np.where(fold_ids != fold)[0]
        check_all_classes_present(data.y, train_index, data.classes)

        x_train = dense_train.copy()

        # burden 2열 — 경계는 fold 의 train 부분에서만 잡는다.
        if burden_positions:
            binner = BurdenBinner().fit(data.rollup_train.iloc[train_index])
            wanted = [names[i] for i in burden_positions]
            x_train[:, burden_positions] = binner.transform(data.rollup_train)[
                wanted
            ].to_numpy(np.float32)

        # 유전자 블록 — chi2 도 fold 의 train 부분에서만 fit 한다.
        for block in gene_blocks:
            columns, gene_train, _ = data.gene[block]
            selector = Chi2TopKSelector(k=topk).fit(
                gene_train[train_index], data.y[train_index]
            )
            x_train = np.hstack([x_train, gene_train[:, selector.indices_]])

        n_features = x_train.shape[1]
        weight = resolve_sample_weight(
            spec["weight"], data.y[train_index], group_keys[train_index]
        )
        model = _fit(params, x_train[train_index], data.y[train_index], weight, seed, use_gpu)

        if list(model.classes_) != list(data.classes):
            raise RuntimeError(f"fold {fold} 의 클래스 순서가 전체와 다르다")

        devices.append(model.describe().get("device", "?"))
        oof[valid_index] = model.predict_proba(x_train[valid_index])
        score = macro_f1(data.y[valid_index], data.classes[oof[valid_index].argmax(axis=1)])
        fold_scores.append(score)

        if track_train:
            fold_train_scores.append(
                macro_f1(data.y[train_index], model.predict(x_train[train_index]))
            )
        del x_train

        if verbose:
            extra = f" train={fold_train_scores[-1]:.4f}" if track_train else ""
            log(f"    fold {fold + 1}/{n_splits}  Macro F1={score:.4f}{extra}")

        # pruning — fold 를 step 으로 삼는다. 같은 step 끼리 비교되므로 fold 난이도
        # 차이(fold 0 이 늘 낮다)는 상쇄된다.
        if trial is not None and prune_state is not None:
            prune_state["scores"].append(score)
            trial.report(float(np.mean(prune_state["scores"])), prune_state["step"])
            prune_state["step"] += 1
            if trial.should_prune():
                raise optuna.TrialPruned()

    oof_pred = data.classes[oof.argmax(axis=1)]
    result = {
        "oof_macro_f1": float(macro_f1(data.y, oof_pred)),
        "oof_macro_f1_singleton": float(
            macro_f1(data.y[data.singleton_mask], oof_pred[data.singleton_mask])
        ),
        "fold_macro_f1": [float(s) for s in fold_scores],
        "fold_mean": float(np.mean(fold_scores)),
        "n_features": int(n_features),
        # fold 중 하나라도 CPU 로 떨어졌으면 그 trial 의 시간은 다른 trial 과 비교할 수
        # 없다. 이 GPU 는 데스크톱도 함께 돌려서 여유 메모리가 출렁이고, 실제로 OOM 이
        # 나면 `_fit` 이 그 fold 만 CPU 로 재시도한다. 점수는 같아도 기록은 남겨야 한다.
        "device": devices[0] if devices else "?",
        "device_mixed": len(set(devices)) > 1,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if track_train:
        result["fold_train_macro_f1"] = [float(s) for s in fold_train_scores]
        result["oof_train_macro_f1"] = float(np.mean(fold_train_scores))
        result["generalization_gap"] = float(
            np.mean(fold_train_scores) - np.mean(fold_scores)
        )
    return result


def evaluate(data: Dataset, params: dict, args, *, trial=None, verbose=False) -> dict:
    """요청한 분할을 전부 돌리고 하나의 결과로 합친다.

    반환값은 분할별 결과를 `skf_`·`sgkf_` 로 접두사를 붙여 평평하게 편 것이다.
    Optuna `user_attrs` 는 중첩 dict 를 넣어도 되지만, 평평해야 나중에 표로 뽑고
    정렬하기가 쉽다.
    """
    prune_state = {"step": 0, "scores": []} if trial is not None else None
    out: dict = {}
    per_cv: dict[str, dict] = {}

    for cv in args.cv_list:
        if verbose:
            log(f"  [{cv}]")
        result = cross_validate(
            data,
            params,
            config=args.config,
            cv=cv,
            topk=args.topk,
            n_splits=args.n_splits,
            seed=args.seed,
            use_gpu=args.device,
            track_train=not args.no_track_train,
            trial=trial,
            verbose=verbose,
            prune_state=prune_state,
        )
        per_cv[cv] = result
        for key, value in result.items():
            out[f"{cv}_{key}"] = value

    # device 가 섞였으면 조용히 넘기지 않는다. 점수는 유효하지만 시간 비교가 깨지고,
    # 무엇보다 "왜 이 trial 만 느렸나" 를 나중에 복기할 수 없게 된다.
    used = {r["device"] for r in per_cv.values()}
    if any(r["device_mixed"] for r in per_cv.values()) or used != {"gpu"}:
        log(f"    경고: device 가 {sorted(used)} 였다 (GPU 폴백 발생) — 시간 비교에 주의")

    scores = {cv: r["oof_macro_f1"] for cv, r in per_cv.items()}
    out["oof_macro_f1_mean"] = float(np.mean(list(scores.values())))
    out["elapsed_seconds"] = float(sum(r["elapsed_seconds"] for r in per_cv.values()))

    # 두 분할을 다 쟀을 때만 의미가 있는 값. sgkf 가 skf 보다 얼마나 위인가 =
    # 그 설정이 쌍둥이 행 암기에 얼마나 기대는가. 클수록 LB 이전이 나쁠 것으로 본다.
    if "skf" in scores and "sgkf" in scores:
        out["sgkf_minus_skf"] = float(scores["sgkf"] - scores["skf"])

    out["objective_value"] = _objective_value(scores, args.objective)
    return out


def _objective_value(scores: dict[str, float], objective: str) -> float:
    """분할별 점수 -> Optuna 가 최대화할 스칼라 하나."""
    if objective == "mean":
        return float(np.mean(list(scores.values())))
    if objective == "min":
        return float(min(scores.values()))
    if objective not in scores:
        raise ValueError(
            f"--objective {objective} 를 쓰려면 --cv 에 {objective} 가 있어야 한다 "
            f"(지금 잰 분할: {sorted(scores)})"
        )
    return float(scores[objective])


def _fit(params, x, y, weight, seed, use_gpu):
    """GPU 로 학습하고 실패하면 CPU 로 재시도. `train_gbdt.fit_with_fallback` 과 같다."""
    full = {**FIXED_PARAMS, **params}
    if use_gpu is not False:
        try:
            model = create_model("xgb", use_gpu=use_gpu, random_state=seed, **full)
            model.fit(x, y, sample_weight=weight)
            return model
        except Exception as error:  # noqa: BLE001
            log(f"    GPU 실패 -> CPU 재시도: {type(error).__name__}: {str(error)[:120]}")
    model = create_model("xgb", use_gpu=False, random_state=seed, **full)
    model.fit(x, y, sample_weight=weight)
    return model


# ---------------------------------------------------------------- 검증
def verify(data: Dataset, args) -> int:
    """기준선 파라미터로 f4r 실측값이 재현되는지 본다.

    이게 통과해야 탐색 결과를 믿을 수 있다. 재현이 안 되면 파이프라인이 어딘가
    달라진 것이고, 그 위에서 고른 최적값은 f4r 의 최적값이 아니다.
    """
    log(f"=== 재현 검증: 기준선 파라미터로 f4r {'/'.join(args.cv_list)} 를 다시 돈다 ===")
    baseline = dict(MODEL_PARAMS["xgb"])
    log(f"파라미터: {baseline}")
    result = evaluate(data, baseline, args, verbose=True)

    ok = True
    for cv in args.cv_list:
        want_all = BASELINE_CV[cv]
        log(f"  --- {cv} ---")
        if result[f"{cv}_n_features"] != BASELINE_N_FEATURES:
            log(f"  [FAIL] 차원 {result[f'{cv}_n_features']} != {BASELINE_N_FEATURES}")
            ok = False
        for i, (got, want) in enumerate(
            zip(result[f"{cv}_fold_macro_f1"], want_all["fold_macro_f1"])
        ):
            mark = "OK " if abs(got - want) < 5e-5 else "FAIL"
            ok &= mark == "OK "
            log(f"  [{mark}] fold {i}  {got:.4f}  기대 {want:.4f}")
        got, want = result[f"{cv}_oof_macro_f1"], want_all["oof_macro_f1"]
        mark = "OK " if abs(got - want) < 5e-5 else "FAIL"
        ok &= mark == "OK "
        log(f"  [{mark}] OOF   {got:.4f}  기대 {want:.4f}")
        gap = result.get(f"{cv}_generalization_gap")
        if gap is not None:
            log(f"  일반화 격차 {gap:+.4f} "
                f"(train {result[f'{cv}_oof_train_macro_f1']:.4f})")

    if "sgkf_minus_skf" in result:
        log(f"\n  sgkf − skf = {result['sgkf_minus_skf']:+.4f}  "
            "(쌍둥이 행 보호막이 주는 이득. 클수록 암기 의존이 크다)")
    log(f"  총 {result['elapsed_seconds']:.0f}초")

    if ok:
        log("\n재현 확인. 이 스크립트의 CV 는 f4r 과 같은 것을 잰다.")
    else:
        log("\n재현 실패. 탐색을 돌리지 말 것 — 파이프라인이 기준선과 다르다.")
    return 0 if ok else 1


# ---------------------------------------------------------------- 탐색
def run_study(data: Dataset, args) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(TUNING_DIR / f'{args.study}.db').as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # trial 0 으로 기준선을 넣는다. 같은 코드·같은 fold 에서 나온 숫자여야 비교가 된다.
    if not study.trials:
        study.enqueue_trial(BASELINE_POINT, user_attrs={"note": "f4r baseline"})
        log("기준선 f4r 을 trial 0 으로 넣었다.")

    log(f"study={args.study}  storage={storage}")
    log(f"분할 {args.cv_list} · 목적함수 {args.objective}")
    log(f"기존 trial {len(study.trials)}개 · 이번에 {args.n_trials}개 추가")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        result = evaluate(data, params, args, trial=trial)
        for key, value in result.items():
            trial.set_user_attr(key, value)
        parts = [f"{cv}={result[f'{cv}_oof_macro_f1']:.4f}" for cv in args.cv_list]
        if "sgkf_minus_skf" in result:
            parts.append(f"차={result['sgkf_minus_skf']:+.4f}")
        gap = result.get(f"{args.cv_list[0]}_generalization_gap")
        if gap is not None:
            parts.append(f"격차={gap:+.4f}")
        log(f"  trial {trial.number}  " + "  ".join(parts)
            + f"  ({result['elapsed_seconds']:.0f}s)")
        return result["objective_value"]

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    report(study, args)


def report(study: optuna.Study, args) -> None:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        log("완료된 trial 이 없다.")
        return

    baseline = next((t for t in done if t.params == BASELINE_POINT), None)
    if baseline is None:
        raise SystemExit(
            "기준선 trial 이 study 에 없다. 비교 기준이 없으면 표가 거짓말을 한다 — "
            "study 를 지우고 다시 돌린다."
        )
    base_score = baseline.value
    gap_key = f"{args.cv_list[0]}_generalization_gap"
    base_gap = baseline.user_attrs.get(gap_key)
    base_split = baseline.user_attrs.get("sgkf_minus_skf")

    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    log("\n" + "=" * 108)
    log(f"완료 {len(done)}개 / 전체 {len(study.trials)}개 (pruned {pruned}개) · "
        f"목적함수 {args.objective}")
    log(f"기준선 f4r trial {baseline.number} = {base_score:.4f}"
        + (f"  격차 {base_gap:+.4f}" if base_gap is not None else "")
        + (f"  sgkf−skf {base_split:+.4f}" if base_split is not None else ""))
    log("-" * 108)
    header = f"{'trial':>6}{'목적':>9}{'대기준선':>10}"
    for cv in args.cv_list:
        header += f"{cv:>9}"
    if base_split is not None:
        header += f"{'sgkf−skf':>10}"
    log(header + f"{'격차':>9}  파라미터")
    for t in sorted(done, key=lambda x: -x.value)[:15]:
        row = f"{t.number:>6}{t.value:>9.4f}{t.value - base_score:>+10.4f}"
        for cv in args.cv_list:
            row += f"{t.user_attrs.get(f'{cv}_oof_macro_f1', float('nan')):>9.4f}"
        if base_split is not None:
            row += f"{t.user_attrs.get('sgkf_minus_skf', float('nan')):>+10.4f}"
        gap = t.user_attrs.get(gap_key)
        row += f"{gap if gap is not None else float('nan'):>+9.4f}  "
        log(row + " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                           for k, v in sorted(t.params.items())))
    log("=" * 108)

    best = study.best_trial
    _announce("CV 최고", best, base_score, args, gap_key)

    # CV 최고를 그대로 제출해 세 번 떨어졌다. 다른 자로 재는 후보를 같이 낸다.
    if base_gap is not None:
        safer = [
            t for t in done
            if t.value > base_score
            and (t.user_attrs.get(gap_key) or 9) < base_gap
        ]
        if safer:
            _announce(
                f"격차가 기준선({base_gap:+.4f})보다 작으면서 목적함수도 높은 "
                f"{len(safer)}개 중 최고",
                max(safer, key=lambda t: t.value), base_score, args, gap_key,
            )
        else:
            log("\n격차가 기준선보다 작으면서 목적함수도 높은 trial 은 없다.")

    # sgkf−skf 가 기준선보다 좁다 = 쌍둥이 암기 의존이 덜하다. LB 이전의 대리 지표다.
    if base_split is not None:
        tighter = [
            t for t in done
            if t.value > base_score
            and (t.user_attrs.get("sgkf_minus_skf") or 9) < base_split
        ]
        if tighter:
            _announce(
                f"sgkf−skf 가 기준선({base_split:+.4f})보다 좁으면서 목적함수도 높은 "
                f"{len(tighter)}개 중 최고",
                max(tighter, key=lambda t: t.value), base_score, args, gap_key,
            )
        else:
            log("\nsgkf−skf 가 기준선보다 좁으면서 목적함수도 높은 trial 은 없다.")

    out = TUNING_DIR / f"{args.study}_summary.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "study": args.study,
                "config": args.config,
                "cv": args.cv_list,
                "objective": args.objective,
                "baseline_trial": baseline.number,
                "baseline_objective": base_score,
                "baseline_gap": base_gap,
                "baseline_sgkf_minus_skf": base_split,
                "n_complete": len(done),
                "best": {"number": best.number, "value": best.value, "params": best.params,
                         "user_attrs": best.user_attrs},
                "trials": [
                    {"number": t.number, "value": t.value, "params": t.params,
                     "user_attrs": t.user_attrs}
                    for t in sorted(done, key=lambda x: -x.value)
                ],
            },
            handle, ensure_ascii=False, indent=2,
        )
    log(f"\n요약: {out}")
    log("제출 파일은 train_gbdt.py 로 다시 뽑는다. DACON 업로드는 사람이 한다.")


def _announce(label: str, trial: optuna.Trial, base_score: float, args, gap_key: str) -> None:
    """후보 하나를 사람이 읽을 수 있게 찍는다. 재실행 커맨드까지 같이 낸다."""
    detail = "  ".join(
        f"{cv} {trial.user_attrs[f'{cv}_oof_macro_f1']:.4f}"
        for cv in args.cv_list
        if f"{cv}_oof_macro_f1" in trial.user_attrs
    )
    log(f"\n{label}: trial {trial.number}  목적 {trial.value:.4f} "
        f"(기준선 대비 {trial.value - base_score:+.4f})  {detail}")
    gap = trial.user_attrs.get(gap_key)
    split = trial.user_attrs.get("sgkf_minus_skf")
    if gap is not None or split is not None:
        bits = []
        if gap is not None:
            bits.append(f"격차 {gap:+.4f}")
        if split is not None:
            bits.append(f"sgkf−skf {split:+.4f}")
        log("  " + "  ".join(bits))
    log(f"  {json.dumps(trial.params, ensure_ascii=False)}")
    log(f"  재실행: {_train_command(trial.params, args)}")


def _train_command(params: dict, args) -> str:
    """최고 설정을 `train_gbdt.py` 로 다시 돌리는 커맨드.

    탐색 스크립트는 OOF·test 예측·제출 파일을 만들지 않는다. 최종 산출물은 팀이
    쓰는 드라이버 한 곳에서만 나와야 파일 이름 규약과 로그 스키마가 어긋나지 않는다.

    **float 을 `%g` 로 줄이면 안 된다.** 유효숫자 6자리로 자르면
    `subsample=0.6037440222139066` 이 `0.603744` 가 되는데, 이건 표시 오차가 아니라
    다른 설정이다. subsample·colsample 은 그 값으로 행·열을 뽑으므로 1e-7 만 달라져도
    샘플이 바뀌고 트리가 통째로 달라진다. 실제로 잘린 값으로 재실행한 trial 53 이
    0.4672 대신 0.4655 를 냈다. `repr` 로 왕복 가능한 전체 자릿수를 쓴다.
    """
    sets = " ".join(f"--set {k}={v!r}" if isinstance(v, float) else f"--set {k}={v}"
                    for k, v in sorted(params.items()))
    return (
        f".\\.venv\\Scripts\\python.exe scripts\\train_gbdt.py --model xgb --tag opt "
        f"--configs {args.config} --cv all --topk {args.topk} --seed {args.seed} {sets}"
    )


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="f4r", help=f"사용 가능: {','.join(CONFIGS)}")
    parser.add_argument(
        "--cv", default="both",
        help="skf / sgkf / both. both 면 trial 마다 둘 다 돌린다 (시간 2배)",
    )
    parser.add_argument(
        "--objective", default=None,
        help="최대화할 값: mean / min / skf / sgkf. 기본값은 both 면 mean, 아니면 그 분할",
    )
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="모델 시드")
    parser.add_argument("--sampler-seed", type=int, default=42, help="TPE 시드")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--study", default="f4r_xgb_both", help="study 이름 = db 파일명")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--no-track-train", action="store_true",
                        help="train 점수를 안 잰다. 빨라지지만 격차를 못 본다")
    parser.add_argument("--verify", action="store_true",
                        help="기준선 파라미터로 f4r 재현만 확인하고 끝낸다")
    parser.add_argument("--show-best", action="store_true",
                        help="학습 없이 기존 study 결과만 다시 출력한다")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.device = {"gpu": True, "cpu": False, "auto": "auto"}[args.device]

    if args.config not in CONFIGS:
        raise SystemExit(f"알 수 없는 config {args.config}. 사용 가능: {list(CONFIGS)}")

    # 분할 목록. 순서를 고정한다 — step 번호와 표의 열 순서가 여기서 정해진다.
    if args.cv == "both":
        args.cv_list = ["skf", "sgkf"]
    else:
        args.cv_list = [c.strip() for c in args.cv.split(",") if c.strip()]
    unknown = [c for c in args.cv_list if c not in BASELINE_CV]
    if unknown:
        raise SystemExit(f"알 수 없는 cv {unknown}. 사용 가능: skf, sgkf, both")

    if args.objective is None:
        args.objective = "mean" if len(args.cv_list) > 1 else args.cv_list[0]
    if args.objective not in ("mean", "min", *args.cv_list):
        raise SystemExit(
            f"--objective {args.objective} 는 지금 잴 분할 {args.cv_list} 로 계산할 수 없다"
        )

    if args.show_best:
        storage = f"sqlite:///{(TUNING_DIR / f'{args.study}.db').as_posix()}"
        report(optuna.load_study(study_name=args.study, storage=storage), args)
        return

    log(f"gpu_available() = {gpu_available()}  ·  요청 device = {args.device}")
    data = Dataset(set(CONFIGS[args.config]["blocks"]), n_splits=args.n_splits)

    if args.verify:
        raise SystemExit(verify(data, args))
    run_study(data, args)


if __name__ == "__main__":
    main()
