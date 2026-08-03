#!/usr/bin/env python
"""GBDT 교차검증 학습 드라이버 — 피처 블록을 조립해 래더로 돌린다.

    python scripts/train_gbdt.py --model xgb --configs f0 --cv skf --no-submission
    python scripts/train_gbdt.py --model xgb                       # 전체 래더

## 피처 블록

    domain    539  도메인 지식 — 드라이버·TMB·유형조성·치환쌍·코돈역추론
    fe25       25  복합변이 샘플 피처 (팀 스키마 `SAMPLE_FEATURE_COLUMNS`)
    rollup     46  복합변이 rollup 44 + burden 2 (fold 안에서 fit)
    rollup16   16  rollup 중 train/test 배율이 안정적인 열만
    enc3    top-K  유전자 3단계 인코딩, fold 안에서 chi2 선택
    gec     top-K  유전자 토큰 수, fold 안에서 chi2 선택
    sigtok  top-K  서명 문서 TF-IDF, fold 안에서 어휘·IDF·chi2 전부 fit
    exacttok top-K 원문 토큰 TF-IDF — sigtok 대조군

## 왜 래더인가

점수 하나만 남으면 복기가 안 된다. 한 번에 한 축만 바꿔 델타를 분리한다 —
`f0`->`f1` 이 가중치, `f1`->`f2` 가 복합변이, `f1`->`f3` 이 유전자 수준의 기여다.

## CV

`data/process/train_folds.parquet` 의 사전계산 폴드를 **읽기만** 한다. 파일이 없으면
만들지 않고 멈춘다 — `scripts/make_folds.py` 가 fold 를 만드는 유일한 곳이다. 그래서
`--seed` 는 모델 시드일 뿐 분할에는 영향을 주지 않는다(분할 시드는
`make_folds.py --seed`). `fold_skf5` 는 `StratifiedKFold(5, shuffle=True,
random_state=42)` 와 배열 단위로 동일함이 확인돼 있어 팀 기존 기록과 그대로 비교된다.

**주 지표는 skf 다.** sgkf 가 늘 0.006 쯤 높지만 그 이득은 전부 쌍둥이 행에서 나온다
(단독 행 5,185개에서는 두 CV 차이가 +0.0006 로 사실상 동률). sgkf 는 변이 프로파일이
같은 샘플을 한 fold 로 묶어 모델이 파트너 라벨을 학습하지 못하게 막아 주는데,
리더보드에는 그 보호막이 없다 — test 2,546행 중 266행(10.45%)이 train 과 프로파일이
완전히 같고, 제출 시점에 그 쌍둥이는 100% 학습에 들어간다. 그래서 sgkf 로 제출을
판단하면 그만큼 속는다. 두 숫자를 다 기록하되 결정은 skf 로 한다.

`oof_macro_f1_singleton` 은 쌍둥이 행을 뺀 단독 행 점수다. 피처 변경의 효과만
보고 싶을 때 이쪽이 민감하다.

## 규정 준수

chi2 선택·`BurdenBinner`·balanced 가중치는 **fold 의 train 부분에서만** fit 한다.
test 는 transform 만 받는다. `eval_set` 은 쓰지 않는다 — valid fold 를 넣고 early
stopping 을 걸면 반복수 선택이 자기 CV 로 새고, test 를 넣는 건 규정 위반이다.

## DACON 제출

이 스크립트는 로컬에 csv 를 만들 뿐이다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.features_basic import (  # noqa: E402
    SAMPLE_FEATURE_COLUMNS,
    BurdenBinner,
)
from cancer_hack.features_domain import (  # noqa: E402
    DOMAIN_PREFIXES,
    align_domain_columns,
)
from cancer_hack.features_sparse import build_fold_tfidf_block  # noqa: E402
from cancer_hack.io import save_csv, write_submission  # noqa: E402
from cancer_hack.metrics import (  # noqa: E402
    build_prediction_frame,
    evaluate_classification,
    macro_f1,
)
from cancer_hack.models_gbdt import (  # noqa: E402
    create_model,
    gpu_available,
    resolve_sample_weight,
)
from cancer_hack.validation import (  # noqa: E402
    CV_SLUG,
    FOLD_META_COLUMNS,
    Chi2TopKSelector,
    check_all_classes_present,
    fold_column,
)

RAW_DIR = PROJECT_ROOT / "data/raw"
PROC_DIR = PROJECT_ROOT / "data/process"
ARTIFACTS = PROJECT_ROOT / "artifacts"

#: fold 안에서 채우는 2열. 자리는 미리 잡아 두고 값은 fold 마다 다시 쓴다.
BURDEN_COLUMNS = ("hypermutated_flag", "burden_quantile_bin")

#: rollup 중 train->test 평균 배율이 0.95~1.28 인 열. 카운트형은 test 에서 최대
#: 3,467배 부푼다 — test 가 같은 변이를 여러 전사체 좌표로 중복 기재하기 때문이다.
#: 비율·플래그·log 만 남기면 CV 는 거의 안 깎이면서 시프트 노출이 사라진다.
#:
#: `duplicate_signature_count` 는 여기 **넣지 않는다.** 샘플 평균이 train 1.30 /
#: test 52.8 로 40배 시프트다. 그 피처는 생물학이 아니라 test 주석 파이프라인
#: 탐지기라서 시프트 내성 블록의 정의에 정면으로 어긋난다. 값어치는 rollup 을
#: 쓰는 `f4` 와 rollup16 을 쓰는 `f4r` 의 델타로 본다.
ROBUST_ROLLUP_COLUMNS = (
    "functional_ratio",
    "missense_ratio",
    "synonymous_ratio",
    "nonsense_ratio",
    "frameshift_ratio",
    "multihit_gene_ratio",
    "has_missense",
    "has_synonymous",
    "has_nonsense",
    "has_frameshift",
    "has_duplicate_token",
    "no_mutation_flag",
    "log1p_mutated_gene_count",
    "log1p_mutation_event_count",
)

#: chi2 top-K 대상 블록. 나머지는 전부 통과시킨다.
GENE_BLOCKS = ("enc3", "gec")

#: fold 안에서 어휘째 다시 만드는 블록. gene 블록과 달리 열의 *정체* 가 fold 마다
#: 바뀌므로 `Dataset` 이 행렬을 미리 못 만든다. 들고 있는 건 문서 문자열이다.
SPARSE_BLOCKS = ("sigtok", "exacttok")

#: sparse 블록 -> (parquet 파일명 템플릿, 문서 열 이름)
SPARSE_SOURCES = {
    "sigtok": ("{split}_signature_mutation_tokens.parquet", "unique_mutation_document"),
    "exacttok": ("{split}_exact_mutation_tokens.parquet", "exact_mutation_document"),
}

BLOCK_SOURCES = {
    "domain": "{split}_domain_features.parquet",
    "fe25": "{split}_sample_mutation_features.parquet",
    "rollup": "{split}_sample_mutation_features_rollup.parquet",
    "rollup16": "{split}_sample_mutation_features_rollup.parquet",
    "enc3": "{split}_mutation_encoded.parquet",
    "gec": "{split}_gene_event_count_matrix.parquet",
}

BLOCK_DESC = {
    "domain": "도메인 539",
    "fe25": "복합변이 25",
    "rollup": "복합변이 rollup 46",
    "rollup16": "rollup 시프트내성 16",
    "enc3": "유전자 3단계",
    "gec": "유전자 토큰수",
    "sigtok": "서명 TF-IDF",
    "exacttok": "원문토큰 TF-IDF (대조군)",
}

#: 래더. 한 번에 한 축만 바꾼다.
CONFIGS: dict[str, dict] = {
    "f0": {
        "blocks": ("domain",),
        "weight": "none",
        "desc": "도메인 539 (가중치 없음) — 재현 기준선",
    },
    "f1": {"blocks": ("domain",), "weight": "balanced", "desc": "도메인 539"},
    "f2": {
        "blocks": ("domain", "rollup"),
        "weight": "balanced",
        "desc": "도메인 + 복합변이",
    },
    "f3": {
        "blocks": ("domain", "enc3"),
        "weight": "balanced",
        "desc": "도메인 + 유전자 3단계",
    },
    "f4": {
        "blocks": ("domain", "rollup", "enc3"),
        "weight": "balanced",
        "desc": "도메인 + 복합변이 + 유전자",
    },
    "f4g": {
        "blocks": ("domain", "rollup", "gec"),
        "weight": "balanced",
        "desc": "도메인 + 복합변이 + 토큰수",
    },
    "f4r": {
        "blocks": ("domain", "rollup16", "enc3"),
        "weight": "balanced",
        "desc": "도메인 + 시프트내성 rollup + 유전자",
    },
    # --- 중복 처리 전략 --------------------------------------------------
    # 서명 TF-IDF 축. f5x 를 대조군으로 함께 둔다 — "서명이 원문 문자열보다 낫다"는
    # 주장이 CV 숫자로 남아야 리뷰가 된다.
    "f5": {
        "blocks": ("domain", "rollup16", "enc3", "sigtok"),
        "weight": "balanced",
        "desc": "f4r + 서명 TF-IDF",
    },
    "f5x": {
        "blocks": ("domain", "rollup16", "enc3", "exacttok"),
        "weight": "balanced",
        "desc": "f4r + 원문토큰 TF-IDF (대조군)",
    },
    # 가중치 축. f4r 과의 델타가 중복 프로파일 감쇠의 순수 효과다.
    # 판정은 oof_macro_f1 이 아니라 oof_macro_f1_singleton 으로 한다 — 전체 OOF 는
    # 쌍둥이 행이 섞여 있어 가중치 변경이 자기 자신을 평가하는 꼴이 된다.
    "f4rw": {
        "blocks": ("domain", "rollup16", "enc3"),
        "weight": "balanced+group",
        "desc": "f4r + 중복 프로파일 감쇠",
    },
    "f4rws": {
        "blocks": ("domain", "rollup16", "enc3"),
        "weight": "balanced+group_sqrt",
        "desc": "f4r + 중복 프로파일 완만 감쇠",
    },
}

#: 모델별 기본 하이퍼파라미터.
#:
#: xgb 값은 `analysis/fe_block_ablation.py` 가 도메인 539열로 0.4272 를 낸 설정
#: 그대로다. `models_gbdt.XGBModel.default_params()` 는 `min_child_weight=2,
#: reg_lambda=5.0` 을 넣는데 그대로 두면 기준선이 조용히 어긋난다. 명시적으로 덮어쓴다.
MODEL_PARAMS: dict[str, dict] = {
    "xgb": dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=1,
        reg_lambda=1.0,
    ),
    "lgbm": dict(n_estimators=800, learning_rate=0.05),
    "catboost": dict(iterations=1000, learning_rate=0.05, depth=6),
}

_LOGGED_PARAMS = (
    "task_type",
    "loss_function",
    "objective",
    "n_estimators",
    "iterations",
    "learning_rate",
    "eta",
    "max_depth",
    "depth",
    "min_child_weight",
    "reg_lambda",
    "l2_leaf_reg",
    "colsample_bytree",
    "rsm",
    "subsample",
    "tree_method",
    "device",
    "random_seed",
    "random_state",
)


def log(message: str) -> None:
    print(message, flush=True)


def effective_params(model) -> dict[str, str]:
    """학습된 estimator 가 **실제로** 쓴 파라미터.

    `BaseGBDT.describe()["params"]` 를 쓰면 안 된다. 그건 래퍼가 들고 있는 원본
    dict 이고, `_build()` 는 그 **사본**에서 GPU 일 때 `rsm` 을 빼기 때문에 둘이
    어긋난다. 실제로 로그에 `rsm=0.4` 가 찍혔는데 GPU 가 받은 값은 `rsm=1`
    (열 샘플링 없음)이었다. 실험 기록이 거짓말을 하면 복기가 불가능해진다.
    """
    actual: dict = {}
    for getter in ("get_all_params", "get_params"):
        try:
            actual = getattr(model.estimator_, getter)()
            break
        except Exception:  # noqa: BLE001 — 백엔드가 이 API 를 안 주면 다음 것을 본다
            continue
    if not actual:
        actual = model.describe().get("params", {})
    return {k: str(actual[k]) for k in _LOGGED_PARAMS if actual.get(k) is not None}


def fit_with_fallback(args, x_train, y_train, weight):
    """GPU 로 학습하고, 실패하면 그 fold 만 CPU 로 다시 돌린다.

    이 GPU 는 데스크톱도 함께 구동하고 있어서 여유 메모리가 출렁인다. 브라우저가
    한 번 튀면 그 순간 OOM 으로 죽는다. fold 마다 모델을 새로 만드는 이유도 같다 —
    폴백으로 `use_gpu=False` 가 된 인스턴스를 재사용하면 이후 fold 가 전부 CPU 로
    끌려가서 비교가 깨진다.
    """
    params = dict(MODEL_PARAMS[args.model])
    params.update(args.override)
    if args.model == "catboost":
        params["gpu_ram_part"] = args.gpu_ram_part

    if args.device is not False:
        try:
            model = create_model(
                args.model, use_gpu=args.device, random_state=args.seed, **params
            )
            model.fit(x_train, y_train, sample_weight=weight)
            return model
        except Exception as error:  # noqa: BLE001
            log(f"    GPU 실패 -> CPU 재시도: {type(error).__name__}: {str(error)[:120]}")

    params.pop("gpu_ram_part", None)
    model = create_model(args.model, use_gpu=False, random_state=args.seed, **params)
    model.fit(x_train, y_train, sample_weight=weight)
    return model


# ---------------------------------------------------------------- 데이터
class Dataset:
    """래더 전체가 공유하는 블록 행렬. 프로세스당 한 번만 만든다.

    블록은 세 갈래다. `dense` 는 그대로 붙이는 블록, `gene` 은 fold 안에서 chi2 로
    걸러 붙이는 블록, `docs` 는 fold 안에서 TF-IDF 를 새로 fit 하는 블록이다.
    `rollup_*` 은 `BurdenBinner` 가 fit 할 원본 프레임이라 따로 들고 있는다.

    `docs` 만 행렬이 아니라 **문서 문자열 배열**을 들고 있다. gene 블록은 열 집합이
    고정이고 fold 가 그중 일부를 고를 뿐이지만, TF-IDF 는 fold 마다 열의 정체가
    바뀌어서 미리 만들어 둘 수가 없다.
    """

    def __init__(self, blocks: set[str], *, n_splits: int) -> None:
        t0 = time.perf_counter()

        # 라벨과 ID 는 도메인 블록에서 받는다 — 어떤 config 든 domain 을 쓴다.
        base = pd.read_parquet(PROC_DIR / "train_domain_features.parquet")
        test_base = pd.read_parquet(PROC_DIR / "test_domain_features.parquet")
        self.train_ids = base["ID"].astype(str).to_numpy()
        self.test_ids = test_base["ID"].astype(str).to_numpy()
        self.y = base["SUBCLASS"].astype(str).to_numpy()
        self.classes = np.unique(self.y)

        self.dense: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
        self.gene: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
        self.docs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.rollup_train: pd.DataFrame | None = None
        self.rollup_test: pd.DataFrame | None = None

        for name in sorted(blocks):
            if name in SPARSE_BLOCKS:
                self.docs[name] = self._load_documents(name)
                log(f"[block] {name:9s} {'문서':>6s}  {BLOCK_DESC[name]}")
                continue
            train_frame, test_frame = self._load_pair(name, base, test_base)
            columns, train_array, test_array = self._select(name, train_frame, test_frame)
            target = self.gene if name in GENE_BLOCKS else self.dense
            target[name] = (columns, train_array, test_array)
            log(f"[block] {name:9s} {len(columns):>6,}열  {BLOCK_DESC[name]}")

        self.folds = self._load_folds(n_splits=n_splits)
        sizes = self.folds.groupby("group_key").size()
        singleton_groups = set(sizes[sizes == 1].index)
        self.singleton_mask = self.folds["group_key"].isin(singleton_groups).to_numpy()

        log(
            f"[data] train {len(self.y)}행 · test {len(self.test_ids)}행 · "
            f"클래스 {len(self.classes)}개 · 그룹 {sizes.size:,}개 · "
            f"단독 행 {int(self.singleton_mask.sum()):,}개  "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    # -- 로딩 -------------------------------------------------------------
    def _load_pair(self, name, base, test_base):
        if name == "domain":
            return base, test_base
        source = BLOCK_SOURCES[name]
        train_frame = pd.read_parquet(PROC_DIR / source.format(split="train"))
        test_path = PROC_DIR / source.format(split="test")
        if not test_path.exists():
            raise FileNotFoundError(
                f"{test_path} 가 없다. scripts/make_features.py 로 먼저 만든다."
            )
        test_frame = pd.read_parquet(test_path)
        # ID 순서가 같다는 걸 가정하지 않고 확인한다. 어긋나면 OOF 가 통째로 밀린다.
        for split, frame, ids in (
            ("train", train_frame, self.train_ids),
            ("test", test_frame, self.test_ids),
        ):
            if not (frame["ID"].astype(str).to_numpy() == ids).all():
                raise ValueError(f"{name}/{split} 의 ID 순서가 도메인 블록과 다르다")
        return train_frame, test_frame

    def _load_documents(self, name) -> tuple[np.ndarray, np.ndarray]:
        """TF-IDF 입력 문서를 train/test 한 쌍으로 읽는다.

        `_load_pair` 와 같은 이유로 ID 순서를 확인한다 — 어긋나면 OOF 가 통째로 밀린다.
        """
        source, column = SPARSE_SOURCES[name]
        documents = []
        for split, ids in (("train", self.train_ids), ("test", self.test_ids)):
            path = PROC_DIR / source.format(split=split)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} 가 없다. scripts/make_features.py 로 먼저 만든다 "
                    f"(sigtok -> sigtokens, exacttok -> tokens)."
                )
            frame = pd.read_parquet(path)
            if not (frame["ID"].astype(str).to_numpy() == ids).all():
                raise ValueError(f"{name}/{split} 의 ID 순서가 도메인 블록과 다르다")
            documents.append(frame[column].to_numpy(dtype=object))
        return documents[0], documents[1]

    def _select(self, name, train_frame, test_frame):
        """블록별 열 선택 + train 기준 정렬. -> (열 이름, train 배열, test 배열)."""
        if name == "domain":
            columns = [c for c in train_frame.columns if c.startswith(DOMAIN_PREFIXES)]
            aligned, filled, dropped = align_domain_columns(test_frame, columns)
            log(
                f"[align] domain: train 에만 있어 0 으로 채운 열 {len(filled)}개, "
                f"test 에만 있어 버린 열 {len(dropped)}개"
            )
            if (len(filled), len(dropped)) != (52, 16):
                log(f"[align] 경고: 기대값(52/16)과 다르다. 버린 열 예시 {dropped[:5]}")
            return (
                columns,
                train_frame[columns].to_numpy(np.float32),
                aligned.to_numpy(np.float32),
            )

        if name == "fe25":
            columns = list(SAMPLE_FEATURE_COLUMNS)
            return (
                columns,
                train_frame[columns].to_numpy(np.float32),
                test_frame[columns].to_numpy(np.float32),
            )

        if name in ("rollup", "rollup16"):
            # BurdenBinner 가 fit 할 원본은 burden 2열을 붙이기 전 상태여야 한다.
            # `SUBCLASS` 도 빼야 한다 — train parquet 에만 있는 라벨이라 남겨 두면
            # test 를 같은 열 목록으로 자를 때 KeyError 가 난다.
            body = [c for c in train_frame.columns if c not in ("ID", "SUBCLASS")]
            self.rollup_train = train_frame[body]
            self.rollup_test = test_frame[body]
            keep = body if name == "rollup" else list(ROBUST_ROLLUP_COLUMNS)
            columns = keep + list(BURDEN_COLUMNS)
            # burden 2열은 자리만 잡는다 — 값은 fold 마다 다시 채운다.
            pad_train = np.zeros((len(train_frame), len(BURDEN_COLUMNS)), np.float32)
            pad_test = np.zeros((len(test_frame), len(BURDEN_COLUMNS)), np.float32)
            return (
                columns,
                np.hstack([train_frame[keep].to_numpy(np.float32), pad_train]),
                np.hstack([test_frame[keep].to_numpy(np.float32), pad_test]),
            )

        # 유전자 블록
        columns = [c for c in train_frame.columns if c not in ("ID", "SUBCLASS")]
        missing = sorted(set(columns) - set(test_frame.columns))
        if missing:
            raise ValueError(f"{name} test 에 없는 유전자 {len(missing)}개: {missing[:5]}")
        aligned = test_frame.reindex(columns=["ID", *columns])
        if aligned[columns].isna().to_numpy().any():
            raise ValueError(f"{name} test 재정렬 후 결측이 생겼다")
        return (
            columns,
            train_frame[columns].to_numpy(np.float32),
            aligned[columns].to_numpy(np.float32),
        )

    def _load_folds(self, *, n_splits: int) -> pd.DataFrame:
        """사전계산 fold 파일을 **읽기만** 한다. 없으면 만들지 않고 멈춘다.

        예전에는 파일이 없으면 여기서 만들어 저장했다. 그러면 같은 이름의 파일이
        두 경로에서 나오고, `artifacts/oof/` 의 예측이 어느 분할에서 나왔는지
        사후에 확인할 수 없다. fold 를 쓰는 쪽과 만드는 쪽을 갈라 둔다.
        """
        path = PROC_DIR / "train_folds.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 가 없다. scripts/make_folds.py 로 먼저 만든다."
            )

        folds = pd.read_parquet(path)
        if not (folds["ID"].astype(str).to_numpy() == self.train_ids).all():
            raise ValueError(
                f"{path.name} 의 ID 순서가 피처 블록과 다르다. "
                "scripts/make_folds.py --overwrite 로 다시 만든다."
            )

        wanted = [fold_column(kind, n_splits) for kind in ("skf", "sgkf")]
        missing = [c for c in (*FOLD_META_COLUMNS, *wanted) if c not in folds.columns]
        if missing:
            raise ValueError(
                f"{path.name} 에 없는 열 {missing}. "
                f"--n-splits {n_splits} 로 만든 파일이 맞는지 확인한다 "
                f"(있는 열: {list(folds.columns)})."
            )
        for column in wanted:
            observed = int(folds[column].max()) + 1
            if observed != n_splits:
                raise ValueError(
                    f"{path.name} 의 {column} 은 {observed}-fold 인데 "
                    f"--n-splits 는 {n_splits} 다."
                )

        meta = path.with_suffix(".json")
        detail = ""
        if meta.exists():
            with open(meta, encoding="utf-8") as handle:
                loaded = json.load(handle)
            detail = f" (seed={loaded.get('seed')} · n_splits={loaded.get('n_splits')})"
        log(f"[folds] {path.name} 재사용{detail}")
        return folds

    # -- 조립 -------------------------------------------------------------
    def assemble(self, config: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        """chi2 대상이 아닌 블록만 먼저 붙인다. 유전자 블록은 fold 안에서 붙는다."""
        names: list[str] = []
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for block in CONFIGS[config]["blocks"]:
            if block in GENE_BLOCKS or block in SPARSE_BLOCKS:
                continue
            columns, train_array, test_array = self.dense[block]
            names += columns
            train_parts.append(train_array)
            test_parts.append(test_array)
        return (
            names,
            np.hstack(train_parts).astype(np.float32, copy=False),
            np.hstack(test_parts).astype(np.float32, copy=False),
        )


# ---------------------------------------------------------------- 학습
def run_config(data: Dataset, *, config: str, cv: str, args) -> dict:
    spec = CONFIGS[config]
    gene_blocks = [b for b in spec["blocks"] if b in GENE_BLOCKS]
    sparse_blocks = [b for b in spec["blocks"] if b in SPARSE_BLOCKS]
    topk = args.topk if gene_blocks else None
    k_slug = f"k{topk}" if topk else "kall"
    # sparse 축을 stem 에 안 넣으면 --sparse-topk 를 바꿔 두 번 돌릴 때 두 번째가
    # 첫 번째 로그를 조용히 덮는다. write_matrix 가 디스크를 재스캔하므로 비교표까지
    # 반쪽이 된다.
    sparse_slug = (
        f"_sp{args.sparse_topk}m{args.tfidf_min_df}" if sparse_blocks else ""
    )
    stem = (
        f"{args.model}_{args.tag}_{config}_{CV_SLUG[cv]}_{k_slug}"
        f"{sparse_slug}_s{args.seed}"
    )

    names, dense_train, dense_test = data.assemble(config)
    burden_positions = [i for i, n in enumerate(names) if n in BURDEN_COLUMNS]
    fold_ids = data.folds[fold_column(cv, args.n_splits)].to_numpy()
    group_keys = data.folds["group_key"].to_numpy()

    oof = np.zeros((len(data.y), len(data.classes)), dtype=np.float64)
    test_proba = np.zeros((len(data.test_ids), len(data.classes)), dtype=np.float64)
    fold_scores: list[float] = []
    fold_seconds: list[float] = []
    devices: list[str] = []
    selected: dict[str, dict[str, list[str]]] = {}
    params: dict = {}
    n_features = len(names)

    started = time.perf_counter()
    for fold in range(args.n_splits):
        t0 = time.perf_counter()
        valid_index = np.where(fold_ids == fold)[0]
        train_index = np.where(fold_ids != fold)[0]
        check_all_classes_present(data.y, train_index, data.classes)

        x_train = dense_train.copy()
        x_test = dense_test.copy()

        # burden 2열 — 경계는 fold 의 train 부분에서만 잡는다.
        if burden_positions:
            binner = BurdenBinner().fit(data.rollup_train.iloc[train_index])
            wanted = [names[i] for i in burden_positions]
            x_train[:, burden_positions] = binner.transform(data.rollup_train)[
                wanted
            ].to_numpy(np.float32)
            x_test[:, burden_positions] = binner.transform(data.rollup_test)[
                wanted
            ].to_numpy(np.float32)

        # 유전자 블록 — chi2 도 fold 의 train 부분에서만 fit 한다.
        for block in gene_blocks:
            columns, gene_train, gene_test = data.gene[block]
            selector = Chi2TopKSelector(k=topk).fit(
                gene_train[train_index], data.y[train_index]
            )
            picked = selector.indices_
            selected.setdefault(block, {})[str(fold)] = [columns[i] for i in picked]
            x_train = np.hstack([x_train, gene_train[:, picked]])
            x_test = np.hstack([x_test, gene_test[:, picked]])

        # TF-IDF 블록 — 어휘·IDF·chi2 를 전부 fold 의 train 문서에서만 fit 한다.
        # train 과 test 를 합쳐 어휘를 만들면 규정 위반이다.
        for block in sparse_blocks:
            train_docs, test_docs = data.docs[block]
            sparse_names, sparse_train, sparse_test = build_fold_tfidf_block(
                train_docs,
                test_docs,
                train_index,
                data.y[train_index],
                prefix=f"tfidf__{block}__",
                topk=args.sparse_topk,
                min_df=args.tfidf_min_df,
            )
            selected.setdefault(block, {})[str(fold)] = sparse_names
            x_train = np.hstack([x_train, sparse_train])
            x_test = np.hstack([x_test, sparse_test])

        n_features = x_train.shape[1]

        # 가중치도 fold 의 train 부분만 보고 만든다. 그룹 크기를 fold 밖에서 세면
        # skf5 에서 fold 를 가로지르는 그룹 364개가 기여보다 과하게 깎인다.
        weight = resolve_sample_weight(
            spec["weight"], data.y[train_index], group_keys[train_index]
        )
        model = fit_with_fallback(args, x_train[train_index], data.y[train_index], weight)

        if list(model.classes_) != list(data.classes):
            raise RuntimeError(
                f"fold {fold} 의 클래스 순서가 전체와 다르다: {list(model.classes_)[:3]}"
            )

        oof[valid_index] = model.predict_proba(x_train[valid_index])
        test_proba += model.predict_proba(x_test) / args.n_splits

        devices.append(model.describe().get("device", "?"))
        params = effective_params(model)

        score = macro_f1(
            data.y[valid_index], data.classes[oof[valid_index].argmax(axis=1)]
        )
        fold_scores.append(score)
        fold_seconds.append(time.perf_counter() - t0)
        log(
            f"  [{stem}] fold {fold + 1}/{args.n_splits}  dim={n_features:,}  "
            f"Macro F1={score:.4f}  ({fold_seconds[-1]:.0f}s)"
        )
        del x_train, x_test

    elapsed = time.perf_counter() - started
    oof_pred = data.classes[oof.argmax(axis=1)]
    summary = evaluate_classification(data.y, oof_pred, data.classes)
    mask = data.singleton_mask
    singleton = macro_f1(data.y[mask], oof_pred[mask])

    result = {
        "stem": stem,
        "model": args.model,
        "config": config,
        "config_desc": spec["desc"],
        "blocks": list(spec["blocks"]),
        "sample_weight": spec["weight"],
        "cv": cv,
        "topk": topk,
        "tfidf": (
            {
                "blocks": sparse_blocks,
                "min_df": args.tfidf_min_df,
                "topk": args.sparse_topk,
            }
            if sparse_blocks
            else None
        ),
        "n_features": int(n_features),
        "n_samples": len(data.y),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "fold_macro_f1": fold_scores,
        "fold_seconds": fold_seconds,
        "oof_macro_f1": summary["macro_f1"],
        "oof_macro_f1_singleton": singleton,
        "n_singleton": int(mask.sum()),
        "oof_accuracy": summary["accuracy"],
        "per_class_f1": summary["per_class_f1"],
        "support": summary["support"],
        "device": devices[0] if devices else "?",
        "device_mixed": len(set(devices)) > 1,
        "model_params": params,
        "elapsed_seconds": elapsed,
    }
    log(
        f"  [{stem}] OOF Macro F1 = {summary['macro_f1']:.4f}  "
        f"단독행 = {singleton:.4f}  Acc = {summary['accuracy']:.4f}  ({elapsed:.0f}s)"
    )
    if result["device_mixed"]:
        log(f"  [{stem}] 경고: fold 마다 device 가 다르다 {devices} — 비교에 주의")

    if args.dry_run:
        return result

    oof_frame = build_prediction_frame(data.train_ids, oof, data.classes, y_true=data.y)
    save_csv(oof_frame, ARTIFACTS / "oof" / f"oof_{stem}.csv")

    test_frame = build_prediction_frame(data.test_ids, test_proba, data.classes)
    save_csv(test_frame, ARTIFACTS / "test_predictions" / f"test_{stem}.csv")

    # 키 이름이 `gene` 이지만 TF-IDF 블록의 항 선택도 여기 실린다. 이름을 바꾸면
    # 과거 로그 스키마가 깨져서 그대로 둔다. 값은 fold 간 Jaccard 평균 하나뿐이라
    # 항 목록(fold 5 x 1,000개)이 JSON 으로 새지는 않는다.
    result["selected_gene_overlap"] = {
        block: _selection_overlap(folds) for block, folds in selected.items()
    }
    log_path = ARTIFACTS / "logs" / f"{stem}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    if args.submission:
        info = write_submission(
            test_frame,
            RAW_DIR / "sample_submission.csv",
            ARTIFACTS / "submissions" / f"submission_{stem}.csv",
        )
        result["submission"] = info
        log(f"  [{stem}] 제출 파일: {info['output_path']}")

    return result


def write_matrix(args) -> Path:
    """비교표를 **디스크의 config 로그를 재스캔해서** 만든다.

    인메모리 `results` 로 쓰면 이번 프로세스가 돌린 것만 담긴다. 래더를 skf 와
    sgkf 로 나눠 두 번 돌리면 뒤엣것이 앞엣것을 덮어써서 비교표가 반쪽이 된다.
    실제로 그렇게 잃었다 — 개별 로그는 남아 있어서 복구는 됐지만, 요약이 조용히
    부분만 담는 건 복기를 망친다.

    재스캔이면 중간에 죽어도, 나눠 돌려도, 나중에 하나만 다시 돌려도 항상 디스크에
    있는 전부가 반영된다.
    """
    directory = ARTIFACTS / "logs"
    prefix = f"{args.model}_{args.tag}_"
    rows = []
    for path in sorted(directory.glob(f"{prefix}*_s{args.seed}.json")):
        if path.name.endswith(f"matrix_s{args.seed}.json"):
            continue
        with open(path, encoding="utf-8") as handle:
            row = json.load(handle)
        if "oof_macro_f1" not in row:  # ERROR 로그는 건너뛴다
            continue
        rows.append({k: v for k, v in row.items() if k != "per_class_f1"})
    rows.sort(key=lambda r: -r["oof_macro_f1"])

    path = directory / f"{prefix}matrix_s{args.seed}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return path


def _selection_overlap(selected: dict[str, list[str]]) -> float:
    """fold 간 선택 항 Jaccard 평균. 낮으면 선택이 불안정하다는 뜻이다.

    유전자 블록의 chi2 선택과 TF-IDF 블록의 어휘 선택에 같은 잣대를 쓴다.
    """
    keys = list(selected)
    if len(keys) < 2:
        return 1.0
    scores = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = set(selected[keys[i]]), set(selected[keys[j]])
            union = a | b
            scores.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------- CLI
def _parse_override(items: list[str]) -> dict:
    """`--set key=value` 를 파이썬 값으로. int -> float -> 문자열 순으로 시도한다."""
    out: dict = {}
    for item in items:
        key, _, raw = item.partition("=")
        for cast in (int, float):
            try:
                out[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key] = raw
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="xgb", help="xgb / lgbm / catboost")
    parser.add_argument(
        "--configs", default="all", help=f"쉼표 구분. 사용 가능: {','.join(CONFIGS)}"
    )
    parser.add_argument("--cv", default="skf", help="skf / sgkf / all")
    parser.add_argument("--topk", type=int, default=500, help="유전자 블록 chi2 상위 K")
    # 유전자 블록의 K 와 분리한다. --topk 를 건드리면 f3/f4 기준선과 비교가 안 된다.
    parser.add_argument(
        "--sparse-topk", type=int, default=1000, help="TF-IDF 블록 chi2 상위 K"
    )
    parser.add_argument(
        "--tfidf-min-df", type=int, default=3, help="TF-IDF 최소 문서 빈도"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="fold 파일에서 읽을 분할 수. 파일과 다르면 멈춘다 (분할은 make_folds.py 담당)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="모델 시드. fold 분할과는 무관하다."
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="모델 하이퍼파라미터 덮어쓰기 (여러 번 지정 가능)",
    )
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--gpu-ram-part", type=float, default=0.4, help="CatBoost 전용")
    parser.add_argument("--tag", default="v2", help="파일명에 들어가는 실험 이름")
    parser.add_argument("--submission", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않는다")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.device = {"gpu": True, "cpu": False, "auto": "auto"}[args.device]
    args.override = _parse_override(args.overrides)

    configs = (
        list(CONFIGS)
        if args.configs == "all"
        else [c.strip() for c in args.configs.split(",") if c.strip()]
    )
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"알 수 없는 config {unknown}. 사용 가능: {list(CONFIGS)}")
    cvs = ["skf", "sgkf"] if args.cv == "all" else [c.strip() for c in args.cv.split(",")]

    log(f"gpu_available() = {gpu_available()}  ·  요청 device = {args.device}")
    log(f"model = {args.model} · config {configs} · cv {cvs} · seed {args.seed}")
    if args.override:
        log(f"파라미터 덮어쓰기: {args.override}")

    needed: set[str] = set()
    for config in configs:
        needed |= set(CONFIGS[config]["blocks"])
    data = Dataset(needed, n_splits=args.n_splits)

    results = []
    for config in configs:
        for cv in cvs:
            log(f"\n=== {config} · {CONFIGS[config]['desc']} · CV {cv} ===")
            try:
                results.append(run_config(data, config=config, cv=cv, args=args))
            except Exception as error:  # noqa: BLE001
                # 하나가 죽어도 래더를 버리지 않는다. 앞선 결과는 이미 디스크에 있다.
                log(f"  [{config}/{cv}] 실패: {type(error).__name__}: {error}")
                traceback.print_exc()
                if not args.dry_run:
                    path = (
                        ARTIFACTS
                        / "logs"
                        / f"{args.model}_{args.tag}_{config}_{CV_SLUG[cv]}_ERROR.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(
                            {"config": config, "cv": cv, "error": traceback.format_exc()},
                            handle,
                            ensure_ascii=False,
                            indent=2,
                        )

    if not results:
        raise SystemExit("성공한 config 가 없다.")

    log("\n" + "=" * 96)
    log(
        f"{'config':<7}{'CV':<7}{'차원':>8}{'MacroF1':>10}{'단독행':>10}"
        f"{'Acc':>9}{'시간':>8}  설명"
    )
    log("-" * 96)
    for r in sorted(results, key=lambda x: -x["oof_macro_f1"]):
        log(
            f"{r['config']:<7}{r['cv']:<7}{r['n_features']:>8,}"
            f"{r['oof_macro_f1']:>10.4f}{r['oof_macro_f1_singleton']:>10.4f}"
            f"{r['oof_accuracy']:>9.4f}{r['elapsed_seconds']:>7.0f}s  {r['config_desc']}"
        )
    log("=" * 96)

    # skf 가 주 지표다. 이번 실행에 skf 가 없으면 있는 것으로 고르되 라벨을 바꾼다 —
    # sgkf 점수를 "주 지표"라고 적으면 로그가 거짓말을 한다.
    ranked = [r for r in results if r["cv"] == "skf"]
    label = "주 지표(skf) 최고"
    if not ranked:
        ranked = results
        label = f"최고 (skf 없음 — {'/'.join(sorted({r['cv'] for r in results}))} 기준)"
    best = max(ranked, key=lambda x: x["oof_macro_f1"])
    log(f"{label}: {best['stem']}  Macro F1 = {best['oof_macro_f1']:.4f}")

    if not args.dry_run:
        matrix_path = write_matrix(args)
        log(f"비교표: {matrix_path}")

    log("\n제출 파일은 로컬에만 만들어 뒀다. DACON 업로드는 직접 한다.")


if __name__ == "__main__":
    main()
