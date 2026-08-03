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
    comut   ~10~20 공변이 유전자 쌍, fold 안에서 지지도·lift·과변이 가드로 선택
            (`--comut-*` 플래그, `cancer_hack.features_graph` 참고)
    lsvd       64  잠재 SVD 성분 — 행을 L2 정규화하고 fold 안에서 기저를 fit
    lnmf       64  잠재 NMF 성분 — 같은 계약, 비음수 혼합
    gmod       24  하드 유전자 모듈 — fold 안에서 KMeans 로 배타 배정
            (`--latent-*`/`--module-*` 플래그, `cancer_hack.features_latent` 참고)
    csig       26  클래스 서명 몫 — fold 안에서 라벨로 클래스별 유전자 집합을 고른다
            (`--signature-*` 플래그, `cancer_hack.features_signature` 참고)

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
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

# Windows 콘솔이 cp949 라 한글과 em dash 가 깨진다. argparse 는 `--help` 를 sys.stdout
# 에 직접 쓰기 때문에 도움말 안의 `—` 하나로 UnicodeEncodeError 가 나서 `--help` 자체가
# 죽는다. 스트림을 **제자리에서** UTF-8 로 바꾼다 — `io.TextIOWrapper` 로 갈아끼우면
# 이 모듈을 import 하는 pytest 의 캡처 버퍼가 닫혀서 수집 단계가 통째로 죽는다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pytest 캡처 등 reconfigure 가 없는 스트림
        pass

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.features_basic import (  # noqa: E402
    ADDITIONAL_BURDEN_FEATURE_COLUMNS,
    MUTATION_STRING_PARSED_COLUMNS,
    SAMPLE_FEATURE_COLUMNS,
    BurdenBinner,
)
from cancer_hack.features_amino_acid import AMINO_ACID_FEATURE_COLUMNS  # noqa: E402
from cancer_hack.features_domain import (  # noqa: E402
    DOMAIN_PREFIXES,
    align_domain_columns,
)
from cancer_hack.features_graph import (  # noqa: E402
    MANUAL_PAIRS,
    build_fold_comutation_block,
)
from cancer_hack.features_latent import (  # noqa: E402
    MODULE_VALUES,
    SHIFT_EXPOSED_VALUES,
    build_fold_latent_block,
    build_fold_module_block,
    module_membership_frame,
    subspace_alignment,
)
from cancer_hack.features_signature import build_fold_signature_block  # noqa: E402
from cancer_hack.features_sparse import (  # noqa: E402
    build_fold_parsed_token_block,
    build_fold_tfidf_block,
)
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
GENE_BLOCKS = ("enc3", "gec", "gtype")

#: fold 안에서 어휘째 다시 만드는 블록. gene 블록과 달리 열의 *정체* 가 fold 마다
#: 바뀌므로 `Dataset` 이 행렬을 미리 못 만든다. 들고 있는 건 문서 문자열이다.
SPARSE_BLOCKS = ("sigtok", "exacttok", "ptok")

#: fold 안에서 쌍을 고르는 블록. 열의 정체와 **개수**가 fold 마다 바뀐다는 점은
#: SPARSE_BLOCKS 와 같지만, 원본 유전자 행렬 자체는 고정이라 GENE_BLOCKS 처럼 미리
#: 읽어 둘 수 있다 — 그래서 `Dataset.pairs` 는 문서 문자열이 아니라 gene 블록과 같은
#: (열 이름, train 배열, test 배열) 모양이다.
PAIR_BLOCKS = ("comut",)

#: fold 안에서 잠재 기저를 fit 하는 블록. PAIR_BLOCKS 와 달리 **폭이 매 fold 고정**이라
#: (열 이름도 `lat__svd__c000` 로 고정) f4r 과의 델타가 차원 변동에 오염되지 않는다.
#: 대신 열 이름이 같아도 열의 *의미* 는 fold 마다 다르다 — 그래서 안정성을
#: `_selection_overlap` 의 Jaccard 로 재면 항상 1.000 이라는 거짓말이 나온다.
#: 이쪽 안정성은 `features_latent.subspace_alignment`(주각 코사인)로 따로 잰다.
LATENT_BLOCKS = ("lsvd", "lnmf")

#: 유전자를 그룹으로 묶어 집계하는 블록. fold 마다 지지도 필터가 다시 걸려 모듈에 든
#: 유전자 집합이 실제로 바뀌므로 `selected` 등록이 의미가 있다.
MODULE_BLOCKS = ("gmod",)

#: **라벨을 보고** 유전자를 묶는 블록. 위 세 가족과 그래서 갈라 둔다 — LATENT/MODULE 는
#: `fit` 시그니처에 `y` 가 아예 없는 게 계약이고, 이쪽은 `y` 가 필수 인자다. 한 상수에
#: 섞으면 그 계약 차이가 코드에서 안 보이고, fold 루프도 `y` 를 어디까지 넘기는지가
#: 흐려진다. 지도 블록이라 `selected` 등록도 의미가 있다(고른 유전자가 fold 마다 바뀐다).
SIGNATURE_BLOCKS = ("csig",)

#: 원본 유전자 행렬을 그대로 들고 있다가 **fold 안에서** 열을 만드는 블록 전부.
#: `Dataset` 은 이들을 `pairs` 주머니에 담고 `assemble` 은 건너뛴다. 가족을 새로 만들 때
#: 여기 한 줄만 더하면 두 자리가 같이 따라온다 — 예전엔 세 상수를 두 곳에서 각각 나열해서
#: 한쪽만 고치면 블록이 조용히 dense 로 떨어졌다(`KeyError` 도 안 난다).
FOLD_MATRIX_BLOCKS = PAIR_BLOCKS + LATENT_BLOCKS + MODULE_BLOCKS + SIGNATURE_BLOCKS

#: sparse 블록 -> (parquet 파일명 템플릿, 문서 열 이름)
SPARSE_SOURCES = {
    "sigtok": ("{split}_signature_mutation_tokens.parquet", "unique_mutation_document"),
    "exacttok": ("{split}_exact_mutation_tokens.parquet", "exact_mutation_document"),
    "ptok": ("{split}_parsed_mutation_tokens.parquet", "parsed_mutation_document"),
}

BLOCK_SOURCES = {
    "domain": "{split}_domain_features.parquet",
    "fe25": "{split}_sample_mutation_features.parquet",
    "rollup": "{split}_sample_mutation_features_rollup.parquet",
    "rollup16": "{split}_sample_mutation_features_rollup.parquet",
    "enc3": "{split}_mutation_encoded.parquet",
    "gec": "{split}_gene_event_count_matrix.parquet",
    "gtype": "{split}_gene_mutation_type_matrix.parquet",
    "parsed19": "{split}_mutation_parsed_features.parquet",
    "burden8": "{split}_additional_burden_features.parquet",
    "aa9": "{split}_amino_acid_features.parquet",
    # enc3 와 같은 파일이다 — Dataset 이 `block_cache_key` 기준으로 캐시해 두 번 안 읽는다.
    "comut": "{split}_mutation_encoded.parquet",
    # 잠재·모듈 블록도 같은 파일이고 `_select` 분기도 같은 유전자 갈래라
    # `block_cache_key` 가 enc3/comut 와 같은 키를 낸다 — 배열 한 벌을 나눠 쓴다.
    "lsvd": "{split}_mutation_encoded.parquet",
    "lnmf": "{split}_mutation_encoded.parquet",
    "gmod": "{split}_mutation_encoded.parquet",
    "csig": "{split}_mutation_encoded.parquet",
}

DENSE_BLOCK_COLUMNS = {
    "parsed19": tuple(MUTATION_STRING_PARSED_COLUMNS),
    "burden8": tuple(ADDITIONAL_BURDEN_FEATURE_COLUMNS),
    "aa9": tuple(AMINO_ACID_FEATURE_COLUMNS),
}

#: `_select` 가 이름을 보고 갈라지는 블록 -> 그 분기의 이름. 여기 없는 블록은 전부
#: 마지막 유전자 분기로 떨어져서, 소스가 같으면 결과도 같다.
#:
#: 캐시 키를 파일 이름만으로 잡으면 `rollup` 과 `rollup16` 이 충돌한다. 둘은 같은
#: parquet 을 읽지만 고르는 열이 46개 / 16개로 다르고, `sorted(blocks)` 에서 `rollup`
#: 이 먼저라 rollup16 자리에 46열이 그대로 들어간다. 그러면 일부러 뺀
#: `duplicate_signature_count`(train 1.30 / test 52.8, 40배 시프트) 가 시프트 내성
#: 블록에 되살아나는데, 예외도 안 나고 fold 로직도 멀쩡히 돌아서 로그의 열 수를
#: 안 보면 모른다. `--configs` 기본값이 `all` 이라 config 를 안 적기만 해도 걸린다.
SELECT_KIND = {
    "domain": "domain",
    "fe25": "fe25",
    "rollup": "rollup46",
    "rollup16": "rollup16",
}


def block_cache_key(name: str) -> tuple[str, str]:
    """`_select` 결과를 공유해도 되는 블록끼리만 같은 값을 낸다.

    `enc3`/`comut` 처럼 소스와 분기가 둘 다 같은 블록은 계속 배열 한 벌을 나눠 쓴다.
    """
    return BLOCK_SOURCES.get(name, name), SELECT_KIND.get(name, "gene")


BLOCK_DESC = {
    "domain": "도메인 539",
    "fe25": "복합변이 25",
    "rollup": "복합변이 rollup 46",
    "rollup16": "rollup 시프트내성 16",
    "enc3": "유전자 3단계",
    "gec": "유전자 토큰수",
    "sigtok": "서명 TF-IDF",
    "exacttok": "원문토큰 TF-IDF (대조군)",
    "ptok": "일반화 ParsedToken CountVectorizer",
    "gtype": "유전자별 6종 변이 유형",
    "parsed19": "Mutation 문자열 구조 19종",
    "burden8": "추가 burden 8종",
    "aa9": "아미노산 치환 페널티 9종",
    "comut": "공변이 쌍 (fold 안 선택)",
    "lsvd": "잠재 SVD (fold 안 fit)",
    "lnmf": "잠재 NMF (fold 안 fit)",
    "gmod": "하드 유전자 모듈 (fold 안 KMeans)",
    "csig": "클래스 서명 몫 (fold 안 라벨 선택)",
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
    # --- 오늘 추가된 피처의 독립 ablation ---------------------------------
    "f4r_parse": {
        "blocks": ("domain", "rollup16", "enc3", "parsed19"),
        "weight": "balanced",
        "desc": "f4r + Mutation 문자열 구조 19종",
    },
    "f4r_burden": {
        "blocks": ("domain", "rollup16", "enc3", "burden8"),
        "weight": "balanced",
        "desc": "f4r + 추가 burden 8종",
    },
    "f4r_gtype": {
        "blocks": ("domain", "rollup16", "enc3", "gtype"),
        "weight": "balanced",
        "desc": "f4r + 유전자별 6종 변이 유형",
    },
    "f4r_ptok": {
        "blocks": ("domain", "rollup16", "enc3", "ptok"),
        "weight": "balanced",
        "desc": "f4r + 일반화 ParsedToken CountVectorizer",
    },
    "f4r_aa": {
        "blocks": ("domain", "rollup16", "enc3", "aa9"),
        "weight": "balanced",
        "desc": "f4r + 아미노산 치환 페널티 9종",
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
    # --- 공변이 쌍 --------------------------------------------------------
    "f4rc": {
        "blocks": ("domain", "rollup16", "enc3", "comut"),
        "weight": "balanced",
        "desc": "f4r + 공변이 쌍",
    },
    # 중복성 대조군. enc3 없이 공변이 쌍만 얹어서, 쌍이 단독 유전자 위에 정말
    # 추가 정보를 얹는지 본다 — 선택된 쌍의 유전자가 PMS2 하나만 빼고 전부 enc3
    # chi2 top-500 안에 이미 있어서, `value=and` 는 depth-6 트리에게 원리상 중복이다.
    "f2c": {
        "blocks": ("domain", "rollup16", "comut"),
        "weight": "balanced",
        "desc": "도메인 + 시프트내성 rollup + 공변이 쌍 (enc3 없음, 중복성 대조군)",
    },
    # --- 유전자 모듈 ------------------------------------------------------
    # 소프트(잠재)와 하드(KMeans)를 같은 사다리에 둔다. 둘 다 같은 공변이 기하를
    # 쓰지만 하드는 유전자를 모듈 하나에만 넣고 소프트는 여러 성분에 걸칠 수 있다.
    "f4rl": {
        "blocks": ("domain", "rollup16", "enc3", "lsvd"),
        "weight": "balanced",
        "desc": "f4r + 잠재 SVD",
    },
    "f4rn": {
        "blocks": ("domain", "rollup16", "enc3", "lnmf"),
        "weight": "balanced",
        "desc": "f4r + 잠재 NMF",
    },
    "f4rm": {
        "blocks": ("domain", "rollup16", "enc3", "gmod"),
        "weight": "balanced",
        "desc": "f4r + 하드 유전자 모듈",
    },
    # --- 클래스 서명 (지도) ------------------------------------------------
    # 위 셋과 달리 그룹을 **라벨로** 만든다. 그래서 CV 를 하나만 보면 안 된다 — skf 는
    # 쌍둥이 행(프로파일이 같은 샘플 842행)이 fold 를 가로질러서 valid 행의 서명에
    # 자기와 같은 프로파일의 라벨이 들어가고, sgkf(Profile Group CV)가 그걸 막는다.
    # 실측은 막은 쪽이 3배 컸다(시드 3개 평균 skf +0.0050 / sgkf +0.0156). 이득의
    # 출처는 쌍둥이가 아니라 희소 클래스다 — sgkf 기준 DLBC +0.26, ACC +0.12 이고
    # macro F1 이 26클래스를 같은 무게로 세니 그 둘만으로 +0.0147 이 설명된다.
    # 자세한 근거와 남은 위험은 `cancer_hack.features_signature` docstring 에 있다.
    "f4rsig": {
        "blocks": ("domain", "rollup16", "enc3", "csig"),
        "weight": "balanced",
        "desc": "f4r + 클래스 서명 몫",
    },
    # 중복성 대조군. f2c 와 같은 역할이다 — "enc3 위에 얹었을 때 이득이 있나"와
    # "블록 자체가 유전자 신호를 담고 있나"는 다른 질문이고, 후자를 이걸로 잰다.
    # f2c 가 -0.0168 을 찍은 게 "공변이 쌍은 사실상 정보가 없다"를 알려준 숫자였다.
    "f2l": {
        "blocks": ("domain", "rollup16", "lsvd"),
        "weight": "balanced",
        "desc": "도메인 + 시프트내성 rollup + 잠재 SVD (enc3 없음, 중복성 대조군)",
    },
    "f2m": {
        "blocks": ("domain", "rollup16", "gmod"),
        "weight": "balanced",
        "desc": "도메인 + 시프트내성 rollup + 하드 모듈 (enc3 없음, 중복성 대조군)",
    },
    "f2sig": {
        "blocks": ("domain", "rollup16", "csig"),
        "weight": "balanced",
        "desc": "도메인 + 시프트내성 rollup + 클래스 서명 (enc3 없음, 중복성 대조군)",
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
        self.pairs: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
        self.docs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.rollup_train: pd.DataFrame | None = None
        self.rollup_test: pd.DataFrame | None = None
        # comut 과 enc3 는 같은 parquet(`{split}_mutation_encoded.parquet`)을 읽어
        # 똑같은 (열 이름, train 배열, test 배열) 을 낸다. `block_cache_key` 로 캐시해
        # 두 번 안 읽는다 — 안 그러면 6,201x4,384 + 2,546x4,384 float32 를 두 벌 들고
        # 153MB 가 306MB 가 된다. fold 루프는 항상 `.copy()`/`hstack` 사본을 쓰므로
        # 두 블록이 같은 배열을 참조해도 안전하다. 키가 파일 이름이 아니라
        # (파일, `_select` 분기) 인 이유는 `SELECT_KIND` 주석 참고.
        self._block_cache: dict[
            tuple[str, str], tuple[list[str], np.ndarray, np.ndarray]
        ] = {}

        for name in sorted(blocks):
            if name in SPARSE_BLOCKS:
                self.docs[name] = self._load_documents(name)
                log(f"[block] {name:9s} {'문서':>6s}  {BLOCK_DESC[name]}")
                continue
            cache_key = block_cache_key(name)
            if cache_key in self._block_cache:
                columns, train_array, test_array = self._block_cache[cache_key]
            else:
                train_frame, test_frame = self._load_pair(name, base, test_base)
                columns, train_array, test_array = self._select(name, train_frame, test_frame)
                self._block_cache[cache_key] = (columns, train_array, test_array)
            # 넷 다 "원본 유전자 행렬을 그대로 들고 있다가 fold 안에서 가공"이라
            # 저장 형태가 같다. `_select` 도 유전자 분기로 그대로 흘러가므로
            # `SELECT_KIND` 에 넣지 않는다 — 갈래를 새로 만들 필요가 없다.
            if name in FOLD_MATRIX_BLOCKS:
                target = self.pairs
            elif name in GENE_BLOCKS:
                target = self.gene
            else:
                target = self.dense
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

        if name in DENSE_BLOCK_COLUMNS:
            columns = list(DENSE_BLOCK_COLUMNS[name])
            missing_train = sorted(set(columns) - set(train_frame.columns))
            missing_test = sorted(set(columns) - set(test_frame.columns))
            if missing_train or missing_test:
                raise ValueError(
                    f"{name} 스키마 불일치: train 누락 {missing_train[:5]}, "
                    f"test 누락 {missing_test[:5]}"
                )
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
            if (
                block in GENE_BLOCKS
                or block in SPARSE_BLOCKS
                or block in FOLD_MATRIX_BLOCKS
            ):
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
    pair_blocks = [b for b in spec["blocks"] if b in PAIR_BLOCKS]
    latent_blocks = [b for b in spec["blocks"] if b in LATENT_BLOCKS]
    module_blocks = [b for b in spec["blocks"] if b in MODULE_BLOCKS]
    signature_blocks = [b for b in spec["blocks"] if b in SIGNATURE_BLOCKS]
    topk = args.topk if gene_blocks else None
    k_slug = f"k{topk}" if topk else "kall"
    # sparse 축을 stem 에 안 넣으면 --sparse-topk 를 바꿔 두 번 돌릴 때 두 번째가
    # 첫 번째 로그를 조용히 덮는다. write_matrix 가 디스크를 재스캔하므로 비교표까지
    # 반쪽이 된다.
    sparse_slug = (
        f"_sp{args.sparse_topk}m{args.tfidf_min_df}" if sparse_blocks else ""
    )
    if "ptok" in sparse_blocks:
        sparse_slug += f"p{args.parsed_min_df}"
    # 공변이 축도 같은 이유로 slug 가 필요하다. 파라미터가 11개라 전부 펴면 파일명을
    # 못 읽으니, 사다리로 실제로 바꾸는 4축(topk·pool·value·mode)만 노출하고 나머지는
    # digest 하나로 접는다. 전체 dict 는 결과 JSON 의 comut.params 에 그대로 남아서
    # digest 를 로그만으로 되짚을 수 있다.
    comut_kwargs = (
        dict(
            pool=args.comut_pool,
            pool_topk=args.comut_pool_topk,
            mode=args.comut_mode,
            value=args.comut_value,
            topk=args.comut_topk,
            min_support=args.comut_min_support,
            min_class_support=args.comut_min_class_support,
            min_purity=args.comut_min_purity,
            min_lift=args.comut_min_lift,
            max_hyper_fraction=args.comut_max_hyper,
            max_pairs_per_gene=args.comut_max_per_gene,
        )
        if pair_blocks
        else {}
    )
    comut_manual_pairs = list(MANUAL_PAIRS) if args.comut_manual else []
    comut_slug = _comut_slug(comut_kwargs, manual=args.comut_manual)
    # 잠재·모듈 축도 같은 이유로 slug 가 필요하다. 셋 다 빈 dict 면 `""` 를 내므로
    # 기존 로그 129개의 파일명이 바이트 단위로 그대로 유지된다.
    # `lnmf` 는 이름 자체가 방식을 정하므로 `--latent-method` 를 덮는다. 그 해석을
    # **여기서** 끝내야 slug 와 로그의 `latent.params.method` 가 실제 돌아간 방식과
    # 일치한다 — fold 루프 안에서만 덮으면 파일명은 svd 라고 적혀 있는데 nmf 가
    # 돌아간다(실제로 한 번 그렇게 나왔다).
    latent_method = "nmf" if "lnmf" in latent_blocks else args.latent_method
    latent_kwargs = (
        dict(
            method=latent_method,
            n_components=args.latent_components,
            row_norm=args.latent_row_norm,
            value=args.latent_value,
            mode=args.latent_mode,
            gene_weight=args.latent_gene_weight,
            min_gene_support=args.latent_min_support,
            random_state=args.latent_random_state,
        )
        if latent_blocks
        else {}
    )
    module_kwargs = (
        dict(
            value=args.module_value,
            n_modules=args.module_n,
            svd_components=args.module_svd_components,
            mode=args.module_mode,
            min_gene_support=args.module_min_support,
            random_state=args.module_random_state,
        )
        if module_blocks
        else {}
    )
    signature_kwargs = (
        dict(
            value=args.signature_value,
            topk=args.signature_topk,
            mode=args.signature_mode,
            min_class_support=args.signature_min_class_support,
            min_lift=args.signature_min_lift,
            max_hyper_fraction=args.signature_max_hyper,
        )
        if signature_blocks
        else {}
    )
    latent_slug = _latent_slug(latent_kwargs)
    module_slug = _module_slug(module_kwargs)
    signature_slug = _signature_slug(signature_kwargs)
    stem = (
        f"{args.model}_{args.tag}_{config}_{CV_SLUG[cv]}_{k_slug}"
        f"{sparse_slug}{comut_slug}{latent_slug}{module_slug}{signature_slug}"
        f"_s{args.seed}"
    )

    # 대조군 설정은 조용히 지나가면 안 된다 — 나중에 로그만 보고 "왜 이건 나빴지"를
    # 되짚을 때 의도였는지 실수였는지가 구별돼야 한다.
    for label, value in (
        ("모듈", module_kwargs.get("value")),
        ("클래스 서명", signature_kwargs.get("value")),
    ):
        if value in SHIFT_EXPOSED_VALUES:
            log(
                f"  [{stem}] 주의: {label} value={value} 는 의도적 시프트 노출 대조군이다. "
                "test 는 변이 유전자가 2.21배라 이 형태는 그대로 부푼다"
            )
    if latent_kwargs.get("row_norm") == "none":
        log(
            f"  [{stem}] 주의: --latent-row-norm none 은 대조군이다. 선두 성분의 "
            "|burden 상관| 이 실측 0.995 로, 변이 부담 축을 그대로 학습한다"
        )
    if latent_kwargs.get("n_components", 0) > 128:
        log(
            f"  [{stem}] 경고: 잠재 성분 {latent_kwargs['n_components']}개. 기준선이 "
            "1,055열이라 이건 블록 추가가 아니라 차원 체제 변경이다"
        )

    names, dense_train, dense_test = data.assemble(config)
    burden_positions = [i for i, n in enumerate(names) if n in BURDEN_COLUMNS]
    fold_ids = data.folds[fold_column(cv, args.n_splits)].to_numpy()
    group_keys = data.folds["group_key"].to_numpy()

    oof = np.zeros((len(data.y), len(data.classes)), dtype=np.float64)
    test_proba = np.zeros((len(data.test_ids), len(data.classes)), dtype=np.float64)
    fold_scores: list[float] = []
    fold_seconds: list[float] = []
    fold_widths: list[int] = []
    devices: list[str] = []
    selected: dict[str, dict[str, list[str]]] = {}
    comut_diagnostics: dict[str, dict[str, list[dict]]] = {}
    comut_widths: list[int] = []
    latent_diagnostics: dict[str, dict[str, list[dict]]] = {}
    module_diagnostics: dict[str, dict[str, list[dict]]] = {}
    signature_diagnostics: dict[str, dict[str, list[dict]]] = {}
    # fold 별 기저·모듈맵. 안정성 진단(주각 코사인)과 모듈맵 CSV 에 쓴다.
    latent_bases: dict[str, list] = {}
    module_maps: dict[str, list] = {}
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
            if block == "ptok":
                sparse_names, sparse_train, sparse_test = (
                    build_fold_parsed_token_block(
                        train_docs,
                        test_docs,
                        train_index,
                        data.y[train_index],
                        prefix="count__ptok__",
                        topk=args.sparse_topk,
                        min_df=args.parsed_min_df,
                    )
                )
            else:
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

        # 공변이 쌍 — 후보 풀·지지도·클래스별 지지도·과변이 의존도·lift 를 전부 fold
        # 의 train 부분에서만 계산한다. test 행렬은 transform 만 받는다
        # (features_graph.select_comutation_pairs 는 test 인자를 아예 받지 않는다).
        for block in pair_blocks:
            columns, pair_train, pair_test = data.pairs[block]
            comut_names, comut_tr, comut_te, comut_diag = build_fold_comutation_block(
                pair_train,
                pair_test,
                train_index,
                data.y[train_index],
                gene_names=columns,
                manual_pairs=comut_manual_pairs,
                **comut_kwargs,
            )
            selected.setdefault(block, {})[str(fold)] = comut_names
            # 강제 포함 5쌍은 fold 와 무관하게 항상 들어가서 Jaccard 를 인위적으로
            # 올린다. 선택 알고리즘의 진짜 안정성은 자동 선별분만으로 따로 잰다.
            selected.setdefault(f"{block}_auto", {})[str(fold)] = [
                d["name"] for d in comut_diag if not d["manual"]
            ]
            comut_diagnostics.setdefault(block, {})[str(fold)] = comut_diag
            comut_widths.append(len(comut_names))
            if len(comut_names) > 30:
                log(
                    f"  [{block}] 경고: fold {fold} 에서 {len(comut_names)}쌍 선택. "
                    "차원 상한(30) 초과, 이 단은 기각 대상이다"
                )
            x_train = np.hstack([x_train, comut_tr])
            x_test = np.hstack([x_test, comut_te])

        # 잠재 기저 — SVD/NMF 를 fold 의 train 부분에서만 fit 한다. test 행렬은
        # transform 만 받는다 (features_latent.fit_latent_basis 는 test 도 y 도
        # 인자로 받지 않는다).
        for block in latent_blocks:
            columns, latent_train, latent_test = data.pairs[block]
            lat_names, lat_tr, lat_te, lat_diag, basis = build_fold_latent_block(
                latent_train,
                latent_test,
                train_index,
                data.y[train_index],
                gene_names=columns,
                **latent_kwargs,
            )
            # `selected` 에 넣지 않는다 — 열 이름이 매 fold `lat__svd__c000` 으로
            # 같아서 Jaccard 가 항상 1.000 이 나온다. 안정성을 모르는 대상에 대해
            # 거짓말하는 숫자다. 실제 안정성은 subspace_alignment 로 잰다.
            latent_bases.setdefault(block, []).append(basis)
            latent_diagnostics.setdefault(block, {})[str(fold)] = lat_diag
            x_train = np.hstack([x_train, lat_tr])
            x_test = np.hstack([x_test, lat_te])

        # 유전자 모듈 — KMeans 배정을 fold 의 train 부분에서만 fit 한다.
        for block in module_blocks:
            columns, module_train, module_test = data.pairs[block]
            mod_names, mod_tr, mod_te, mod_diag, modules = build_fold_module_block(
                module_train,
                module_test,
                train_index,
                data.y[train_index],
                gene_names=columns,
                **module_kwargs,
            )
            frame = module_membership_frame(modules, columns)
            module_maps.setdefault(block, []).append(frame)
            # 모듈에 들어간 유전자 집합은 fold 마다 진짜 바뀐다(지지도 필터가
            # fold-train 에서 다시 걸린다). 다만 모듈 *번호* 는 KMeans 초기화에
            # 따라 뒤섞이니 여기서 재는 건 "어떤 유전자가 모듈 체계에 포함됐나"
            # 까지다. 배정 자체의 안정성(seed 간 ARI)은 inspect_latent.py 가 잰다.
            selected.setdefault(block, {})[str(fold)] = list(frame["gene"])
            module_diagnostics.setdefault(block, {})[str(fold)] = mod_diag
            x_train = np.hstack([x_train, mod_tr])
            x_test = np.hstack([x_test, mod_te])

        # 클래스 서명 — 라벨을 보는 유일한 블록이다. 넘기는 `y` 는 chi2·comut 과
        # 같은 `data.y[train_index]` 이고, 선택 함수는 test 행렬을 인자로 받지 않는다.
        for block in signature_blocks:
            columns, sig_train, sig_test = data.pairs[block]
            sig_names, sig_tr, sig_te, sig_diag = build_fold_signature_block(
                sig_train,
                sig_test,
                train_index,
                data.y[train_index],
                gene_names=columns,
                **signature_kwargs,
            )
            # 클래스별로 고른 유전자를 한 줄에 편다. 열 이름은 fold 마다 같지만
            # (`sig_share__BRCA`) 그 안의 유전자 집합이 실제로 바뀌므로, 잠재 블록과
            # 달리 Jaccard 가 거짓말을 하지 않는다.
            selected.setdefault(block, {})[str(fold)] = [
                f"{d['class']}:{gene}" for d in sig_diag for gene in d["genes"]
            ]
            signature_diagnostics.setdefault(block, {})[str(fold)] = sig_diag
            x_train = np.hstack([x_train, sig_tr])
            x_test = np.hstack([x_test, sig_te])

        n_features = x_train.shape[1]
        fold_widths.append(n_features)

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
                "parsed_min_df": args.parsed_min_df,
                "topk": args.sparse_topk,
            }
            if sparse_blocks
            else None
        ),
        "comut": (
            {
                "blocks": pair_blocks,
                "manual": args.comut_manual,
                "params": comut_kwargs,
                "n_pairs_per_fold": comut_widths,
                # 쌍별 n_AB·purity·lift·hyper_fraction 과, 선택이 끝난 뒤에만 계산하는
                # train_rate/test_rate/rate_ratio(감사 기록 — 어떤 필터도 안 읽는다).
                "diagnostics": comut_diagnostics.get("comut", {}),
            }
            if pair_blocks
            else None
        ),
        "latent": (
            {
                "blocks": latent_blocks,
                "params": latent_kwargs,
                # 열 이름이 고정이라 Jaccard 는 무의미하다. fold 쌍마다 주각 코사인
                # 평균을 내서 "같은 부분공간을 보고 있나"를 잰다 — 1 이면 완전 일치.
                "subspace_alignment": {
                    block: _pairwise_alignment(bases)
                    for block, bases in latent_bases.items()
                },
                "diagnostics": latent_diagnostics,
            }
            if latent_blocks
            else None
        ),
        "module": (
            {
                "blocks": module_blocks,
                "params": module_kwargs,
                "diagnostics": module_diagnostics,
            }
            if module_blocks
            else None
        ),
        "signature": (
            {
                "blocks": signature_blocks,
                "params": signature_kwargs,
                # 클래스별로 고른 유전자와 lift·지지도, 그리고 선택이 끝난 뒤에만
                # 계산하는 train/test 평균비(감사 기록 — 어떤 필터도 안 읽는다).
                "diagnostics": signature_diagnostics,
            }
            if signature_blocks
            else None
        ),
        "n_features": int(n_features),
        "n_features_per_fold": fold_widths,
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
    # 모듈맵은 **fold 별로만** 쓴다. 전체 train 으로 fit 한 맵은 그 자체로 규정 위반은
    # 아니지만(test 를 안 본다) fold 를 넘는 산출물이라 나중에 무심코 재사용하기 쉽다.
    # 필요하면 scripts/inspect_latent.py 가 진단용으로 따로 만든다.
    for block, frames in module_maps.items():
        directory = ARTIFACTS / "features" / "modules" / stem
        for fold_id, frame in enumerate(frames):
            save_csv(frame.assign(fold=fold_id), directory / f"{block}_fold_{fold_id}.csv")

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


def _comut_slug(comut_kwargs: dict, *, manual: bool) -> str:
    """공변이 파라미터 -> stem 슬러그. `comut_kwargs` 가 비어 있으면(블록 없음) `""`.

    슬러그를 빠뜨리면 두 번째 실행이 첫 번째 로그를 조용히 덮는다 — 이 버그는
    이미 한 번 나서 `sparse_slug` 가 생겼다. 파라미터가 11개라 전부 펴면 파일명을
    못 읽으니, 사다리로 실제로 바꾸는 4축(topk·pool·value·mode)만 노출하고 나머지는
    digest 하나로 접는다. 전체 dict 는 결과 JSON 의 `comut.params` 에 그대로 남아서
    digest 를 로그만으로 되짚을 수 있다.
    """
    if not comut_kwargs:
        return ""
    digest = hashlib.blake2s(
        json.dumps(comut_kwargs, sort_keys=True).encode("utf-8"), digest_size=3
    ).hexdigest()
    manual_flag = "m" if manual else "x"
    return (
        f"_cm{comut_kwargs['topk']}{comut_kwargs['pool'][:3]}{comut_kwargs['value'][:3]}"
        f"{comut_kwargs['mode'][0]}{manual_flag}{digest}"
    )


def _digest(payload: dict) -> str:
    """파라미터 dict -> 3바이트 blake2s. `_comut_slug` 와 같은 방식이다."""
    return hashlib.blake2s(
        json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=3
    ).hexdigest()


def _latent_slug(latent_kwargs: dict) -> str:
    """잠재 파라미터 -> stem 슬러그. 비어 있으면(블록 없음) `""`.

    `_comut_slug` 와 같은 이유로 필요하다 — 슬러그가 없으면 `--latent-components` 만
    바꿔 두 번 돌릴 때 두 번째가 첫 번째 로그를 조용히 덮고, `write_matrix` 가
    디스크를 재스캔하므로 비교표까지 반쪽이 된다.

    사다리로 실제로 바꾸는 3축(성분 수·method·row_norm)만 펴고 나머지는 digest 로
    접는다. 전체 dict 는 결과 JSON 의 `latent.params` 에 그대로 남는다.
    """
    if not latent_kwargs:
        return ""
    return (
        f"_lt{latent_kwargs['n_components']}{latent_kwargs['method']}"
        f"{latent_kwargs['row_norm']}{_digest(latent_kwargs)}"
    )


def _module_slug(module_kwargs: dict) -> str:
    """모듈 파라미터 -> stem 슬러그. 비어 있으면 `""`."""
    if not module_kwargs:
        return ""
    return (
        f"_gm{module_kwargs['n_modules']}{module_kwargs['value'][:3]}"
        f"{_digest(module_kwargs)}"
    )


def _signature_slug(signature_kwargs: dict) -> str:
    """클래스 서명 파라미터 -> stem 슬러그. 비어 있으면 `""`.

    `_module_slug` 와 접두사(`_gm`/`_sg`)로 갈린다. 두 블록이 같은 config 에 같이
    들어가도 슬러그가 이어 붙으므로 충돌하지 않는다.
    """
    if not signature_kwargs:
        return ""
    return (
        f"_sg{signature_kwargs['topk']}{signature_kwargs['value'][:3]}"
        f"{_digest(signature_kwargs)}"
    )


def _pairwise_alignment(bases: list) -> float:
    """fold 쌍마다 주각 코사인 평균을 내고 다시 평균. 1 에 가까우면 안정적이다.

    `_selection_overlap` 의 잠재 블록 판이다. 열 이름이 fold 마다 같아서 Jaccard 는
    항상 1.000 을 내므로 여기서는 부분공간이 실제로 겹치는지를 본다.
    """
    if len(bases) < 2:
        return 1.0
    return float(
        np.mean(
            [
                subspace_alignment(bases[i], bases[j])
                for i in range(len(bases))
                for j in range(i + 1, len(bases))
            ]
        )
    )


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
        "--parsed-min-df",
        type=int,
        default=2,
        help="ParsedToken CountVectorizer 최소 문서 빈도",
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

    # --- 공변이 쌍 블록 (comut) ------------------------------------------
    parser.add_argument(
        "--comut-pool",
        choices=["drivers", "chi2"],
        default="drivers",
        help="쌍 후보 유전자 풀. drivers=features_domain.DRIVERS 92개(fold 무관). "
        "chi2 는 대조군. BRAF x passenger 축퇴를 일부러 재현한다",
    )
    parser.add_argument(
        "--comut-pool-topk", type=int, default=300, help="--comut-pool chi2 전용"
    )
    parser.add_argument(
        "--comut-mode", choices=["mutated", "functional"], default="mutated"
    )
    parser.add_argument(
        "--comut-value",
        choices=["and", "share", "gate"],
        default="share",
        help="쌍 피처 값. share/gate 는 행의 변이 유전자 수로 정규화해 시프트 내성이 있다",
    )
    parser.add_argument(
        "--comut-topk", type=int, default=20, help="자동 선별 쌍 상한 (필터가 먼저 물린다)"
    )
    parser.add_argument("--comut-min-support", type=int, default=20)
    parser.add_argument("--comut-min-class-support", type=int, default=8)
    parser.add_argument("--comut-min-purity", type=float, default=0.25)
    parser.add_argument("--comut-min-lift", type=float, default=0.15)
    parser.add_argument(
        "--comut-max-hyper",
        type=float,
        default=0.50,
        help="쌍이 과변이 샘플에 몰린 정도의 상한. 기저율 0.05",
    )
    parser.add_argument("--comut-max-per-gene", type=int, default=3)
    parser.add_argument(
        "--comut-manual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="§9 강제 포함 5쌍(IDH1+ATRX 등)을 필터와 무관하게 항상 넣을지",
    )

    # --- 잠재 모듈 블록 (lsvd / lnmf) -------------------------------------
    parser.add_argument(
        "--latent-method",
        choices=["svd", "nmf"],
        default="svd",
        help="분해 방식. config lnmf 는 이 값과 무관하게 nmf 로 고정된다",
    )
    parser.add_argument("--latent-components", type=int, default=64, help="잠재 성분 수")
    parser.add_argument(
        "--latent-row-norm",
        choices=["l2", "none"],
        default="l2",
        help="분해 전 행 정규화. l2 는 조성만 남겨 test 의 2.21배 희석에 내성이 있다. "
        "none 은 **의도적 시프트 노출 대조군** — 선두 성분의 |burden 상관| 이 실측 "
        "0.995 로, 변이 부담 축을 그대로 학습한다 (l2 는 0.458)",
    )
    parser.add_argument(
        "--latent-value",
        choices=["proj", "share"],
        default="proj",
        help="proj=투영 그대로, share=행별 |값| 합으로 정규화(NMF 면 혼합 비율)",
    )
    parser.add_argument(
        "--latent-mode", choices=["mutated", "functional"], default="mutated"
    )
    parser.add_argument(
        "--latent-gene-weight",
        choices=["none", "idf"],
        default="none",
        help="idf 면 fold-train 희귀도로 유전자를 가중한 뒤 분해한다",
    )
    parser.add_argument(
        "--latent-min-support",
        type=int,
        default=5,
        help="fold-train 에서 이 행수 미만 변이된 유전자는 분해 전에 뺀다",
    )
    parser.add_argument("--latent-random-state", type=int, default=0)

    # --- 하드 유전자 모듈 블록 (gmod) --------------------------------------
    parser.add_argument("--module-n", type=int, default=24, help="KMeans 모듈 수")
    parser.add_argument(
        "--module-svd-components", type=int, default=64, help="군집 전 SVD 축소 차원"
    )
    parser.add_argument(
        "--module-value",
        choices=list(MODULE_VALUES),
        default="share",
        help="모듈 집계 값. share/enrich/wshare 는 행의 변이 유전자 수로 나눠 시프트 "
        f"내성이 있다. {'/'.join(SHIFT_EXPOSED_VALUES)} 는 **의도적 시프트 노출 대조군** "
        "— test 에서 그대로 부푼다 (fraction 은 분모가 모듈 크기라는 상수라 비율처럼 "
        "보여도 카운트다)",
    )
    parser.add_argument(
        "--module-mode", choices=["mutated", "functional"], default="mutated"
    )
    parser.add_argument("--module-min-support", type=int, default=5)
    parser.add_argument("--module-random-state", type=int, default=0)

    # --- 클래스 서명 블록 (csig) -------------------------------------------
    parser.add_argument(
        "--signature-value",
        choices=list(MODULE_VALUES),
        default="share",
        help="서명 집계 값. --module-value 와 같은 규약이다. 기본 share 가 리뷰에서 "
        "요청된 signature_share — 행의 변이 유전자 수로 나눠 변이 부담 차이를 지운다",
    )
    parser.add_argument(
        "--signature-topk",
        type=int,
        default=30,
        help="클래스당 서명 유전자 상한 (필터가 먼저 물린다)",
    )
    parser.add_argument(
        "--signature-mode", choices=["mutated", "functional"], default="mutated"
    )
    parser.add_argument(
        "--signature-min-class-support",
        type=int,
        default=5,
        help="fold-train 의 그 클래스에서 이 행수 미만 변이된 유전자는 후보에서 뺀다. "
        "DLBC 는 fold-train 이 30행뿐이라 이 값이 실질 하한을 정한다",
    )
    parser.add_argument(
        "--signature-min-lift",
        type=float,
        default=0.05,
        help="클래스 안 변이율 - 밖 변이율의 하한. comut 의 --comut-min-lift 와 같은 "
        "덧셈 규약이다",
    )
    parser.add_argument(
        "--signature-max-hyper",
        type=float,
        default=0.50,
        help="서명 유전자의 클래스 안 변이가 과변이 샘플에 몰린 정도의 상한. "
        "--comut-max-hyper 와 같은 가드다",
    )
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
