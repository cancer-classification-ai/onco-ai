# 유전자 변이 정보 기반 암 아형(서브타입) 예측 AI 알고리즘


# 프로젝트 개요

환자의 유전자별 변이 정보(6,199개 유전자, WT/변이 표기)를 기반으로 해당 환자의 암 아형(SUBCLASS)을 예측하는 **다중 분류(multi-class classification)** AI 알고리즘을 개발합니다.

암은 유전자 수준에서 발생하는 다양한 변이(mutation)에 의해 아형(subtype)이 결정되며, 아형에 따라 치료 방침과 예후가 크게 달라집니다. 본 프로젝트는 방대한 유전자 변이 정보 속에서 암의 정확한 아형을 판별하는 AI 모델을 개발하여 정밀 의료 발전에 기여하는 것을 목표로 합니다.


# 팀 구성

| 이름 | 역할 |
|---|---|
| 권민재 | |
| 권병학 | |
| 정현우 | |
| 한금준 | |


# 데이터 설명

| 변수 | 설명 |
|---|---|
| 유전자 변이 컬럼 (6,199개) | 각 유전자의 변이 여부 (WT: 정상 / 변이: 돌연변이) |
| SUBCLASS | 예측 대상 암 아형 (다중 클래스) |

- `train.csv` : 학습 데이터 (SUBCLASS 레이블 포함)
- `test.csv` : 예측 대상 데이터 (SUBCLASS 미포함)


# 데이터 전처리



# 모델링

- 사용 모델


# 실험 결과

| 모델 | Score |
|---|---|


# 최종 모델


- 선택 이유


# 실행 방법

### 환경 설치

```bash
pip install -r requirements.txt
```

### 모델 학습

Kaggle에서는 한 명령으로 세 단계 DL ladder를 실행한다.

```bash
python scripts/run_dl_ladder.py --device cuda --seed 42
```

기본 실행 순서는 다음과 같다.

1. full dense feature를 사용하는 MLP baseline
2. mutation token만 사용하는 Hierarchical Gene Set Encoder
3. 두 표현을 결합하는 full-feature Hybrid

세 모델은 동일한 `train_folds.parquet`을 사용하며 OOF, 로그, test 확률,
submission을 `artifacts/{oof,logs,test_predictions,submissions}/`에 각각 저장한다.
특정 모델만 실행하려면 `--models`를 사용한다.

```bash
python scripts/run_dl_ladder.py --models mlp --device cuda --seed 42
python scripts/run_dl_ladder.py --models set_encoder --device cuda --seed 42
python scripts/run_dl_ladder.py --models hybrid --device cuda --seed 42
```

Frequency와 latent 피처를 fold 안에서 fit하는 full-feature DL은 설정을 분리해
비교한다. 세 설정 모두 `freq21`과 `aatrans9`를 사용하며 latent 방식만 다르다.

| 실험 | MLP 설정 | Hybrid 설정 |
|---|---|---|
| SVD 64 | `mlp_full_features_svd.yaml` | `hybrid_set_mlp_v2_full_features_svd.yaml` |
| NMF 64 | `mlp_full_features_nmf.yaml` | `hybrid_set_mlp_v2_full_features_nmf.yaml` |
| SVD 64 + NMF 64 | `mlp_full_features.yaml` | `hybrid_set_mlp_v2_full_features.yaml` |

Kaggle에서는 계산이 빠르고 안정적인 SVD부터 실행한 뒤, 같은 fold의 OOF Macro F1로
NMF와 결합 설정을 비교한다.

```bash
# 1) 권장 기준선: frequency + SVD
python scripts/train_dl.py \
  --model mlp \
  --config configs/mlp_full_features_svd.yaml \
  --cv skf --device cuda --seed 42 --tag mlp_full_svd_s42

# 2) frequency + NMF
python scripts/train_dl.py \
  --model mlp \
  --config configs/mlp_full_features_nmf.yaml \
  --cv skf --device cuda --seed 42 --tag mlp_full_nmf_s42

# 3) frequency + SVD + NMF
python scripts/train_dl.py \
  --model mlp \
  --config configs/mlp_full_features.yaml \
  --cv skf --device cuda --seed 42 --tag mlp_full_svd_nmf_s42
```

같은 `frequency + SVD` 피처를 고정하고 weighted CE, smoothing, Focal Loss,
완화 class weight를 한 번에 비교하려면 loss ladder를 실행한다. A-D를 먼저 모두
학습한 뒤 C/D 중 OOF Macro F1이 가장 높은 설정에 smoothing 0.02를 적용한 E를
자동 실행한다.

```bash
python scripts/run_dl_loss_ladder.py \
  --model mlp \
  --base-config configs/mlp_full_features_svd.yaml \
  --device cuda \
  --seed 42
```

중단 후 이어서 돌릴 때는 `--skip-existing`을 추가한다. 최종 비교표는
`artifacts/comparisons/dl_loss_ladder_mlp_skf5_s42.{csv,md}`에 저장되며 OOF
Macro F1, singleton Macro F1, fold 표준편차, 희귀 5개 클래스 Macro F1을 함께
보여준다. 이 ladder는 비교 실험이라 submission은 만들지 않는다.

Gene Set Encoder의 유전자별 count·중복·유형·위치 통계를 누적 비교하려면 별도
ablation ladder를 실행한다.

```bash
# set_v0 → set_v1_count → set_v2_type → set_v3_position → set_v4_full → hybrid_v2
python scripts/run_gene_rule_ladder.py --device cuda --seed 42

# 최종 강화형 두 모델만 실행
python scripts/run_gene_rule_ladder.py \
  --experiments set_v4_full hybrid_v2 \
  --device cuda \
  --seed 42
```

### 예측 생성

```bash
python src/predict.py
```

### 하이퍼파라미터 탐색

`scripts/tune_optuna.py` 는 `train_gbdt.py` 와 같은 피처·fold 를 쓰고 파라미터만 흔든다.
탐색 결과를 믿기 전에 `--verify` 로 기준선 재현부터 확인한다 — 여기서 어긋나면
파이프라인이 바뀐 것이고 그 위에서 고른 최적값은 무효다.

```bash
python scripts/tune_optuna.py --verify --cv both     # 기준선 재현 확인 (선행 필수)
python scripts/tune_optuna.py --cv both --n-trials 60  # 탐색
python scripts/tune_optuna.py --show-best             # 기존 study 결과만 다시 출력
```

study 는 `artifacts/tuning/<study>.db` 에 쌓이고 git 에는 올리지 않는다.
제출 파일은 이 스크립트가 만들지 않는다 — 최고 설정을 `train_gbdt.py` 로 재실행해 뽑는다.


# 프로젝트 구조

```
onco-ai/
├── data/
│   ├── raw/                     # 원본 데이터 (train.csv, test.csv, sample_submission.csv)
│   ├── interim/                 # 가공 중간 결과 (예: raw parser output)
│   └── processed/               # 모델 학습용 정제 피처 (Parquet/CSV)
│
├── artifacts/                   # 결과 및 로그 폴더
│   ├── features/                # 생성된 feature 파일 (features_v*.parquet)
│   ├── embeddings/              # 사전학습 임베딩 결과
│   ├── models/                  # 학습된 모델 weights/pkl
│   ├── oof/                     # OOF 예측 CSV
│   ├── test_predictions/        # Test 데이터 최종 예측 확률/클래스
│   ├── submissions/            # 최종 제출 파일
│   └── logs/                   # 학습/평가 로그 파일
│
├── pretrained/                 # 허용된 사전학습 모델 관리
│   ├── README.md                # 다운로드/사용 방법 명시
│   └── .gitkeep                 # 빈 디렉토리 유지 (weights는 Git제외)
│
├── notebooks/                  # 탐색적 분석 및 프로토타입 노트북
│   ├── 00_data_audit.ipynb      # 데이터 구조·통계 탐색(EDA)
│   ├── 01_baseline_parser.ipynb # 변이 파싱 및 baseline 분류
│   ├── 02_sparse_token_model.ipynb
│   ├── 03_latent_feature_experiments.ipynb
│   ├── 04_tabular_models.ipynb
│   ├── 05_deep_models.ipynb
│   └── 06_ensemble_analysis.ipynb
│
├── src/
│   └── cancer_hack/            # 프로젝트 모듈 패키지
│       ├── __init__.py
│       ├── io.py               # 데이터 로드/저장 (parquet 변환 등)
│       ├── validation.py       # StratifiedKFold 생성, CV 관리
│       ├── parser.py           # 변이 문자열 파서(기본 특성 추출)
│       ├── features_basic.py   # Gene binary, burden 등 Tier1 피처 생성
│       ├── features_frequency.py # Train-fold 빈도·IDF·희귀도 피처
│       ├── features_amino_acid.py # 아미노산 물리화학적 치환 피처
│       ├── features_sparse.py  # Token TF-IDF 등 희소 피처 생성
│       ├── features_latent.py  # NMF, SVD, Autoencoder 등 잠재 피처
│       ├── features_graph.py   # 공변이 네트워크 피처 등
│       ├── models_linear.py    # 로지스틱/선형 SVM 모델 클래스
│       ├── models_gbdt.py      # LightGBM, CatBoost, XGBoost 클래스
│       ├── models_dl.py       # 딥러닝 모델 정의 (Autoencoder, MLP, CNN, GNN 등)
│       ├── ensemble.py        # OOF 집계 및 Stacking/Calibration 메타 모델
│       ├── calibration.py     # 클래스별 확률 보정, 로그잇 바이어스 적용
│       └── metrics.py         # Macro F1 등 평가 지표 함수
│
├── configs/                   # 하이퍼파라미터 및 경로 설정 YAML
│   ├── paths.yaml
│   ├── folds.yaml
│   ├── features.yaml
│   ├── lgbm.yaml
│   ├── catboost.yaml
│   ├── linear.yaml
│   ├── autoencoder.yaml
│   └── ensemble.yaml
│
├── scripts/                   # 실행 스크립트
│   ├── audit_data.py           # 데이터 검사·EDA
│   ├── make_folds.py           # Stratified K-Fold 생성
│   ├── make_features.py        # 피처 생성 파이프라인 실행
│   ├── train_linear.py         # 로지스틱/Linear 모델 학습 (OOF 생성)
│   ├── train_gbdt.py           # GBDT 모델 학습 (LightGBM/CatBoost/XGB)
│   ├── train_dl.py             # 딥러닝 모델 학습 (CNN/GNN/Autoencoder)
│   ├── train_meta.py           # Level-2 메타 모델 학습 (Stacking)
│   └── make_submission.py      # 테스트 예측 및 최종 제출 파일 생성
│
├── manifests/                 # 실험·피처·제출 기록
│   ├── feature_registry.csv    # 생성한 피처 목록 및 버전
│   ├── experiment_registry.csv # 실험 설정(모델·피처 조합) 기록
│   ├── pretrained_model_registry.csv  # 사용 사전학습 모델 목록(출처,버전)
│   └── submission_registry.csv # 제출 이력 및 결과 기록
│
├── compliance/                # 규정 준수 문서
│   ├── COMPETITION_RULES.md    # 대회 규칙 요약
│   ├── DATA_POLICY.md         # 데이터 사용 정책
│   ├── PRETRAINED_MODEL_POLICY.md # 사전학습 모델 정책
│   └── data_leakage_checklist.md  # 누수 방지 확인서
│
├── tests/                     # 자동화된 무결성 테스트
│   ├── test_raw_data_integrity.py   # 원본 데이터 파일 무결성
│   ├── test_fold_integrity.py       # Fold 별 데이터 일관성
│   ├── test_feature_alignment.py    # Train/Val/OOF 피처 정렬 일치
│   ├── test_fold_fit_only.py        # Trainer가 Fold Train만 사용
│   ├── test_no_external_features.py # 외부 데이터 사용 차단 테스트
│   ├── test_oof_completeness.py     # 모든 Sample에 OOF 예측 포함 여부
│   └── test_submission_schema.py    # 제출 파일 스키마 검증
│
├── README.md                 # 프로젝트 개요 및 실행 가이드
└── Makefile                  # 자동화 명령(예: make features, make train, make submit)

```


# 사용 기술


# 향후 개선 방향
