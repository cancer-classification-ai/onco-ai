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

진입점은 `scripts/train_gbdt.py` 다. 피처셋은 `--configs`(스크립트 안 `CONFIGS` dict),
하이퍼파라미터는 기본값 → `--params` 프리셋 → `--set` 순으로 겹친다.

```bash
python scripts/train_gbdt.py --model catboost --configs f16 --cv sgkf --seed 42
python scripts/train_gbdt.py --model catboost --configs f16 --params cbopt10   # 이름 붙은 프리셋
python scripts/train_gbdt.py --model xgb --configs f16 --set n_estimators=500  # 한 축만 덮어쓰기
```

`--params` 로 고를 수 있는 프리셋은 `PARAM_PRESETS` 에 있고, 각 항목의 `desc` 에 그게
채택된 값인지 기각된 값인지 적어 뒀다. **기각된 것도 등록돼 있다** — 그 구성으로 낸
제출본을 재현할 수 있어야 해서다.

### 예측 생성 · 제출 파일

```bash
# 확률 파일에서 제출 csv (여러 개 주면 평균)
python scripts/make_submission.py --predictions artifacts/test_predictions/test_X.csv

# 짝 라벨 규칙까지 한 번에 (LB +0.0829, docs/pair_rule.md)
python scripts/make_submission.py --predictions artifacts/test_predictions/test_X.csv --pair-rule

# 이미 만들어 둔 제출 csv 에 규칙만 얹기 (여러 개 한 번에)
python scripts/apply_pair_rule.py --submission a.csv b.csv --out-dir artifacts/submissions
```

### 제출본 재현 노트북

대회는 코드를 `.ipynb` 로 낸다. 두 노트북이 원본 csv 에서 실제 제출 파일까지 간다.

| 노트북 | 구성 | LB |
|---|---|---|
| `notebooks/09_final_submission.ipynb` | f16 · seed 42 · 기본 파라미터 + 짝 규칙 | **0.4818** (최고) |
| `notebooks/10_seed_ensemble_submission.ipynb` | f16 · seed 42/7/2024 · `cbopt10` + 짝 규칙 | 0.4725 |

10 은 기각된 구성이지만 재현은 되어야 한다 — 재현이 안 되면 그 판정 자체를 못 믿는다.
`REUSE_CACHE = True` 면 캐시된 예측으로 결합부터만 다시 해 몇 초에 끝난다.

`tests/test_final_notebook_reproduces.py` 가 두 노트북의 산출물을 실제 제출 파일과
전 행 대조한다. **노트북을 고쳤으면 반드시 다시 실행한다** — 안 그러면 이 테스트가 낡은
산출물을 보고 통과한다.

```bash
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebooks/09_final_submission.ipynb
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
