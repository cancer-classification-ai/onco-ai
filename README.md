# 유전자 변이 정보 기반 암 아형(서브타입) 예측 AI 알고리즘


# 프로젝트 개요

환자의 유전자별 변이 정보(6,199개 유전자, WT/변이 표기)를 기반으로 해당 환자의 암 아형(SUBCLASS)을 예측하는 **다중 분류(multi-class classification)** AI 알고리즘을 개발합니다.

암은 유전자 수준에서 발생하는 다양한 변이(mutation)에 의해 아형(subtype)이 결정되며, 아형에 따라 치료 방침과 예후가 크게 달라집니다. 본 프로젝트는 방대한 유전자 변이 정보 속에서 암의 정확한 아형을 판별하는 AI 모델을 개발하여 정밀 의료 발전에 기여하는 것을 목표로 합니다.

> [초격차] AI 헬스케어 5기 해커톤 — 암환자 유전체 데이터의 변이 정보를 활용한 암종 분류 AI 모델 개발


# 팀 구성

모든 팀원이 데이터 전처리·피처 엔지니어링·모델링·앙상블·제출까지 전 과정을 함께 수행했습니다. 각자 독립적으로 실험(EXP 시리즈)을 진행하고 결과를 공유·비교하며 최종 모델을 선정했습니다.

| 이름 | 역할 |
|---|---|
| 한금준 (팀장) | 데이터 전처리 · 피처 엔지니어링 · 모델링 · 앙상블 · 제출 (전 과정 공동 수행) |
| 권민재 | 데이터 전처리 · 피처 엔지니어링 · 모델링 · 앙상블 · 제출 (전 과정 공동 수행) |
| 정현우 | 데이터 전처리 · 피처 엔지니어링 · 모델링 · 앙상블 · 제출 (전 과정 공동 수행) |
| 권병학 | 데이터 전처리 · 피처 엔지니어링 · 모델링 · 앙상블 · 제출 (전 과정 공동 수행) |


# 데이터 설명

| 변수 | 설명 |
|---|---|
| 유전자 변이 컬럼 (6,199개) | 각 유전자의 변이 여부 (WT: 정상 / 변이: 돌연변이) |
| SUBCLASS | 예측 대상 암 아형 (다중 클래스) |

- `train.csv` : 학습 데이터 (SUBCLASS 레이블 포함)
- `test.csv` : 예측 대상 데이터 (SUBCLASS 미포함)


# 데이터 전처리

원본 데이터는 6,199개 유전자 컬럼에 각 유전자의 변이 여부가 `WT`(정상) 또는 변이 문자열로 들어 있습니다. 문자열을 그대로 쓰지 않고 파서로 변이 사건(event) 단위까지 분해한 뒤 여러 층위의 파생변수를 만듭니다. 통계량을 이용하는 모든 전처리는 **train fold에만 `fit()`, validation/test에는 `transform()`만** 적용해 데이터 누수를 차단합니다.

- **변이 문자열 파싱** (`parser.py`) — 각 유전자 셀을 변이 사건으로 분해하고 유형(missense·nonsense·frameshift·indel·synonymous·complex), 위치, 참조·대체 아미노산을 추출합니다.
- **유전자별 Wide 피처** — 각 유전자를 3단계로 인코딩(WT=0 / 동의변이=1 / 기능성 변이=2)하고, 변이 존재 여부(0/1)·사건 수·상대빈도·유형별 존재 플래그를 생성합니다(현재 4,384개 유전자 기준).
- **행 단위 파생변수** — 변이 부담(변이 유전자 수·전체 사건 수 및 로그 변환), 유형별 수·비율, 복합변이(multihit), deletion/indel/LoF, 변이 문자열 구조 통계(위치 mean/std/median, hotspot, 아미노산 다양성) 등.
- **아미노산 물리화학 피처** — 치환 페널티 합·평균·최대, 급진적/보존적 치환 수 등 9종.
- **도메인 피처(542개)** — 드라이버 유전자 변이(`A_`, `A2_`), 변이 부담(`B_`), 치환 조성(`C_`, `D_`), 돌연변이 서명 역추론(`N_`: SBS6·CpG·transition), MSI/면역회피(`M_`).
- **Fold 동적 피처** — 누수 방지를 위해 fold-train에서만 학습: 빈도·희귀도(`freq21`, `aatrans9`), TF-IDF/Count 토큰(`sigtok`, `exacttok`, `ptok`), 공변이 쌍(`comut`), 잠재표현 SVD·NMF(`lsvd`, `lnmf`), KMeans 유전자 모듈(`gmod`), 클래스 서명(`csig`), EB-shrinkage 근거 점수(`ebovr`, `ebbnb`).
- **검증 틀** — SUBCLASS 기준 Stratified K-Fold(5/10), 일부 실험은 StratifiedGroupKFold(`group5`)로 fold를 고정합니다. 원본 CSV는 `.parquet`으로 정제해 공유합니다.


# 모델링

- **사용 모델**
  - 트리 계열(주력): XGBoost, CatBoost, LightGBM, RandomForest — 피처셋 프리셋(`f0`~`f16`)과 `group5` CV, Optuna 하이퍼파라미터 튜닝, SHAP 기반 변수 선택.
  - 선형: Logistic Regression (베이스라인 및 메타 모델).
  - 딥러닝 ladder: full dense **MLP baseline** → mutation token만 쓰는 **Hierarchical Gene Set Encoder** → 두 표현을 결합한 **Hybrid**. SVD/NMF 잠재피처 결합, weighted CE·Label smoothing·Focal Loss 비교.
  - 앙상블·후처리: OOF 기반 스태킹/블렌딩, 기하평균, 그리디 가중, 교차적합(cross-fit), 시드 앙상블, 클래스별 확률 보정(calibration), 짝 라벨 규칙(pair rule, LB +0.0829), 클래스쌍 residual/gate 보정.
- **평가지표**: Macro F1 (OOF·singleton·희귀 클래스 별도 관리)


# 실험 결과

주요 마일스톤 기준이며, 점수는 대회 리더보드 Macro F1(Public / Private)입니다.

| Experiment | 모델 | Public | Private | 결과 |
|---|---|---|---|---|
| EXP_001 | XGBoost (token) | 0.0921 | 0.130 | 채택 |
| EXP_002 | Encoding · LightGBM | 0.266 | 0.330 | 채택 |
| EXP_006 | XGBoost `f0` baseline (SKF5) | 0.290 | 0.428 | 채택 |
| EXP_015 | XGBoost `f4r` + group5 | 0.352 | 0.479 | 채택 |
| EXP_026 | 앙상블 (XGB + CatBoost + RF) | 0.390 | 0.517 | 채택 |
| EXP_040 | f16 + group5 + 앙상블(XGB/Cat/RF) + 짝 규칙 | 0.482 | 0.517 | 채택 |
| EXP_042 | 그리디 앙상블(Cat/XGB/RF/MLP) + 교차적합 | 0.476 | 0.536 | 채택 |
| EXP_055 | OOF-driven Adaptive Ensemble | 0.487 | 0.529 | 채택 |
| EXP_053 | Geo60-ACat20-AMLP7.5-Greedy2.5-XCR10 | 0.494 | 0.553 | 채택 |
| EXP_048 | Geo85-XCR-Crossfit10-Greedy05 + group5 | 0.495 | 0.546 | 채택 |
| **EXP_065** | **V036 → V035 + PCPG→BRCA pair residual** | **0.495** | **0.562** | **최종 채택** |

전체 실험 이력(EXP_001~066)은 노션 「모델 성능 기록」 DB와 `manifests/experiment_registry.csv`에 기록되어 있습니다.


# 최종 모델

**EXP_065 — `V036 → V035 + PCPG→BRCA pair residual`** (Public 0.4953 / Private 0.5623)

NMF32 잠재표현과 GBMLGG·SARC 전용(specialist) 모델을 결합한 기하평균 기반 앙상블(V030)을 뼈대로, 검증된 클래스쌍 residual·gate 보정을 누적 적용한 V-시리즈의 최종 버전입니다. `V033`(COAD/GBMLGG/TGCT residual trio) → `V034`(OV→BRCA quantile gate) → `V035`(LUSC→LUAD NMF32 gate)를 거쳐, 마지막으로 **PCPG→BRCA 짝 residual** 보정을 얹었습니다.

- **선택 이유**
  - Public LB 최고 구간(0.4953)이면서 Private LB 0.5623으로 상위권을 유지해, 단일 모델·초기 앙상블 대비 일반화 성능이 가장 안정적이었습니다.
  - 오분류가 잦은 특정 암종 쌍(예: PCPG↔BRCA, LUSC↔LUAD, OV↔BRCA)을 근거 기반 residual/gate로 개별 보정해, 희귀·혼동 클래스에서 Macro F1을 끌어올렸습니다.
  - 뒤이은 `V037`(EXP_066, Private 0.5627)은 Private가 근소하게 높았으나 Public 검증 값이 없어, Public·Private 균형과 재현성이 확인된 EXP_065를 최종본으로 채택했습니다.


# 실행 방법

### 환경 설치

```bash
pip install -r requirements.txt
```

### 모델 학습 — GBDT

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

### 모델 학습 — 딥러닝

한 명령으로 세 단계 DL ladder 를 실행한다. torch 는 CUDA 빌드가 필요하다
(`requirements.txt` 상단 주석 참고).

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

Full-feature 설정은 `freq21`과 `aatrans9`를 항상 사용한다. 중복 YAML을 만들지 않고
`--latent-methods`로 SVD/NMF를 선택한다.

```bash
# 1) 권장 기준선: frequency + SVD
python scripts/train_dl.py \
  --model mlp \
  --config configs/mlp_full_features.yaml \
  --latent-methods svd \
  --cv skf --device cuda --seed 42 --tag mlp_full_svd_s42

# 2) frequency + NMF
python scripts/train_dl.py \
  --model mlp \
  --config configs/mlp_full_features.yaml \
  --latent-methods nmf \
  --cv skf --device cuda --seed 42 --tag mlp_full_nmf_s42

# 3) frequency + SVD + NMF
python scripts/train_dl.py \
  --model mlp \
  --config configs/mlp_full_features.yaml \
  --latent-methods svd nmf \
  --cv skf --device cuda --seed 42 --tag mlp_full_svd_nmf_s42
```

같은 `frequency + SVD` 피처를 고정하고 weighted CE, smoothing, Focal Loss,
완화 class weight를 한 번에 비교하려면 loss ladder를 실행한다. A-D를 먼저 모두
학습한 뒤 C/D 중 OOF Macro F1이 가장 높은 설정에 smoothing 0.02를 적용한 E를
자동 실행한다.

```bash
python scripts/run_dl_loss_ladder.py \
  --model mlp \
  --base-config configs/mlp_full_features.yaml \
  --latent-methods svd \
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

DL OOF 도 `fold_group5` 를 쓰므로 GBDT OOF 와 그대로 블렌딩된다. fold 파일이 같은지는
`cancer_hack.provenance.check_fold_fingerprint` 가 확인한다.

### 예측 생성 · 제출 파일

```bash
# 확률 파일에서 제출 csv (여러 개 주면 평균)
python scripts/make_submission.py --predictions artifacts/test_predictions/test_X.csv

# 짝 라벨 규칙까지 한 번에 (LB +0.0829, docs/pair_rule.md)
python scripts/make_submission.py --predictions artifacts/test_predictions/test_X.csv --pair-rule

# 이미 만들어 둔 제출 csv 에 규칙만 얹기 (여러 개 한 번에)
python scripts/apply_pair_rule.py --submission a.csv b.csv --out-dir artifacts/submissions
```

### 앞으로의 기본 경로 — `11_full_pipeline.ipynb`

새 결과를 낼 때는 이 노트북 하나만 쓴다. 원본 csv 에서 fold · 피처 파켓 22개 · 모델 ·
앙상블 · 짝 규칙 · 제출 파일 · 실행 로그까지 전부 만든다. 로직은 노트북에 복사하지 않고
`scripts/` 와 `src/cancer_hack/` 을 불러 쓴다.

**기존 산출물을 덮어쓰지 않는다.** `cancer_hack.paths.use_run_dirs(RUN_TAG)` 가 출력 위치를
실행별로 가른다.

```
data/process_<RUN_TAG>/      이번에 만든 피처 파켓
artifacts/runs/<RUN_TAG>/    oof · test_predictions · logs · submissions
  ├── run.json               설정·환경·파켓 지문·단계별 시간·점수 전부
  └── run.md                 사람이 읽는 요약
```

원본 `data/raw/*.csv` 는 읽기만 한다. 경로는 환경변수(`ONCO_PROCESS_DIR`·
`ONCO_ARTIFACTS_DIR`)로 정해지고 **쓰는 시점에** 확인하므로, import 를 끝낸 뒤에 바꿔도
반영된다.

`data/process/` 와 `artifacts/` 는 **지금까지 낸 제출본의 근거**라 그대로 둔다. 그쪽
숫자를 다시 확인할 때만 `reset_run_dirs()` 로 되돌린다.

> 왜 새로 만드나 — `features_basic.encode_mutation` 이 바뀌어(`*931*` 같은 동의 정지코돈을
> 2 가 아니라 1 로 센다) 기존 `mutation_encoded.parquet` 과 어긋난다. `enc3`·`comut`·
> `lsvd`·`lnmf`·`gmod` 다섯 블록이 그 파일 하나에서 나오므로 **두 쪽 OOF 를 한 블렌드에
> 넣으면 안 된다.**

### 제출본 재현 노트북 (기존 결과 재검증용)

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

- **언어·데이터**: Python, pandas, NumPy, PyArrow(Parquet)
- **머신러닝**: scikit-learn, XGBoost, CatBoost, LightGBM
- **딥러닝**: PyTorch (MLP · Gene Set Encoder · Hybrid)
- **피처·표현학습**: TF-IDF / CountVectorizer, TruncatedSVD, NMF, KMeans, SHAP
- **튜닝·검증**: Optuna, StratifiedKFold / StratifiedGroupKFold, OOF 스태킹, 확률 Calibration
- **협업·인프라**: Git/GitHub(Git Flow), Kaggle(GPU) · Colab(GPU) · VS Code
- **재현·자동화**: YAML 설정, nbconvert, pytest(무결성 테스트)


# 향후 개선 방향

- **딥러닝 표현 고도화** — Gene Set Encoder / Hybrid의 유전자 count·유형·위치 통계를 추가 반영하고, GNN·Autoencoder 임베딩을 앙상블에 정식 편입.
- **클래스 불균형 대응** — singleton·희귀 클래스 전용 손실(Focal/weighted CE)과 클래스별 임계값 보정을 표준화해 Macro F1 안정화.
- **앙상블 자동화** — residual/gate 규칙을 수작업이 아닌 검증 기반 자동 탐색으로 전환하고, 다중 seed·다중 fold의 편차를 줄이는 나이브한 규칙 정리.
- **재현성·문서화** — 실험 매니페스트와 제출 레지스트리를 CI에 연동해, 노트북 산출물과 제출 파일의 일치를 자동 검증.
