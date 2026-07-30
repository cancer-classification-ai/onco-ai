# onco-ai
# 유전자 변이 정보 기반 암 아형(서브타입) 예측 AI 알고리즘

---

# 프로젝트 개요

환자의 유전자별 변이 정보(6,199개 유전자, WT/변이 표기)를 기반으로 해당 환자의 암 아형(SUBCLASS)을 예측하는 **다중 분류(multi-class classification)** AI 알고리즘을 개발합니다.

암은 유전자 수준에서 발생하는 다양한 변이(mutation)에 의해 아형(subtype)이 결정되며, 아형에 따라 치료 방침과 예후가 크게 달라집니다. 본 프로젝트는 방대한 유전자 변이 정보 속에서 암의 정확한 아형을 판별하는 AI 모델을 개발하여 정밀 의료 발전에 기여하는 것을 목표로 합니다.

---

# 팀 구성

| 이름 | 역할 |
|---|---|
| 권민재 | |
| 권병학 | |
| 정현우 | |
| 한금준 | |

---

# 데이터 설명

| 변수 | 설명 |
|---|---|
| 유전자 변이 컬럼 (6,199개) | 각 유전자의 변이 여부 (WT: 정상 / 변이: 돌연변이) |
| SUBCLASS | 예측 대상 암 아형 (다중 클래스) |

- `train.csv` : 학습 데이터 (SUBCLASS 레이블 포함)
- `test.csv` : 예측 대상 데이터 (SUBCLASS 미포함)

---

# 데이터 전처리



---

# 모델링

- 사용 모델

---

# 실험 결과

| 모델 | Score |
|---|---|

---

# 최종 모델


- 선택 이유

---

# 실행 방법

### 환경 설치

```bash
pip install -r requirements.txt
```

### 모델 학습

```bash
python src/train.py
```

### 예측 생성

```bash
python src/predict.py
```

---

# 프로젝트 구조

```
onco-ai/
├── data/                  # 원본 및 정제된 피처 (.parquet, .csv) - Git 관리 제외
│   ├── raw/               # train.csv, test.csv 등 원본 데이터
│   └── processed/         # features_v1.parquet 등 전처리된 데이터
├── notebooks/             # 팀원별 프로토타입/실험용 Jupyter Notebook
│   ├── eda_and_baseline.ipynb
│   ├── feature_exp_rwr.ipynb
│   └── model_exp_lgb.ipynb
├── oof/                   # 모델별 Out-Of-Fold 예측값 (.csv) - Stacking 입력용
│   ├── oof_lightgbm_v1.csv
│   ├── oof_catboost_v1.csv
│   └── oof_gcn_v1.csv
├── src/                   # 모듈화된 파이썬 코드
│   ├── __init__.py
│   ├── data_loader.py     # Data Loading 및 Parquet 변환 로직
│   ├── features.py        # CCF, CADD, NMF, RWR 등 피처 생성 클래스
│   ├── models.py          # LightGBM, CatBoost, GNN 등 모델 클래스
│   └── utils.py           # Seed 고정, Stratified K-Fold, Evaluation Metric
├── models/                # 학습된 모델 저장
│   └── model.pkl
├── submission/            # 해커톤 제출용 파일
│   └── submission.csv
├── config.py              # 하이퍼파라미터 및 경로 설정
├── main_train.py          # 1차 모델 학습 및 OOF 생성 실행 스크립트
├── main_stacking.py       # Level-2 Stacking 및 최종 제출 파일 생성 스크립트
├── requirements.txt       # 의존성 패키지 목록
├── .gitignore             # data/, oof/, *.parquet 등 대용량 파일 제외 설정
└── README.md
```

---

# 사용 기술

---

# 향후 개선 방향
