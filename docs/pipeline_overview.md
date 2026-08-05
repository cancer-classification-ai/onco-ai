# 파이프라인 전체 지도

이 저장소에서 **무엇을 써서 어떤 점수를 냈는지**, 그리고 **지금 무엇을 더 쓸 수 있는지**
한 곳에 모은 문서다. 숫자는 전부 `artifacts/logs/` 의 원본 로그와 코드에서 뽑았다.

기준 날짜 2026-08-05.

---

## 1. 한눈에 — LB 를 받은 경로

현재 최고 제출은 이 조합이다. 재현 노트북은 `notebooks/09_final_submission.ipynb`.

| 층 | 내용 | 기여 |
|---|---|---|
| 1 | `f16` 피처 → XGBoost·CatBoost·RandomForest → 0.45/0.45/0.10 + 로짓 보정 | LB 0.3896 |
| 2 | **짝 라벨 규칙** — test 214행의 코호트 라벨 교체 | **+0.0922 → LB 0.4818** |

여기서 배운 것이 이 대회의 핵심이다. **CV 를 올린 시도는 거의 전부 LB 에서 떨어졌고,
CV 에 아예 보이지 않던 규칙 하나가 +0.09 를 벌었다.**

### LB 실측 열 점

| 구성 | CV | LB |
|---|---|---|
| f4r (xgb 단독) | sgkf 0.4786 | 0.3520 |
| f4 | sgkf 0.4787 | 0.3411 |
| f6x | sgkf 0.4847 | 0.3400 |
| f4rs | sgkf 0.4806 | 0.3317 |
| xc 앙상블 | sgkf 0.4937 | 0.3384 |
| team3 앙상블 | group5 0.5077 | 0.3414 |
| mix4 앙상블 | group5 0.5070 | 0.3531 |
| v002 f16 앙상블 | group5 0.5165 | 0.3896 |
| v010 cbopt10 3-seed | group5 0.5210 | 0.3812 |
| **v002 + 짝 규칙** | group5 0.5165 | **0.4818** |
| v010 + 짝 규칙 | group5 0.5210 | 0.4725 |

CV 순위와 LB 순위가 맞은 적이 거의 없다. **CV 는 채택 근거로 약하다.**

---

## 2. 데이터와 분할 — 모든 실험의 고정 조건

| 항목 | 값 |
|---|---|
| train | 6,201행 × 4,386열 (`ID` + `SUBCLASS` + 유전자 4,384) |
| test | 2,546행 × 4,385열 |
| 클래스 | 26종, 최대 BRCA 786 / 최소 DLBC 38 |
| 희소성 | 전체 셀의 non-WT 0.81%, 샘플당 변이 중앙값 14개 |

분할은 `data/process/train_folds.parquet` 한 파일에 두 벌이 들어 있다.
`scripts/make_folds.py` 가 만들고, seed 42 · 5분할이다.

| 컬럼 | 방식 | 쓰는 곳 |
|---|---|---|
| `fold_skf5` | StratifiedKFold | 초기 실험 |
| `fold_group5` | **Stratified*Group*KFold** — 같은 변이 프로파일을 한 fold 로 묶는다 | 지금 기본 |

`fold_group5` 크기는 1241 / 1240 / 1241 / 1239 / 1240 이고 그룹은 5,636개다.

**이 파일은 2026-07-31 이후 바뀐 적이 없다.** `artifacts/oof/` 의 예측 100여 개가 전부
이 분할에 묶여 있어서, 다시 만들면 예전 OOF 와 새 OOF 를 섞는 순간 교차적합 보정이
valid fold 를 보게 된다. 그래서 `cancer_hack.provenance` 가 내용 지문을 재고
(`997d89a20595cc23`) 노트북과 테스트가 시작할 때 대조한다.

### 규정 — 무엇이 금지인가

평가 데이터를 **학습에** 쓰면 실격이다. 인코더·스케일러·집계 통계는 전부 fold 의 train
부분에서만 fit 하고 test 는 transform 만 받는다. train+test 를 합쳐 카테고리 사전을
만드는 흔한 트릭은 실격 사유다.

test 를 **행 단위로** 손보는 건 leakage 가 아니다. 판정 기준은 "test 한 행만 따로 넣어도
같은 결과가 나오는가"다. 짝 라벨 규칙이 이 기준을 만족한다(§5).

---

## 3. 피처 — 블록 23종

`scripts/train_gbdt.py` 안에 정의돼 있다. YAML 이 아니라 코드다.

### 캐시에서 읽는 블록

`data/process/*.parquet` 에 미리 만들어 두고 그대로 붙인다.
`scripts/make_features.py` 가 만든다.

| 블록 | 열 | 내용 |
|---|---|---|
| `domain` | 539 | 도메인 지식 기반 파생 |
| `rollup` | 46 | 복합변이 rollup |
| `rollup16` | 16 | rollup 중 **train/test 배율이 안정적인 열만** |
| `parsed19` | 19 | Mutation 문자열 구조 |
| `burden8` | 8 | 추가 변이 부담 |
| `aa9` | 9 | 아미노산 치환 페널티 |

`rollup` vs `rollup16` 이 중요하다. 뺀 30열에 `duplicate_signature_count`
(train 1.30 / test 52.8, **40배 시프트**) 같은 게 들어 있다. f4 → f4r 로 교체했을 때
CV 는 +0.0002 로 안 보였는데 **LB 는 +0.0109** 였다.

### fold 안에서 chi2 상위 K 만 고르는 블록

| 블록 | 원본 열 | 내용 |
|---|---|---|
| `enc3` | 4,384 | 유전자 3단계 (WT / 동의 / 기능) |
| `gec` | 4,384 | 유전자별 토큰 수 |
| `gecr` | 4,384 | 토큰 수 **행 정규화**(상대빈도) |
| `gtype` | 26,304 | 유전자별 6종 변이 유형 |

기본 `--topk 500`. `gtype` 은 원본이 26,304열이라 메모리를 900MB 쯤 더 쓴다.

### fold 안에서 기저를 새로 만드는 블록

**여기가 leakage 방지의 핵심이다.** 전부 fold 의 train 부분에서만 fit 한다.
`tests/test_fold_fit_only.py` 가 이 계약을 지킨다.

| 블록 | 내용 |
|---|---|
| `sigtok` · `exacttok` · `ptok` | 문서 TF-IDF / CountVectorizer (기본 `--sparse-topk 1000`) |
| `comut` | 공변이 쌍 선택 |
| `lsvd` · `lnmf` | 잠재 64성분 (SVD / NMF) |
| `gmod` | KMeans 24모듈 |
| `csig` | 클래스 서명 몫 |
| `freq21` · `aatrans9` | fold-train 빈도·IDF·희귀도 |
| `ebovr` · `ebbnb` | EB shrinkage supervised gene evidence 26종 |

### 피처셋 프리셋 36개

`--configs <이름>` 으로 고른다. 사다리 규칙은 **한 번에 한 축만 움직이기**다.

| 이름 | 블록 | 쓰임 |
|---|---|---|
| `f0` · `f1` | domain | 재현 기준선 |
| `f4r` | domain + rollup16 + enc3 | **단일 모델 기준선** (sgkf 0.4786 / LB 0.3520) |
| `f4r_*` | f4r + 한 블록 | 축 하나씩 재는 사다리 (11종) |
| `f2*` | enc3 를 뺀 대조군 | 중복성 확인 |
| `f11` | 캐시 11블록 | 앙상블 입력 |
| **`f16`** | **16블록 5,309열** | **LB 0.3896 제출의 입력** |
| `f16r` | f16 의 rollup → rollup16 | 시프트 노출 30열 제거 |
| `f16n` | f16 의 gec → gecr | 행 정규화 |
| `full_all` | 15블록 | 비중복 full feature 기준선 |

---

## 4. 모델과 하이퍼파라미터

### 기본값 (`MODEL_PARAMS`)

```python
xgb       n_estimators=300, learning_rate=0.1, max_depth=6,
          subsample=0.8, colsample_bytree=0.5, min_child_weight=1, reg_lambda=1.0
catboost  iterations=1000, learning_rate=0.05, depth=6
rf        n_estimators=500, max_features="sqrt"
lgbm      n_estimators=800, learning_rate=0.05
```

`sample_weight` 는 대부분의 config 에서 `balanced` 다(26클래스 불균형 대응).
`f4rw` · `f4rws` 만 중복 프로파일 감쇠를 추가로 건다.

### 겹치는 순서

```
MODEL_PARAMS 기본값  →  --params 프리셋  →  --set KEY=VALUE
```

`--set` 이 프리셋을 이긴다. 프리셋 한 축만 바꿔 보는 게 흔한 사용이라서다.

### 이름 붙은 프리셋 (`PARAM_PRESETS`)

| 이름 | 모델 | 판정 |
|---|---|---|
| `cbopt10` | catboost | **기각** — OOF +0.016 / LB −0.0084. 재현 전용 |

`cbopt10` 은 EXP_039 의 Optuna trial 10 이다. 저장소에 없고 셸 명령줄에만 있어서
EXP_041 제출본을 재현할 수 없었던 걸 학습 로그에서 되찾아 등록했다. **여기 있다고
쓰라는 뜻이 아니다** — 각 프리셋의 `desc` 에 채택/기각을 적어 둔다.

```
iterations=1600, learning_rate=0.1002086, depth=6,
l2_leaf_reg=1.0630, subsample=0.5105, rsm=1
```

### 실측 OOF (f16 · fold_group5)

| 모델 | seed 42 | seed 7 | seed 2024 |
|---|---|---|---|
| XGBoost | 0.4922 | 0.4880 | 0.4878 |
| CatBoost 기본 | 0.4932 | 0.4932 | 0.4914 |
| CatBoost cbopt10 | 0.5089 | 0.5097 | 0.5111 |
| RandomForest | 0.4774 | 0.4797 | 0.4687 |

**cbopt10 이 세 seed 모두에서 기본값을 1.6%p 앞서는데 LB 는 더 낮다.** 튜닝 이득이
train 분포에 붙어 있다는 뜻이고, 이 판정은 EXP_039·EXP_041 에서 두 번 나왔다.

---

## 5. 결합 — 앙상블과 후처리

### 고정 가중 + 로짓 보정

`scripts/calibrate_ensemble.py`. 가중치는 **0.45 / 0.45 / 0.10** (xgb / catboost / rf) 고정이다.
seed 를 여러 개 쓰면 모델 가중을 seed 수로 나눈다.

보정은 클래스별로 로짓에 상수를 더해 결정 경계를 옮긴다. macro F1 이 26클래스를 동등하게
세는 걸 이용한다. **교차적합으로 한다** — fold 를 뺀 나머지에서 바이어스를 찾고 그 fold 에만
적용한다. 전체 OOF 에 한 번에 맞추면 0.5210 이 0.5438 로 부풀고 그 값으로는 구성을 못 고른다.

구현은 `cancer_hack.ensemble.crossfit_calibrated_blend` 하나이고 **CLI 와 재현 노트북이
같은 함수를 부른다.**

| 단계 (v010 기준) | macro F1 |
|---|---|
| 균등 가중 | 0.5196 |
| 고정 가중 raw | 0.5179 |
| **고정 가중 + 보정** | **0.5210** |

균등 가중이 raw 보다 높다는 게 읽을 거리다 — 보정 이득의 상당 부분이 가중치가 어긋난 걸
되돌리는 것이다. 그래도 LB 로 검증된 쪽(.45/.45/.10)을 그대로 둔다.

### Caruana 그리디

`scripts/greedy_blend.py`. 사람이 멤버 3개를 고르는 대신, `artifacts/oof/` 의 라이브러리
전체(현재 group5 기준 102개)에서 점수가 가장 오르는 멤버를 하나씩 **복원 허용**으로 담는다.
담긴 횟수가 곧 가중치다. 재학습이 없어 몇 분이면 끝난다.

교차적합 0.5337 까지 나온다(v002 대비 +0.0172). **다만 LB 로 확인된 적이 없다.**

### 결합 전략 매트릭스 — 무엇이 잡음인지 먼저 잰다

`scripts/blend_matrix.py`. 위 세 스크립트를 **서브프로세스로 그대로 부르면서** 조건만
바꿔 가며 돌리고 결과를 한 표에 모은다. 결합 로직을 새로 쓰지 않는다.

같은 후보군(`--members-file` 로 고정) 위에서 전략 33종을 돌린 결과가
`research/14_ensemble_strategy_matrix.md` 에 있다. 요점 셋.

1. **선택 잡음 폭이 0.0027~0.0028 이다.** 나머지를 고정하고 `--random-state` 만 0/1/2/3
   으로 바꾸면 교차적합이 그만큼 흔들린다. **이보다 작은 전략 간 차이는 읽지 않는다.**
   상위 8종이 전부 이 폭 안에 들어와 서로 구별되지 않는다.
2. **cbopt10 을 빼도 CV 를 안 잃는다.** 후보 98개에서 셋을 빼면 0.5337 → 0.5330 이고
   난수 형제 분포가 겹친다. 그리디가 그 몫을 남은 95개로 메운다. 잡음 폭은 오히려
   0.0028 → 0.0010 으로 줄어든다. 제출 후보로 만들어 뒀다 —
   `Models/pairrule_candidates/submission_blendmx_fullnocb_pairrule_m3.csv`.
3. **후보를 단독 점수 상위 K 로 자르면 안 된다.** `top20` 은 상위권이 catboost·xgb 로만
   차서 rf·DL 가중이 0% 가 된다(0.5256). 계열당 상한을 거는 쪽이 0.5344 다.

```powershell
.\.venv\Scripts\python.exe scripts\blend_matrix.py --run-tag blendmx1 --jobs 5
.\.venv\Scripts\python.exe scripts\blend_matrix.py --run-tag blendmx1 --verify --skip-first
```

`--verify` 는 같은 조건으로 2회차를 돌려 산출물을 sha256 으로 맞댄다. **198개 대조에서
다른 것 0개**다. csv 186개는 바이트 동일이고, meta·calib 로그 12개만 기록된 입력 경로
문자열이 다르다(점수·가중치·바이어스는 전부 일치).

### 짝 라벨 규칙 — 이 대회에서 유일하게 크게 통한 것

test 행의 유전자 프로파일 4,384열이 train 의 **유일한** 행과 바이트 단위로 같고 그 train
라벨이 `KIPAN`·`KIRC`·`GBMLGG`·`LGG` 중 하나면, 예측을 **짝 코호트 라벨**로 바꾼다.
매칭된 라벨이 아니라 **반대쪽**을 쓴다는 게 핵심이다.

```
KIPAN = KICH ∪ KIRC ∪ KIRP        GBMLGG = GBM ∪ LGG
```

근거·규정 판정·측정 불가 이유는 `docs/pair_rule.md`. 요약하면 train 안에서 프로파일이 같은
422 묶음이 예외 없이 라벨이 갈리고 예외 없이 코호트 짝이며, train 의 짝 없는 KIRC 57개·
LGG 50개가 test 매칭 수와 **정확히** 일치한다.

기본 `--min-mut 3` 에서 **214행**이 걸린다. 두 베이스에서 +0.0922 와 +0.0913 으로
거의 같은 값을 벌었다 — **베이스와 무관한 상수 가산**으로 봐도 된다.

**로컬 CV 로는 잴 수 없다.** group CV 가 같은 프로파일을 한 fold 로 묶어 이 상황 자체를
못 만든다. 그래서 두 진입점 모두 실행할 때마다 전제 네 가지를 다시 재고, 하나라도 깨지면
파일을 쓰지 않고 멈춘다.

---

## 6. 딥러닝

`torch==2.11.0+cu128` 이 `code/.venv` 에 깔려 있다. CUDA 12.8 빌드이고 RTX 5070 에서
동작을 확인했다. PyPI 기본 휠은 CPU 전용이라 인덱스를 지정해야 한다
(`requirements.txt` 주석 참고).

### 모델 3종

| 모델 | 입력 |
|---|---|
| `mlp` | dense 피처만 |
| `set_encoder` | mutation token 만 (Hierarchical Gene Set Encoder) |
| `hybrid` | 둘 다 |

### config 17종

`configs/*.yaml` 이고 축은 넷이다 — `domain_feature_set`(all 여부),
`frequency_blocks`(freq21·aatrans9), `latent`(svd / nmf / 둘 다, 64성분), 학습 파라미터.

```
mlp          epochs 60  lr 0.0007  batch 128   early stopping patience 8 (min 15 epoch)
set_encoder  epochs 80  lr 0.0005  batch 64
hybrid       epochs 80  lr 0.0005  batch 64
```

### 실측 OOF (fold_group5)

| DL OOF | macro F1 | GBDT 3종과 라벨 불일치 |
|---|---|---|
| `hybrid_set_mlp_svd64` | **0.4388** | 46.1% |
| `gene_set_encoder_full` | 0.4092 | 49.0% |
| `mlp_full_features` (svd+nmf) | 0.3957 | 49.8% |
| `mlp_full_features_svd` | 0.3872 | **53.0%** |
| (참고) xgb / catboost | 0.4922 / 0.4932 | — |

**DL 은 단독으로 GBDT 를 못 이긴다. 값어치는 다양성이다.** GBDT 끼리는 27~39% 만 갈리는데
DL 은 GBDT 와 44~56% 가 갈린다. 그래서 그리디가 단독 꼴찌인 `mlp_full_svd` 를 11회나
집는다(가중치 3위).

다만 **그 다양성 축은 포화로 보인다.** 새 멤버를 넣어도 그리디 교차적합이
0.5337 → 0.5333 으로 움직이지 않는다.

---

## 7. 실행 진입점

| 스크립트 | 하는 일 |
|---|---|
| `make_folds.py` | fold 생성. **다시 돌리면 기존 OOF 전부 무효** |
| `make_features.py` | 피처 parquet 캐시 생성. 위와 같은 주의 |
| `train_gbdt.py` | GBDT 학습. 실질적 메인 진입점 (105KB) |
| `train_dl.py` · `run_dl_ladder.py` · `run_dl_loss_ladder.py` · `run_gene_rule_ladder.py` | DL 학습·사다리 |
| `train_rf.py` · `tune_optuna_rf.py` · `run_rf_b_optuna.py` | RF 스태킹 계열 |
| `calibrate_ensemble.py` | 고정/학습 가중 + 로짓 보정 |
| `greedy_blend.py` | Caruana 그리디 선택 |
| `blend_strategies.py` | 결합 방식 비교 (랭크평균·온도보정 등) |
| `blend_matrix.py` | 결합 전략 매트릭스 + 산출물 재현 대조 (`--verify`) |
| `train_meta.py` | OOF 스태킹 |
| `apply_pair_rule.py` | 짝 규칙 후처리 |
| `make_submission.py` | 제출 csv (`--pair-rule` 로 규칙 포함) |
| `tune_optuna.py` · `tune_sweep.py` | 하이퍼파라미터 탐색 |
| `run_seed_sweep.py` · `report_seeds.py` | seed 스윕 |
| `run_block_ablation.py` | 블록 절제 |
| `export_for_drive.py` · `external_members.py` · `colab_setup.py` | 팀 공유 |
| `check_models.py` · `check_dl.py` | 설치·버전 확인 |

### 노트북

**새 결과는 `11_full_pipeline.ipynb` 하나로 낸다.** 원본 csv 에서 fold · 피처 파켓 22개 ·
모델 · 앙상블 · 짝 규칙 · 제출 · 실행 로그까지 전부 만들고, 출력을
`data/process_<RUN_TAG>/` 와 `artifacts/runs/<RUN_TAG>/` 로 갈라 기존 산출물을 안 건드린다.
`run.json` 하나에 설정·환경·파켓 지문·단계별 시간·점수가 다 들어간다.

아래 셋은 **기존 제출본 재검증용**이다. 돌리기 전에 `reset_run_dirs()` 로 기본 경로로
돌아가야 한다.

| 노트북 | 구성 | LB |
|---|---|---|
| `09_final_submission.ipynb` | f16 · seed 42 · 기본 파라미터 + 짝 규칙 | **0.4818** |
| `09_pair_rule_postprocess.ipynb` | 짝 규칙만, 표준 라이브러리 + pandas 자체 완결 | — |
| `10_seed_ensemble_submission.ipynb` | f16 · seed 42/7/2024 · cbopt10 + 짝 규칙 | 0.4725 |

10 은 기각된 구성이지만 재현은 된다 — 재현이 안 되면 그 판정 자체를 못 믿는다.
`REUSE_CACHE = True` 면 캐시된 예측으로 결합부터만 다시 해 몇 초에 끝난다.

---

## 8. 기각된 것 — 반복하지 않도록

| 시도 | 결과 | 근거 |
|---|---|---|
| Optuna 튜닝 (60 trial) | **LB −0.0084** | EXP_039. 세 모델 다 채택 기준 미달 |
| cbopt10 + 3-seed | **LB −0.0093** (규칙 얹은 뒤에도) | EXP_041 |
| 피처 선별 (chi2 topk 축소) | CV 단조 하락 | 500→200→100 |
| 랭크 평균 | CV −0.0448 | 스케일이 정보였다 |
| 온도 보정 | 산술평균과 동점 | fold 5개 전부 격자 하한 |
| 클래스별 가중 | CV −0.0042 | 78 파라미터 대 DLBC 38행 |
| 핫스팟·치환쌍 블록 | LB −0.012 ~ −0.020 | CV 는 올랐다 |
| 한 방향만 짝 규칙 | 추정 +0.023 (양방향은 +0.066) | 클래스 총량이 무너진다 |
| OOF 스태킹 (로지스틱·릿지) | 최고 0.5102, 낙관격차 +0.028~+0.31 | EXP_042. 26클래스 × 멤버수 계수 대 DLBC 38행 |
| 스태킹 + 로짓 변환 | 0.3926, 낙관격차 **+0.3134** | 확률 꼬리를 선형 모델이 외운다 |
| 그리디 뒤에 로짓 보정 | −0.0008 ~ −0.0036 | 그리디가 이미 macro F1 을 직접 본다 |
| 단독 점수 상위 K 로 후보 자르기 | rf·DL 가중 0% | 다양성을 정확히 반대로 깎는다 |
| 그리디 라운드 60 · bagging 끄기 | 0.5306 · 0.5291 | 복잡도를 늘리면 손해 |

---

## 9. 지금 무엇을 쓸 수 있나

**바로 돌아가는 것**

- `f16` · `f16r` · `f16n` · `f11` · `full_all` 을 xgb / catboost / rf 로 학습
- `--params cbopt10` 프리셋, `--set` 으로 임의 하이퍼파라미터
- seed 스윕 (42 / 7 / 2024 검증됨, 균등 블렌드에서 +0.0022)
- 고정 가중 + 로짓 보정, Caruana 그리디, OOF 스태킹
- 결합 전략 매트릭스 (`blend_matrix.py`) — 전략을 나란히 돌리고 산출물까지 재현 대조
- 짝 규칙 후처리 (어떤 제출 csv 에도, 여러 개 한 번에)
- DL 3종 × config 17종, loss 사다리, gene rule 사다리
- 두 재현 노트북

**주의해서 써야 하는 것**

- `make_folds.py` · `make_features.py` 재실행 — 기존 OOF 100여 개가 무효가 된다.
  특히 `mutation_encoded.parquet` 은 **지금 코드와 이미 어긋나 있다**
  (`encode_mutation` 이 `*931*` 같은 동의 정지코돈을 2 가 아니라 1 로 센다).
  재생성은 전부 다시 학습할 때만 한다. `cancer_hack.provenance` 가 지문으로 막는다.
- Optuna 탐색 — 결과를 믿기 전에 `--verify` 로 기준선 재현부터 확인한다.

**아직 안 해 본 것**

- 계층형 2단계 분류 (24그룹 → 코호트 쌍 이진). 짝 규칙이 못 고치는 잔여분을 겨냥한다.
  다만 train 의 KIRC 334행 중 254행이 KIPAN 쌍둥이를 갖고 있어 이진 분류기의 학습
  데이터 대부분이 모순 쌍이다 — 착수 전에 이 점부터 재야 한다.
- SARC(F1 0.210) 원인 분석. BRCA·PRAD·KIPAN·OV 가 전부 SARC 로 새는 흡수구다.

---

## 10. 검증 장치

| 테스트 | 지키는 것 |
|---|---|
| `test_fold_fit_only.py` | fold 안 fit 블록이 train 부분만 본다 |
| `test_data_provenance.py` | fold·피처 파켓이 예측을 만들 때와 같다 |
| `test_pair_rule.py` | 짝 규칙의 전제 네 가지 |
| `test_param_presets.py` | 프리셋 값이 제출본 로그와 같다 |
| `test_crossfit_blend.py` | 보정이 실제로 fold 를 빼고 fit 한다 |
| `test_final_notebook_reproduces.py` | 노트북 산출물이 제출본과 전 행 같다 |
| `test_submission_schema.py` · `test_oof_completeness.py` | 제출·OOF 형식 |

현재 **1,028개 전부 통과**한다.

**DACON 업로드는 사람이 직접 한다.** 이 저장소의 코드는 로컬 csv 만 만든다.
