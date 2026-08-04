# Random Forest Stacking Base Model — 명세

담당: 권병학 · 상태: 설계 확정(승인 완료), 미구현 · 브랜치: `agent/rf-stacking-baseline`
base commit: `28437689300bf3776042b934b50b01b0e7ab421a` (origin/develop, 2026-08-03 fetch 기준)

이 문서는 `/grill-with-docs` 세션에서 합의한 결정을 구현 가능한 명세로 정리한 것이다.
GitHub Issue나 외부 tracker는 만들지 않는다 — 이 파일이 정본이다.

---

## 1. 문제와 목적

DACON 암종(유전자 변이 기반 암 아형) 다중분류 대회에서 권병학이 담당하는 모델은
Random Forest다. **이 모델의 1차 목적은 단독 리더보드 경쟁이 아니라, 권민재가 진행하는
최종 stacking에 넣을 다양성 있는 base model을 만드는 것**이다.

따라서 평가 기준은 "Random Forest가 XGBoost/CatBoost보다 높은 점수를 내는가"가 아니라
다음 두 가지다.

1. 신뢰할 수 있는 OOF/test 클래스 확률을 재현 가능하게 생성하는가
2. 그 확률이 기존 모델과 겹치지 않는 오류 패턴을 가지는가(stacking에 정보를 더하는가)

Macro F1이 기존 모델보다 낮아도, 다양성이 입증되면 전달 후보로 남을 수 있다. 반대로
Macro F1이 같더라도 기존 모델과 예측이 거의 동일하면 stacking 가치가 없다.

### 1.1 알려진 위험: OOF와 Public LB의 괴리

권민재의 EXP_024(Notion, 2026-08-02) 실측 기록: `f4r`(1,055열) XGBoost 기준 OOF Macro
F1(Group5) = 0.4786, train Macro F1 = 0.9255(격차 +0.4526, 정규화로 거의 못 줄임 —
6,201행 × 26클래스 × 최소 클래스 38행이라는 데이터 규모 자체의 과적합으로 진단됨),
Public LB = 0.3520(현재까지 최고 LB). **CV(sgkf)와 LB의 상관관계는 실측 n=4 기준
-0.312(음의 상관)** — OOF를 올린 시도(피처 추가, 정규화, CatBoost 앙상블 블렌드
OOF+0.0151)가 전부 LB를 깎았다(블렌드는 LB -0.0136).

이 프로젝트에서는 "OOF 상승이 Public LB 상승을 보장하지 않는다"는 원칙이 실측으로
뒷받침된다. 따라서 이 명세 전체에서 test/Public LB는 어떤 모델 선택·튜닝·pruning
판단에도 쓰지 않는다. OOF는 유일한 내부 신호이되, 그 한계를 알고 쓴다.

---

## 2. 데이터 계약

### 2.1 원본 데이터 위치

`train.csv`(82MB) / `test.csv`(35MB) / `sample_submission.csv`(35KB)는 저장소 밖의
외부 디렉터리에 있다(조사 시점 기준 팀 프로젝트 루트의 sibling `data/` 디렉터리,
비공개 권한). `onco-ai/data/raw/`는 `.gitkeep`만 있고 비어 있다(`.gitignore`가
`data/raw/*` 제외).

**저장소 안으로 복사하거나 심볼릭 링크를 만들지 않는다.** 대신:

- 실행 스크립트(`scripts/train_rf.py`, `scripts/tune_optuna_rf.py` 등)는
  `--data-dir <external-data-dir>` CLI 인자 또는 `RF_DATA_DIR` 환경변수로 외부
  경로를 받는다(우선순위: CLI > 환경변수). 실제 로컬 경로는 커밋되지 않는 실행
  환경에서만 전달한다 — 이 문서를 포함해 저장소의 어떤 파일에도 절대경로를
  적지 않는다.
- config/로그/provenance JSON에는 **절대경로를 기록하지 않는다.** 파일명
  (`train.csv`/`test.csv`/`sample_submission.csv`)과 SHA-256만 남긴다.
- 기본값은 없다 — `--data-dir`/`RF_DATA_DIR` 둘 다 없으면 즉시 에러로 중단한다.

### 2.2 fold

`data/process/train_folds.parquet`의 `fold_group5` 열(canonical Group5,
`StratifiedGroupKFold` + `profile_hash` groups, `scripts/make_folds.py`가 유일한 생성
경로, 0-based). 파일이 없으면 `python scripts/make_folds.py`로 생성한다 — RF 작업이 fold
를 직접 만들지 않는다(`train_gbdt.py`와 동일 원칙).

### 2.3 class order

26개 SUBCLASS를 알파벳순으로 고정한다(`sklearn.LabelEncoder` 정렬과 동일 관례,
저장소 전체에서 일관됨):

```
ACC, BLCA, BRCA, CESC, COAD, DLBC, GBMLGG, HNSC, KIPAN, KIRC, LAML, LGG, LIHC,
LUAD, LUSC, OV, PAAD, PCPG, PRAD, SARC, SKCM, STES, TGCT, THCA, THYM, UCEC
```

이 순서를 `artifacts/.../class_order.json`(단순 리스트)에 기록하고, 모든 확률 열
순서·`classes_` 정렬을 이 파일 기준으로 검증한다.

---

## 3. 피처 계약

### 3.1 채택: `f4r` (팀 표준, 그대로)

RF-A/B/C 전부 다음 입력을 쓴다 — 확장하지 않는다.

| 블록 | 열 수 | 내용 |
|---|---|---|
| domain (`A_`+`A2_`+`B_`+`C_`+`D_`+`N_`) | 539 | 드라이버92 + 드라이버×기능결과184 + TMB7 + 유형조성6 + 치환쌍241 + 코돈역추론9 |
| rollup16 | 16 | 비율·플래그14 + fold별 burden 2 |
| enc3 (3단계 유전자 인코딩, fold-local chi2 top500) | 500 | `Chi2TopKSelector(k=500)` |
| **합계** | **1,055** | dense |

`scripts/train_gbdt.py`의 `CONFIGS["f4r"]`(`train_gbdt.py:309`)와 동일 조합이며,
`scripts/tune_optuna.py`의 `BASELINE_N_FEATURES = 1055`와 일치한다. 기존 XGBoost/CatBoost
`f4r` 벤치마크(§1.1)와 조건을 동일하게 맞춰 비교 가능하게 하는 것이 목적이다.

**제외(이번 범위 밖)**: class-signature(`csig`, PR#31, 라벨을 보는 supervised 블록),
sparse 토큰 블록(`sigtok`/`exacttok`/`ptok`), latent/module/comut 블록, rollup46,
aa9/parsed19/burden8 등 다른 fold-safe 블록. Pathway 블록(PR#29)은 PR#31에서 이미
삭제되어 저장소에 존재하지 않는다 — 제외할 대상 자체가 없다(재도입 시 감시 필요).

### 3.2 fold-local 계약 (누수 방지, 확인됨)

`Chi2TopKSelector`는 **매 fold의 train 부분에서만** `.fit()`된다. 확인 근거:
`origin/develop`의 `scripts/train_gbdt.py:926-955` — `for fold in range(args.n_splits):`
루프 안에서 `selector = Chi2TopKSelector(k=topk).fit(gene_train[train_index],
data.y[train_index])`, `train_index`는 `fold_ids != fold`로 매 fold 다시 계산됨. 전역에서
한 번 고른 500열을 모든 fold가 재사용하는 경로가 아니다. RF 구현은 이 fold-local 계약을
반드시 재사용한다(재구현 금지, §6 참고).

burden 2열(BurdenBinner)도 동일하게 fold-train에서만 fit한다.

---

## 4. RF-A — 재현 가능한 기준선

```python
RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=<환경 확인 후 안전하게 설정>,
)
```

- `class_weight="balanced_subsample"` **고정**. 이 외 sample_weight는 추가하지 않는다
  (class_weight와 동일 목적 가중치 중복 적용 금지 — PR#27 결론에 따라 중복 그룹 가중치도
  사용하지 않는다).
- fold: Group5(`fold_group5`), seed=42.
- 5-fold 각각 새 모델을 fit하고, OOF와 test predict_proba(5-fold 평균)를 생성한다.
- **RF-A 5-fold 총 실행시간을 실측하고 기록한다** — RF-B Optuna 예산 산정의 전제
  (§5.2). 이 값을 spec 밖 provenance에 남긴다(현재 미확인).

---

## 5. RF-B — Optuna 탐색

### 5.1 탐색 범위 (8개 파라미터, 보수적)

XGBoost `f4r`이 train 0.93/valid 0.48(격차 +0.45)로 정규화가 거의 안 듣는 과적합을
보였다(§1.1). RF는 배깅 구조라 과적합 양상이 다를 수 있지만 같은 데이터 규모 제약
(6,201행×26클래스×최소클래스38행)은 동일하게 적용되므로, 탐색 범위를 보수적으로 제한한다.

| 파라미터 | 범위 |
|---|---|
| `n_estimators` | 400~1,000, step=100 |
| `max_depth` | [8, 12, 16, 20, 24, None] — `None`은 RF-A 대조용으로만 포함, 무제한 깊이를 기본 권장값으로 해석하지 않음 |
| `min_samples_split` | [2, 5, 10, 20] |
| `min_samples_leaf` | [1, 2, 4, 8] |
| `max_features` | 문자열 key로 저장 후 매핑 (§5.4) |
| `max_samples` | [0.65, 0.80, 1.0] |
| `criterion` | ["gini", "log_loss"] |
| `class_weight` | ["balanced", "balanced_subsample"] |

고정값: `bootstrap=True`, `random_state=42`, Group5 fold, test 미사용, sample_weight
미사용(class_weight 외 중복 가중치 금지).

### 5.2 예산

1. **smoke**: 최대 3 trial, 축소된 `n_estimators`(예: 100). 결과는 모델 선택에 쓰지
   않는다 — 데이터·fold·확률·Optuna storage 파이프라인 검증 전용.
2. RF-A의 Group5 5-fold 실측 시간(§4)과 탐색 공간 `n_estimators` 중앙값을 근거로
   trial당 예상 시간을 보수적으로 계산한다.
3. **본 탐색**: hard timeout **3시간**, trial 상한 **40**, 목표 **20 이상**
   (TPE가 의미 있게 동작하려면 최소 이 정도 필요).
4. 예상 가능 trial 수가 **10 미만**이면 TPE 탐색을 강행하지 않는다 — 탐색 공간 축소 또는
   RandomizedSearch 수준의 제한 실험으로 전환할지 사용자에게 보고하고 승인받는다.
5. trial은 병렬 실행하지 않는다(RF 내부 `n_jobs`만 사용해 CPU/메모리 중첩 방지).
6. Optuna sampler seed=42. RF-A 설정을 **trial 0으로 강제 enqueue**한다
   (`study.enqueue_trial(...)`, PR#32 `tune_optuna.py` 관례 재사용).
7. SQLite storage(`load_if_exists=True`)로 영속화 — 중단 후 재개 가능.
8. Stop: 연속 3 trial이 동일 원인으로 실패, 또는 메모리/swap 이상 발생 시 즉시 중단하고
   보고한다.
9. 탐색 중 파라미터가 경계값에 계속 몰리면 범위를 임의로 확장하지 않는다 — 어떤
   파라미터가 경계에 몰렸는지 보고하고 추가 탐색 여부를 사용자에게 확인한다.

### 5.3 Pruning

**기본값은 `optuna.pruners.NopPruner()`(pruning 미사용)다.** Group5는 fold별 클래스
구성이 흔들릴 수 있어 1개 fold 결과만으로 공격적으로 pruning하면 안 되기 때문이다.

RF-A 실측 결과 3시간 안에 최소 10 trial도 어렵다고 판단될 때만, 사용자에게 보고하고
승인받은 뒤 `MedianPruner`를 적용한다. 적용하더라도:

- 최소 2~3개 fold가 끝난 뒤에만 prune 판단을 한다(1 fold 결과로 끊지 않는다).
- **trial 0(RF-A 기준선)은 절대 prune하지 않는다.**

### 5.4 `max_features` 저장 형식

문자열과 실수를 SQLite categorical에 그대로 섞지 않는다. 문자열 key로 저장한 뒤
sklearn 값으로 매핑한다.

| key | sklearn 값 |
|---|---|
| `"sqrt"` | `"sqrt"` |
| `"log2"` | `"log2"` |
| `"frac_005"` | `0.05` |
| `"frac_010"` | `0.10` |
| `"frac_020"` | `0.20` |

### 5.5 로깅

objective는 OOF Macro F1만 최적화하되, 모든 trial에 다음을 진단값으로 함께 저장한다:
train Macro F1, validation OOF Macro F1, train-validation 격차, fold별 Macro F1, fold
표준편차, 클래스별 F1, 실행시간, peak memory, 실패/pruning 사유.

### 5.6 RF-B 채택 조건

다음을 **모두** 만족해야 "명확한 개선"으로 채택 후보가 된다.

1. RF-A 대비 OOF Macro F1 **+0.005 이상** 개선
2. 5-fold 중 **최소 3개** fold에서 개선
3. 특정 소수 클래스의 대규모 붕괴 없음
4. fold 표준편차가 과도하게 증가하지 않음

개선폭이 +0.005 미만이면 "동률/보류"로 기록한다(명확한 개선으로 취급하지 않는다).

---

## 6. RF-C — ExtraTrees 다양성 후보

### 6.1 고정 설정

```python
ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    bootstrap=False,
    class_weight="balanced",
    random_state=42,
    # 나머지는 sklearn 기본값
)
```

`class_weight="balanced"` 고정(RF-A/B의 `balanced_subsample`과 다름 — ExtraTrees는
`balanced_subsample`을 지원하지 않음). 별도 sample_weight는 사용하지 않는다.
입력은 RF-A/B와 완전히 동일한 `f4r` + Group5 + class order + seed=42.

`bootstrap=False`와 노드마다의 무작위 분할점 선택이 ExtraTrees의 구조적 다양성을
만든다(단일 트리가 아니라 다수의 극단적으로 무작위화된 트리를 평균하는 앙상블임을
문서에 명확히 기록한다).

### 6.2 평가 — 목적은 단독 최고점이 아니라 stacking 다양성 검증

먼저 위 고정 설정으로 기준선만 실행하고 다음을 비교한다.

- OOF Macro F1 및 fold별 편차
- 클래스별 F1과 예측 분포
- RF-B 대비 예측 불일치율(disagreement rate)
- RF-B 대비 OOF 확률 상관
- (가능하면) 팀 XGBoost OOF 대비 불일치율·확률 상관 — §9.4 조건부
- 단순 확률 평균(RF-B와 RF-C를 1:1 blend) 시 OOF Macro F1 변화

**RF-B보다 성능이 현저히 낮거나 예측 다양성이 부족하면 ExtraTrees용 Optuna는 진행하지
않는다.** 성능이 수용 가능하고 다양성이 확인될 때만 별도 튜닝 후보로 보고한다.
다양성 판단은 고정된 숫자 임계값이 아니라 위 지표들을 근거로 한 판단이며, 근거를 반드시
기록한다. RF-C Optuna 진행 여부는 기준선 결과를 사용자에게 보고한 뒤 별도 승인 사항이다.

---

## 7. Loop 계약

**Trigger**: 사용자가 승인한 티켓을 `/implement`로 시작할 때만 실행한다.

**State**: 이 spec 파일, tickets 문서와 blocking 관계, PROGRESS/실험 진행 기록,
`data/process/train_folds.parquet`(fold 파일과 provenance), Optuna SQLite study,
trials CSV, metrics JSON, 현재 최고 config와 결정 로그.

**Plan**: 한 번에 하나의 티켓 또는 하나의 제한된 실험군만 수행한다.

**Work**: 데이터 계약 검증 → RF-A 기준선(+ 5-fold 실행시간 실측) → 제한된 RF-B Optuna
탐색 → 최종 Group5 재학습 → RF-C 기준선(+ 조건부 승인 시 Optuna) → OOF/test 확률 및
submission 생성 → stacking 전달 번들 생성.

**Evaluate**: §4~6에 정의된 지표 전부(OOF Macro F1, fold별 평균/표준편차, 26클래스별
F1, singleton/duplicate 성능, 예측 클래스 분포, 확률 행 합/결측, 모델 간 불일치율·확률
상관, 실행시간·메모리, OOF 커버리지·Group5 누수 여부, confusion matrix).

**Gate**: §9.5 acceptance criteria + §5.6/§6.2 채택 기준.

**Stop**: 원본 데이터/`sample_submission` 없음 · Group5 누수 발견 · class order 미확정 ·
메모리 부족/비정상 swap · 연속 3 trial 동일 원인 실패 · trial/timeout 예산 소진 ·
기준선 재현 실패 · 외부 데이터 규정 위반 가능성 발견. 각 Stop 상황에서 임의 우회하지
않고 진행 상태와 필요한 사용자 결정을 보고한다.

**Memory**: 이 spec, tickets 문서, provenance JSON, Optuna study DB가 세션 간 상태를
보존한다. `/handoff`로 세션이 끊기면 이 네 가지부터 확인한다.

---

## 8. 산출물 스키마

팀 표준 `cancer_hack.metrics.build_prediction_frame`을 **rename 없이 그대로** 사용한다
(`proba__{class}`/`prediction` 같은 개인 workspace 관례나, 원 브리핑의 `pred_label`
관례는 채택하지 않는다 — 팀 유틸리티와의 호환을 우선한다).

### 8.1 OOF 확률

| 컬럼 | 설명 |
|---|---|
| `ID` | 원본 train.csv 행 순서 |
| `fold` | 0~4 |
| `y_true` | 실제 SUBCLASS |
| `y_pred` | argmax(probability) |
| `p_{class}` × 26 | canonical class order(§2.3) |

6,201행, 중복·누락 없음. 각 train 행은 정확히 한 fold의 validation에서만 예측된다.

### 8.2 Test 확률

| 컬럼 | 설명 |
|---|---|
| `ID` | |
| `p_{class}` × 26 | canonical class order |

sample_submission과 동일한 2,546개 ID·순서. 5개 fold 모델 predict_proba의 평균.

### 8.3 단독 submission

`ID, SUBCLASS`(test 확률 argmax). **UTF-8-sig 인코딩**(권민재 EXP_024 실측 선례 반영),
sample_submission과 ID 순서 완전 일치, 결측 0. 사용자가 직접 제출할 후보일 뿐 자동
제출 금지.

### 8.4 class order 기록

`class_order.json`(단순 리스트, §2.3)을 별도로 저장한다.

### 8.5 동봉 정보

config YAML(또는 `--data-dir` 기반 CLI 인자 기록), Group5 fold provenance
(fold_group5 해시 포함), 전체/fold별/클래스별 metrics, confusion matrix(raw +
normalized), Optuna study DB·trials 요약·best_params, feature 목록·순서(f4r 계약),
class order, **Git commit(full SHA)**, Python·핵심 라이브러리 버전, 원본 데이터
파일명+SHA-256(절대경로 아님), 실행 방법 README, 파일별 SHA-256.

---

## 9. 테스트 seam과 acceptance criteria

### 9.1 코드 구조

- `src/cancer_hack/models_rf.py`: `RandomForestClassifier`/`ExtraTreesClassifier`를
  감싸는 래퍼. `BaseGBDT`는 상속하지 않지만 다음 인터페이스 계약을 유지한다.
  - `fit(X, y, sample_weight=None)`
  - `predict_proba(X)`
  - `classes_`
  - `feature_importances_`
  - 고정 `random_state`, fit 전 predict 호출 방어, 확률 shape·유한값·class order 검증
  - RandomForest/ExtraTrees 공통 로직은 복붙하지 않고 내부 공통 wrapper/factory로 공유
  - 책임 범위: 모델 생성, 파라미터 검증, fit 상태 관리, canonical class order 정렬,
    누락/예상외 클래스 방어, sample_weight 전달, seed/n_jobs 설정
- `scripts/train_rf.py`: `train_gbdt.py`와 같은 CLI 관례(`--config`, `--cv`, `--tag`,
  overwrite 방지 등). canonical f4r 피처·Group5 fold·`build_prediction_frame`을
  재사용한다(재구현 금지). 책임 범위: config/cv/tag 옵션, f4r 피처 생성·로드, fold별
  모델 생성·학습, OOF/test predict_proba 생성, metrics 저장, test 확률 fold 평균,
  submission·provenance 저장, 기존 파일 덮어쓰기 방지.
- `scripts/tune_optuna_rf.py`: PR#32 `tune_optuna.py`의 SQLite/trial 기록 관례를
  재사용하는 별도 CLI. 노트북 셀에 Optuna를 넣지 않는다.
- 노트북은 정본이 아니다 — 필요하면 CLI가 만든 metrics/OOF를 읽어 설명하는 얇은 분석
  노트북만 별도로 둘 수 있다.

### 9.2 필수 테스트 (RF 전용 파일 — 팀 공용 빈 스텁은 건드리지 않음, §9.3)

`tests/test_models_rf.py`, `tests/test_train_rf.py`(필요 시
`tests/test_rf_artifact_schema.py`)에 최소 다음을 포함한다.

- RF/ExtraTrees fit 및 predict_proba shape
- 확률 행 합과 NaN/Inf 검사
- `classes_` 및 canonical class order 정렬
- 동일 seed 결정론성
- sample_weight 전달
- fit 전 predict 호출 시 에러
- 잘못된 모델 종류/파라미터 에러
- 작은 합성 데이터로 Group5 OOF 완전 커버리지
- 동일 group의 fold 교차 0
- test가 fit에 전달되지 않는지
- CLI smoke run과 산출물 스키마

**OOF 검증**: 원본 train과 행 수·ID 집합 일치, ID 중복/누락 0, 각 train 행이 정확히 한
fold의 validation에만 포함, fold 값 0~4, 동일 profile_hash의 fold 교차 0, y_true/y_pred가
canonical class 집합에 포함, `p_{class}` 26열과 class order 일치, NaN/Inf 0, 확률 범위
[0,1], 행별 확률 합≈1, y_pred가 argmax와 일치, 저장된 Macro F1이 OOF에서 재계산한 값과
일치.

**Test/submission 검증**: sample_submission과 ID 수·집합·순서 일치, ID 중복/결측 0,
`p_{class}` 26열과 canonical 순서 일치, 확률 행 합≈1, submission 컬럼이 정확히
`ID,SUBCLASS`, SUBCLASS가 canonical 26클래스 중 하나, SUBCLASS가 test probability
argmax와 일치, index/Unnamed 컬럼 없음, CSV 저장 후 재읽기해도 스키마·값 유지.

테스트는 실제 대회 데이터에 의존하지 않는 작은 합성 데이터와 임시 디렉터리로 실행
가능해야 한다. 실제 RF 산출물은 실행 단계에서 동일 validator를 재호출해 별도 검증한다.

### 9.3 팀 공용 빈 스텁 처리

`tests/test_oof_completeness.py`, `tests/test_submission_schema.py`는 팀 저장소에서
현재 0바이트 빈 스텁이다(강제되는 스키마 검증이 전혀 없음). **이번 RF 작업 범위에서는
수정하지 않는다** — 공용 파일의 의도·담당자가 확인되지 않았고, 채우면 RF 구현을 넘어
팀 전체 산출물 계약을 변경하게 된다. 대신 §9.2의 RF 전용 테스트에서 동등한 검증을
빠짐없이 수행한다. 최종 보고서에 "확인된 미구현 공용 검증"으로 기록하고, 필요하면 RF
PR 이후 별도 공통 작업/PR로 제안한다.

### 9.4 XGBoost 비교 (조건부, 재학습 금지)

팀 XGBoost OOF가 **신뢰할 수 있는 class order와 ID 스키마로 제공되면** RF-C 다양성
비교(§6.2)에 사용한다. 제공되지 않으면 RF-A/RF-B/RF-C 사이의 다양성만 평가하고, XGBoost
비교 항목은 "자료 미확보"로 명시적으로 기록한다. **이번 RF 작업 범위에서 XGBoost를
재학습하지 않는다** — 팀 XGBoost OOF가 없다는 이유로 새로 학습하는 것은 범위 밖이다.

### 9.5 Acceptance criteria (Gate)

- 6,201개 train 행 각각 정확히 한 번 OOF 예측
- 동일 profile_hash가 fold를 넘는 사례 0개
- test가 fit, early stopping, parameter selection에 사용된 사례 0개
- 모든 probability 열 순서가 canonical 26개 class order와 일치
- NaN/Inf 0개, 각 행 확률 합이 1
- sample_submission의 ID 순서·집합과 submission 완전 일치
- RF-B는 §5.6, RF-C는 §6.2 채택 기준 적용
- OOF 상승이 Public LB 상승을 보장한다고 해석하지 않는다(§1.1)

---

## 10. Stacking 전달 번들

`share/random_forest_stacking_<YYYYMMDD>/` — PR#27 번들 구조
(`share/duplicate_group_weight_experiment_20260803/`)를 확장한다.

```
share/random_forest_stacking_<YYYYMMDD>/
├── README.md
├── BUNDLE_MANIFEST.json
├── SHA256SUMS.txt
├── class_order.json
├── configs/
├── scripts/
├── lib/
├── artifacts/
│   ├── folds/
│   ├── oof/
│   ├── test_predictions/
│   ├── submissions/
│   ├── metrics/
│   ├── confusion_matrices/
│   ├── tuning/
│   │   ├── optuna_study.db
│   │   ├── trials.csv
│   │   └── best_params.json
│   └── provenance/
└── reference_results/
    ├── model_comparison.csv
    └── decision_summary.json
```

**필수 내용**: RF-A/B/C별 OOF·test 확률, 각 모델의 단독 submission 후보, 전체/fold별/
클래스별 Macro F1, raw+normalized confusion matrix, 모델 간 예측 불일치율·확률 상관,
RF-A/B/C 채택·보류·기각 판정과 근거, Optuna study DB·전체 trials CSV·best params,
canonical class order, f4r 피처 계약, Group5 fold provenance(fold hash 포함), **Git
commit full SHA**, 데이터 파일명+SHA-256(절대경로 아님), 환경·라이브러리 버전, 각 파일의
상대경로 기반 SHA-256.

**`BUNDLE_MANIFEST.json` RF 전용 필드**: `canonical_model`, `model_variants`(RF-A/B/C),
`decision_status`/`decision_reason`, `class_order_path`, `feature_config_path`,
`fold_provenance_path`, `optuna_study_path`, `oof_probability_paths`,
`test_probability_paths`, `submission_paths`, `stacking_recommended_paths`,
`base_git_commit`(full SHA), `data_hashes`, `created_at`, `schema_version`.

**README 첫 부분**: (1) stacking에 권장하는 OOF/test 파일, (2) `ID`+`p_{class}` 스키마,
(3) canonical class order, (4) RF-A/B/C 차이, (5) 최종 판정, (6) checksum 검증 명령,
(7) 재실행 명령.

**제외**: train/test/sample_submission 원본, 개인 절대경로, API key/토큰/환경변수 값,
pytest·notebook cache, 임시 Optuna WAL/SHM 파일, 필요성이 확인되지 않은 대용량 모델
pickle/joblib, 중간 parquet과 중복 아티팩트.

번들 생성 후 절대경로·개인정보를 스크럽하고, `SHA256SUMS.txt`를 마지막에 생성해 전
파일을 재검증한다. ZIP 생성 후에도 다시 열어 manifest·파일 수·체크섬을 확인한다.

**이번 단계는 로컬 번들·ZIP 생성까지만이다. Google Drive 업로드는 하지 않는다** —
검증 보고 후 사용자가 수동으로 업로드한다.

---

## 11. 재현성과 provenance

- Git commit(브랜치 base **full SHA**: `28437689300bf3776042b934b50b01b0e7ab421a`, 작업
  커밋은 각 티켓에서 추가 기록)
- Python·scikit-learn(1.7.2 확인됨)·optuna(4.8.0 확인됨) 등 핵심 라이브러리 버전
- 데이터 파일명 + SHA-256(§2.1, 절대경로 기록 금지)
- Group5 fold provenance(`train_folds.json`, 생성 시 seed/n_splits 포함)
- Optuna SQLite study(재개 가능)
- 실행 환경: CPU 8코어(물리 8), RAM 16GB(확인됨) — n_jobs/병렬 trial 설정의 근거

---

## 12. 제외 범위

- Pathway/PPI/BLOSUM/Grantham/외부 annotation 피처(현재 저장소에 존재하지 않음 — 재도입
  감시만 필요)
- class-signature(`csig`), sparse 토큰, latent/module/comut 블록, rollup46 등 f4r 외
  안전 블록 확장(별도 실험으로만 가능, RF-A/B/C에 섞지 않음)
- 중복 그룹 역수 가중치(PR#27 결론에 따라 미사용)
- 팀 XGBoost 재학습(§9.4)
- `tests/test_oof_completeness.py`/`test_submission_schema.py` 등 팀 공용 스텁 수정
  (§9.3)
- 원본 데이터의 저장소 내 복사/심볼릭 링크(§2.1)
- 실제 DACON 제출(사용자가 수동으로 함)

---

## 13. 권한 경계

- GitHub push/PR/merge, Google Drive 업로드, Notion 수정, DACON 제출은 **승인 전
  금지**한다.
- 로컬 커밋은 가능하나(사용자 승인 시), branch/commit 자체도 명시적 승인 후에만
  수행한다(이번 세션에서 `agent/rf-stacking-baseline` 브랜치 생성은 사용자 승인 완료).
- `main`/`develop`에는 직접 push하지 않는다.
- 개인 절대경로와 원본 데이터는 커밋하지 않는다.

---

## 14. 미확인 항목 (구현 세션에서 확인 필요)

- `f4r`(1,055열, 6,201행)에서 RF `n_estimators=500` Group5 5-fold 실측 소요시간 — RF-B
  trial 수 확정의 전제(§5.2)
- 팀 XGBoost의 Group5 f4r OOF 파일이 팀 저장소 또는 신뢰 가능한 위치에 실제로 존재하는지
  (§9.4) — 권민재의 EXP_024 수치는 Notion 기록일 뿐, 파일 자체는 개인 로컬 workspace
  산출물로만 확인됨
