# Random Forest Stacking Model — Tickets

정본 spec: `docs/specs/random_forest_stacking_model.md`. 이 문서는 그 spec을 구현
가능한 tracer-bullet 티켓으로 분할한 것이며, GitHub Issue나 외부 tracker를 대체하지
않는다 — 이 파일이 정본이다.

모든 티켓 공통 금지 사항: GitHub push/PR/merge, Google Drive 업로드, Notion 수정,
DACON 제출, 원본 데이터의 저장소 내 복사·심볼릭 링크. 이 문서 어디에도 개인 절대
경로를 쓰지 않는다 — 외부 데이터 경로는 항상 `--data-dir <external-data-dir>` CLI
인자 또는 `RF_DATA_DIR` 환경변수로만 표현하고, 실제 로컬 경로는 커밋되지 않는 실행
환경에서만 전달한다. provenance에는 파일명과 SHA-256만 기록한다(절대경로 금지).

## DAG

```
Ticket 1a
  └─ Ticket 1b
       ├─ Ticket 2 → [사용자 예산 승인 gate] → Ticket 3
       └─ Ticket 4
Ticket 3 + Ticket 4
  └─ Ticket 5
       └─ Gate 6 (새 세션, /code-review, report-only)
```

| 티켓 | 예상 세션 | 선행 |
|---|---|---|
| 1a | 1 | 없음 |
| 1b | 1 | 1a |
| 2 | 1(가벼움) | 1b |
| 3 | 1(단, 최대 3시간 실행시간) | 2 + 사용자 예산 승인 |
| 4 | 1 | 1b (3와 독립적으로 병렬 가능) |
| 5 | 1 | 3 + 4 |
| Gate 6 | 1(새 세션) | 5 |

총 7개 세션.

---

## Ticket 1a — RF 학습 인터페이스 및 합성 데이터 end-to-end

**목적**: 실제 대회 데이터 없이 RF/ET 공통 학습 인터페이스, CLI 뼈대, artifact
validator를 합성 데이터로 검증한다. 이 티켓에서는 실제 대회 데이터 전체 학습을
하지 않는다.

**입력**: 승인된 spec §2.3(class order), §3(피처 계약), §9.1(코드 구조), §9.2(테스트
목록). 데이터는 전부 합성(테스트 코드 안에서 생성).

**작업 범위**:
- `--data-dir <external-data-dir>` CLI 계약과 `RF_DATA_DIR` 환경변수 지원(우선순위:
  CLI > 환경변수, 둘 다 없으면 즉시 에러). 이 티켓에서는 존재 여부 검사·에러 처리
  로직만 합성 경로로 테스트하고, 실제 파일 유무 판정은 Ticket 1b에서 실데이터로
  확인한다.
- `src/cancer_hack/models_rf.py`: `RandomForestClassifier`/`ExtraTreesClassifier`
  공통 wrapper/factory. 계약:
  - `fit(X, y, sample_weight=None)`
  - `predict_proba(X)`
  - `classes_`
  - `feature_importances_`
  - 고정 `random_state`, 지정 가능한 `n_jobs`
  - fit 전 `predict`/`predict_proba` 호출 시 에러
  - `predict_proba` 출력이 canonical class order(§4)에 맞게 정렬되는지 검증,
    누락/예상외 클래스 방어
  - `BaseGBDT`는 상속하지 않는다. RF/ET 공통 로직은 복붙하지 않고 내부 공통
    wrapper 또는 factory 함수로 공유한다.
  - **이 티켓에서 RF/ET 양쪽 경로를 전부 구현·테스트한다** — Ticket 1b는 RF 경로만
    실행하고, Ticket 4는 이 모듈을 수정하지 않고 ET 경로를 그대로 재사용한다.
- `artifacts/metadata/class_order.json` 결정 로직:
  1. 저장소 안에 이미 기록된 팀 canonical class order가 있는지 먼저 확인한다
     (예: 다른 모델의 metadata, `LabelEncoder`가 실제로 산출한 `classes_` 순서를
     기록한 기존 산출물 등). 있으면 그 정본을 그대로 재사용한다.
  2. 없을 때만 train `SUBCLASS` 알파벳 정렬 결과로 새로 생성한다.
  3. 어느 경로를 택했는지와 근거를 함께 기록한다(새 정본을 임의로 만든 것처럼
     보이지 않게 한다).
  4. 파일 경로: `artifacts/metadata/class_order.json`(단순 리스트).
- `scripts/train_rf.py`의 최소 CLI 구조(`--config`, `--cv`, `--tag`, `--data-dir`,
  overwrite 방지 플래그 등) — 이 티켓에서는 합성 데이터 경로만 구동한다.
- RF artifact validator(예: `src/cancer_hack/rf_artifact_validator.py`): spec
  §9.2의 OOF/Test/submission 검증 항목을 재사용 가능한 함수로 구현한다. Ticket
  1b/3/4/5가 전부 이 validator를 그대로 호출한다(재구현 금지).
- 합성 데이터 기반 CLI smoke: 소규모 행/열/26클래스 축소본, profile_hash 중복
  그룹 구조를 흉내 낸 합성 group 포함.
- 단위·통합 테스트(모두 합성 데이터).

**산출물(정본 파일)**:
- `src/cancer_hack/models_rf.py`
- `scripts/train_rf.py`(CLI 뼈대)
- `src/cancer_hack/rf_artifact_validator.py`
- `tests/test_models_rf.py`, `tests/test_train_rf.py`, `tests/test_rf_artifact_schema.py`
- `artifacts/metadata/class_order.json`(신규 생성 또는 기존 재사용 확인 기록)

**테스트**: spec §9.2 전부(합성 데이터) — fit/predict_proba shape, 확률 합/NaN·Inf,
class order 정렬, 동일 seed 결정론성, sample_weight 전달, fit-전-predict 에러,
잘못된 모델 종류/파라미터 에러, 합성 Group5류 구조로 OOF 완전 커버리지, 동일 group의
fold 교차 0, test가 fit에 전달되지 않는지, CLI smoke, OOF/submission 스키마
validator 자체의 정확성(고의로 깨진 합성 산출물을 넣어 validator가 실패를 잡아내는지
포함).

**Acceptance criteria**: 전체 `pytest -q` green. 합성 데이터 CLI smoke가 validator를
통과. `class_order.json` 결정 로직(기존 재사용 vs 신규 생성)이 근거와 함께 기록됨.
RF/ET 양쪽 경로 모두 테스트로 커버됨.

**Stop 조건**: 없음(합성 데이터만 다루므로 데이터 계약 관련 Stop이 없다). 반복되는
코드 결함은 일반적인 diagnose-and-report 대상이다.

**선행 티켓**: 없음(승인된 spec만 전제).

---

## Ticket 1b — 실제 데이터 RF-A 기준선

**목적**: 실제 외부 데이터로 canonical f4r(1,055열)·Group5 fold 기반 RF-A 전체
5-fold를 실행하고 실측 실행시간·메모리를 남긴다.

**입력**: Ticket 1a의 `models_rf.py`/`train_rf.py`/`rf_artifact_validator.py`.
외부 데이터는 `--data-dir <external-data-dir>` 또는 `RF_DATA_DIR` 환경변수로
실행 환경에서만 전달한다(문서·config·로그에 절대경로를 쓰지 않는다).

**작업 범위**:
- 전달된 데이터 디렉터리에서 `train.csv`/`test.csv`/`sample_submission.csv`
  존재를 확인하고, **파일명 + SHA-256만** provenance에 기록한다(절대경로 기록
  금지).
- **Group5 fold 처리**(모순 해소, 확정):
  1. `data/process/train_folds.parquet`가 있으면 스키마(열 이름, 행 수,
     `fold_group5` 열 존재)를 검증한 뒤 재사용한다.
  2. 없으면 canonical `scripts/make_folds.py`를 호출해 생성한다. **fold 생성
     로직을 `train_rf.py`에 재구현하지 않는다.**
  3. 생성/재사용된 fold 파일은 runtime artifact이며 Git commit 대상이 아니다
     (`.gitignore`가 이미 `data/process/*`를 제외한다).
  4. fold 파일의 SHA-256과 생성 provenance(seed, n_splits, 생성 시각, 원본 소스
     파일명)를 기록한다.
  5. **실제로 사용한 열이 `fold_group5`인지 코드에서 assert한다.**
  6. `--cv` 값을 이 문서에서 미리 단정하지 않는다 — 구현 시점에
     `src/cancer_hack/validation.py`의 `fold_column()`/`CV_SLUG`를 직접 확인해
     `fold_group5`에 대응하는 실제 accepted value를 사용한다(코드 조사 시점
     기준으로는 `"sgkf"`였으나 재확인이 필요하며, 값이 바뀌었으면 이 티켓에서
     문서를 갱신한다).
- canonical f4r(도메인539+rollup16+enc3 fold-local chi2 top500=1,055열) 조립,
  fold-local chi2(재사용, 재구현 금지 — `Chi2TopKSelector`가 매 fold의 train
  부분에서만 fit되는 것을 spec §3.2에서 코드로 이미 확인함).
- RF-A 고정 설정(spec §4: `n_estimators=500, max_features="sqrt",
  class_weight="balanced_subsample", random_state=42`)으로 Group5 5-fold 전체
  실행.
- OOF/test 확률/submission 후보 생성, `build_prediction_frame` 그대로 사용
  (rename 없음).
- Ticket 1a의 `rf_artifact_validator`로 실제 산출물을 검증.
- 5-fold 총 실행시간·peak memory 실측 및 기록(Ticket 2의 입력).
- **Macro F1 계산 시 `labels=canonical_class_order`를 명시**해 26개 클래스를
  항상 일관되게 평가한다(특정 fold의 validation에 없는 클래스가 있어도 해당
  클래스는 0점으로 정확히 반영되도록 한다).

**산출물(정본 파일)**:
- `data/process/train_folds.parquet`(runtime artifact, 없었을 경우 생성),
  `data/process/train_folds.json`(provenance)
- `artifacts/oof/oof_rf_a_f4r_group5_s42.csv`
- `artifacts/test_predictions/test_rf_a_f4r_group5_s42.csv`
- `artifacts/submissions/submission_rf_a_f4r_group5_s42.csv`
- `artifacts/logs/rf_a_f4r_group5_s42.json`(metrics + 실행시간 + peak memory +
  클래스 결측 진단)

**테스트**: validator를 실제 산출물에 재호출. fold 파일 "이미 있음"과 "없어서
생성" 양쪽 경로 테스트. `fold_group5` 열 사용 assert 테스트. `labels=`
명시 여부를 검증하는 metrics 계산 단위 테스트.

**Acceptance criteria**: spec §9.5 전부 + Ticket 1a validator 통과 + provenance에
절대경로가 전혀 없음(파일명+SHA-256만) + 실행시간·peak memory 기록 완료 + 6,201/
2,546행 정합.

**Stop 조건**(수정): `--data-dir`/`RF_DATA_DIR` 둘 다 없음 · 원본 3개 파일 중
하나라도 없음 · fold 파일 검증 실패(재생성해도 `fold_group5` 불일치) ·
**train partition에서 canonical 26개 클래스 중 하나라도 완전히 빠져 26개 확률을
생성할 수 없는 경우** · 메모리 부족·비정상 swap.
**Stop 아님**: validation fold에 일부 클래스가 없는 것 — Group5 구조상 가능하므로
중단하지 않고 `artifacts/logs/rf_a_...json`의 진단 필드에만 기록한다.

**선행 티켓**: Ticket 1a

---

## Ticket 2 — Optuna smoke 및 본 탐색 예산 산정

**목적**: RF-B 본 탐색 전에 파이프라인(데이터·fold·확률·Optuna storage)이 실제로
동작하는지 검증하고, Ticket 1b 실측치로 3시간 예산 안에 가능한 trial 수를 계산해
사용자 승인을 받는다. **본 탐색은 이 티켓에서 실행하지 않는다.**

**입력**: Ticket 1b의 `models_rf.py`, `class_order.json`, Group5 fold, 5-fold
실행시간·peak memory 실측치.

**작업 범위**:
- `scripts/tune_optuna_rf.py` 생성(PR#32 `tune_optuna.py` 관례 재사용 — SQLite
  storage, TPE sampler seed=42, `NopPruner` 기본값).
- **별도의 smoke 전용 study**(`artifacts/tuning/rf_b_smoke_study.db`)로 최대
  3 trial, `n_estimators` 축소(예: 100).
- **SQLite 재개 검증(강제 종료 없이)**:
  1. 첫 실행에서 1개 trial을 완료하고 프로세스를 정상 종료한다.
  2. 같은 `study_name`과 `storage`로 다시 열어 나머지 trial(최대 2개)을
     실행한다.
  3. 기존 trial이 보존되고 trial 번호가 이어지는지(`len(study.trials)`, trial
     순번 연속성) 검증한다.
- Ticket 1b 실측 시간 + 탐색 공간 `n_estimators` 중앙값(spec §5.1:
  400~1,000, step100)으로 trial당 예상 시간을 계산 → 3시간 상한 안에서 가능한
  trial 수를 산정한다.
- 예상 trial 수, peak memory, 계산 근거를 보고 문서로 정리한다. 예상 trial 수가
  10 미만이면 탐색 축소 또는 RandomizedSearch 대안을 함께 보고한다.
- **smoke DB는 폐기물이 아니라 audit용 진단 산출물로 보존한다** — 파이프라인이
  실제로 동작했다는 증거로 남기되, **모델 선택과 Ticket 5의 최종 reference
  result에는 사용하지 않는다**고 `rf_b_budget_report.json`에 명시한다.

**산출물(정본 파일)**:
- `scripts/tune_optuna_rf.py`
- `artifacts/tuning/rf_b_smoke_study.db`(보존, audit 전용 — 본 탐색과 별개 파일)
- `artifacts/tuning/rf_b_budget_report.json`(실측 시간, trial당 예상 시간,
  산정된 trial 수, 메모리 근거, smoke DB의 audit-only 지위 명시)

**테스트**: 재개 시나리오(정상 종료 → 재오픈) 검증, trial 번호 연속성 검증,
objective가 OOF Macro F1만 사용하고 test/LB를 참조하지 않는지 정적 검사,
`NopPruner` 기본 적용 확인.

**Acceptance criteria**: 예상 trial 수·근거가 사용자에게 보고됨.

**Stop 조건**: smoke 3 trial 중 2개 이상 동일 원인 실패 · SQLite 재개 실패 ·
메모리·swap 이상.

**Gate**: 이 티켓 완료 후 **사용자가 본 탐색 trial 수(≤40)를 명시적으로 승인해야
Ticket 3를 시작할 수 있다.** 승인 없이 Ticket 3로 넘어가지 않는다.

**선행 티켓**: Ticket 1b

---

## Ticket 3 — RF-B 제한 Optuna 탐색

**목적**: 사용자 승인된 예산으로 실제 RF-B 탐색을 수행하고 RF-A 대비 채택/보류/
기각을 판정한다.

**입력**: Ticket 2에서 사용자가 승인한 trial 수, `scripts/tune_optuna_rf.py`,
Ticket 1b의 RF-A OOF(비교 기준선).

**작업 범위**:
- **새 study**(`artifacts/tuning/rf_b_optuna_study.db`, smoke DB와 별개 파일,
  처음부터 시작) 생성. spec §5.1 탐색 공간(8개 파라미터) 적용:
  `n_estimators` 400~1,000(step100), `max_depth` [8,12,16,20,24,None],
  `min_samples_split` [2,5,10,20], `min_samples_leaf` [1,2,4,8], `max_features`
  (spec §5.4 문자열 key → sklearn 값 매핑: `sqrt`/`log2`/`frac_005`=0.05/
  `frac_010`=0.10/`frac_020`=0.20), `max_samples` [0.65,0.80,1.0], `criterion`
  [gini, log_loss], `class_weight` [balanced, balanced_subsample].
- RF-A 설정을 trial 0으로 강제 enqueue(`study.enqueue_trial(...)`) → 재학습
  결과가 Ticket 1b의 RF-A OOF와 **다음 기준으로** 일치하는지 검증한다(바이트
  단위 일치를 요구하지 않는다):
  - OOF의 ID/fold/class order가 완전히 일치
  - 저장된 확률이 허용오차 내 일치(허용오차는 테스트 상수로 명시 — 구현
    시점에 실제 재현 정밀도를 관찰해 합리적 값을 고정하고 테스트
    docstring에 근거를 남긴다. 시작 제안값: 확률 절대오차 `atol=1e-6`,
    OOF Macro F1 절대오차 `atol=1e-6` — 병렬 `n_jobs`로 인한 부동소수점
    합산 순서 차이가 실측되면 근거와 함께 완화한다)
  - 설정(하이퍼파라미터)·fold·seed가 일치
  - **이 재현 검증에 실패하면 본 탐색을 중단한다.**
- 승인된 trial 수만큼 순차 실행(병렬 없음, RF 내부 `n_jobs`만 사용), 매 trial
  마다 spec §5.5 진단값(train Macro F1, validation OOF Macro F1, 격차, fold별
  점수, fold 표준편차, 클래스별 F1, 실행시간, peak memory, 실패/pruning 사유)을
  기록.
- pruning은 기본 `NopPruner`. 만약 사용자 승인 하에 `MedianPruner`로 전환했다면
  최소 2~3 fold 완료 후에만 판단하고 trial 0은 절대 prune하지 않는다.
- 연속 3 trial이 동일 원인으로 실패하면 즉시 중단.
- best trial로 최종 Group5 5-fold 재학습 → OOF/test 확률/submission 후보 생성.
- spec §5.6 채택 기준으로 판정: RF-A 대비 OOF Macro F1 **+0.005 이상** 개선
  **and** 5-fold 중 **최소 3개**에서 개선 **and** 특정 소수 클래스 대규모
  붕괴 없음 **and** fold 표준편차 과도 증가 없음. 미달 시 "동률/보류"로 기록.

**산출물(정본 파일)**:
- `artifacts/tuning/rf_b_optuna_study.db`, `artifacts/tuning/rf_b_trials.csv`,
  `artifacts/tuning/rf_b_best_params.json`
- `artifacts/oof/oof_rf_b_f4r_group5_s42.csv`
- `artifacts/test_predictions/test_rf_b_f4r_group5_s42.csv`
- `artifacts/submissions/submission_rf_b_f4r_group5_s42.csv`
- `artifacts/logs/rf_b_f4r_group5_s42.json`(fold·클래스별 metrics, 채택/보류/
  기각 판정 + 근거)

**테스트**: trial 0 재현 검증(위 허용오차 기준), objective가 test/LB를 참조하지
않는지 정적 검사, study 재개 가능성, `rf_artifact_validator` 재사용 검증.

**Acceptance criteria**: spec §9.5 전부 + §5.6 채택 기준 명시적 판정 + 파라미터가
경계값에 계속 몰릴 경우 범위를 임의로 확장하지 않고 보고.

**Stop 조건**: trial 0 재현 실패 · 연속 3 trial 동일 원인 실패 · 3시간 hard
timeout 도달 · 메모리·swap 이상.

**선행 티켓**: Ticket 2 + **사용자 예산 승인 gate**

---

## Ticket 4 — RF-C ExtraTrees 다양성 기준선

**목적**: spec §6.1 고정 설정으로 ExtraTrees 기준선을 실행하고 **RF-A 대비**
다양성을 확인한다. RF-B 대비 비교는 Ticket 5로 이연한다(Ticket 3와 병렬 진행
가능하도록). 성능·다양성이 부족하면 여기서 멈춘다(ExtraTrees Optuna는 이
티켓에서 진행하지 않는다).

**입력**: Ticket 1a/1b의 `models_rf.py`(ET 경로, 수정하지 않고 그대로 재사용),
Group5 fold, class order, RF-A OOF(Ticket 1b).

**작업 범위**:
- spec §6.1 고정 설정으로 Group5 5-fold 실행:
  `ExtraTreesClassifier(n_estimators=500, max_features="sqrt", bootstrap=False,
  class_weight="balanced", random_state=42)`, 나머지 sklearn 기본값, 별도
  sample_weight 없음.
- OOF/test 확률/submission 후보 생성, `rf_artifact_validator`로 검증.
- **RF-A 대비**만 계산: disagreement rate, OOF 확률 상관, 단순 평균 blend
  (RF-A + RF-C 1:1) 시 OOF Macro F1 변화.
- 팀 XGBoost Group5 f4r OOF를 저장소 `artifacts/oof/`에서 탐색(패턴 예:
  `xgb_*_f4r_*group5*`). 없고 사용자가 이 티켓 시작 시점에 별도 파일을 제공하지
  않으면 **재학습하지 않고 "자료 미확보"로 기록한다**(Notion/Drive를 추가로
  뒤지지 않는다 — 이 티켓 범위 밖).
- ExtraTrees Optuna는 **실행하지 않는다** — 결과를 보고한 뒤 별도 승인 대상으로만
  남긴다.

**산출물(정본 파일)**:
- `artifacts/oof/oof_rf_c_f4r_group5_s42.csv`
- `artifacts/test_predictions/test_rf_c_f4r_group5_s42.csv`
- `artifacts/submissions/submission_rf_c_f4r_group5_s42.csv`
- `artifacts/logs/rf_c_f4r_group5_s42.json`
- `artifacts/metrics/rf_c_vs_rf_a_diversity.json`(disagreement/상관/blend 결과,
  XGBoost 비교는 있으면 포함·없으면 "자료 미확보" 명시)

**테스트**: ET wrapper가 Ticket 1a의 기존 계약을 그대로 만족하는지(신규 모델
테스트 최소화, 기존 스위트 재사용), diversity 계산 함수 단위 테스트(합성 데이터로
disagreement/상관 계산 정확성 검증).

**Acceptance criteria**: spec §9.5 전부(RF-C에도 동일 적용) + RF-A 대비 다양성
지표 보고 완료 + ExtraTrees Optuna 미실행 확인.

**Stop 조건**: 메모리·swap 이상 · RF-A와 예측이 완전히 동일(다양성 0, 코드 버그
의심 — 중단하고 원인 보고).

**선행 티켓**: Ticket 1b (Ticket 3와 독립적으로 병렬 진행 가능)

---

## Ticket 5 — 최종 stacking 전달물 및 번들

**목적**: RF-A/B/C 결과를 종합해 권장 모델(들)을 확정하고 spec §10 번들을 만든다.

**입력**: Ticket 3(RF-B 최종 산출물+판정), Ticket 4(RF-C 최종 산출물+RF-A 대비
다양성).

**작업 범위**:
- RF-B vs RF-C 상호 비교(disagreement/OOF 확률 상관) 계산 — Ticket 4에서 이연된
  "RF-B 대비" 항목을 여기서 완성한다.
- RF-A/B/C 전체 비교표(`model_comparison.csv`) 및 confusion matrix(raw+
  normalized, 3개 모델).
- 최종 권장 모델·판정 근거(`decision_summary.json`) — spec §5.6/§6.2 기준을
  기계적으로 적용하되, **규칙과 근거를 투명하게 기록**한다(숨기지 않는다).
  구성 예:
  - `default_candidate`: RF-A(항상 유효한 기본 전달 후보로 유지)
  - `rf_b_status` / `rf_b_reason`: 채택/보류/기각 + 근거
  - `rf_c_status` / `rf_c_reason`: 채택/보류/기각 + 근거
  - `auxiliary_candidates`: 단독 점수는 낮지만 stacking 다양성이 확인된 모델
    (근거 포함, 강제 채택 아님)
  - 최종 판정은 Gate 6에서 사람이 검토할 수 있도록 근거 전체를 남긴다(자동
    확정이 아니라 "권장"으로 표기).
- spec §10 구조로 `share/random_forest_stacking_<YYYYMMDD>/` 번들 조립(README/
  BUNDLE_MANIFEST.json/SHA256SUMS.txt/class_order.json/configs/scripts/lib/
  artifacts/reference_results).
- 절대경로·개인정보 스크럽 → SHA256SUMS 마지막 생성 → ZIP 생성 → ZIP 재오픈해
  manifest·파일 수·checksum 재검증.

**산출물(정본 파일)**: `share/random_forest_stacking_<YYYYMMDD>/` 전체 + 동일
이름의 `.zip`.

**테스트**: 번들 내 모든 파일이 `SHA256SUMS.txt`와 일치, ZIP 재오픈 후
`BUNDLE_MANIFEST.json`의 `model_variants`/경로 필드가 실제 파일과 일치, README의
checksum 검증 명령이 실제로 통과.

**Acceptance criteria**: spec §10 필수 내용·제외 항목 전부 충족,
`BUNDLE_MANIFEST.json` RF 전용 필드 전부 채워짐, Drive 업로드 없음,
`decision_summary.json`에 규칙·근거가 전부 기록됨.

**Stop 조건**(수정): RF-B와 RF-C가 모두 RF-A를 이기지 못해도 번들 생성을
중단하지 않는다 — RF-A를 기본 전달 후보로 유지하고, RF-B/RF-C는 보류 또는 기각
상태로 함께 기록한다. stacking 다양성이 확인된 모델은 단독 점수가 낮더라도
`auxiliary_candidates`로 표시할 수 있다. **유효한 모델이 하나도 없거나(예: 전
모델이 artifact validator 실패) 산출물 검증 자체가 실패한 경우에만 Stop한다.**

**선행 티켓**: Ticket 3 + Ticket 4

---

## Gate 6 — 독립 acceptance review

**목적**: 새 세션에서 `/code-review`로 producer가 아닌 verifier 역할을 수행한다.
spec 충족·누수·OOF 완전성·확률/submission 스키마·재현성·checksum을 독립
재검증한다.

**입력**: `docs/specs/random_forest_stacking_model.md`, 이 tickets 문서,
Ticket 5의 번들.

**작업 범위**: spec §9.5 전부 독립 재검증(가능하면 저장된 metrics를 OOF 파일에서
직접 재계산해 비교), fold-local chi2/Group5 누수 재검증, checksum 재검증, clean
environment에서 README 재현 절차가 실제로 재현 가능한지 확인,
`decision_summary.json`의 근거가 실제 데이터와 일치하는지 검토.

**산출물**: 대화 보고(Blocking/Important/Minor/검증 완료/재확인 필요/전달 가능
여부 최종 판정). **파일 수정·commit은 하지 않는다.**

**선행 티켓**: Ticket 5
