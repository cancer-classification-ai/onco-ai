# XGBoost·CatBoost·RandomForest 3종 앙상블 (2026-08-03)

담당: 권민재 · 관련 PR: 이 문서를 포함한 PR(본 실험) · 관련 브랜치: `feat/앙상블`

## 1. 실험 목적

세 모델(XGBoost·CatBoost·RandomForest)을 캐시된 피처 11블록 전부(config `f11`,
3,742~3,750열)로 각각 학습하고, 균등 평균·학습 가중치·로짓 보정 세 가지 방식으로
앙상블했을 때 단일 최고 모델보다 실제로 나은지 확인한다.

이 PR의 목적은 **앙상블을 운영 기본값으로 채택하는 것이 아니다.** 통제 실험을
재현 가능한 팀 자산으로 남기고, "앙상블이 이득인가"에 대한 정직한 답을 공유하는
것이다. `scripts/train_gbdt.py`의 기본 학습 동작은 이 실험으로 바뀌지 않는다.
DACON 제출은 사람이 직접 한다 — 이 실험은 로컬 파일만 만들었다.

## 2. 이번 PR에서 같이 고친 것

앙상블을 돌리기 전에 재료 세 가지가 빠져 있었다.

1. **RandomForest가 저장소에 없었다.** `models_gbdt._REGISTRY`가 xgb·lgbm·catboost
   뿐이라 `RFModel(BaseGBDT)`을 새로 구현했다.
2. **앙상블 도구가 develop에 없었다.** `ensemble.py`·`calibration.py`가 0바이트
   스텁이라, push 안 된 로컬 커밋(`70995d9`)을 체리픽했다.
3. **모든 블록을 켠 config가 없었다.** 25개 config 중 최대가 4블록·2,055열(`f5`)
   이라 캐시가 있는 11블록 전부를 켠 `f11`을 새로 만들었다.

여기에 더해 **기존 코드의 버그를 하나 고쳤다**: `lsvd`(SVD)와 `lnmf`(NMF)를 같은
config에 넣으면 잠재 방식이 config 전체에 하나로만 정해져 둘 다 NMF로 돌았다 —
소스 parquet·시드·성분 수가 같아서 결과가 바이트 단위로 같은 64열 두 벌이 됐고
SVD는 조용히 사라졌다. `_resolve_latent_methods`로 블록마다 방식을 풀도록 고쳤다.

## 3. 데이터와 피처 조건

- config `f11`: `domain, rollup16, enc3, gec, sigtok, exacttok, comut, lsvd, lnmf,
  gmod, csig` — 11블록, 열 3,742~3,750(comut이 fold마다 선택 폭이 달라 흔들린다)
- 모델별 하이퍼파라미터는 xgb·catboost는 팀 기존값(`MODEL_PARAMS`) 그대로, RF만
  새로 정했다(`n_estimators=500, max_features='sqrt', min_samples_leaf=1,
  bootstrap=True, criterion='gini'`) — 근거는 `models_gbdt.RFModel.default_params()`
  docstring과 실험용 fold 0 실측(§8 참고).
- fold: 팀 표준 `train_folds.parquet`의 `fold_skf5`/`fold_group5`, seed 42
- test 데이터는 예측만 만들었고(제출 후보 csv), **DACON 업로드는 하지 않았다**

## 4. 단일 모델 결과 (skf가 주 지표, sgkf는 기록용)

| 모델 · config | skf OOF Macro F1 | sgkf OOF Macro F1 | 학습 시간(skf, 5-fold) |
|---|---|---|---|
| xgb f4r (기준선) | 0.4563 | 0.4786 | — |
| catboost f4r (기준선) | 0.4647 | 0.4728 | — |
| **rf f4r (통제군, 이번 실험)** | 0.4415 | — | 9초 |
| xgb f11 | 0.4581 | 0.4769 | 216초 |
| catboost f11 | **0.4742** | 0.4866 | 200초 |
| rf f11 | 0.4457 | 0.4714 | 64초 |

f11(넓은 입력)의 효과는 모델마다 다르다. catboost는 f4r 대비 +0.0095로 확실히
올랐고, xgb는 +0.0018로 잡음 범위(아래 §7 기준 0.005 미만)다. RF는 f11에서
+0.0042로 역시 잡음 범위 — **넓은 입력 자체가 RF에는 별 도움이 안 됐다.**

## 5. 앙상블 결과 (calibrate_ensemble.py, skf 기준)

세 가지 산출물이 매 실행마다 같이 나온다: 균등 평균(uniform), fold 교차적합
학습 가중치(raw blend), 학습 가중치+클래스별 로짓 보정(calibrated).

| 조합 | uniform | raw blend | calibrated | 최고 멤버 대비 Δ | test TVD (최고 멤버 solo 대비) |
|---|---|---|---|---|---|
| **xgb+cat+rf, f11 (본 실험)** | 0.4662 | 0.4744 | **0.4861** | **+0.0119** | 0.304→0.293 (개선) |
| cat+rf, f11 | 0.4678 | 0.4734 | 0.4852 | +0.0110 | 0.304→0.311 (악화) |
| xgb+cat, f11 (과거 실패 조합 재현) | 0.4660 | 0.4747 | 0.4852 | +0.0110 | 0.295→0.294 (거의 그대로) |
| xgb+cat+rf, f4r 통제군 | 0.4647 | 0.4624 | 0.4771 | +0.0124 | 0.298→0.311 (악화) |
| xgb+cat+rf, f11, **sgkf**(기록용) | 0.5053 | 0.5050 | 0.5102 | +0.0236 | 0.281→0.304 (악화) |

균등 평균은 매번 최고 멤버보다 **낮다** — RF의 낮은 확신도(§6)가 평균을 끌어
내린다. 학습 가중치만으로는 이득이 잡음 수준이고, 실질적인 개선은 전부 로짓
보정 단계에서 나온다.

sgkf(기록용) 델타(+0.0236)가 skf 델타(+0.0119)의 두 배 가까이 크다 — 과거
문서가 경고한 "쌍둥이 행 과대평가" 패턴이 이번에도 그대로 재현됐다. **판단은
skf로만 한다.**

## 6. 분리 실험 — 이득이 앙상블에서 오는가, 보정에서 오는가

CatBoost f11 단독에 **앙상블 없이** 로짓 보정만 적용해 봤다(`--oof`에 catboost
하나만 넣음).

| | skf Macro F1 |
|---|---|
| catboost f11 단독 | 0.4742 |
| catboost f11 + 로짓 보정만 (앙상블 없음) | **0.4848** |
| xgb+catboost+rf 3종 앙상블 + 보정 | 0.4861 |

로짓 보정 하나만으로 +0.0106이 나온다 — 3종 앙상블의 전체 이득(+0.0119) 중
**89%가 보정 효과이고, 모델을 세 개로 늘려서 얻는 진짜 앙상블 이득은 +0.0013
(잡음 수준, §7 기준 0.005 미만)이다.** 과거 실험 기록(`04_session_handover`)의
"로짓보정만 skf에서 +0.0127로 실질이었다"는 관찰과 일치하는 패턴이 다시
나타났다.

fold 0 평균 최대확률을 재 보면 xgb 0.575 · catboost 0.418 · **rf 0.265**로,
RF가 확신도가 극단적으로 낮다. `MacroF1Blender`가 학습한 fold별 가중치도
이걸 반영한다 — catboost 가중치가 fold마다 0.72~0.95로 흔들리고(폭 0.23,
"0.2 이상 벌어지면 못 믿는다"는 판정 기준을 살짝 넘는다), RF 가중치는
0.00~0.14로 사실상 무시된다.

## 7. 승격 조건과 판정

사전에 정한 조건(넷 다 만족해야 제출 후보로 승격):

1. skf에서 최고 멤버 대비 +0.005 이상
2. 이득이 `support < 60` 소수 클래스에 몰려 있지 않음
3. 단독행(singleton) macro F1도 같이 오름
4. test 예측 TVD가 최고 멤버 단독보다 나빠지지 않음

**xgb+cat+rf, f11, skf, calibrated** 조합만 넷을 전부 통과했다.

- Δ = +0.0119 (조건 1 통과)
- 이득 상위: TGCT(124행) +0.127, CESC(155행) +0.074, LIHC(158행) +0.049,
  BRCA(786행) +0.047 — `support < 60`(DLBC 38행 등)에서는 오히려 **악화**됐다
  (DLBC −0.051, ACC −0.062). 소수 클래스 착시가 아니다(조건 2 통과)
- 단독행 macro F1: 0.4873 (catboost 단독 0.4800 대비 +0.0073, 조건 3 통과)
- test TVD: 0.304 → 0.293로 개선(조건 4 통과)

나머지 조합(cat+rf, f4r 통제군, xc)은 전부 TVD가 나빠져 조건 4에서 탈락했다.

**다만 §6의 분리 실험이 보여주듯, 이 조합이 통과한 진짜 이유의 대부분은
로짓 보정이지 3-모델 다양성이 아니다.** 이 사실을 숨기지 않고 같이 보고한다.

## 8. RandomForest 하이퍼파라미터 근거

`max_features='sqrt'`가 3,749열 중 대부분(2,382열)이 99% 이상 0인 TF-IDF
희소열인 상황에서 죽은 열만 뽑을 거라 우려했는데, fold 0 실측에서 기각됐다.
난수열을 걷어내고 재면 `sqrt`가 `0.05`, `0.1`보다 **더 낫다** — 후보 열이
늘수록 탐욕 분할기가 표본 몇 개를 완벽히 가르는 희귀 TF-IDF 지시자를 집어서
과적합한다. `n_estimators=500`은 1000과 실측 차이가 거의 없어(0.4480 vs
0.4440) 메모리만 두 배 드는 1000을 기각했다. `class_weight`는 넣지 않았다 —
파이프라인이 이미 balanced sample_weight를 주므로 이중 가중이 되어 실측
macro F1을 0.4480→0.3917로 떨어뜨렸다.

## 9. 알려진 한계

- 이 실험은 seed 42 하나에서만 확인했다. 세 모델의 fold별 가중치가 흔들리는
  걸 봤을 때 다른 seed에서도 같은 조합이 이길지는 확인하지 않았다.
- `f11`은 gtype·parsed19·burden8·aa9·ptok을 뺀 11블록이다 — 캐시가 없어서다.
  이 블록들을 채운 뒤의 앙상블은 이번 범위 밖이다.
- RF는 GPU 백엔드가 없어 CPU로만 돌았다. xgb·catboost의 GPU 학습과 시간
  비교는 공정하지 않다(RF가 압도적으로 빠른 건 GPU 없이도 그렇다는 뜻이다).
- **LB로 검증하지 않았다.** 과거 XGB+CatBoost f4r 균등평균은 CV +0.0151인데
  LB는 −0.0136으로 떨어졌다(팀 기록). 이번 조합이 로컬 게이트를 통과했다고
  LB에서도 이긴다는 보장은 없다 — 오히려 팀의 과거 제출 4건 전부가 CV 상승
  → LB 하락 패턴이었다는 걸 감안하면 조심스럽게 봐야 한다.
- 스태킹(메타 모델)은 팀원 데이터를 기다리는 중이라 이번 범위 밖이다. 이번에
  만든 `artifacts/oof/oof_*_ens11_*.csv`(`ID` + `p_*×26` + `y_true`)가 그
  입력 스키마를 그대로 만족한다.

## 10. 재현 방법

```powershell
# 세 모델 학습 (skf, 주 지표)
.venv/Scripts/python.exe scripts/train_gbdt.py --model xgb      --configs f11 --cv skf --tag ens11
.venv/Scripts/python.exe scripts/train_gbdt.py --model catboost --configs f11 --cv skf --tag ens11
.venv/Scripts/python.exe scripts/train_gbdt.py --model rf       --configs f11 --cv skf --tag ens11

# 앙상블
.venv/Scripts/python.exe scripts/calibrate_ensemble.py `
  --oof  artifacts/oof/oof_xgb_ens11_f11_skf5_..._s42.csv `
         artifacts/oof/oof_catboost_ens11_f11_skf5_..._s42.csv `
         artifacts/oof/oof_rf_ens11_f11_skf5_..._s42.csv `
  --test artifacts/test_predictions/test_xgb_ens11_f11_skf5_..._s42.csv `
         artifacts/test_predictions/test_catboost_ens11_f11_skf5_..._s42.csv `
         artifacts/test_predictions/test_rf_ens11_f11_skf5_..._s42.csv `
  --fold-column fold_skf5 --tag ens11_xcr_skf5 --submission
```

`--verify` 성격의 재현 검증은 pytest로 대신했다 — `tests/test_models_rf.py`,
`tests/test_model_params_rf.py`, `tests/test_fold_fit_only.py`의 잠재 방식·f11
관련 테스트 19개가 이번 PR에 포함돼 있다. 전체 스위트 667개 통과(체리픽 이전
639개 + 이번 PR 신규 28개).

## 11. 대용량 산출물

OOF·test 예측·제출 후보 csv, `artifacts/logs/*.json`은 이 저장소에 커밋하지
않았다(팀 관례상 `artifacts/`는 `.gitkeep`만 추적). 제출 후보 파일 경로:

- `artifacts/submissions/submission_ens11_xcr_skf5.csv` (calibrated, 승격 조건 통과)
- `artifacts/submissions/submission_ens11_catboost_solo_cal_skf5.csv` (§6 분리
  실험 — catboost 단독 + 보정만, 앙상블보다 단순하고 로컬 점수는 거의 동일)

두 후보 다 **DACON에 제출하지 않았다.** 어느 쪽을 시도할지, 시도할지 말지는
사람이 판단한다. 팀 하루 제출 4회 제한을 감안해 신중히 고르길 권한다.

## 12. DACON 미제출 확인

**실제 DACON 제출을 하지 않았다.** `calibrate_ensemble.py`·`train_gbdt.py`
모두 로컬 csv만 만들고 업로드 API를 호출하지 않는다.

## 13. 외부 데이터 미사용 확인

**외부 데이터를 사용하지 않았다.** 전부 팀 표준 파이프라인의 캐시된 피처
parquet에서 유도했다. chi2 선택·BurdenBinner·balanced 가중치는 fold의 train
부분에서만 fit했고 test는 예측에만 썼다.
