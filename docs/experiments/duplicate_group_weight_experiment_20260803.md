# 중복 그룹 가중치 비교 실험 (2026-08-03)

담당: 권병학 · 관련 PR: 이 문서를 포함한 PR(본 실험) · 관련: PR #17(`feat/#17`, 병합됨)

## 1. 실험 목적

동일하거나 반복되는 변이 프로필이 학습에 과도한 영향을 주는지 확인하고, 중복 그룹
가중치가 Group5 OOF Macro F1 및 일반화 안정성에 실제로 도움이 되는지 검증한다.

이 PR의 목적은 **중복 가중치를 운영 기본값으로 채택하는 것이 아니다.** 통제 실험을
재현 가능한 팀 자산으로 남기고, "클래스 균형 가중치만 사용한 방식이 가장 좋았으며
중복 가중치는 기각됐다"는 결론을 공유하는 것이다. `scripts/train_gbdt.py`의 기본
학습 동작은 이 실험으로 바뀌지 않는다.

## 2. 데이터와 피처 조건

권병학 개인 파이프라인 기준(팀 표준 `scripts/train_gbdt.py`의 피처 블록과는 다르다):

- 3단계 유전자 인코딩 4,384열 + sample-level stateless 17열 + fold-safe
  ratio-transform 10열 = 모델 입력 4,411열
- 모델: XGBoost, 하이퍼파라미터 고정(튜닝 없음)
- seed 42, 외부 데이터 미사용
- test 데이터는 이 실험에서 로드하지 않는다(제출 예측 생성 없음)

민재님의 EXP_015/f4r 결과는 참고 자료일 뿐, 이 실험의 기준선으로 재사용하지 않았다
— 피처·fold·산출물 조건이 완전히 다르기 때문이다. 이 실험은 A(무가중)/B(클래스균형)
자체를 직접 만들어 기준선으로 썼다.

## 3. Group5 정의

Group CV의 그룹은 **원본 `train.csv`의 4,384개 변이 문자열 열**로 만든 canonical
`profile_hash`를 쓴다 — enc3 인코딩값이나 최종 4,411 피처로 다시 해시하지 않는다.

```python
from cancer_hack.validation import make_profile_hash, make_profile_group_kfold
# gene_columns = 원본 train.csv에서 ID/SUBCLASS를 뺀 4,384개 열
```

`make_profile_group_kfold`(`StratifiedGroupKFold` + `groups=make_profile_hash(...)`)를
그대로 재사용했다 — 새로 구현하지 않았다. 0-based fold, 고유 그룹 5,636개, 중복
그룹(크기>1) 451개, 최대 그룹 크기 94(all-WT). fold를 가로지른 그룹 수 0건(생성
스크립트 + 독립 재검증 양쪽에서 확인).

## 4. A~E 수식

fold의 train 부분만 $N$행, $K$개 클래스가 있다고 하자.

- **A_none**: `sample_weight = None`
- **B_balanced**: `resolve_sample_weight("balanced", y, profile_hash)` — `cancer_hack.models_gbdt` canonical 재사용
- **C_balanced_profile**: `resolve_sample_weight("balanced+group", y, profile_hash)` — canonical 재사용, PR #17 `f4rw`와 동일 정의(profile_hash 단독 그룹핑)
- **D_balanced_same_label_naive**: `balanced_sample_weight(y) * same_label_duplicate_weight(y, profile_hash)` — `(SUBCLASS, profile_hash)` 조인, Notion 원안, 단순곱
- **E_same_label_rebalanced**: $weight_i = \dfrac{N/K}{G_c \cdot n_{cg}}$ — 클래스별 총weight를 $N/K$로 보존하는 교정식($G_c$ = 클래스 $c$ 안의 고유 그룹 수, $n_{cg}$ = 그 행이 속한 그룹 크기)

`same_label_duplicate_weight`/`same_label_rebalanced_weight`만 이번에 새로 구현했다
(`src/cancer_hack/sample_weights.py`). 나머지는 기존 `models_gbdt.py`를 재사용한다.

## 5. 전체 결과표

| Scheme | OOF Macro F1 | Δ vs A | Δ vs B | 판정 |
|---|---|---|---|---|
| A_none | 0.3971 | — | -0.0099 | 무가중 기준선 |
| **B_balanced** | **0.4070** | +0.0099 | — | **채택** |
| C_balanced_profile | 0.4053 | +0.0082 | -0.0017 | 기각 |
| D_balanced_same_label_naive | 0.4033 | +0.0062 | -0.0037 | 기각 |
| E_same_label_rebalanced | 0.4042 | +0.0071 | -0.0028 | 기각 |

C/D/E가 A보다는 개선됐지만, 진짜 비교 기준인 B보다는 전부 낮다. 사전 판정 기준
(B 대비 0.005 이상 개선 & 5개 fold 중 3개 이상 개선)을 셋 다 충족하지 못해
**기각**했다.

## 6. fold별 요약

| fold | A | B | C | D | E |
|---|---|---|---|---|---|
| 0 | 0.3816 | 0.4093 | 0.3978 | 0.4033 | 0.4077 |
| 1 | 0.4006 | 0.3898 | 0.3925 | 0.3816 | 0.3843 |
| 2 | 0.3860 | 0.4094 | 0.4030 | 0.4108 | 0.3996 |
| 3 | 0.4066 | 0.4230 | 0.4303 | 0.4224 | 0.4254 |
| 4 | 0.3931 | 0.3878 | 0.3821 | 0.3838 | 0.3844 |
| 평균±표준편차 | 0.394±0.010 | 0.404±0.015 | 0.401±0.018 | 0.400±0.018 | 0.400±0.017 |

B 대비 개선된 fold 수: C=2/5, D=1/5, E=1/5 (전부 3/5 미달).

## 7. singleton/duplicate 결과

profile_hash 그룹 크기가 1인 행(singleton, 5,185행/83.6%)과 2 이상인 행(duplicate,
1,016행/16.4%)을 나눠 봤다.

| scheme | singleton F1 | duplicate F1 | Δsingleton vs A |
|---|---|---|---|
| A | 0.3890 | 0.1293 | — |
| B | 0.4060 | 0.1210 | +0.0170 |
| C | 0.4075 | 0.1195 | +0.0185 |
| D | 0.4031 | 0.1208 | +0.0141 |
| E | 0.4048 | 0.1145 | +0.0159 |

singleton에서는 전부 A보다 개선(부작용 없음). duplicate에서는 어느 스킴도 A보다
나아지지 못했다 — 가중치가 정작 겨냥한 "중복 행" 자체의 성능은 못 살렸다.

## 8. weight 분포와 Effective Sample Size (fold-평균)

| scheme | norm_std | norm_min | norm_max | ESS | ESS비율 | 클래스별총weight std |
|---|---|---|---|---|---|---|
| B | 0.702 | 0.30 | 6.30 | 3323 | 67.0% | 0.0000 |
| C | 0.791 | 0.03 | 6.80 | 3053 | 61.5% | 26.62 |
| D | 0.720 | 0.08 | 6.45 | 3267 | 65.9% | 11.70 |
| E | 0.735 | 0.08 | 6.30 | 3222 | 65.0% | 0.0000 |

## 9. Notion 단순곱의 클래스 총가중치 불균형

**D(Notion 원안, 단순곱)의 클래스별 총weight 불균형 우려는 실측으로 확인됐다**
(std 11.70, class 총weight가 138~195 사이로 벌어짐). B와 E는 설계·정의상 정확히
0(균등). 다만 **D와 E의 OOF 성능 차이는 -0.0009로, 이 데이터·모델 조합에서
뚜렷한 차이라고 주장할 근거는 아니다** — 불균형이 실측됐다는 사실과, 그 불균형이
이번 실험에서 성능 차이로 이어졌는지는 별개의 결론이다.

## 10. 채택·기각 판정

**B_balanced 채택. C/D/E 전부 기각(개선폭 부족).** 결론을 과장하지 않는다:

- C/D/E가 A보다는 개선
- 하지만 진짜 비교 기준 B보다 모두 낮음(-0.0017 ~ -0.0037)
- D의 클래스별 총 weight 불균형은 확인
- D와 E 성능 차이 0.0009는 뚜렷한 차이라고 주장하지 않음
- 이번 피처·모델·fold 조건에서는 중복 가중치 채택 근거 부족
- 클래스 균형 가중치만 사용한 B가 최선

## 11. 재현 방법

원본 데이터·feature parquet·OOF는 이 저장소에 포함돼 있지 않다(대회 규정·저장소
용량 정책). 재현하려면:

1. `configs/duplicate_group_weight_experiment.yaml`의 `paths.raw_train_csv`,
   `paths.train_feature_parquet`을 로컬 경로로 채운다.
2. `cancer_hack.validation.make_profile_hash` + `make_profile_group_kfold`로
   Group5 fold를 만든다(원본 4,384개 변이 문자열 열 기준, 0-based, 그룹 무교차
   검증 필수).
3. `cancer_hack.sample_weights.resolve_experiment_weight(scheme, y_train_fold,
   profile_hash_train_fold)`로 fold-local 가중치를 계산한다 — **fold의 train
   부분만** 넘긴다.
4. `configs/duplicate_group_weight_experiment.yaml`의 XGBoost 하이퍼파라미터로
   5-fold × 5-scheme(A~E) 학습, OOF 수집.
5. 전체 OOF Macro F1, fold별 점수, singleton/duplicate 점수, 클래스별 F1을 비교.

코드 골격은 `notebooks/08_duplicate_group_weight_experiment.ipynb`(개인 workspace
실행 기록의 사본, 실행 결과 보존)를 참고한다 — 이 사본은 개인 폴더 구조를 전제로
한 경로 코드를 포함하므로 그대로 재실행되지는 않는다(노트북 상단 안내 참고).

## 12. 알려진 한계

- Group5 CV 자체의 순수 효과(무가중 SKF vs 무가중 Group5)는 이 실험만으로는
  분리되지 않는다 — A~E 다섯 스킴이 전부 Group5를 공통으로 쓰기 때문이다.
- 노트북 실행은 develop HEAD `26f8cc5`(PR #17 머지 직전) 기준이며, 권병학 개인
  파이프라인은 `scripts/train_gbdt.py`가 아니라 별도 XGBoost 코드를 쓴다 — PR
  #17이 이후 머지됐지만 이 실행의 숫자·판정에는 영향이 없다.
- `power`(그룹 크기 감쇠 강도) 파라미터 탐색은 이번 범위 밖이다.
- 이 실험은 XGBoost 한 가지 백엔드·한 가지 하이퍼파라미터 설정에서만 확인했다 —
  LightGBM/CatBoost나 다른 피처 블록(팀 표준 도메인 539 등)에서도 같은 결론이
  나오는지는 확인하지 않았다.

## 13. DACON 미제출 확인

**실제 DACON 제출을 하지 않았다.** 이 실험은 test 데이터 자체를 로드하지 않았고
(config `test_predictions.generate: false`), submission CSV도 생성하지 않았다.

## 14. 외부 데이터 미사용 확인

**외부 데이터를 사용하지 않았다.** 전부 대회가 제공한 `train.csv`의 원본 변이
문자열 열에서 유도한 피처·그룹핑만 사용했다.

## 15. 대용량 산출물

전체 OOF CSV, metrics JSON, 재현 패키지(4,411열 feature parquet 등)는 이
저장소에 커밋하지 않았다. 필요 시 Google Drive로 별도 공유할 예정이다.

## 16. PR #17과의 관계

PR #17(`feat/중복_처리_전략`, 병합됨)에는 profile_hash 단독 중복 감쇠 구현
(`group_size_inverse_weight`, `f4rw`/`f4rws` config)이 이미 존재하지만, 독립 실행
아티팩트는 확인되지 않았다(팀 공유 Drive를 조사했을 때 `f4rw`/`f4rws` 실행 로그가
전혀 없었다 — 코드는 있지만 검증된 적 없는 상태였다). 본 실험은 동일 조건의 A~E
통제 실험과 same-label/교정식(D/E)을 추가해 검증한 결과를 공유한다. 기본 학습에는
중복 가중치를 연결하지 않으며, 최종 판정은 class-balanced only(B) 채택이다.
