# exact-match 짝 라벨 규칙

제출 csv 의 214행을 짝 코호트 라벨로 바꾸는 후처리다. 재학습이 필요 없고, 붙이는 데 1초쯤
걸리며, **Public LB 를 0.3896 → 0.4725 로 올렸다.**

구현은 `src/cancer_hack/pair_rule.py`, 진입점은 `scripts/apply_pair_rule.py` 와
`scripts/make_submission.py --pair-rule` 둘이다.

---

## 쓰는 법

이미 만들어 둔 제출 csv 에 얹기:

```powershell
.\.venv\Scripts\python.exe scripts\apply_pair_rule.py `
  --submission artifacts\submissions\submission_x.csv `
  --out artifacts\submissions\submission_x_pairrule_m3.csv
```

후보가 여럿이면 한 번에 준다. train 스캔을 한 번만 하므로 12개 기준 15.8초가 1.3초가 되고,
무엇보다 `--min-mut` 이 후보끼리 어긋날 일이 없다.

```powershell
.\.venv\Scripts\python.exe scripts\apply_pair_rule.py `
  --submission artifacts\submissions\cand1.csv artifacts\submissions\cand2.csv `
  --out-dir artifacts\submissions
```

확률 파일에서 제출을 새로 만드는 중이라면 그 자리에서 얹는 쪽이 낫다:

```powershell
.\.venv\Scripts\python.exe scripts\make_submission.py `
  --predictions artifacts\test_predictions\test_x.csv --pair-rule
```

두 진입점 모두 **규칙의 전제가 깨지면 파일을 쓰지 않고 멈춘다**. 아래 「전제」 참고.

---

## 규칙

test 행의 유전자 프로파일 4,384열이 train 의 **유일한** 행과 바이트 단위로 완전히 같고,
그 train 라벨이 `KIPAN`·`KIRC`·`GBMLGG`·`LGG` 중 하나면, 예측을 **짝 코호트 라벨**로 바꾼다.

```
KIPAN ↔ KIRC        GBMLGG ↔ LGG
```

매칭된 라벨을 그대로 쓰는 게 아니라 **반대쪽**을 쓴다는 게 핵심이다. 평범한 1-NN 과 방향이
반대다.

### 왜 반대인가

TCGA 에서 상위 코호트는 하위 코호트의 합집합이다.

```
KIPAN = KICH ∪ KIRC ∪ KIRP        GBMLGG = GBM ∪ LGG
```

같은 환자가 상위·하위 두 라벨로 두 번 들어가 있다. train 안에서 프로파일이 같은 422 묶음이
**예외 없이** 라벨이 갈리고 **예외 없이** 코호트 짝이다(같은 라벨 묶음 0개). 그 짝 중 한쪽만
train 에 있는 "고아" 프로파일의 반대편이 test 로 간 것이다.

결정적인 근거는 수가 맞는다는 점이다.

| 라벨 | train 안에 짝 있음 | 짝이 train 밖(고아) | test 완전일치 |
|---|---|---|---|
| KIRC | 254 | 57 | **57** |
| LGG | 168 | 50 | **50** |
| KIPAN | 254 | 233 | 59 |
| GBMLGG | 168 | 280 | 48 |

train 의 짝 없는 KIRC 57개는 전부 그 KIPAN 사본이 test 에 있다. LGG 50개도 같다.
고아 수와 매칭 수가 우연히 맞을 확률은 사실상 0이다.

### 왜 `--min-mut` 기본값이 3인가

프로파일이 우연히 일치할 확률은 변이 수가 적을수록 크다. 매칭된 test 행을 변이 수로 갈라
train 라벨 구성을 보면 이렇다.

| 변이 수 | 매칭 행 | 짝 4종 | 짝 4종이 아닌 것 |
|---|---|---|---|
| 1 | 5 | 4 | **1 (LAML)** |
| 2 | 2 | 2 | 0 |
| 3~5 | 25 | 25 | 0 |
| 6~10 | 84 | 84 | 0 |
| 11+ | 105 | 105 | 0 |

우연 일치가 실재한다는 증거가 변이 1개 구간에만 있다. 2개부터는 209행 전부가 짝 4종이다.
여유를 둬서 3 으로 잡았고, 그러면 **214행**이 걸린다. `--min-mut 6` 으로 좁히면 189행이다.

---

## 결과

```
v002_seed42_f16_group5              CV 0.5165 · LB 0.3896    규칙 없음
ens16_cbopt10_seed3 + 짝 규칙       CV 0.5210 · LB 0.4725    규칙 있음
```

**+0.0829.** 사전 추정치 +0.05~0.08 범위 안이다.

두 점 사이에서 축이 셋 움직였다(짝 규칙 · CatBoost 튜닝 · 3-seed 평균). 그래도 출처는 사실상
갈린다 — 나머지 두 축의 CV 기여를 합치면 +0.0045 뿐이고, 이 대회에서 그 크기의 CV 델타가
LB 를 0.08 옮긴 적이 없다. 앞선 여덟 점에서는 오히려 CV 가 오르고 LB 가 떨어지는 쪽이 잦았다.

추정식도 남겨 둔다. 예측 총량이 짝 안에서 보존되므로(`P_after ≈ P_now`) 모델이 지금 얼마나
잘하는지가 거의 상쇄된다.

```
델타 ≈ (1/26) · Σ_c  2·A_c / (P_c + T_c)        A_c = c 로 바뀌어 들어오는 행 수
```

| 클래스 | A | P + T | 기여 |
|---|---|---|---|
| KIRC | 59 | 82 + 137 | +0.539 |
| LGG | 48 | 65 + 94 | +0.604 |
| GBMLGG | 50 | 113 + 189 | +0.331 |
| KIPAN | 57 | 248 + 211 | +0.248 |
| | | **/26** | **+0.066** |

**양방향으로 간다.** 한 방향만 바꾸면(KIRC→KIPAN·LGG→GBMLGG) 클래스 총량이 무너져
추정 델타가 +0.023 으로 떨어진다.

---

## 규정

규칙은 `train.csv` 에서만 유도되고, 적용에는 test 한 행이면 된다(프로파일을 train 테이블에
조회). test 통계도 test 라벨도 쓰지 않으므로 대회 규정의 판정 기준인 "test 한 행만 따로
넣어도 같은 결과가 나오는가"를 만족한다. 사실상 짝 라벨로 변환하는 1-NN 이다.

**하지 않기로 한 것: test 내부 중복 매칭.** test 두 행의 프로파일이 서로 같으면 구조상 한쪽은
KIPAN, 한쪽은 KIRC 여야 한다. 하지만 이건 다른 test 행을 봐야 정해지므로 행 단위 독립이
깨진다. 규정의 포괄 조항에 걸릴 소지가 있어 쓰지 않는다.

---

## 로컬 CV 로는 잴 수 없다

group CV 는 같은 프로파일을 한 fold 로 묶어 valid 프로파일의 짝을 train 에서 빼 버린다.
**이 상황 자체를 못 만든다.** 예측 분포도 짝 안에서 스왑이라 총량이 보존돼 TVD 로도 안 보인다.

그래서 구현이나 데이터가 조용히 틀어져도 제출 한 번을 태우기 전까지 아무도 모른다.
대신 `PairRule.verify_premises()` 가 매 실행마다 전제 네 가지를 다시 재고, 하나라도 깨지면
두 진입점 모두 파일을 쓰지 않고 멈춘다.

1. 프로파일이 같은데 라벨도 같은 묶음이 0개인가
2. 코호트 짝 묶음이 존재하는가 (실측 422)
3. KIRC·LGG 의 고아 수가 test 매칭 수와 같은가 (57 = 57, 50 = 50)
4. 짝 4종이 아닌 매칭이 0개인가

같은 내용을 `tests/test_pair_rule.py` 가 원본 csv 로 다시 확인한다. 원본이 없는 환경에서는
해당 테스트만 자동으로 건너뛴다.

---

## 남은 것

이 규칙이 고치는 건 test 에서 완전일치가 잡히는 214행뿐이다. OOF 혼동 상위 셋이 전부 코호트
쌍 내부이고(`GBMLGG→LGG` 164, `KIPAN→KIRC` 162, `KIRC→KIPAN` 92), 네 클래스 안에서만
481건이 어긋난다. 나머지는 입력이 같은데 라벨이 다른 구조라 **어떤 모델도 원리적으로
구분할 수 없다.**
