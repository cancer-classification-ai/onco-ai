# Logistic Regression baseline — 2026-08-04

## 목적

GBDT와 동일한 fold-safe feature assembly로 multinomial Logistic Regression 기준선을
만들고, 이후 block forward ablation과 probability ensemble에 재사용한다.

## 공통 조건

- 데이터: train 6,201행, test 2,546행, SUBCLASS 26개
- fold: 사전 계산된 SKF 5-fold, seed 42
- 기준 feature config: `f4r = domain + rollup16 + enc3`
- 차원: 모든 fold 1,055열
- scaler: `StandardScaler(with_mean=False)`, outer-train에서만 fit
- solver: `lbfgs`, L2 multinomial Logistic Regression
- balanced sample weight: outer-train에서만 계산
- `C` 후보: 0.001, 0.003, 0.01, 0.03, 0.1
- `C` 선택: 각 outer-train 내부 15% stratified validation
- outer validation은 scaler, `C` 선택, feature 선택에 사용하지 않음

## 팀 공용 pipeline 결과

실행:

```bash
python scripts/train_linear.py \
  --configs f4r \
  --cv skf \
  --tag pipeline_smoke
```

| 지표 | 값 |
|---|---:|
| OOF Macro F1 | 0.435691 |
| Singleton Macro F1 | 0.442909 |
| OOF Accuracy | 0.424125 |
| fold별 선택 C | 0.01, 0.003, 0.003, 0.001, 0.003 |
| 수렴 fold | 5 / 5 |

실행 환경:

- Python 3.12.11
- NumPy 2.3.4
- SciPy 1.17.1
- scikit-learn 1.8.0

## 다음 단계

`train_linear.py --configs ...`로 `f4r`에 블록 하나씩 추가하되, 같은 outer fold에서
선택한 `C`를 모든 config에 공통 적용한다. 주 판단은 SKF Macro F1, 보조 판단은
singleton Macro F1과 SGKF로 한다.
