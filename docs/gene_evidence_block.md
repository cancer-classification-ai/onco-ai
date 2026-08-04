# Supervised Gene Evidence Encoding

`ebovr`와 `ebbnb`는 4,384-gene binary mutation matrix를 canonical class별
26개 evidence score로 변환하는 모델 독립적인 feature block이다.

| block | 계산 | 출력 |
|---|---|---:|
| `ebovr` | global-prevalence EB shrinkage + OVR log-odds | 26 |
| `ebbnb` | global-prevalence EB shrinkage + Bernoulli likelihood | 26 |

## Leakage contract

두 block은 label을 feature 값 계산에 사용한다.

- Outer-train feature: inner stratified K-fold cross-fitting
- Outer-validation feature: outer-train 전체로 encoder fit 후 transform
- Test feature: outer-train 전체로 encoder fit 후 transform
- `fit(X_outer_train, y).transform(X_outer_train)` 사용 금지

## Model-independent use

```python
from cancer_hack.features_gene_evidence import build_gene_evidence_fold

evidence = build_gene_evidence_fold(
    X_train=X_gene[train_idx],
    y_train=y[train_idx],
    X_valid=X_gene[valid_idx],
    X_test=X_gene_test,
    method="eb_ovr",  # or "eb_bnb"
    classes=class_order,
    strength=20.0,
    prior_weight=0.0,
    inner_n_splits=5,
    random_state=42 + outer_fold,
)

model.fit(evidence.train, y[train_idx])
valid_probability = model.predict_proba(evidence.valid)
test_probability = model.predict_proba(evidence.test)
```

기존 feature와 사용할 때는 같은 outer fold에서 만든 배열끼리 결합한다.

```python
X_train_model = np.hstack([X_base_train, evidence.train])
X_valid_model = np.hstack([X_base_valid, evidence.valid])
X_test_model = np.hstack([X_base_test, evidence.test])
```

## GBDT integration

```bash
python scripts/train_gbdt.py \
  --model xgb \
  --configs f4r_ebovr \
  --cv all \
  --tag gene_evidence

python scripts/train_gbdt.py \
  --model lgbm \
  --configs f4r_ebbnb \
  --cv all \
  --tag gene_evidence
```

관련 옵션:

```text
--evidence-strength 20
--evidence-prior-weight 0
--evidence-inner-splits 5
```

Evidence config는 experimental이므로 `--configs all`과 `full_all`에 포함되지 않는다.

## v018b reference result

| config | SKF Macro F1 | Δ vs f4r | SGKF Macro F1 | Δ vs f4r |
|---|---:|---:|---:|---:|
| `f4r` | 0.4582 | — | 0.4819 | — |
| `f4r + ebovr` | 0.5182 | +0.0600 | 0.4762 | −0.0058 |
| `f4r + ebbnb` | 0.4677 | +0.0094 | 0.4827 | +0.0008 |

현재 상태는 `experimental`이다. 기본 `full_all`이나 제출 모델로 자동 승격하지 않는다.
