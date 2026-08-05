# LightGBM text + NMF branch — Colab 실행

## 고정 조건

- CV: `data/process/train_folds.parquet`의 `fold_group5`, 5-fold (`--cv sgkf`)
- class order: 현재 26개 클래스 알파벳순
- seed: 42
- 비교 점수: Pair rule 적용 전 OOF Macro F1
- 기준선: 권민재 팀원 v002 + Pair rule, LB 0.4818
- exact token: 낮은 coverage의 대조군이므로 이 branch에는 넣지 않음
- Pair rule: 이 실험에서는 호출하지 않고 최종 제출에만 적용

## 1. Colab 준비

이 변경을 `feat/rwr-stacking` 원격 브랜치에 올린 뒤 Colab에서 해당 브랜치를 받는다.

```bash
git checkout feat/rwr-stacking
git pull
python scripts/colab_setup.py
```

`colab_setup.py`가 패키지 버전을 바꿨다면 런타임을 재시작하고 다시 실행한다. 다음
캐시가 `data/process/`에 있어야 한다.

```text
train_folds.parquet
train/test_domain_features.parquet
train/test_sample_mutation_features_rollup.parquet
train/test_signature_mutation_tokens.parquet
train/test_parsed_mutation_tokens.parquet
train/test_mutation_encoded.parquet
```

## 2. 기본 실험

LightGBM 하이퍼파라미터는 기본값을 그대로 쓰며 튜닝하지 않는다. `--no-submission`은
중간 실험에서 제출 CSV만 막고 OOF와 test 확률은 모두 저장한다.

```bash
python scripts/train_gbdt.py \
  --model lgbm \
  --configs lgbm_text_nmf \
  --cv sgkf \
  --n-splits 5 \
  --seed 42 \
  --tag stack_lgbm_v1 \
  --no-submission
```

기본값은 NMF 64, signature TF-IDF top-K 1000, parsed-token top-K 1000이다.

## 3. 단일 블록 제거 ablation

```bash
python scripts/train_gbdt.py \
  --model lgbm \
  --configs lgbm_text_nmf_no_sigtok,lgbm_text_nmf_no_ptok,lgbm_text_nmf_no_lnmf \
  --cv sgkf \
  --n-splits 5 \
  --seed 42 \
  --tag stack_lgbm_v1 \
  --no-submission
```

각 config는 기본 구성에서 이름에 표시된 블록 하나만 제거한다.

## 4. NMF 차원

64는 기본 실험 결과를 재사용한다. 추가 실행은 32와 128뿐이다.

```bash
python scripts/train_gbdt.py \
  --model lgbm --configs lgbm_text_nmf --cv sgkf --n-splits 5 \
  --seed 42 --tag stack_lgbm_v1 --latent-components 32 --no-submission

python scripts/train_gbdt.py \
  --model lgbm --configs lgbm_text_nmf --cv sgkf --n-splits 5 \
  --seed 42 --tag stack_lgbm_v1 --latent-components 128 --no-submission
```

## 5. Signature TF-IDF top-K

1000은 기본 실험 결과를 재사용한다. `--tfidf-topk`만 바꾸므로 parsed-token top-K는
1000으로 고정된다.

```bash
python scripts/train_gbdt.py \
  --model lgbm --configs lgbm_text_nmf --cv sgkf --n-splits 5 \
  --seed 42 --tag stack_lgbm_v1 --tfidf-topk 500 --no-submission

python scripts/train_gbdt.py \
  --model lgbm --configs lgbm_text_nmf --cv sgkf --n-splits 5 \
  --seed 42 --tag stack_lgbm_v1 --tfidf-topk 2000 --no-submission
```

`--sparse-topk`는 이전 실험 재현을 위해 두 sparse family를 함께 바꾸는 기존 동작을
유지한다. 이번 비교에는 사용하지 않는다.

## 6. 판정

산출물은 다음 위치에 생긴다.

```text
artifacts/oof/oof_lgbm_stack_lgbm_v1_*.csv
artifacts/test_predictions/test_lgbm_stack_lgbm_v1_*.csv
artifacts/logs/lgbm_stack_lgbm_v1_*.json
```

각 후보에 대해 다음 두 값을 함께 기록한다.

1. 후보 단일 raw OOF Macro F1
2. 동일 `fold_group5`에서 만든 v002 raw OOF와 후보 OOF를 확률 50:50 평균한 Macro F1 및
   v002 단독 대비 증분

내부 예측 스키마(`ID`, `p_<CLASS>`, `y_true`)의 v002 OOF/test가 있으면 기존 검증 도구로
ID, class order, 정답을 대조하면서 50:50 결과를 만들 수 있다.

```bash
python scripts/calibrate_ensemble.py \
  --oof V002_OOF.csv CANDIDATE_OOF.csv \
  --test V002_TEST.csv CANDIDATE_TEST.csv \
  --folds data/process/train_folds.parquet \
  --fold-column fold_group5 \
  --weights 0.5 0.5 \
  --tag stack_lgbm_v1_CANDIDATE_5050
```

판정에는 출력 중 `source OOF`와 `uniform blend OOF`만 사용한다. 로짓 보정 점수와
Pair rule 결과는 후보 선택에 사용하지 않는다. v002 파일이 팀 공유 형식
(`sample_id`, `prob_class_N`)뿐이면 먼저 내부 class-name 스키마로 변환하거나 두 후보를
같은 팀 공유 형식으로 맞춘 뒤 평균해야 한다.
