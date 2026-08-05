# 재현 가능성 — 무엇이 보장되고 무엇이 안 되나

이 대회는 **여러 사람이 각자 뽑은 OOF 를 한데 모아 블렌딩한다.** 같은 코드·같은 seed 가
같은 숫자를 내지 못하면 각자의 CV 는 멀쩡해 보이는데 합쳐 놓은 결과만 조용히 어긋난다.
심사에서 제출 코드를 다시 돌렸을 때 우리가 낸 점수가 안 나오는 것도 문제다.

그래서 **seed 를 고정하는 것만으로 충분한지 실측했다.** 결론부터 — 충분하지 않다.

---

## 1. 고정돼 있는 난수원

전부 기본값이 박혀 있다. 인자로 받을 수 있게만 해 두면 코드만 받은 사람이 다른 값을 쓰게
되므로, **기본값 자체**가 재현의 조건이다. `tests/test_seeds_are_pinned.py` 가 지킨다.

| 무엇 | 어디 | 값 |
|---|---|---|
| fold 분할 | `cancer_hack.validation.SEED` | 42 |
| 모델 | `train_gbdt --seed` | 42 |
| 잠재 SVD/NMF | `--latent-random-state` | 0 |
| KMeans 모듈 | `--module-random-state` | 0 |
| 그리디 블렌드 | `greedy_blend --random-state` | 0 |
| 그리디 탐색 | `--n-rounds` · `--bag-fraction` · `--bag-rounds` | 30 · 0.6 · 5 |
| DL | `train_dl --seed` → random·numpy·torch·cuda + cuDNN | 42 |

모델 파라미터(`MODEL_PARAMS`·`PARAM_PRESETS`)에는 seed 를 넣지 않는다. fit 경로에서
`args.seed` 로 주입하므로 `--seed` 하나가 전부를 통제한다.

DL 은 `torch.manual_seed` 만으로 부족하다. cuDNN 이 실행마다 알고리즘을 자동 선택하면
같은 seed 로도 커널이 달라진다. 그래서 `seed_everything` 이 다음을 같이 건다.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

DataLoader 셔플도 난수라 `torch.Generator().manual_seed(seed)` 를 넘긴다.

---

## 2. 실측 — 같은 seed·같은 설정으로 두 번 돌렸다

`f2l` config(domain + rollup16 + lsvd) · `fold_group5` · seed 42 · 5-fold.
OOF 확률 행렬을 통째로 비교했다.

| 모델 | 1회차 | 2회차 | 확률 최대차 | 라벨 다른 행 | 판정 |
|---|---|---|---|---|---|
| XGBoost | 0.460094 | 0.460094 | **0.00e+00** | 0 | **비트 동일** |
| RandomForest | 0.464918 | 0.464918 | 1.11e-16 | 0 | 라벨 동일 (부동소수점 오차) |
| CatBoost (`--gpu-ram-part auto`) | 0.474239 | 0.474054 | 7.49e-02 | **56** | **결과 다름** |

CatBoost 만 갈렸다. OOF 가 0.000185 흔들리고 6,201행 중 56행(0.9%)의 라벨이 달랐다.
seed 를 42 로 고정했는데도 그렇다.

### 원인은 CatBoost GPU 학습 자체다 (설정이 아니다)

처음에는 `--gpu-ram-part auto` 를 원인으로 봤다. `resolve_gpu_ram_part` 가 실행 시점의 GPU
여유 메모리로 값을 정하니 그럴듯했고, 값을 박은 두 실행이 일치하기도 했다.

**그 판단은 틀렸다.** 표본을 늘리니 무너졌다 — `--device gpu` 와 `--gpu-ram-part 0.4` 를
**둘 다 숫자로 박고** f2l 을 3회 돌린 결과다.

| 실행 | OOF macro F1 | 1회차 대비 |
|---|---|---|
| gpu0 | 0.474054000 | — |
| gpu1 | 0.474054000 | 최대차 0.0e+00 |
| gpu2 | **0.473744407** | **최대차 7.9e-02** |

3회 중 1회가 갈렸다. 간헐적이라 표본이 적으면 재현되는 것처럼 보인다. 앞서 "일치했다"고
적힌 관측들은 전부 이 확률을 통과한 것이었다.

추론은 결정론적이다(같은 모델 3회 예측 차이 0, 배치를 500행씩 쪼개도 0). fold 별 device
폴백도 아니다(`device=gpu`·`mixed=False`). 남는 건 학습 커널이다.

### CPU 는 비트 단위로 재현된다

| 실행 | OOF macro F1 | oof 최대차 | test 최대차 | 시간 |
|---|---|---|---|---|
| cpu0 | 0.483017344 | — | — | 10.6분 |
| cpu1 | 0.483017344 | **0.0e+00** | **0.0e+00** | 10.6분 |

값이 소수점 아홉 자리까지 같고 확률 행렬이 통째로 일치한다.

### 대가가 둘 있다

**속도** — f2l 기준 1.5분 → 10.6분으로 **7배**다. f16·3seed 파이프라인이면 CatBoost 몫이
12분에서 85분 정도로 늘어난다.

**점수가 달라진다** — CPU 가 0.4830, GPU 가 0.4741 로 **CPU 쪽이 0.0089 높다.** 재현성
문제가 아니라 **다른 모델**이기 때문이다. `models_gbdt.CatBoostModel._build` 가 GPU 에서
`rsm`(열 샘플링)을 빼기 때문에, GPU 는 `rsm=1`(샘플링 없음)로 CPU 는 `rsm=0.4` 로 학습한다.

즉 CPU 로 바꾸는 건 "느리지만 같은 걸 얻는" 교환이 아니라 **다른 구성으로 갈아타는 것**이다.
f2l 에서는 그쪽이 더 높았지만 f16 에서도 그런지는 따로 재야 한다.

### LB 최고점 제출본을 통째로 다시 만들어 봤다 (2026-08-05)

`submission_ens_v002_pairrule_m3.csv`(LB 0.4818, 지금까지 받은 점수 중 최고)를
`notebooks/09_final_submission.ipynb` 로 원본 csv 부터 끝까지 재생성했다.
CatBoost 는 **기존 산출물과 같은 조건인 GPU** 로 돌렸다.

| 단계 | 이번 | 기록 | 시간 |
|---|---|---|---|
| XGBoost | 0.4922 | 0.4922 | 4.4분 |
| CatBoost | 0.4932 | 0.4932 | 4.3분 |
| RandomForest | 0.4774 | 0.4774 | 1.5분 |
| 블렌드 (보정 전) | 0.5027 | 0.5027 | — |
| **블렌드 (보정 후)** | **0.5165** | **0.5165** | — |
| 짝 규칙 | 214행 (213행이 train 라벨 복사) | 동일 | 1초 미만 |

제출 csv 는 1층·2층 **둘 다 2,546행 전부 일치**했다.

```
전체 10.8분 (피처 파켓이 이미 있는 상태)
파켓부터 새로 만들면 +16분 -> 약 27분
```

**다만 이건 보장이 아니라 3분의 2 확률을 통과한 것이다.** 같은 조건 3회 중 1회는
어긋난다(§2). 이 제출본을 GPU 로 다시 만들면 대체로 같지만 가끔 다르다.

### 곁들여 확인된 것

- **CLI 와 노트북은 같은 결과를 낸다.** 같은 파켓으로 `train_gbdt.py` 를 CLI 로 돌린 것과
  `11_full_pipeline.ipynb` 가 `run_config` 를 부른 것을 비교했다 — xgb 확률까지 비트 동일,
  rf 1e-16, catboost 는 위 편차 범위이고 **라벨 차이 0**.
- **피처 파이프라인은 결정론적이다.** 파켓 22개를 새로 만들어 기존과 대조했더니 20개가
  지문까지 같았다. 다른 2개(`mutation_encoded`)는 코드가 실제로 바뀐 것이고 차이도
  2,718만 셀 중 68셀로 설명된다.
- **그리디 블렌드는 결정론적이다.** 라이브러리 98개를 고정하면 교차적합 macro F1 이
  `0.5336875292416042` 로 소수점 끝까지 재현된다.

---

## 3. 실무 지침

### 반드시 지켜야 하는 것

- **제출본을 만드는 실행에서는 CatBoost 를 CPU 로 돌린다.** `PipelineConfig.device_by_model`
  기본값이 `{"catboost": "cpu"}` 이고 `run_pipeline.py --catboost-device` 로 바꾼다.
  GPU 로 뽑은 CatBoost OOF 는 "다시 만들면 달라질 수 있는 것"으로 다룬다 — 탐색·비교에는
  써도 되지만 제출본의 근거로 삼지 않는다.
- **`--gpu-ram-part` 를 숫자로 준다.** `auto` 는 실행마다 설정 자체가 달라진다. CatBoost 를
  CPU 로 내리면 이 축은 XGBoost 에만 남지만, 기본값 0.4 를 그대로 둔다.
- **`--threads` 를 바꾸지 않는다.** XGBoost 는 스레드 수가 바뀌면 트리가 달라진다.
  기본값(전 코어)을 그대로 쓴다. CatBoost 도 `thread_count` 가 로그에 남는다.
- **seed 는 `cancer_hack.validation.SEED_ENSEMBLE` 에서 온다.** 스크립트마다 적어 두면
  사람마다 다른 값으로 돌리게 된다. 늘릴 때는 뒤에 덧붙여 앞 셋을 유지한다.

### 다른 기기에서 돌릴 때

같은 기기에서는 위 셋만 지키면 비트 단위로 재현된다. 기기가 바뀌면 한 축이 더 붙는다 —
`gpu_ram_part` 는 **비율**이라 GPU 메모리 총량이 다르면 절대값이 달라진다. 그래도 라벨이
갈릴 정도의 차이는 관찰되지 않았지만(노트북 ↔ CLI 대조에서 확률 3.3e-3, 라벨 차이 0),
숫자를 소수점까지 맞춰야 하는 자리라면 CPU 로 돌리는 쪽이 안전하다. CPU 도 비트 동일이
확인돼 있다.

라이브러리 버전이 맞아야 하는 건 §4.

---

## 4. 재현에 필요한 나머지 조건

seed 말고도 세 가지가 맞아야 같은 숫자가 나온다.

### 라이브러리 버전

예측을 바꾸는 넷은 `requirements.txt` 에 `==` 로 박혀 있다.

```
scikit-learn==1.9.0  xgboost==3.3.0  lightgbm==4.7.0  catboost==1.2.10
```

버전이 다르면 같은 seed 로도 트리가 달라진다. 제출 노트북들이 시작할 때 이 넷을 대조한다.

### fold 파일

`data/process/train_folds.parquet` 지문 `997d89a20595cc23`. 다시 만들면 `artifacts/oof/` 의
예측 100여 개가 전부 무효가 된다. `cancer_hack.provenance` 가 막는다.

### 피처 파켓

f16 이 읽는 22개의 내용 지문이 `provenance.EXPECTED_FEATURE_FINGERPRINTS` 에 박혀 있다.
`make_features.py` 재실행은 전부 다시 학습할 때만 한다. 새로 만들 때는
`paths.use_run_dirs()` 로 폴더를 갈라 기존 것을 남긴다.

---

## 5. 확인하는 법

```powershell
cd code
.\.venv\Scripts\python.exe -m pytest tests/test_seeds_are_pinned.py -q       # seed 고정
.\.venv\Scripts\python.exe -m pytest tests/test_data_provenance.py -q        # fold·파켓 지문
.\.venv\Scripts\python.exe -m pytest tests/test_final_notebook_reproduces.py -q  # 제출본 대조
```

마지막 것은 **노트북이 마지막으로 만든 산출물**을 본다. 노트북을 고쳤으면 다시 실행해야
의미가 있다.
