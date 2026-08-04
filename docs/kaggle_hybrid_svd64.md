# Hybrid + fold-local SVD64 Kaggle 실행

이 노트북은 기존 Hybrid 설정을 다시 정의하지 않고
`configs/hybrid_set_mlp_svd64.yaml`만 실행한다. SVD의 support gene 선택과 basis
fit은 각 outer-fold train 행에서만 수행된다. Kaggle Notebook의 Accelerator를 GPU로
설정한 뒤 아래 셀을 순서대로 실행한다.

## Cell 1 — Git clone

```bash
!git clone \
  --branch codex/hybrid-svd64-kaggle \
  --single-branch \
  https://github.com/cancer-classification-ai/onco-ai.git

%cd /kaggle/working/onco-ai

!git rev-parse HEAD
!git status --short
```

## Cell 2 — Dependency installation

```bash
!pip install -q -r requirements.txt
```

```python
import torch
import numpy as np
import pandas as pd
import sklearn

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)

assert torch.cuda.is_available(), "Kaggle Notebook GPU를 활성화해야 합니다."
```

## Cell 3 — Kaggle Dataset 경로 확인과 안전한 연결

사용자가 수정할 값은 `DATASET_SLUG` 하나뿐이다. source 후보와 target 상태를 먼저
출력한다. 저장소 target에 실제 데이터가 있으면 자동 삭제하지 않고 중단하며,
비어 있는 Git placeholder 디렉터리만 symlink로 교체한다.

```python
from pathlib import Path

DATASET_SLUG = "<KAGGLE-DATASET-SLUG>"

dataset_root = Path("/kaggle/input") / DATASET_SLUG
repo_data = Path("/kaggle/working/onco-ai/data")
layouts = [
    (dataset_root / "raw", dataset_root / "process"),
    (dataset_root / "data/raw", dataset_root / "data/process"),
]
raw_source, process_source = next(
    ((raw, process_) for raw, process_ in layouts if raw.is_dir()),
    (None, None),
)
print("dataset_root:", dataset_root)
print("raw candidates:", [str(raw) for raw, _ in layouts])
print("process candidates:", [str(process_) for _, process_ in layouts])
if raw_source is None:
    raise FileNotFoundError("지원하는 raw 디렉터리를 찾지 못했습니다.")


def safe_link(source: Path, target: Path) -> None:
    print(f"link request: {target} -> {source}")
    if target.is_symlink():
        if target.resolve() == source.resolve():
            print("already linked")
            return
        raise RuntimeError(f"다른 symlink가 존재합니다: {target} -> {target.resolve()}")
    if target.exists():
        entries = list(target.iterdir())
        non_placeholders = [entry for entry in entries if entry.name != ".gitkeep"]
        if non_placeholders:
            raise RuntimeError(
                f"{target}에 실제 파일이 있습니다. 자동 교체하지 않습니다: "
                f"{[entry.name for entry in non_placeholders]}"
            )
        for placeholder in entries:
            placeholder.unlink()
        target.rmdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
    print("linked:", target, "->", target.resolve())


safe_link(raw_source, repo_data / "raw")
if process_source.is_dir():
    safe_link(process_source, repo_data / "process")
else:
    (repo_data / "process").mkdir(parents=True, exist_ok=True)
    print("process source 없음; 저장소 data/process에서 필요한 피처를 생성하세요.")
```

## Cell 4 — 필수 파일 전체 검사

`load_dense_feature_bundle()`, `load_latent_source()`, fold loader 및 submission
writer가 실제로 읽는 파일을 한 번에 검사한다.

```python
from pathlib import Path

required = [
    "data/raw/train.csv",
    "data/raw/test.csv",
    "data/raw/sample_submission.csv",
    "data/process/train_folds.parquet",
    "data/process/train_domain_features.parquet",
    "data/process/test_domain_features.parquet",
    "data/process/train_sample_mutation_features_rollup.parquet",
    "data/process/test_sample_mutation_features_rollup.parquet",
    "data/process/train_mutation_parsed_features.parquet",
    "data/process/test_mutation_parsed_features.parquet",
    "data/process/train_additional_burden_features.parquet",
    "data/process/test_additional_burden_features.parquet",
    "data/process/train_amino_acid_features.parquet",
    "data/process/test_amino_acid_features.parquet",
    "data/process/train_mutation_encoded.parquet",
    "data/process/test_mutation_encoded.parquet",
]
missing = [path for path in required if not Path(path).is_file()]
print("required files:")
for path in required:
    print(" OK " if path not in missing else "MISS", path)
if missing:
    raise FileNotFoundError("누락 파일 전체:\n" + "\n".join(missing))
```

## Cell 5 — 1-fold, 1-epoch GPU smoke test

`--dry-run`은 실제 parser가 지원하며 1 fold/1 epoch로 제한하고 artifact를 쓰지 않는다.
학습 로그의 `dense_dim=655 latent_dim=64`를 검사해 CUDA forward/backward와 차원을
동시에 확인한다.

```bash
!bash -o pipefail -c 'python scripts/train_dl.py \
  --model hybrid \
  --config configs/hybrid_set_mlp_svd64.yaml \
  --cv skf \
  --n-splits 5 \
  --seed 42 \
  --device cuda \
  --dry-run \
  --no-submission \
  --tag hybrid_svd64_smoke \
  | tee /kaggle/working/hybrid_svd64_smoke.log'

!grep -q "dense_dim=655 latent_dim=64" /kaggle/working/hybrid_svd64_smoke.log
```

## Cell 6 — Full training

이 명령은 SKF 5-fold 전체를 학습하고 OOF/test prediction, JSON log와 submission을
만든다. submission 파일 생성은 허용되지만 대회에는 제출하지 않는다.

```bash
!python scripts/train_dl.py \
  --model hybrid \
  --config configs/hybrid_set_mlp_svd64.yaml \
  --cv skf \
  --n-splits 5 \
  --seed 42 \
  --device cuda \
  --tag hybrid_svd64_s42_skf5
```

## Cell 7 — Result verification

위 tag와 현재 artifact naming 규약에 따른 로그는
`artifacts/logs/dl_hybrid_hybrid_svd64_s42_skf5_skf5.json`이다.

```python
import json
from pathlib import Path
import numpy as np

result_path = Path(
    "artifacts/logs/dl_hybrid_hybrid_svd64_s42_skf5_skf5.json"
)
result = json.loads(result_path.read_text(encoding="utf-8"))
folds = np.asarray(result["fold_macro_f1"], dtype=float)

print("OOF Macro F1:", result["oof_macro_f1"])
print("singleton-only Macro F1:", result["oof_macro_f1_singleton"])
print("OOF Accuracy:", result["oof_accuracy"])
print("fold Macro F1:", folds.tolist())
print("fold mean:", folds.mean())
print("fold standard deviation:", folds.std())
print("best epoch per fold:", [fold["best_epoch"] for fold in result["fold_training"]])
print("dense dimension per fold:", result["fold_dense_dimensions"])
print("latent dimension per fold:", result["latent_dimensions"])
print("latent methods per fold:", result["latent_methods"])
print("latent fit rows per fold:", result["latent_fit_row_counts"])
print("latent support genes per fold:", result["latent_support_gene_counts"])
print("elapsed seconds:", result["elapsed_seconds"])
print("per-class F1:", result["per_class_f1"])

assert result["fold_dense_dimensions"] == [655, 655, 655, 655, 655]
assert result["latent_dimensions"] == [64, 64, 64, 64, 64]
assert np.isfinite(result["oof_macro_f1"])
assert np.isfinite(result["oof_macro_f1_singleton"])
```

## Cell 8 — Baseline comparison

```python
import numpy as np

BASELINE_OOF_MACRO_F1 = 0.42429162605366255
BASELINE_SINGLETON_F1 = 0.41942993257160466
BASELINE_FOLDS = np.asarray([
    0.42490941096916524,
    0.41723778362489977,
    0.42391928905699310,
    0.42610789161461000,
    0.43129868936685710,
])

candidate_folds = np.asarray(result["fold_macro_f1"], dtype=float)
oof_delta = result["oof_macro_f1"] - BASELINE_OOF_MACRO_F1
singleton_delta = result["oof_macro_f1_singleton"] - BASELINE_SINGLETON_F1
paired_delta = candidate_folds - BASELINE_FOLDS
improved = int((paired_delta > 0).sum())
worsened = int((paired_delta < 0).sum())
std_change = candidate_folds.std() - BASELINE_FOLDS.std()

if oof_delta >= 0.005 and improved >= 3 and singleton_delta >= -0.003:
    decision = "ADOPT_CANDIDATE"
elif oof_delta <= 0 or improved <= 2 or singleton_delta < -0.005:
    decision = "REJECT"
elif 0 < oof_delta < 0.005:
    decision = "WEAK_POSITIVE"
else:
    decision = "REJECT"

print("OOF delta:", oof_delta)
print("singleton delta:", singleton_delta)
print("fold paired delta:", paired_delta.tolist())
print("improved folds:", improved)
print("worsened folds:", worsened)
print("baseline fold std:", BASELINE_FOLDS.std())
print("candidate fold std:", candidate_folds.std())
print("fold std change:", std_change)
print("decision:", decision)
```

JSON baseline log가 있으면 저장소의 비교 도구로 per-class delta와 상·하위 5개
클래스까지 확인할 수 있다.

```bash
!python scripts/compare_dl_runs.py \
  --baseline-log <hybrid-baseline-json> \
  --candidate-log artifacts/logs/dl_hybrid_hybrid_svd64_s42_skf5_skf5.json
```

baseline JSON이 없다면 `--baseline-log`를 생략한다. 이 경우 기록된 baseline scalar와
fold 값으로 비교하며 per-class 비교는 출력하지 않는다.

## Cell 9 — Artifact archive

존재하는 artifact 디렉터리의 파일만 zip에 넣는다.

```python
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

archive = Path("/kaggle/working/hybrid_svd64_artifacts.zip")
roots = [
    Path("artifacts/oof"),
    Path("artifacts/test_predictions"),
    Path("artifacts/submissions"),
    Path("artifacts/logs"),
]
files = [path for root in roots if root.is_dir() for path in root.rglob("*") if path.is_file()]
if not files:
    raise FileNotFoundError("압축할 artifact가 없습니다.")
with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
    for path in files:
        output.write(path, path.as_posix())
print("archive:", archive, "files:", len(files), "bytes:", archive.stat().st_size)
```

## Process feature가 없을 때

Cell 4가 process 파일 누락을 보고하면 아래 셀로 Hybrid 591개 dense block과 SVD
source에 필요한 파일만 생성한다. 각 명령은 output이 없을 때만 실행하므로 기존
feature나 fold 파일을 덮어쓰지 않는다.

```python
from pathlib import Path
import subprocess

commands = []
for split in ("train", "test"):
    specs = [
        (f"data/process/{split}_domain_features.parquet",
         ["python", "scripts/make_features.py", "domain", "--split", split]),
        (f"data/process/{split}_sample_mutation_features_rollup.parquet",
         ["python", "scripts/make_features.py", "sample", "--split", split,
          "--include-cell-rollup"]),
        (f"data/process/{split}_mutation_parsed_features.parquet",
         ["python", "scripts/make_features.py", "parsed", "--split", split]),
        (f"data/process/{split}_additional_burden_features.parquet",
         ["python", "scripts/make_features.py", "burden-extra", "--split", split]),
        (f"data/process/{split}_amino_acid_features.parquet",
         ["python", "scripts/make_features.py", "amino", "--split", split]),
        (f"data/process/{split}_mutation_encoded.parquet",
         ["python", "scripts/make_features.py", "enc3", "--split", split]),
    ]
    commands.extend(command for output, command in specs if not Path(output).exists())

for command in commands:
    print("+", " ".join(command))
    subprocess.run(command, check=True)

fold_path = Path("data/process/train_folds.parquet")
if not fold_path.exists():
    command = ["python", "scripts/make_folds.py", "--n-splits", "5", "--seed", "42"]
    print("+", " ".join(command))
    subprocess.run(command, check=True)
else:
    print("fold 파일 유지:", fold_path)
```

feature 생성 후 Cell 4부터 다시 실행한다. `train_folds.parquet`은 저장소의 공식
`scripts/make_folds.py`만 사용하며 별도 split 로직을 만들지 않는다.
