#!/usr/bin/env python
"""fold 파일 생성 — 이 저장소에서 fold 를 만드는 **유일한 곳**이다.

    python scripts/make_folds.py                       # 이미 있으면 거부한다
    python scripts/make_folds.py --overwrite
    python scripts/make_folds.py --n-splits 10 --seed 0 --out /tmp/folds.parquet

`data/process/train_folds.parquet` 에 네 열을 쓴다.

    ID           원본 train.csv 의 행 순서 그대로
    group_key    변이 프로파일이 같은 행을 묶는 정수
    fold_skf5    StratifiedKFold                          0-based
    fold_group5  StratifiedGroupKFold(groups=group_key)   0-based

`train_gbdt.py` 는 이 파일을 **읽기만 한다.** fold 방식을 바꿀 일이 생기면 여기만
고쳐서 다시 돌리면 되고, 학습 스크립트가 몰래 다른 분할을 만들 여지가 없다.

## 덮어쓰기를 막아 둔 이유

`artifacts/oof/` 의 예측이 전부 이 파일의 분할에서 나왔다. 열 이름·기준·시드 중
하나라도 바뀌면 과거 점수와 비교가 끊긴다. 그래서 이미 있는 파일은 `--overwrite`
없이는 건드리지 않고, 어떤 설정으로 만들었는지 `train_folds.json` 에 함께 남긴다.

## 1-based `fold` 열은 어디 갔나

`PR#16` 의 `assign_fold_column`(1-based, group CV 하나)은 노트북에서 fold 를 눈으로
확인하는 용도로 `validation.py` 에 그대로 있다. 학습 경로가 쓰는 건 이 스크립트가
내는 0-based 두 열이다 — `train_gbdt` 의 fold 루프가 `range(n_splits)` 로 돌고
skf 도 함께 필요해서다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.io import save_parquet  # noqa: E402
from cancer_hack.validation import (  # noqa: E402
    SEED,
    build_fold_frame,
    fold_class_distribution,
    fold_column,
)

RAW_TRAIN = PROJECT_ROOT / "data/raw/train.csv"
PROC_DIR = PROJECT_ROOT / "data/process"
OUT_PATH = PROC_DIR / "train_folds.parquet"
GROUP_CACHE = PROC_DIR / "train_group_keys.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="fold 분할 시드. 모델 시드(`train_gbdt --seed`)와 별개다.",
    )
    parser.add_argument("--label-col", type=str, default="SUBCLASS")
    parser.add_argument("--input", type=Path, default=RAW_TRAIN)
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="저장할 Parquet 경로")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 있는 fold 파일을 덮어쓴다. 과거 OOF 와 비교가 끊긴다.",
    )
    return parser.parse_args()


def meta_path(out_path: Path) -> Path:
    """fold 파일 옆에 두는 설정 기록 경로."""
    return out_path.with_suffix(".json")


def main() -> None:
    args = parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(
            f"{args.out} 가 이미 있다. 덮어쓰려면 --overwrite 를 준다.\n"
            "  주의: fold 가 바뀌면 artifacts/oof/ 의 과거 점수와 비교가 끊긴다."
        )
    if not args.input.exists():
        raise SystemExit(f"{args.input} 가 없다. 원본 csv 를 먼저 놓는다.")

    print(f"input       : {args.input}")
    print(f"n_splits    : {args.n_splits}")
    print(f"seed        : {args.seed}")

    folds = build_fold_frame(
        args.input,
        label_column=args.label_col,
        n_splits=args.n_splits,
        seed=args.seed,
        group_cache_path=GROUP_CACHE,
    )
    columns = [fold_column(kind, args.n_splits) for kind in ("skf", "sgkf")]
    n_groups = folds["group_key"].nunique()
    print(f"rows        : {len(folds):,}")
    print(f"groups      : {n_groups:,}")
    print(f"fold columns: {', '.join(columns)}")

    # 분포는 label 을 다시 붙여야 볼 수 있다. 파일에는 라벨을 싣지 않는다 —
    # 피처 parquet 이 이미 들고 있고, 두 벌 두면 한쪽만 고쳤을 때 어긋난다.
    labeled = folds.merge(
        pd.read_csv(
            args.input, usecols=["ID", args.label_col], dtype=str, na_filter=False
        ),
        on="ID",
        how="left",
        validate="one_to_one",
    )
    for column in columns:
        dist = fold_class_distribution(
            labeled, label_column=args.label_col, fold_column=column
        )
        print(f"\n[{column} × class 분포 (비율 %)]\n")
        print(dist["ratio"].unstack(level=args.label_col).round(2).to_string())

    save_parquet(folds, args.out)
    meta = {
        # posix 로 적어 둔다 — 팀원 OS 마다 구분자가 달라지면 diff 가 지저분해진다.
        "source": args.input.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "label_column": args.label_col,
        "n_rows": int(len(folds)),
        "n_groups": int(n_groups),
        "fold_columns": columns,
    }
    meta_file = meta_path(args.out)
    with open(meta_file, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print(f"\nSaved → {args.out}")
    print(f"       {meta_file}")


if __name__ == "__main__":
    main()
