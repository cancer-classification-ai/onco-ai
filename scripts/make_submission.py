#!/usr/bin/env python
"""test 확률 파일에서 제출 csv 를 만든다.

    python scripts/make_submission.py --predictions artifacts/test_predictions/test_X.csv
    python scripts/make_submission.py --predictions a.csv b.csv --output blend.csv

확률 파일을 여러 개 주면 ID 기준으로 정렬해 평균한 뒤 argmax 한다. 재학습 없이
블렌딩만 다시 해 볼 수 있게 `train_gbdt.py` 와 분리해 뒀다. 실제 로직은 두 진입점이
`cancer_hack.io` 의 같은 함수를 쓴다.

제출 파일은 `sample_submission.csv` 와 **ID 로 병합**해 만든다. 두 파일의 행 순서가
같은 건 확인됐지만 순서 가정에 기대면 어느 한쪽 생성 경로가 바뀔 때 조용히 어긋난다.

## DACON 제출

이 스크립트는 로컬에 csv 를 만들 뿐이다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.io import average_probabilities, write_submission  # noqa: E402
from cancer_hack.metrics import build_prediction_frame, read_prediction_frame  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data/raw"
SUBMISSION_DIR = PROJECT_ROOT / "artifacts/submissions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="artifacts/test_predictions/ 의 확률 csv. 여러 개면 평균한다.",
    )
    parser.add_argument("--output", type=Path, default=None, help="출력 제출 csv 경로")
    parser.add_argument(
        "--sample-submission", type=Path, default=RAW_DIR / "sample_submission.csv"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames, class_sets = [], []
    for path in args.predictions:
        if not path.exists():
            raise FileNotFoundError(path)
        frame, classes = read_prediction_frame(path)
        frames.append(frame)
        class_sets.append(tuple(classes))

    if len(set(class_sets)) != 1:
        raise ValueError("확률 파일마다 클래스 구성이 다르다 — 평균낼 수 없다")
    classes = list(class_sets[0])

    if len(frames) == 1:
        merged = frames[0]
    else:
        proba = average_probabilities(frames, classes)
        merged = build_prediction_frame(frames[0]["ID"].astype(str), proba, classes)

    if args.output is not None:
        output = args.output
    elif len(args.predictions) == 1:
        stem = args.predictions[0].stem.removeprefix("test_")
        output = SUBMISSION_DIR / f"submission_{stem}.csv"
    else:
        output = SUBMISSION_DIR / f"submission_blend{len(args.predictions)}.csv"

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"이미 있다: {output}. --overwrite 를 준다.")

    info = write_submission(merged, args.sample_submission, output)
    for key, value in info.items():
        print(f"{key}: {value}")
    print(f"sources: {len(args.predictions)}개")
    print("\n로컬 파일만 만들었다. DACON 업로드는 직접 한다.")


if __name__ == "__main__":
    main()
