#!/usr/bin/env python
"""제출 csv 에 exact-match 짝 라벨 규칙을 적용한다.

    python scripts/apply_pair_rule.py --submission a.csv --out a_pairrule.csv
    python scripts/apply_pair_rule.py --submission a.csv b.csv c.csv --out-dir artifacts/submissions
    python scripts/apply_pair_rule.py --submission a.csv --out a_m6.csv --min-mut 6

규칙이 무엇이고 왜 맞는지는 `cancer_hack.pair_rule` 모듈 docstring 과 `docs/pair_rule.md`
에 있다. 이 스크립트는 그 모듈을 부르는 얇은 진입점이다.

파일을 여러 개 주면 train 스캔은 **한 번만** 한다. 후보를 여러 개 만들 때 이쪽이 빠르고,
무엇보다 `--min-mut` 이 후보끼리 어긋날 여지가 없다.

규칙의 전제가 깨지면 파일을 쓰지 않고 멈춘다. 이 규칙은 로컬 CV 로 잴 수 없어서, 구현이나
데이터가 틀어져도 제출 한 번을 태우기 전까지 아무도 모르기 때문이다.

## DACON 제출

이 스크립트는 로컬에 csv 를 만들 뿐이다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# 경로는 `cancer_hack.paths` 가 정한다 — 값은 쓰는 시점에 정해진다.
from cancer_hack.paths import LAZY_RAW, raw_dir  # noqa: E402
from cancer_hack.pair_rule import (  # noqa: E402
    DEFAULT_MIN_MUT,
    apply_to_submission,
    build_pair_rule,
)

RAW_DIR = LAZY_RAW
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--submission", type=Path, nargs="+", required=True, help="입력 제출 csv (여러 개 가능)"
    )
    parser.add_argument("--out", type=Path, default=None, help="출력 경로. 입력이 하나일 때만.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="출력 폴더. 파일명은 <원본>_pairrule_m<min-mut>.csv 가 된다.",
    )
    parser.add_argument("--train", type=Path, default=RAW_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=RAW_DIR / "test.csv")
    parser.add_argument(
        "--min-mut",
        type=int,
        default=DEFAULT_MIN_MUT,
        help=f"매칭에 요구하는 최소 변이 수. 낮출수록 우연 일치가 섞인다 (기본 {DEFAULT_MIN_MUT})",
    )
    parser.add_argument("--log", type=Path, default=None, help="진단 json 을 남길 경로")
    parser.add_argument(
        "--allow-premise-violation",
        action="store_true",
        help="전제 검사에 걸려도 그대로 쓴다. 제출용으로는 쓰지 말 것.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_outputs(args: argparse.Namespace) -> list[Path]:
    if args.out is not None and args.out_dir is not None:
        raise SystemExit("--out 과 --out-dir 은 같이 못 쓴다")
    if args.out is not None:
        if len(args.submission) != 1:
            raise SystemExit("--out 은 입력이 하나일 때만 쓴다. 여러 개면 --out-dir 을 준다.")
        return [args.out]

    suffix = f"_pairrule_m{args.min_mut}"
    out_dir = args.out_dir
    return [
        (out_dir if out_dir is not None else path.parent) / f"{path.stem}{suffix}.csv"
        for path in args.submission
    ]


def main() -> None:
    args = parse_args()
    outputs = resolve_outputs(args)

    for path in args.submission:
        if not path.exists():
            raise FileNotFoundError(path)
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"이미 있다: {path}. --overwrite 를 준다.")

    rule = build_pair_rule(args.train, args.test, args.min_mut)
    diagnostics = rule.diagnostics

    print(f"train 중복 묶음  같은 라벨 {diagnostics['train_dup_groups_same_label']} / "
          f"짝 라벨 {diagnostics['train_dup_groups_pair_label']}")
    print(f"고아 수      {diagnostics['train_orphans_by_label']}")
    print(f"test 매칭    {diagnostics['test_matched_by_train_label']}")
    print(f"규칙 대상    {diagnostics['n_flipped']}행 (변이 {args.min_mut}개 이상)")

    reasons = rule.verify_premises()
    if reasons:
        print("\n전제 검사 실패 — 이대로 제출하면 안 된다:")
        for reason in reasons:
            print(f"  - {reason}")
        if not args.allow_premise_violation:
            raise SystemExit("파일을 쓰지 않았다. 근거를 확인하고 --allow-premise-violation 을 준다.")
        print("  (--allow-premise-violation 이라 그대로 진행한다)")

    results = []
    for submission, output in zip(args.submission, outputs):
        info = apply_to_submission(rule, submission, output)
        results.append(info)
        print(f"\n{submission.name}")
        print(f"  바뀐 행 {info['n_changed']} · 바꾸기 전 train 라벨 복사였던 행 "
              f"{info['n_was_copying_train_label']}")
        print(f"  → {output}")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        payload = {**diagnostics, "premise_violations": reasons, "applied": results}
        args.log.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n진단 로그 → {args.log}")

    print("\n로컬 파일만 만들었다. DACON 업로드는 직접 한다.")


if __name__ == "__main__":
    main()
