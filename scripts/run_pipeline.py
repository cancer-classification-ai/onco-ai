#!/usr/bin/env python
"""제출까지 한 번에 — fold · 피처 · seed 앙상블 학습 · 블렌드 · 짝 규칙 · 제출.

    python scripts/run_pipeline.py --run-tag run01
    python scripts/run_pipeline.py --run-tag quick --seeds 42 --no-rebuild-features
    python scripts/run_pipeline.py --run-tag v002 --seeds 42 --reference ../Models/...csv

로직은 `cancer_hack.pipeline` 에 있다. 이 스크립트와 `notebooks/13_pipeline.ipynb` 가
**같은 함수를 부른다** — 진입점이 둘이어도 코드 경로는 하나여야 결과가 어긋나지 않는다.

출력은 실행별로 갈린다. 기존 `data/process/`·`artifacts/` 는 건드리지 않는다.

    data/process_<RUN_TAG>/      피처 파켓
    artifacts/runs/<RUN_TAG>/    oof · test · logs · submissions + run.json · run.md

## 재현 가능성

난수원을 전부 고정하고 `--gpu-ram-part` 도 숫자로 박는다(`auto` 는 실행 시점 GPU 여유로
값을 정해서 같은 seed 로도 CatBoost 결과가 갈린다). `docs/reproducibility.md` 참고.

## DACON 제출

로컬에 csv 를 만들 뿐이다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from cancer_hack.validation import SEED as FOLD_SEED, SEED_ENSEMBLE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-tag", required=True, help="출력 폴더 이름. 공백·경로문자 금지")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEED_ENSEMBLE),
        help=f"seed 앙상블에 쓸 시드 (기본 {' '.join(map(str, SEED_ENSEMBLE))}). "
             "기본값을 바꾸려면 cancer_hack.validation.SEED_ENSEMBLE 을 고친다 — "
             "여기서 다른 값을 주면 기존 OOF 와 못 섞는다",
    )
    parser.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    parser.add_argument("--config", dest="feature_config", default="f16",
                        help="train_gbdt.CONFIGS 의 피처셋 이름")
    parser.add_argument("--cv", default="sgkf", choices=["skf", "sgkf"])
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--weights", type=float, nargs=3, default=[0.45, 0.45, 0.10],
                        metavar=("XGB", "CAT", "RF"),
                        help="모델 가중. seed 수로 나눠 배분한다 (기본 0.45 0.45 0.10)")
    parser.add_argument("--catboost-params", default=None,
                        help="CatBoost 에 쓸 PARAM_PRESETS 이름 (예: cbopt10)")
    parser.add_argument("--min-mut", type=int, default=3)
    parser.add_argument("--no-pair-rule", action="store_true",
                        help="짝 규칙을 끈다. LB +0.09 짜리라 보통 켜 둔다")
    parser.add_argument("--gpu-ram-part", type=float, default=0.4,
                        help="숫자로 고정한다. auto 는 재현을 깬다 (기본 0.4)")
    parser.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"],
                        help="기본 device. CatBoost 는 아래 이유로 따로 CPU 로 내린다")
    parser.add_argument(
        "--catboost-device", default="cpu", choices=["auto", "gpu", "cpu"],
        help="CatBoost GPU 학습은 같은 seed 로도 결과가 간헐적으로 갈린다(3회 중 1회, "
             "OOF 최대차 7.9e-02). CPU 는 비트 단위로 재현된다. 제출본을 만들 때는 cpu 를 "
             "쓴다. 빠른 탐색이면 gpu 로 바꾸되 그 OOF 는 재현 대상이 아니다 (기본 cpu)",
    )
    parser.add_argument("--no-rebuild-folds", action="store_true")
    parser.add_argument("--no-rebuild-features", action="store_true")
    parser.add_argument("--no-train", action="store_true",
                        help="이 실행 폴더의 기존 OOF 로 결합부터 한다")
    parser.add_argument("--reference", type=Path, default=None,
                        help="대조할 기준 제출 csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig(
        run_tag=args.run_tag,
        seeds=tuple(args.seeds),
        fold_seed=args.fold_seed,
        feature_config=args.feature_config,
        cv=args.cv,
        topk=args.topk,
        model_weights={"xgb": args.weights[0], "catboost": args.weights[1],
                       "rf": args.weights[2]},
        params_presets=({"catboost": args.catboost_params}
                        if args.catboost_params else {}),
        min_mut=args.min_mut,
        apply_pair_rule=not args.no_pair_rule,
        gpu_ram_part=args.gpu_ram_part,
        device=args.device,
        device_by_model={"catboost": args.catboost_device},
        rebuild_folds=not args.no_rebuild_folds,
        rebuild_features=not args.no_rebuild_features,
        train_models=not args.no_train,
        reference=args.reference,
    )

    result = run_pipeline(config)
    log = result["log"]

    print("\n" + "=" * 62)
    print(log.summary())
    print(f"\nOOF macro F1 = {result['oof_macro_f1']:.4f}")
    for label, path in log.artifacts.items():
        print(f"  {label:22s} {path}")
    for label, path in result["run_files"].items():
        print(f"  {label:22s} {path}")
    if "reference_rows_differ" in log.values:
        n = log.values["reference_rows_differ"]
        print(f"\n기준 파일과 다른 행: {n}" + ("  — 재현 성공" if n == 0 else ""))
    print("\n로컬 파일만 만들었다. DACON 업로드는 사람이 직접 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
