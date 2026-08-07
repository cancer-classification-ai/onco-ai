#!/usr/bin/env python
"""잠재·모듈 블록 사전 진단 — **학습 없이** 이 방향이 살아 있는지 먼저 본다.

    python scripts/inspect_latent.py --method svd --n-components 64 --row-norm l2
    python scripts/inspect_latent.py --method svd --n-components 64 --row-norm none
    python scripts/inspect_latent.py --method kmeans --n-modules 24 --seeds 0,1,2

## 왜 이게 먼저인가

공변이 쌍(`comut`) 패밀리는 30초짜리 진단으로 알 수 있었을 걸 파라미터 6종 x 2 CV
학습으로 배웠다 — 6종 전부 sgkf 음수, skf 는 전부 noise 이하. 기록된 원인은
"chi2 후보 풀이 `BRAF x 긴 passenger 유전자` 로 축퇴해 생물학이 아니라 유전자 길이·
TMB 를 학습했다"였다. 같은 행렬을 쓰는 이상 잠재·모듈도 같은 축을 찾을 수 있다.

그래서 학습 전에 세 숫자를 본다.

    burden 상관   |corr(성분, 변이 유전자 수)|. 선두 성분이 0.9 를 넘으면 그건 TMB
                  축이고, comut 의 실패 모드를 새 좌표계에서 재현한 것이다.
    시프트 배율   mean|성분_test| / mean|성분_train|. 설계 목표는 1.0.
                  comut 실측으로 원시 AND 는 3.12, share 정규화 후 1.46 이었다.
    부분공간 정렬 fold 쌍의 주각 코사인 평균. 이름이 고정된 기저에는 Jaccard 가
                  무의미하므로 이쪽을 쓴다.

하드 모듈은 여기에 **seed 간 ARI** 를 더한다. 배정이 seed 마다 흔들리면 CV 델타와
무관하게 기각 대상이다 — 이 저장소가 채택한 것 중 제일 불안정한 게
`comut_auto` 의 fold 간 Jaccard 0.448~0.547 이고, `enc3` chi2 는 0.719 다.

## 중단 규칙

선두 성분이 burden 정렬(|r| > 0.9)이고 시프트 배율이 1.8 을 넘으면 거기서 멈춘다.
그건 *방법* 이 아니라 *행렬* 이 문제라는 뜻이라 L3·L4 를 같이 정리한다.

## 이 스크립트가 쓰는 것

읽기만 한다 — 모델도 제출도 없고, 출력은 `artifacts/features/modules/preflight/` 뿐이다.
fold 는 `data/process/train_folds.parquet` 을 읽기만 한다(만들지 않는다).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Windows 콘솔이 cp949 라 한글·기호가 깨진다. 스트림을 제자리에서 UTF-8 로 바꾼다
# (갈아끼우면 이 모듈을 import 하는 쪽의 캡처 버퍼가 닫힌다 — train_gbdt.py 참고).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# 경로는 `cancer_hack.paths` 가 정한다 — 값은 쓰는 시점에 정해진다.
from cancer_hack.paths import LAZY_PROCESS, LazyDir, artifacts_dir, process_dir  # noqa: E402
from cancer_hack.features_latent import (  # noqa: E402
    _binarize,
    fit_gene_modules,
    fit_latent_basis,
    subspace_alignment,
    transform_latent_basis,
)
from cancer_hack.validation import fold_column  # noqa: E402

PROC_DIR = LAZY_PROCESS
OUT_DIR = LazyDir(lambda: artifacts_dir() / "features/modules/preflight")
#: 중단 규칙 — 선두 성분이 이만큼 burden 에 정렬되고 시프트가 이만큼이면 멈춘다.
BURDEN_CORR_LIMIT = 0.90
SHIFT_RATIO_LIMIT = 1.80


def log(message: str) -> None:
    print(message, flush=True)


def load_matrices() -> tuple[list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    """enc3 행렬 train/test 와 fold 표. `train_gbdt.Dataset` 과 같은 파일을 읽는다."""
    train = pd.read_parquet(PROC_DIR / "train_mutation_encoded.parquet")
    test = pd.read_parquet(PROC_DIR / "test_mutation_encoded.parquet")
    folds = pd.read_parquet(PROC_DIR / "train_folds.parquet")
    if not (folds["ID"].astype(str).to_numpy() == train["ID"].astype(str).to_numpy()).all():
        raise ValueError("train_folds.parquet 의 ID 순서가 enc3 와 다르다")
    columns = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    return (
        columns,
        train[columns].to_numpy(np.float32),
        test[columns].to_numpy(np.float32),
        folds,
    )


def inspect_latent(args, columns, train_matrix, test_matrix, folds) -> dict:
    """fold 마다 기저를 fit 하고 성분별 burden 상관·시프트 배율을 낸다."""
    fold_ids = folds[fold_column(args.cv, args.n_splits)].to_numpy()
    burden = np.asarray(_binarize(train_matrix, args.mode).sum(axis=1)).ravel()

    bases = []
    rows = []
    for fold in range(args.n_splits):
        train_index = np.where(fold_ids != fold)[0]
        t0 = time.perf_counter()
        basis = fit_latent_basis(
            train_matrix,
            train_index,
            gene_names=columns,
            method=args.method,
            n_components=args.n_components,
            mode=args.mode,
            row_norm=args.row_norm,
            gene_weight=args.gene_weight,
            min_gene_support=args.min_support,
            random_state=args.random_state,
        )
        bases.append(basis)
        train_projected = transform_latent_basis(train_matrix, basis)[train_index]
        test_projected = transform_latent_basis(test_matrix, basis)
        fold_burden = burden[train_index].astype(np.float64)

        for k in range(basis.n_components):
            column = train_projected[:, k].astype(np.float64)
            train_mean = float(np.abs(column).mean())
            test_mean = float(np.abs(test_projected[:, k]).mean())
            rows.append(
                {
                    "fold": fold,
                    "component": k,
                    "burden_corr": (
                        float(np.corrcoef(column, fold_burden)[0, 1])
                        if column.std() > 1e-12
                        else 0.0
                    ),
                    "train_mean_abs": train_mean,
                    "test_mean_abs": test_mean,
                    "shift_ratio": (test_mean / train_mean) if train_mean > 0 else np.nan,
                    "top_genes": " ".join(basis.stats[k]["top_genes"][:20]),
                }
            )
        log(
            f"  fold {fold}: 유전자 {basis.gene_index.size:,}개 -> 성분 "
            f"{basis.n_components}개  ({time.perf_counter() - t0:.1f}s)"
        )

    frame = pd.DataFrame(rows)
    alignment = [
        subspace_alignment(bases[i], bases[j])
        for i in range(len(bases))
        for j in range(i + 1, len(bases))
    ]

    leading = frame[frame["component"] < 3]
    summary = {
        "method": args.method,
        "row_norm": args.row_norm,
        "n_components": int(bases[0].n_components),
        "n_genes_kept": int(bases[0].gene_index.size),
        "leading_burden_corr_absmax": float(leading["burden_corr"].abs().max()),
        "burden_corr_absmean": float(frame["burden_corr"].abs().mean()),
        "shift_ratio_median": float(frame["shift_ratio"].median()),
        "shift_ratio_leading": float(leading["shift_ratio"].median()),
        "subspace_alignment_mean": float(np.mean(alignment)) if alignment else 1.0,
    }

    log("")
    log(f"  선두 3성분 |burden 상관| 최대 = {summary['leading_burden_corr_absmax']:.3f}")
    log(f"  전 성분  |burden 상관| 평균 = {summary['burden_corr_absmean']:.3f}")
    log(f"  시프트 배율 중앙값 = {summary['shift_ratio_median']:.3f}  (설계 목표 1.0)")
    log(f"  선두 3성분 시프트 배율 = {summary['shift_ratio_leading']:.3f}")
    log(f"  fold 간 부분공간 정렬 = {summary['subspace_alignment_mean']:.3f}  (1.0 이 완전 일치)")

    stop = (
        summary["leading_burden_corr_absmax"] > BURDEN_CORR_LIMIT
        and summary["shift_ratio_leading"] > SHIFT_RATIO_LIMIT
    )
    summary["stop_rule_triggered"] = bool(stop)
    log("")
    if stop:
        log(
            f"  [중단 규칙 발동] 선두 성분이 변이 부담 축이고(|r| > {BURDEN_CORR_LIMIT}) "
            f"시프트가 {SHIFT_RATIO_LIMIT} 를 넘는다. comut 의 실패 모드를 새 좌표계에서 "
            "재현한 것이다 — 방법이 아니라 행렬이 문제라는 뜻이라 여기서 멈춘다."
        )
    else:
        log("  [중단 규칙 통과] 학습 사다리로 넘어가도 된다.")

    _write(frame, f"latent_{args.method}_{args.row_norm}_c{args.n_components}")
    return summary


def inspect_modules(args, columns, train_matrix, test_matrix, folds) -> dict:
    """seed 를 바꿔 가며 KMeans 배정을 재고 ARI 로 안정성을 잰다."""
    from sklearn.metrics import adjusted_rand_score

    fold_ids = folds[fold_column(args.cv, args.n_splits)].to_numpy()
    train_index = np.where(fold_ids != 0)[0]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    fitted = []
    for seed in seeds:
        t0 = time.perf_counter()
        modules = fit_gene_modules(
            train_matrix,
            train_index,
            gene_names=columns,
            n_modules=args.n_modules,
            svd_components=args.svd_components,
            mode=args.mode,
            min_gene_support=args.min_support,
            random_state=seed,
        )
        fitted.append(modules)
        log(
            f"  seed {seed}: 유전자 {modules.gene_index.size:,}개 -> 모듈 "
            f"{modules.n_modules}개 · 최대 {int(modules.module_sizes.max()):,} · "
            f"단일원소 {int((modules.module_sizes == 1).sum())}개  "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    # ARI 는 같은 유전자 집합 위에서만 비교된다. fold 가 같으니 gene_index 도 같다.
    pairs = [
        (seeds[i], seeds[j], float(adjusted_rand_score(fitted[i].labels, fitted[j].labels)))
        for i in range(len(fitted))
        for j in range(i + 1, len(fitted))
    ]
    aris = [p[2] for p in pairs]

    summary = {
        "n_modules": int(fitted[0].n_modules),
        "n_genes_kept": int(fitted[0].gene_index.size),
        "largest_module": int(fitted[0].module_sizes.max()),
        "singleton_modules": int((fitted[0].module_sizes == 1).sum()),
        "seed_ari_mean": float(np.mean(aris)) if aris else 1.0,
        "seed_ari_min": float(np.min(aris)) if aris else 1.0,
        "seed_pairs": [{"a": a, "b": b, "ari": v} for a, b, v in pairs],
    }

    log("")
    for a, b, value in pairs:
        log(f"  ARI(seed {a}, seed {b}) = {value:.3f}")
    log(f"  seed 간 ARI 평균 = {summary['seed_ari_mean']:.3f}")
    log("")
    log(
        "  비교: enc3 chi2 의 fold 간 Jaccard 0.719 / comut_auto 0.448~0.547 이 "
        "이 저장소가 받아들인 선택 안정성의 하한이다."
    )
    if summary["seed_ari_mean"] < 0.40:
        log(
            f"  [경고] ARI {summary['seed_ari_mean']:.3f} 은 그 하한보다 낮다. 배정을 "
            "생물학적 실체로 해석하면 안 되고, CV 델타와 무관하게 기각 근거가 된다."
        )

    frame = pd.DataFrame(
        {
            "gene": [columns[i] for i in fitted[0].gene_index],
            **{f"module_seed{seed}": m.labels for seed, m in zip(seeds, fitted)},
        }
    )
    _write(frame, f"modules_n{args.n_modules}_seeds{'-'.join(map(str, seeds))}")
    return summary


def _write(frame: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    log(f"  -> {path.relative_to(PROJECT_ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", choices=["svd", "nmf", "kmeans"], default="svd")
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument(
        "--row-norm",
        choices=["l2", "none"],
        default="l2",
        help="none 은 의도적 시프트 노출 대조군이다",
    )
    parser.add_argument("--gene-weight", choices=["none", "idf"], default="none")
    parser.add_argument("--n-modules", type=int, default=24, help="--method kmeans 전용")
    parser.add_argument("--svd-components", type=int, default=64)
    parser.add_argument(
        "--seeds", default="0,1,2", help="--method kmeans 의 ARI 비교용 시드 목록"
    )
    parser.add_argument("--mode", choices=["mutated", "functional"], default="mutated")
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--cv", choices=["skf", "sgkf"], default="skf")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None, help="요약을 저장할 경로")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    t0 = time.perf_counter()
    log(f"[load] enc3 행렬 + fold 표 읽는 중 ...")
    columns, train_matrix, test_matrix, folds = load_matrices()
    log(
        f"[load] train {train_matrix.shape} · test {test_matrix.shape} · "
        f"유전자 {len(columns):,}개  ({time.perf_counter() - t0:.1f}s)"
    )
    log("")

    if args.method == "kmeans":
        log(f"[하드 모듈] n_modules={args.n_modules} · seeds={args.seeds}")
        summary = inspect_modules(args, columns, train_matrix, test_matrix, folds)
    else:
        log(
            f"[잠재 기저] method={args.method} · n_components={args.n_components} · "
            f"row_norm={args.row_norm}"
        )
        summary = inspect_latent(args, columns, train_matrix, test_matrix, folds)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        log(f"  -> {args.json}")
    log(f"\n총 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
