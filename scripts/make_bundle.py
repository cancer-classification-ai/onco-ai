#!/usr/bin/env python
"""실험 하나의 산출물을 **폴더 하나로** 묶는다.

    python scripts/make_bundle.py --tag gf_nocb_geo_rs0 --version v012 \
        --config greedy_nocb_geometric --note "그리디 + 기하평균 + 짝 규칙"

우리 산출물은 종류별로 흩어져 있다 — `artifacts/oof/`, `artifacts/test_predictions/`,
`artifacts/submissions/`, `artifacts/logs/`. 실험 하나를 남에게 넘기거나 나중에 다시
보려면 그 넷을 손으로 모아야 하는데, 그러다 **다른 실험의 파일이 섞인다.** 확률은 A
실험 것이고 제출 라벨은 B 실험 것인 폴더가 만들어져도 파일만 봐서는 아무도 모른다.

그래서 태그 하나를 받아 그 태그의 것만 모은다. 형식은 팀 공유 드라이브 규격
(`폴더구조.md`)을 그대로 따르고, 거기에 세 가지를 더한다.

| 더한 것 | 왜 |
|---|---|
| `submission_pairrule.csv` | 짝 라벨 규칙까지 적용한 최종 후보. LB 를 +0.09 올린 후처리다 |
| `*.parquet` | csv 와 같은 내용의 파켓. 다음 실험이 읽을 때 빠르고 dtype 이 안 흔들린다 |
| `README.md` | 어떤 전략·파라미터로 나온 점수인지. 이게 없으면 폴더가 숫자 더미다 |

`submission.csv` 는 `test_pred.csv` 에서 새로 만든다. 기존 제출 파일을 복사해 오면
확률과 라벨이 서로 다른 실험 조합이 될 수 있어서다 — `export_for_drive.py` 와 같은 이유다.

DACON 업로드는 하지 않는다. 로컬 폴더만 만든다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cancer_hack.pair_rule import (  # noqa: E402
    DEFAULT_MIN_MUT,
    apply_to_submission,
    build_pair_rule,
)
from cancer_hack.paths import LazyDir, artifacts_dir, process_dir, raw_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = LazyDir(raw_dir)
ARTIFACTS = LazyDir(artifacts_dir)
PROCESS = LazyDir(process_dir)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_readme(args, log: dict, metrics: dict, pair_summary: dict, files: list[tuple[str, str]]) -> str:
    """폴더만 보고도 무슨 실험인지 알 수 있게 적는다."""

    lines = [
        f"# {args.version} — {args.config}",
        "",
        args.note or "(설명 없음)",
        "",
        "## 점수",
        "",
        "| 지표 | 값 |",
        "|---|---|",
    ]
    crossfit = log.get("crossfit_macro_f1")
    if crossfit is not None:
        lines.append(f"| **교차적합 macro F1** | **{crossfit:.4f}** |")
    if log.get("crossfit_fold_macro_f1"):
        folds = log["crossfit_fold_macro_f1"]
        lines.append(f"| fold 별 | {', '.join(f'{v:.4f}' for v in folds)} |")
        lines.append(f"| fold 표준편차 | {np.std(folds):.4f} |")
        lines.append(f"| 최저 fold | {min(folds):.4f} |")
    full = log.get("full_fit_macro_f1")
    if isinstance(full, dict):
        lines.append(f"| 전체 적합 (낙관적 — 선택 금지) | {full['value']:.4f} |")

    lines += [
        "",
        "교차적합이 보고할 값이다. fold 를 뺀 나머지에서 멤버·가중치를 정하고 그 fold 에서만",
        "잰다. 전체 OOF 에 한 번에 맞춘 값은 선택에 쓴 행을 다시 보는 것이라 부풀어 있다.",
        "",
        "## 전략",
        "",
        "| 축 | 값 |",
        "|---|---|",
        f"| 분할 | `{log.get('fold_column', '?')}` (StratifiedGroupKFold 5분할, seed 42) |",
        f"| 결합 | Caruana 그리디 · 복원 허용 (담긴 횟수 = 가중치) |",
        f"| 합치는 방식 | `{log.get('form', 'mean')}` |",
        f"| 라운드 | {log.get('n_rounds', '?')} |",
        f"| bagging | 비율 {log.get('bag_fraction', '?')} × {log.get('bag_rounds', '?')}회 |",
        f"| 난수 | `--random-state {log.get('random_state', '?')}` |",
        f"| 후보 | {log.get('library_size', '?')}개 |",
        f"| 선택된 멤버 | {len(log.get('selected', {}))}개 |",
        f"| 후처리 | 짝 라벨 규칙 (`min_mut={pair_summary['min_mut']}`) |",
    ]

    if log.get("selected"):
        top = sorted(log["selected"].items(), key=lambda kv: -kv[1])[:12]
        total = sum(log["selected"].values())
        lines += ["", "### 가중이 큰 멤버", "", "| 가중 | 횟수 | 멤버 |", "|---|---|---|"]
        for name, count in top:
            lines.append(f"| {count / total:.3f} | {count} | `{name[:72]}` |")

    lines += [
        "",
        "## 짝 라벨 규칙",
        "",
        "KIPAN=KICH+KIRC+KIRP, GBMLGG=GBM+LGG 로 암 분류 체계가 겹쳐서 같은 환자가 두 줄로",
        "들어가 있다. 유전자 4,384열이 글자까지 같고 라벨만 다르다. 그 두 줄이 train 과 test 로",
        "갈리면 모델은 train 쪽 라벨을 그대로 답하는데, train 이 그 라벨을 이미 쓰고 있으므로",
        "정답은 반드시 반대편이다. 그래서 제출 파일의 해당 행만 뒤집는다.",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 바뀐 행 | {pair_summary['changed']}행 (test {pair_summary['total']}행의 "
        f"{pair_summary['changed'] / pair_summary['total'] * 100:.1f}%) |",
        f"| 그중 train 라벨을 복사하고 있던 행 | {pair_summary['copied']}행 |",
        f"| 최소 변이 수 | {pair_summary['min_mut']} |",
        "",
        "test 한 행만 따로 넣어도 같은 결과가 나온다 — train 명단만 조회하므로 다른 test 행을",
        "보지 않는다. 대회 규정의 판정 기준을 만족한다.",
        "",
        "## 파일",
        "",
        "| 파일 | sha256 (앞 16) |",
        "|---|---|",
    ]
    for name, digest in files:
        lines.append(f"| `{name}` | `{digest[:16]}` |")

    lines += [
        "",
        "`submission.csv` 는 짝 규칙 **전**, `submission_pairrule.csv` 가 최종 후보다.",
        "둘을 같이 두는 이유는 규칙의 기여를 나중에도 따로 읽을 수 있게 하기 위해서다.",
        "",
        "DACON 업로드는 사람이 직접 한다. 이 폴더는 로컬 산출물일 뿐이다.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="artifacts/*/ 에서 이 태그의 파일을 모은다")
    parser.add_argument("--version", required=True, help="v012 처럼")
    parser.add_argument("--config", required=True, help="폴더 이름에 들어가는 구성 이름")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--note", default="", help="README 첫 줄에 들어갈 한 줄 설명")
    parser.add_argument("--min-mut", type=int, default=DEFAULT_MIN_MUT)
    parser.add_argument("--out", type=Path, default=None,
                        help="기본은 artifacts/bundles/<version>_<config>_macroF1_<점수>")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log_path = ARTIFACTS / "logs" / f"{args.tag}.json"
    if not log_path.exists():
        raise SystemExit(f"로그가 없다: {log_path}")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    score = log.get("crossfit_macro_f1")

    out = args.out or (ARTIFACTS / "bundles" /
                       f"{args.version}_seed{args.seed}_{args.config}_macroF1_{score:.4f}")
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"이미 있다: {out}  (--overwrite 로 덮어쓴다)")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1) 팀 규격 파일 5종 — export_for_drive.py 를 그대로 부른다. 변환 로직이 두 벌이
    #    되면 팀에 올린 것과 여기 것이 언제 갈렸는지 알 수 없게 된다.
    argv = [sys.executable, str(ROOT / "scripts/export_for_drive.py"),
            "--oof", str(ARTIFACTS / "oof" / f"oof_{args.tag}.csv"),
            "--test", str(ARTIFACTS / "test_predictions" / f"test_{args.tag}.csv"),
            "--log", str(log_path), "--version", args.version, "--config", args.config,
            "--seed", str(args.seed), "--out", str(out), "--overwrite"]
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if done.returncode != 0:
        print(done.stdout[-2000:], done.stderr[-2000:])
        raise SystemExit("export_for_drive 실패")
    # export 가 자기 이름의 하위 폴더를 만들면 내용만 끌어올린다
    nested = [p for p in out.iterdir() if p.is_dir() and (p / "submission.csv").exists()]
    for folder in nested:
        for item in folder.iterdir():
            shutil.move(str(item), out / item.name)
        folder.rmdir()

    # 2) 짝 라벨 규칙. 전제가 깨졌으면 조용히 넘어가지 않고 멈춘다 — 규칙이 안 맞는
    #    데이터에 얹으면 맞던 행까지 틀리게 바꾼다.
    rule = build_pair_rule(RAW / "train.csv", RAW / "test.csv", min_mut=args.min_mut)
    broken = rule.verify_premises()
    if broken:
        raise SystemExit("짝 규칙 전제가 깨졌다:\n  " + "\n  ".join(broken))
    result = apply_to_submission(rule, out / "submission.csv", out / "submission_pairrule.csv")
    pair_summary = {"changed": int(result["n_changed"]), "total": int(result["rows"]),
                    "copied": int(result["n_was_copying_train_label"]),
                    "min_mut": args.min_mut}
    changed = pair_summary["changed"]

    # 3) 파켓 — csv 와 같은 내용. 다음 실험이 읽을 때 dtype 이 안 흔들린다.
    for name in ("oof", "test_pred"):
        frame = pd.read_csv(out / f"{name}.csv", encoding="utf-8-sig")
        frame.to_parquet(out / f"{name}.parquet", index=False)

    # 4) 원본 로그
    (out / "logs").mkdir(exist_ok=True)
    shutil.copy2(log_path, out / "logs" / log_path.name)

    # 5) README
    files = sorted((p.name, sha256(p)) for p in out.iterdir() if p.is_file())
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    (out / "README.md").write_text(
        build_readme(args, log, metrics, pair_summary, files), encoding="utf-8")

    print(f"묶음: {out}")
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out).as_posix():28s} {path.stat().st_size:>10,}B")
    print(f"\n짝 규칙으로 바뀐 행 {changed} / {pair_summary['total']}"
          f" (그중 train 라벨 복사 중이던 행 {pair_summary['copied']})")
    print("DACON 업로드는 사람이 직접 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
