#!/usr/bin/env python
r"""결합 전략 매트릭스 — 같은 라이브러리 위에서 여러 앙상블 전략을 돌려 한 표에 모은다.

    # 1회차 — 전략 전부를 격리된 실행 폴더에서 돌린다
    .\.venv\Scripts\python.exe scripts\blend_matrix.py --run-tag blendmx1

    # 2회차 — 같은 조건으로 다시 돌려 산출물이 글자 하나까지 같은지 본다
    .\.venv\Scripts\python.exe scripts\blend_matrix.py --run-tag blendmx1 --verify

    # 표만 다시 본다 (재실행 없음)
    .\.venv\Scripts\python.exe scripts\blend_matrix.py --run-tag blendmx1 --report-only

## 무엇을 하는가

`greedy_blend.py` · `train_meta.py` · `calibrate_ensemble.py` 를 **그대로 서브프로세스로
돌린다.** 결합 로직을 새로 쓰지 않는다 — 이미 검증된 그 코드가 답을 내고, 이 스크립트는
조건을 바꿔 가며 부르고 결과를 모으는 일만 한다. 로직이 두 벌이 되면 여기 표의 숫자와
실제 제출본이 언제 갈렸는지 알 수 없게 된다.

## 왜 표로 만드는가

CV 를 올린 시도는 LB 에서 거의 전부 떨어졌다(`docs/pipeline_overview.md` §1). "가장 높은
교차적합 점수" 하나를 고르는 건 그래서 근거가 약하다. 대신 네 가지를 같이 잰다.

    교차적합      fold 를 뺀 나머지에서 결합을 정하고 그 fold 에서만 잰 값 — 보고할 값
    낙관 격차     전체 적합 − 교차적합. 결합이 선택에 쓴 행을 얼마나 외웠는지
    fold 편차     fold 5개 점수의 표준편차. 한 fold 가 떠받치는 구성인지
    선택 잡음     같은 전략을 `--random-state` 만 바꿔 돌렸을 때의 흔들림

넷째가 기준선이다. **전략 간 차이가 선택 잡음보다 작으면 그 순위에는 읽을 것이 없다.**

## 라이브러리를 왜 고정하는가

그리디 결과는 후보가 무엇이었느냐에 통째로 달려 있는데 `artifacts/oof/` 는 실험할 때마다
늘어난다. 그래서 예전 실행의 로그에서 목록을 그대로 가져와(`--library-log`) 모든 전략이
**같은 후보군**을 보게 한다. 전략끼리 비교가 되려면 이게 먼저다.

## 왜 격리 폴더인가

기준선 `artifacts/` 에는 지금까지 낸 제출본의 근거가 들어 있다. 여기에 전략 수십 개의
산출물을 쏟아 부으면 다음 사람이 무엇이 무엇인지 못 가린다. `cancer_hack.paths.use_run_dirs`
로 `artifacts/runs/<tag>/` 를 따로 잡고, 입력 라이브러리는 **하드링크**로 깐다 —
복사가 아니라서 디스크를 더 쓰지 않고 내용이 같다는 게 파일 시스템 수준에서 보장된다.

## DACON 제출

제출 후보 csv 를 로컬에 만들 뿐 어디에도 올리지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.metrics import macro_f1  # noqa: E402
from cancer_hack.paths import (  # noqa: E402
    artifacts_dir,
    process_dir,
    raw_dir,
    reset_run_dirs,
    use_run_dirs,
)
from cancer_hack.provenance import check_fold_fingerprint  # noqa: E402

SCRIPTS = PROJECT_ROOT / "scripts"

# --- 라이브러리 변형 -------------------------------------------------------
#
# 그리디의 과적합은 라운드 수보다 **후보 수**에서 온다. 98개 중 63개가 xgb 라
# 서로 거의 같은 예측인데, 그중 어느 것을 담느냐는 상당 부분 fold 의 우연이다.
# 그래서 후보를 줄이는 축을 점수 축과 나란히 둔다.

#: LB 로 검증된 수동 3인방(`0.45/0.45/0.10`). 순서가 곧 가중치 순서다.
TRIO_PATTERNS = (
    r"^xgb_repo16_f16_group5_.*_s42$",
    r"^catboost_repo16_f16_group5_.*_s42$",
    r"^rf_repo16_f16_group5_.*_s42$",
)


def _family(name: str) -> str:
    """멤버 이름에서 모델 계열을 뽑는다. `dl_hybrid_...` 는 두 토큰까지 본다."""
    return "dl_" + name.split("_")[1] if name.startswith("dl_") else name.split("_")[0]


def _top(n: int):
    return lambda profile: [row["name"] for row in profile[:n]]


def _per_family(n: int, *, drop: str | None = None):
    """계열마다 단독 점수 상위 n 개. `drop` 에 걸리는 이름은 후보에서 먼저 뺀다.

    `drop` 이 필요한 이유 — 계열 상한만 걸면 catboost 세 자리가 전부 `cbopt10` 로 찬다.
    OOF 는 그게 가장 높지만 LB 는 −0.0084 였다(EXP_039·041). 단독 점수로 고르는 규칙이
    LB 에서 이미 기각된 구성을 자동으로 끌어올리는 셈이라, 뺀 판을 나란히 잰다.
    """
    pattern = re.compile(drop) if drop else None

    def pick(profile):
        seen: dict[str, int] = {}
        out = []
        for row in profile:
            if pattern is not None and pattern.search(row["name"]):
                continue
            fam = row["family"]
            if seen.get(fam, 0) >= n:
                continue
            seen[fam] = seen.get(fam, 0) + 1
            out.append(row["name"])
        return out

    return pick


def _matching(patterns):
    def pick(profile):
        names = [row["name"] for row in profile]
        out = []
        for pattern in patterns:
            hits = [n for n in names if re.match(pattern, n)]
            if len(hits) != 1:
                raise SystemExit(
                    f"패턴 {pattern!r} 에 맞는 멤버가 {len(hits)}개다 — 1개여야 한다."
                    + (f" 예: {hits[:3]}" if hits else "")
                )
            out.append(hits[0])
        return out

    return pick


#: 이름 -> 프로필(단독 점수 내림차순)에서 멤버를 고르는 함수.
LIBRARIES = {
    "full": lambda profile: [row["name"] for row in profile],
    "top40": _top(40),
    "top20": _top(20),
    "fam5": _per_family(5),
    "fam3": _per_family(3),
    "fam2": _per_family(2),
    "fam1": _per_family(1),
    "fam3nocb": _per_family(3, drop=r"cbopt10"),
    "fullnocb": lambda profile: [row["name"] for row in profile if "cbopt10" not in row["name"]],
    "trio": _matching(TRIO_PATTERNS),
}

# --- 전략 -----------------------------------------------------------------
#
# `kind` 가 어느 스크립트를 부르는지 정한다. 나머지 키는 그 스크립트의 플래그다.
# 한 번에 한 축만 움직인다 — 두 축을 같이 바꾸면 어느 쪽이 효과인지 못 가린다.

GREEDY_BASE = {"n_rounds": 30, "bag_fraction": 0.6, "bag_rounds": 5, "random_state": 0}

STRATEGIES: list[dict] = [
    # 기준선 — 지금까지 나온 최고 교차적합(0.5337)을 그대로 재현한다.
    {"tag": "g_full_r30", "kind": "greedy", "library": "full"},
    # 축 1: 복잡도 상한. 라운드가 곧 파라미터 수다.
    {"tag": "g_full_r10", "kind": "greedy", "library": "full", "n_rounds": 10},
    {"tag": "g_full_r60", "kind": "greedy", "library": "full", "n_rounds": 60},
    # 축 2: bagging. 라운드마다 후보를 일부만 보여 우연히 좋아 보이는 멤버를 막는다.
    {"tag": "g_full_nobag", "kind": "greedy", "library": "full",
     "bag_fraction": 1.0, "bag_rounds": 1},
    {"tag": "g_full_bag03", "kind": "greedy", "library": "full",
     "bag_fraction": 0.3, "bag_rounds": 10},
    # 축 3: 후보 수. 여기가 그리디 과적합의 진짜 손잡이다.
    {"tag": "g_top40_r30", "kind": "greedy", "library": "top40"},
    {"tag": "g_top20_r30", "kind": "greedy", "library": "top20"},
    {"tag": "g_fam5_r30", "kind": "greedy", "library": "fam5"},
    {"tag": "g_fam3_r30", "kind": "greedy", "library": "fam3"},
    # 축 4: 선택 잡음. 나머지 조건을 고정하고 난수만 바꾼다. 이 셋의 폭이
    #       위 축들의 차이를 읽을 수 있는지 없는지를 정한다.
    {"tag": "g_full_rs1", "kind": "greedy", "library": "full", "random_state": 1},
    {"tag": "g_full_rs2", "kind": "greedy", "library": "full", "random_state": 2},
    {"tag": "g_full_rs3", "kind": "greedy", "library": "full", "random_state": 3},
    # 스태킹 — 클래스별로 다른 가중치를 배운다. 멤버를 계열별 최고 1개로 줄여
    # 파라미터를 억제하고, 정규화 세기(C)를 사다리로 잰다.
    {"tag": "s_fam1_c001", "kind": "meta", "library": "fam1", "meta": "logistic",
     "C": 0.01, "transform": "prob"},
    {"tag": "s_fam1_c01", "kind": "meta", "library": "fam1", "meta": "logistic",
     "C": 0.1, "transform": "prob"},
    {"tag": "s_fam1_c1", "kind": "meta", "library": "fam1", "meta": "logistic",
     "C": 1.0, "transform": "prob"},
    {"tag": "s_fam1_c1_logit", "kind": "meta", "library": "fam1", "meta": "logistic",
     "C": 1.0, "transform": "logit"},
    {"tag": "s_fam1_ridge", "kind": "meta", "library": "fam1", "meta": "ridge",
     "C": 1.0, "transform": "prob"},
    {"tag": "s_fam3_c01", "kind": "meta", "library": "fam3", "meta": "logistic",
     "C": 0.1, "transform": "prob"},
    # 고정/학습 가중 + 로짓 보정 — LB 로 검증된 경로다. 비교의 바닥.
    {"tag": "c_trio_fixed", "kind": "calib", "library": "trio",
     "weights": [0.45, 0.45, 0.10]},
    {"tag": "c_trio_learned", "kind": "calib", "library": "trio"},
    {"tag": "c_fam1_learned", "kind": "calib", "library": "fam1"},
    # 축 5: 계열 상한을 더 조인다. 후보 수와 점수의 관계를 두 점 더 찍는다.
    {"tag": "g_fam2_r30", "kind": "greedy", "library": "fam2"},
    {"tag": "g_fam3_r10", "kind": "greedy", "library": "fam3", "n_rounds": 10},
    # 축 6: LB 가 기각한 cbopt10 을 뺀 판. 단독 점수로 고르는 규칙이 그 구성을
    #       자동으로 끌어올리기 때문에, 뺐을 때 얼마나 잃는지 재 둔다.
    {"tag": "g_fam3nocb_r30", "kind": "greedy", "library": "fam3nocb"},
    # 축 7: 작은 라이브러리의 선택 잡음. 후보가 15개면 그리디가 덜 흔들릴 텐데,
    #       그렇다면 fam3 계열의 차이는 full 의 잡음 폭이 아니라 이쪽으로 읽어야 한다.
    {"tag": "g_fam3_rs1", "kind": "greedy", "library": "fam3", "random_state": 1},
    {"tag": "g_fam3_rs2", "kind": "greedy", "library": "fam3", "random_state": 2},
    {"tag": "g_fam3_rs3", "kind": "greedy", "library": "fam3", "random_state": 3},
    # 2단계 — 그리디로 고른 블렌드 위에 로짓 보정을 얹는다. 그리디에는 보정이 없고,
    # 보정은 LB 0.3896 경로의 구성 요소였다. 둘을 잇는 건 아직 안 해 본 조합이다.
    # 주의: 그리디 OOF 는 이미 fold 교차적합 산물이라 그 위의 보정 점수는 약간
    # 낙관적이다. 순위를 뒤집을 크기는 아니지만 값 자체를 LB 예측으로 읽지 않는다.
    # 축 8: 팀이 이미 "쓰지 않는다"고 정한 cbopt10 을 전체 라이브러리에서도 뺀다.
    #       `Models/pairrule_candidates/README.md` 가 근거다 — CV 는 그쪽이 높은데
    #       LB 는 두 번 다 낮았다. 규칙을 지킨 판의 값을 알아야 고를 수 있다.
    {"tag": "g_fullnocb_r30", "kind": "greedy", "library": "fullnocb"},
    {"tag": "g_fullnocb_rs1", "kind": "greedy", "library": "fullnocb", "random_state": 1},
    {"tag": "g_fullnocb_rs2", "kind": "greedy", "library": "fullnocb", "random_state": 2},
    # 축 9: bagging 횟수를 5 -> 25 로. `counts_` 는 bag 별 독립 실행의 합이라 회수를
    #       늘리면 큰 수의 법칙으로 난수 의존이 줄어든다. 점수를 올리려는 게 아니라
    #       **잡음 폭을 줄이려는** 것이다. 폭이 줄면 최종 제출을 임의의 seed 하나에
    #       맡기지 않아도 된다.
    {"tag": "g_full_bag25", "kind": "greedy", "library": "full", "bag_rounds": 25},
    {"tag": "g_full_bag25_rs1", "kind": "greedy", "library": "full",
     "bag_rounds": 25, "random_state": 1},
    {"tag": "g_fam3_bag25", "kind": "greedy", "library": "fam3", "bag_rounds": 25},
    {"tag": "g_fam3_bag25_rs1", "kind": "greedy", "library": "fam3",
     "bag_rounds": 25, "random_state": 1},
    {"tag": "g_fam3_bag25_rs2", "kind": "greedy", "library": "fam3",
     "bag_rounds": 25, "random_state": 2},
    {"tag": "x_cal_g_fam3", "kind": "calib2", "source": "g_fam3_r30", "stage": 2},
    {"tag": "x_cal_g_full", "kind": "calib2", "source": "g_full_r30", "stage": 2},
    {"tag": "x_cal_g_fam3nocb", "kind": "calib2", "source": "g_fam3nocb_r30", "stage": 2},
]


def log(message: str = "") -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- 라이브러리 프로필 -----------------------------------------------------


def build_profile(base: Path, members: list[str], cache: Path) -> list[dict]:
    """멤버마다 단독 macro F1 을 재고 내림차순으로 정렬한다.

    동점일 때 이름으로 한 번 더 정렬한다 — 안 하면 파일 시스템 순서가 결과에 새고,
    같은 명령이 다른 라이브러리를 만들어 재현이 깨진다.
    """
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    labels = pd.read_csv(raw_dir() / "train.csv", usecols=["ID", "SUBCLASS"])
    labels = labels.set_index("ID")["SUBCLASS"]
    classes = sorted(labels.unique())
    columns = [f"p_{c}" for c in classes]
    folds = pd.read_parquet(process_dir() / "train_folds.parquet")
    ids = folds["ID"].to_numpy()
    y = labels.reindex(ids).astype(str).to_numpy()
    class_array = np.asarray(classes)

    rows = []
    for name in members:
        frame = pd.read_csv(base / "oof" / f"oof_{name}.csv")
        values = np.asarray(frame[columns], dtype=float)
        rows.append({
            "name": name,
            "family": _family(name),
            "solo": float(macro_f1(y, class_array[values.argmax(axis=1)])),
        })
    rows.sort(key=lambda row: (-row["solo"], row["name"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def stage(base: Path, target: Path, members: list[str]) -> list[str]:
    """라이브러리 입력을 실행 폴더에 하드링크로 깐다. 실패하면 복사한다."""
    staged = []
    for name in members:
        for sub, prefix in (("oof", "oof_"), ("test_predictions", "test_")):
            source = base / sub / f"{prefix}{name}.csv"
            destination = target / sub / f"{prefix}{name}.csv"
            if not source.exists():
                raise SystemExit(f"라이브러리 멤버의 {sub} 가 없다: {source}")
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            staged.append(str(destination.relative_to(target)).replace("\\", "/"))
    return staged


# --- 실행 -----------------------------------------------------------------


def argv_for(strategy: dict, run: Path, libraries: dict[str, list[str]],
             cv: str, fold_column: str) -> list[str]:
    """전략 하나를 기존 스크립트의 명령줄로 옮긴다."""
    kind, tag = strategy["kind"], strategy["tag"]
    python = [sys.executable]

    if kind == "calib2":
        # 앞 단계가 만든 블렌드 하나를 멤버로 받아 로짓 보정만 얹는다.
        source = strategy["source"]
        return python + [
            str(SCRIPTS / "calibrate_ensemble.py"),
            "--oof", str(run / "oof" / f"oof_{source}.csv"),
            "--test", str(run / "test_predictions" / f"test_{source}.csv"),
            "--fold-column", fold_column, "--tag", tag,
            "--submission", "--overwrite",
        ]

    members = libraries[strategy["library"]]

    if kind == "greedy":
        options = {**GREEDY_BASE, **{k: v for k, v in strategy.items() if k in GREEDY_BASE}}
        return python + [
            str(SCRIPTS / "greedy_blend.py"),
            "--cv", cv,
            "--members-file", str(run / "logs" / f"library_{strategy['library']}.json"),
            "--n-rounds", str(options["n_rounds"]),
            "--bag-fraction", str(options["bag_fraction"]),
            "--bag-rounds", str(options["bag_rounds"]),
            "--random-state", str(options["random_state"]),
            "--tag", tag, "--top", "10",
        ]

    oof = [str(run / "oof" / f"oof_{name}.csv") for name in members]
    test = [str(run / "test_predictions" / f"test_{name}.csv") for name in members]

    if kind == "meta":
        return python + [
            str(SCRIPTS / "train_meta.py"),
            "--oof", *oof, "--test", *test,
            "--fold-column", fold_column, "--tag", tag,
            "--meta", strategy["meta"], "--C", str(strategy["C"]),
            "--transform", strategy["transform"],
            "--submission", "--overwrite",
        ]

    if kind == "calib":
        argv = python + [
            str(SCRIPTS / "calibrate_ensemble.py"),
            "--oof", *oof, "--test", *test,
            "--fold-column", fold_column, "--tag", tag,
            "--submission", "--overwrite",
        ]
        if "weights" in strategy:
            argv += ["--weights", *[str(w) for w in strategy["weights"]]]
        return argv

    raise SystemExit(f"모르는 kind {kind!r}")


def run_one(strategy: dict, run: Path, libraries: dict, cv: str, fold_column: str) -> dict:
    argv = argv_for(strategy, run, libraries, cv, fold_column)
    env = dict(os.environ)
    env["ONCO_ARTIFACTS_DIR"] = str(run)     # 출력만 옮긴다. 원본 csv·fold 는 기준선 그대로.
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=PROJECT_ROOT, env=env)
    return {
        "tag": strategy["tag"],
        "returncode": done.returncode,
        "stdout": done.stdout or "",
        "stderr": done.stderr or "",
        "argv": argv,
    }


def run_matrix(strategies: list[dict], run: Path, libraries: dict, cv: str,
               fold_column: str, jobs: int) -> None:
    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_one, s, run, libraries, cv, fold_column): s["tag"]
            for s in strategies
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            mark = "ok  " if result["returncode"] == 0 else "실패"
            log(f"  [{index:>2d}/{len(strategies)}] {mark} {result['tag']}")
            if result["returncode"] != 0:
                failures.append(result["tag"])
                log((result["stdout"] or "")[-1500:])
                log((result["stderr"] or "")[-1500:])
    if failures:
        raise SystemExit(f"전략 {len(failures)}개가 실패했다: {failures}")


# --- 리포트 ---------------------------------------------------------------


def read_scores(run: Path, strategy: dict) -> dict:
    """스크립트마다 다른 로그 형식에서 같은 네 값을 뽑는다."""
    payload = json.loads((run / "logs" / f"{strategy['tag']}.json").read_text(encoding="utf-8"))
    kind = strategy["kind"]

    if kind == "greedy":
        return {
            "crossfit": payload["crossfit_macro_f1"],
            "fullfit": payload["full_fit_macro_f1"]["value"],
            "folds": payload["crossfit_fold_macro_f1"],
            "n_members": payload["library_size"],
            "n_selected": len(payload["selected"]),
        }
    if kind == "meta":
        return {
            "crossfit": payload["crossfit_stacked"]["macro_f1"],
            "fullfit": payload["full_oof_fit"]["macro_f1"],
            "folds": [row["macro_f1"] for row in payload["crossfit_folds"]],
            "n_members": len(payload["oof_sources"]),
            "n_selected": len(payload["oof_sources"]),
        }
    return {
        "crossfit": payload["crossfit_calibrated"]["macro_f1"],
        "fullfit": payload["final_full_oof_fit"]["macro_f1"],
        "folds": [row["calibrated_macro_f1"] for row in payload["crossfit_folds"]],
        "n_members": len(payload["oof_sources"]),
        "n_selected": len(payload["oof_sources"]),
        "raw_blend": payload["crossfit_raw_blend"]["macro_f1"],
        "uniform": payload["uniform_blend_oof_macro_f1"],
    }


#: `--random-state` 만 다른 형제들. 이 묶음의 폭이 그 라이브러리에서 읽을 수 있는
#: 차이의 하한이다. 전략을 늘릴 때 형제도 같이 늘린다.
NOISE_GROUPS = {
    "full": ("g_full_r30", "g_full_rs1", "g_full_rs2", "g_full_rs3"),
    "fam3": ("g_fam3_r30", "g_fam3_rs1", "g_fam3_rs2", "g_fam3_rs3"),
    "fullnocb": ("g_fullnocb_r30", "g_fullnocb_rs1", "g_fullnocb_rs2"),
    "full+bag25": ("g_full_bag25", "g_full_bag25_rs1"),
    "fam3+bag25": ("g_fam3_bag25", "g_fam3_bag25_rs1", "g_fam3_bag25_rs2"),
}


def report(run: Path, strategies: list[dict]) -> dict:
    rows = []
    for strategy in strategies:
        scores = read_scores(run, strategy)
        scores.update({
            "tag": strategy["tag"],
            "kind": strategy["kind"],
            "library": strategy.get("library", f"<-{strategy.get('source', '')}"),
            "gap": scores["fullfit"] - scores["crossfit"],
            "fold_std": float(np.std(scores["folds"], ddof=1)),
            "fold_min": float(min(scores["folds"])),
        })
        rows.append(scores)

    by_tag = {row["tag"]: row for row in rows}
    noise = {}
    for name, tags in NOISE_GROUPS.items():
        values = [by_tag[t]["crossfit"] for t in tags if t in by_tag]
        if len(values) > 1:
            noise[name] = {"values": values, "range": max(values) - min(values)}

    log()
    log(f"{'전략':18s} {'종류':7s} {'후보':>5s} {'선택':>5s} {'교차적합':>10s} "
        f"{'낙관격차':>9s} {'fold편차':>9s} {'최저fold':>9s}")
    log("-" * 88)
    for row in sorted(rows, key=lambda r: -r["crossfit"]):
        log(f"{row['tag']:18s} {row['kind']:7s} {row['n_members']:>5d} "
            f"{row['n_selected']:>5d} {row['crossfit']:>10.4f} {row['gap']:>+9.4f} "
            f"{row['fold_std']:>9.4f} {row['fold_min']:>9.4f}")

    log()
    for name, item in noise.items():
        log(f"선택 잡음 [{name}] — random_state 만 바꾼 {len(item['values'])}회: "
            f"{min(item['values']):.4f} ~ {max(item['values']):.4f}  (폭 {item['range']:.4f})")
    if noise:
        log("  이 폭보다 작은 전략 간 차이는 순위로 읽지 않는다.")

    payload = {
        "run": str(run),
        "rows": rows,
        "selection_noise": noise,
    }
    out = run / "logs" / "blend_matrix_report.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n리포트 {out}")
    return payload


# --- 재현성 대조 ------------------------------------------------------------


def _normalize_paths(text: str, source: Path, target: Path) -> str:
    """json 본문에 박힌 실행 폴더 경로를 대조 대상 폴더 이름으로 바꾼다.

    세 표기를 전부 처리한다 — 원문(`D:\\a\\b`), json 이스케이프(`D:\\\\a\\\\b`),
    슬래시(`D:/a/b`). 첫 판에서는 이스케이프 표기를 빠뜨려 로그 12개가 "다름"으로
    잡혔는데, 실제 차이는 그 문자열 하나뿐이었다.
    """
    for form in (str, lambda p: json.dumps(str(p))[1:-1], lambda p: str(p).replace("\\", "/")):
        text = text.replace(form(source), form(target))
    return text


def _json_key_diff(left: Path, right: Path) -> str:
    """어느 키가 갈렸는지 붙여 준다. "다름" 만 찍으면 원인을 다시 찾아야 한다."""
    if not left.name.endswith(".json"):
        return ""
    try:
        a = json.loads(left.read_text(encoding="utf-8"))
        b = json.loads(right.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return ""
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return f"  (키: {', '.join(keys)})" if keys else "  (키 차이 없음 — 표기만 다름)"


def compare_runs(first: Path, second: Path, staged: set[str]) -> bool:
    """두 실행이 만든 파일을 sha256 으로 맞대 본다.

    로그 json 에는 입력 파일의 **절대 경로**가 적힌다. 실행 폴더 이름이 다르면 그
    문자열만으로 sha256 이 갈리므로, 원문 그대로와 폴더 이름을 맞춘 뒤 둘 다 잰다.
    csv 는 경로를 담지 않아 원문 그대로 같아야 한다.
    """
    skip = staged | {"logs/blend_matrix_report.json", "logs/library_solo.json"}
    produced = sorted(
        str(p.relative_to(first)).replace("\\", "/")
        for p in first.rglob("*") if p.is_file()
    )
    produced = [name for name in produced if name not in skip]

    same_raw, same_norm, missing, differing = 0, 0, [], []
    for name in produced:
        left, right = first / name, second / name
        if not right.exists():
            missing.append(name)
            continue
        if sha256(left) == sha256(right):
            same_raw += 1
            same_norm += 1
            continue
        if name.endswith(".json"):
            # 텍스트끼리 견준다. 파일은 개행이 `\r\n` 인데 `read_text` 는 `\n` 으로 읽어
            # 오므로, 정규화한 문자열의 해시를 원본 **파일** 해시와 맞대면 늘 어긋난다.
            try:
                left_text = left.read_text(encoding="utf-8")
                right_text = _normalize_paths(right.read_text(encoding="utf-8"), second, first)
            except (OSError, UnicodeDecodeError):
                left_text = right_text = None
            if left_text is not None and left_text == right_text:
                same_norm += 1
                continue
        differing.append(name)

    log()
    log(f"산출물 {len(produced)}개 대조")
    log(f"  원문 그대로 일치      {same_raw}")
    log(f"  경로 정규화 후 일치   {same_norm}")
    log(f"  다름                  {len(differing)}")
    log(f"  2회차에 없음          {len(missing)}")
    for name in differing[:10]:
        log(f"    다름: {name}{_json_key_diff(first / name, second / name)}")
    for name in missing[:10]:
        log(f"    없음: {name}")

    csv_ok = all(not name.endswith(".csv") for name in differing + missing)
    log()
    log("csv(oof·test·submission)는 전부 바이트 단위로 같다" if csv_ok
        else "csv 가 갈렸다 — 위 목록을 본다")
    return not differing and not missing


# --- 진입점 ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", required=True, help="출력 폴더 이름 (artifacts/runs/<tag>)")
    parser.add_argument("--cv", default="group5", choices=["group5", "skf5"])
    parser.add_argument("--library-log", default="greedy_repro98.json",
                        help="기준선 logs/ 안의 json. `library` 키를 후보 목록으로 읽는다")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--only", nargs="+", default=None, help="이 태그들만 돌린다")
    parser.add_argument("--verify", action="store_true",
                        help="같은 조건으로 2회차를 돌려 산출물을 sha256 으로 대조한다")
    parser.add_argument("--skip-first", action="store_true",
                        help="1회차가 이미 있을 때 --verify 와 같이 준다. 2회차만 돌려 대조한다")
    parser.add_argument("--report-only", action="store_true", help="재실행 없이 표만")
    parser.add_argument("--compare-only", action="store_true",
                        help="재실행 없이 1회차와 <tag>_verify 산출물만 대조한다")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    fold_column = f"fold_{args.cv}"

    reset_run_dirs()
    base = artifacts_dir()
    folds_path = process_dir() / "train_folds.parquet"

    # fold 파켓이 예측을 만들 때와 같은 분할인지 먼저 본다. 다르면 라이브러리의
    # OOF 를 섞는 것 자체가 무의미하다.
    fingerprint = check_fold_fingerprint(folds_path)
    log(f"fold 지문 {fingerprint}  ({folds_path})")

    library_log = base / "logs" / args.library_log
    if not library_log.exists():
        raise SystemExit(f"라이브러리 로그가 없다: {library_log}")
    members = json.loads(library_log.read_text(encoding="utf-8"))["library"]
    log(f"고정 라이브러리 {len(members)}개  ({library_log.name})")

    strategies = STRATEGIES
    if args.only:
        wanted = set(args.only)
        strategies = [s for s in STRATEGIES if s["tag"] in wanted]
        unknown = wanted - {s["tag"] for s in STRATEGIES}
        if unknown:
            raise SystemExit(f"모르는 태그: {sorted(unknown)}")

    run = Path(use_run_dirs(args.run_tag)["artifacts"])
    reset_run_dirs()

    profile = build_profile(base, members, run / "logs" / "library_solo.json")
    libraries = {name: pick(profile) for name, pick in LIBRARIES.items()}
    needed = sorted({s["library"] for s in strategies if "library" in s})
    for name in needed:
        manifest = run / "logs" / f"library_{name}.json"
        manifest.write_text(
            json.dumps({"library": libraries[name]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"  라이브러리 {name:6s} {len(libraries[name]):>3d}개")

    staged = set(stage(base, run, sorted({m for n in needed for m in libraries[n]})))

    if args.compare_only:
        second = Path(use_run_dirs(f"{args.run_tag}_verify")["artifacts"])
        reset_run_dirs()
        return 0 if compare_runs(run, second, staged) else 1

    if args.report_only:
        report(run, strategies)
        return 0

    def run_all(target: Path, label: str) -> None:
        """1단계를 먼저 다 끝내고 2단계를 돌린다. 2단계는 1단계 산출물을 입력으로 받는다."""
        for stage in (1, 2):
            batch = [s for s in strategies if s.get("stage", 1) == stage]
            if not batch:
                continue
            log(f"\n{label} {stage}단계 — 전략 {len(batch)}개 (동시 {args.jobs})")
            run_matrix(batch, target, libraries, args.cv, fold_column, args.jobs)

    if not args.skip_first:
        run_all(run, "1회차")
    else:
        log("\n1회차는 건너뛴다 — 이미 있는 산출물을 기준으로 삼는다")
    payload = report(run, strategies)

    if args.verify:
        second = Path(use_run_dirs(f"{args.run_tag}_verify")["artifacts"])
        reset_run_dirs()
        shutil.copy2(run / "logs" / "library_solo.json", second / "logs" / "library_solo.json")
        for name in needed:
            shutil.copy2(run / "logs" / f"library_{name}.json",
                         second / "logs" / f"library_{name}.json")
        stage(base, second, sorted({m for n in needed for m in libraries[n]}))
        run_all(second, "2회차")
        identical = compare_runs(run, second, staged)
        payload["verify"] = {"second_run": str(second), "identical": identical}
        (run / "logs" / "blend_matrix_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not identical:
            return 1

    log("\n로컬 파일만 만들었다. DACON 업로드는 사람이 직접 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
