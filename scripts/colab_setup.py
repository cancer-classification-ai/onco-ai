#!/usr/bin/env python
"""Colab(또는 새 환경)에서 이 저장소를 돌릴 수 있게 맞춰 준다.

노트북 첫 셀에 이 한 줄만 넣으면 된다.

    !python scripts/colab_setup.py

또는 파이썬에서:

    import sys; sys.path.insert(0, "scripts")
    import colab_setup; colab_setup.main()

## 무엇을 하나

1. **예측을 바꾸는 라이브러리 4종의 버전을 맞춘다.** `requirements.txt` 맨 위 주석의
   이유 그대로다 — 멤버마다 다른 기계에서 뽑은 OOF 를 한데 모아 블렌딩하는데,
   GBDT 3종과 scikit-learn 은 버전이 바뀌면 같은 seed·같은 파라미터로도 트리가
   달라진다. 각자의 CV 는 멀쩡해 보이는데 **합쳐 놓은 OOF 만 조용히 어긋난다.**
2. 나머지(numpy·pandas 등)는 **건드리지 않는다.** Colab 은 파이썬이 3.14 가 아니라
   `requirements.txt` 를 그대로 install 하면 휠을 못 찾고 런타임 스택이 먼저 깨진다.
3. 원본 csv 3종이 `data/raw/` 에 있는지 확인하고, 없으면 어디에 둬야 하는지 알려 준다.
   **데이터는 자동으로 받지 않는다** — 대회 데이터라 저장소에 없고, 외부에서 끌어오면
   규정 문제가 된다.
4. `import cancer_hack` 이 되게 `src` 를 `sys.path` 에 넣는다.

## 설치 후 런타임 재시작

pip 로 버전을 바꿨으면 **런타임을 재시작해야 실제로 반영된다.** 이 스크립트는 재시작이
필요한지 판정해서 알려 준다. 재시작 없이 이어서 돌리면 이전 버전이 메모리에 남아 있어
"버전은 맞다고 찍히는데 트리는 옛날 것" 인 상태가 된다 — 이게 제일 잡기 어려운 어긋남이다.

## DACON 제출

이 스크립트는 제출과 무관하다. 업로드는 사람이 직접 한다.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 예측을 바꾸는 것들. `requirements.txt` 와 **같은 값이어야 한다** —
#: `tests/test_colab_setup.py` 가 어긋나면 실패한다.
PINNED = {
    "scikit-learn": "1.9.0",
    "xgboost": "3.3.0",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
}

#: pip 이름 -> import 이름
IMPORT_NAME = {"scikit-learn": "sklearn"}

RAW_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def log(message: str) -> None:
    print(message, flush=True)


def installed_version(package: str) -> str | None:
    try:
        module = importlib.import_module(IMPORT_NAME.get(package, package))
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def check_versions() -> dict[str, tuple[str | None, str]]:
    """{패키지: (설치된 버전, 기대 버전)} — 어긋난 것만 담는다."""
    mismatched = {}
    for package, wanted in PINNED.items():
        found = installed_version(package)
        if found != wanted:
            mismatched[package] = (found, wanted)
    return mismatched


def install(packages: dict[str, tuple[str | None, str]]) -> None:
    specs = [f"{name}=={wanted}" for name, (_, wanted) in packages.items()]
    log(f"[install] {' '.join(specs)}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *specs], check=True)


def check_data() -> list[str]:
    raw = PROJECT_ROOT / "data" / "raw"
    return [name for name in RAW_FILES if not (raw / name).exists()]


def ensure_importable() -> None:
    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="설치하지 않고 진단만 한다")
    args = parser.parse_args(argv)

    log(f"파이썬 {sys.version.split()[0]}  ·  {PROJECT_ROOT}")

    mismatched = check_versions()
    if not mismatched:
        log("[ok] 예측을 바꾸는 라이브러리 4종이 전부 기준 버전이다")
    else:
        for name, (found, wanted) in mismatched.items():
            log(f"[diff] {name}: {found or '미설치'} -> {wanted}")
        if args.check_only:
            log("[check-only] 설치는 하지 않았다")
        else:
            install(mismatched)
            log("")
            log("=" * 72)
            log("  런타임을 재시작한 뒤 이 셀을 한 번 더 돌린다.")
            log("  재시작 없이 이어 가면 이전 버전이 메모리에 남아, 버전은 맞다고")
            log("  찍히는데 트리는 옛날 것인 상태가 된다.")
            log("=" * 72)
            return 1

    missing = check_data()
    if missing:
        log(f"[data] 원본 csv 가 없다: {', '.join(missing)}")
        log(f"       {PROJECT_ROOT / 'data' / 'raw'} 에 둔다.")
        log("       Colab 이면 Drive 를 마운트해 복사하는 쪽이 편하다:")
        log("         from google.colab import drive; drive.mount('/content/drive')")
        log("         !cp '/content/drive/MyDrive/<경로>/train.csv' data/raw/")
        log("       (대회 데이터라 저장소에 없다. 자동으로 받지 않는다)")
    else:
        log("[ok] 원본 csv 3종 확인")

    ensure_importable()
    try:
        import cancer_hack  # noqa: F401
        log("[ok] import cancer_hack")
    except ImportError as error:
        log(f"[fail] import cancer_hack: {error}")
        return 1

    fold_file = PROJECT_ROOT / "data" / "process" / "train_folds.parquet"
    if fold_file.exists():
        log("[ok] fold 파일 확인 — 팀과 같은 분할을 쓴다")
    else:
        log(f"[data] {fold_file.name} 이 없다. `python scripts/make_folds.py` 로 만든다.")
        log("       ※ 다른 분할로 뽑은 OOF 는 블렌딩에 넣지 않는다 — 팀원 Colab fold 가")
        log("         우리 것과 20% 밖에 안 겹쳐 스태킹을 접은 전례가 있다.")

    log("")
    log("준비 끝. 예:")
    log("  !python scripts/train_gbdt.py --model xgb --configs f16 --cv sgkf --topk 500 --tag colab")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
