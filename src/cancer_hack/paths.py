"""경로를 한 곳에서 정한다. 환경변수로 갈아끼울 수 있다.

왜 필요한가
-----------
`data/process/` 의 파켓과 `artifacts/` 의 예측 100여 개는 **지금까지 낸 제출본의 근거**다.
피처 코드가 바뀐 뒤에 그 자리에 새로 만들면 예전 것과 비교가 끊긴다
(`provenance.EXPECTED_FEATURE_FINGERPRINTS` 주석 참고).

그래서 새로 돌릴 때는 출력 위치를 옮긴다. 기존 디렉터리는 **재검증용으로 그대로 둔다.**

    ONCO_PROCESS_DIR    피처·fold parquet          기본 data/process
    ONCO_ARTIFACTS_DIR  oof·test·log·submission    기본 artifacts
    ONCO_RAW_DIR        원본 csv                   기본 data/raw

환경변수는 **읽는 시점에** 확인한다. 노트북이 셀 중간에 바꿔도 그다음 호출부터 반영된다.
경로를 모듈 상수로 캐시하면 import 순서에 따라 조용히 옛 값을 쓰게 된다.

노트북에서 쓰는 법
------------------
    from cancer_hack.paths import use_run_dirs
    dirs = use_run_dirs("nb11_s42")     # data/process_nb11_s42 · artifacts/runs/nb11_s42
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ENV = "ONCO_RAW_DIR"
PROCESS_ENV = "ONCO_PROCESS_DIR"
ARTIFACTS_ENV = "ONCO_ARTIFACTS_DIR"

#: `artifacts/` 안에서 실행 산출물이 들어가는 하위 폴더들.
ARTIFACT_SUBDIRS = ("oof", "test_predictions", "logs", "submissions", "models")


def _resolve(env: str, default: str) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser() if value else PROJECT_ROOT / default


def raw_dir() -> Path:
    """원본 csv. 여기는 **절대 덮어쓰지 않는다.**"""
    return _resolve(RAW_ENV, "data/raw")


def process_dir() -> Path:
    """피처·fold parquet."""
    return _resolve(PROCESS_ENV, "data/process")


def artifacts_dir() -> Path:
    """oof·test 예측·로그·제출."""
    return _resolve(ARTIFACTS_ENV, "artifacts")


class LazyDir(os.PathLike):
    """`PROC_DIR / "x.parquet"` 자리를 그대로 두면서 값만 **쓰는 시점에** 정한다.

    스크립트들이 `PROC_DIR = PROJECT_ROOT / "data/process"` 를 모듈 상수로 잡고 있었다.
    그대로 두면 노트북이 환경변수를 언제 세팅했느냐에 따라 조용히 옛 경로를 쓴다 —
    import 순서에 결과가 달리는 건 추적이 안 되는 종류의 버그다.

    Path 를 상속하지 않는다. 상속하면 `Path.__new__` 가 생성 시점에 문자열을 굳혀
    지연이 깨진다. 대신 실제로 쓰이는 연산만 위임한다.
    """

    __slots__ = ("_resolver",)

    def __init__(self, resolver):
        self._resolver = resolver

    def resolve_now(self) -> Path:
        return self._resolver()

    def __truediv__(self, other) -> Path:
        return self._resolver() / other

    def __fspath__(self) -> str:
        return str(self._resolver())

    def __str__(self) -> str:
        return str(self._resolver())

    def __repr__(self) -> str:
        return f"LazyDir({self._resolver()})"

    def __eq__(self, other) -> bool:
        return self._resolver() == other

    def __hash__(self) -> int:
        return hash(self._resolver())

    def __getattr__(self, name: str):
        """`.name`·`.parent`·`.exists()` 처럼 Path 에만 있는 것은 그때 풀어서 넘긴다.

        `Path` 는 불변이라 위임이 안전하다. 이 fallback 이 없으면 호출부마다 어떤
        속성을 쓰는지 미리 다 알아야 하는데, 그건 결국 하나씩 터지면서 알게 된다.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._resolver(), name)


#: 스크립트가 모듈 상수로 잡아 쓰는 자리용. `str()` 이나 `/` 를 만나야 값이 정해진다.
LAZY_RAW = LazyDir(raw_dir)
LAZY_PROCESS = LazyDir(process_dir)
LAZY_ARTIFACTS = LazyDir(artifacts_dir)


def use_run_dirs(run_tag: str, *, create: bool = True) -> dict[str, Path]:
    """이번 실행의 출력 위치를 정하고 환경변수에 심는다.

    원본 csv 는 건드리지 않는다. 피처와 산출물만 실행별로 갈라 둬서, 기존
    `data/process/`·`artifacts/` 는 재검증 기준선으로 남는다.

    돌려주는 dict 는 로그에 그대로 적어 두면 된다 — 나중에 "이 점수가 어느 파켓에서
    나왔나"를 되짚는 유일한 단서다.
    """
    if not run_tag or any(c in run_tag for c in r'\/:*?"<>| '):
        raise ValueError(f"run_tag 에 경로 구분자나 공백을 넣지 않는다: {run_tag!r}")

    process = PROJECT_ROOT / f"data/process_{run_tag}"
    artifacts = PROJECT_ROOT / f"artifacts/runs/{run_tag}"
    os.environ[PROCESS_ENV] = str(process)
    os.environ[ARTIFACTS_ENV] = str(artifacts)

    if create:
        process.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_SUBDIRS:
            (artifacts / name).mkdir(parents=True, exist_ok=True)

    return {"run_tag": run_tag, "raw": raw_dir(), "process": process, "artifacts": artifacts}


def reset_run_dirs() -> None:
    """기본 경로(`data/process`·`artifacts`)로 되돌린다 — 재검증할 때 쓴다."""
    os.environ.pop(PROCESS_ENV, None)
    os.environ.pop(ARTIFACTS_ENV, None)
