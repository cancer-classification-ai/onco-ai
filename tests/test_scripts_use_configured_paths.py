"""스크립트가 출력 경로를 하드코딩하지 않는지 본다.

왜 필요한가
-----------
`use_run_dirs()` 로 출력 위치를 옮겨 놓고 `calibrate_ensemble.py` 를 돌렸더니, 그 스크립트만
`ARTIFACTS = PROJECT_ROOT / "artifacts"` 를 들고 있어서 **기준선 `artifacts/` 에 파일 10개를
써 버렸다.** 지금까지 낸 제출본의 근거가 있는 디렉터리다.

에러도 안 나고 로그도 정상이라, 다음 단계에서 "왜 파일이 없지" 하고 넘어졌을 때에야 알았다.
그래서 정적으로 막는다 — 새 스크립트가 같은 실수를 하면 여기서 걸린다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

#: `PROJECT_ROOT / "artifacts..."` · `PROJECT_ROOT / "data/process..."` 꼴.
HARDCODED = re.compile(
    r'PROJECT_ROOT\s*/\s*(?:f?")(?:artifacts|data/process)|'
    r'PROJECT_ROOT\s*/\s*"data"\s*/\s*"process"|'
    r'PROJECT_ROOT\s*/\s*"artifacts"'
)

#: 예외. 전부 **CLI 플래그로 출력 위치를 받는** 스크립트라 기본값이 저장소 경로여도 된다.
#: 예외를 늘릴 때는 그 스크립트가 정말 플래그를 받는지 확인하고 이유를 여기 적는다.
ALLOWED = {
    # --out 으로 내보낼 위치를 받는다. 팀 드라이브 공유용이라 실행 폴더와 무관하다.
    "export_for_drive.py",
    # Colab 부트스트랩. 저장소 레이아웃 자체를 점검하는 스크립트다.
    "colab_setup.py",
    # --out-dir 을 받는다. RF 계열은 자체 실험 폴더 규약을 쓴다.
    "run_rf_b_optuna.py",
    "tune_optuna_rf.py",
    "train_rf.py",
    # 검증 도구다. `artifacts/runs/<tag>` 와 `data/process_<tag>` 를 **입력으로 지정**하는
    # 게 목적이라 경로를 직접 만든다. 출력은 환경변수로 서브프로세스에 넘긴다.
    "verify_reproducible.py",
}


def _scripts() -> list[Path]:
    return sorted(p for p in SCRIPTS.glob("*.py") if p.stat().st_size > 0)


@pytest.mark.parametrize("path", _scripts(), ids=lambda p: p.name)
def test_script_does_not_hardcode_the_output_root(path: Path):
    if path.name in ALLOWED:
        pytest.skip(f"{path.name} 는 CLI 플래그로 출력 위치를 받는다")
    hits = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if HARDCODED.search(line)
    ]
    assert not hits, (
        f"{path.name} 이 출력 경로를 하드코딩한다 — `cancer_hack.paths` 를 쓴다.\n  "
        + "\n  ".join(hits)
    )


def test_the_allowlist_only_names_scripts_that_exist():
    """지운 스크립트가 예외 목록에 남아 있으면 다음 사람이 헷갈린다."""
    missing = [name for name in ALLOWED if not (SCRIPTS / name).exists()]
    assert not missing, f"예외 목록에 없는 파일이 있다: {missing}"
