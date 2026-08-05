"""출력 경로 전환과 실행 로그의 계약.

`notebooks/11_full_pipeline.ipynb` 가 이 둘 위에 서 있다. 경로 전환이 조용히 안 먹으면
새로 만든 파켓이 **기존 캐시를 덮어써서** 지금까지 낸 제출본의 근거가 사라진다.
되돌릴 방법이 없는 종류의 사고라 여기서 못 박는다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cancer_hack.paths import (  # noqa: E402
    ARTIFACTS_ENV,
    PROCESS_ENV,
    LazyDir,
    artifacts_dir,
    process_dir,
    raw_dir,
    reset_run_dirs,
    use_run_dirs,
)
from cancer_hack.runlog import RunLog  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env():
    """테스트가 서로의 환경변수를 물려받지 않게 한다."""
    saved = {k: os.environ.get(k) for k in (PROCESS_ENV, ARTIFACTS_ENV)}
    reset_run_dirs()
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_defaults_point_at_the_verification_baseline():
    assert process_dir() == PROJECT_ROOT / "data" / "process"
    assert artifacts_dir() == PROJECT_ROOT / "artifacts"


def test_use_run_dirs_moves_output_but_not_raw():
    """원본 csv 는 절대 안 움직인다 — 읽기 전용이다."""
    before_raw = raw_dir()
    dirs = use_run_dirs("unittag", create=False)
    assert dirs["process"].name == "process_unittag"
    assert dirs["artifacts"].parts[-2:] == ("runs", "unittag")
    assert raw_dir() == before_raw


def test_reset_returns_to_the_baseline():
    use_run_dirs("unittag", create=False)
    reset_run_dirs()
    assert process_dir() == PROJECT_ROOT / "data" / "process"


def test_run_tag_rejects_path_separators():
    """`../` 같은 값이 들어오면 엉뚱한 곳을 지운다."""
    for bad in ("../evil", "a/b", "with space", ""):
        with pytest.raises(ValueError):
            use_run_dirs(bad, create=False)


def test_lazy_dir_resolves_at_use_time_not_import_time():
    """스크립트가 import 시점에 상수로 굳히면 노트북의 전환이 안 먹는다."""
    lazy = LazyDir(process_dir)
    baseline = str(lazy / "train_folds.parquet")
    use_run_dirs("unittag", create=False)
    moved = str(lazy / "train_folds.parquet")
    assert baseline != moved
    assert "process_unittag" in moved


def test_train_gbdt_picks_up_the_moved_directory():
    """이미 import 된 모듈에도 반영되어야 한다 — 여기가 실제 사고 지점이다."""
    import train_gbdt

    use_run_dirs("unittag", create=False)
    assert "process_unittag" in str(train_gbdt.PROC_DIR / "x.parquet")
    assert "unittag" in str(train_gbdt.ARTIFACTS / "oof")
    reset_run_dirs()
    assert "process_unittag" not in str(train_gbdt.PROC_DIR / "x.parquet")


def test_runlog_writes_both_files(tmp_path: Path):
    log = RunLog(tmp_path, run_tag="t", config={"seed": 42})
    with log.step("한 단계"):
        log.record("oof", 0.5165)
    log.artifact("submission", tmp_path / "s.csv")
    paths = log.save()

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["run_tag"] == "t"
    assert payload["config"]["seed"] == 42
    assert payload["steps"][0]["status"] == "완료"
    assert payload["values"]["oof"] == 0.5165
    assert "scikit-learn" in payload["environment"]

    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "0.5165" in markdown and "한 단계" in markdown
    assert "DACON" in markdown


def test_runlog_records_a_failed_step_and_reraises(tmp_path: Path):
    """예외를 삼키면 안 된다 — 실패한 실행이 성공처럼 보이는 게 최악이다."""
    log = RunLog(tmp_path, run_tag="t")
    with pytest.raises(RuntimeError, match="터졌다"):
        with log.step("망가진 단계"):
            raise RuntimeError("터졌다")

    assert log.steps[0]["status"] == "실패"
    assert "RuntimeError" in log.steps[0]["error"]
    assert "실패한 단계" in log.summary()


def test_runlog_step_times_are_recorded(tmp_path: Path):
    log = RunLog(tmp_path, run_tag="t")
    with log.step("a"):
        pass
    assert isinstance(log.steps[0]["seconds"], float)
