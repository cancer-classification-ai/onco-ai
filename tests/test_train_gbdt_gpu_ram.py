"""`--gpu-ram-part auto` 의 계약.

이 GPU 는 화면 출력도 함께 한다(`display_active: Enabled`, 연결 프로세스 31개).
그래서 CatBoost 가 잡을 몫을 고정 비율로 박으면 두 가지 중 하나가 된다 — 낮으면
평소에 손해고, 높으면 브라우저 스파이크에 OOM 이 난다. `fit_with_fallback` 은 그때
CPU 로 폴백하는데, 그러면 그 fold 만 느려지는 게 아니라 시간 비교가 통째로 깨진다.

그래서 실행 시점의 여유에서 정한다. 여기서는 그 산식의 경계만 고정한다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("train_gbdt_gpu", PROJECT_ROOT / "scripts" / "train_gbdt.py")
tg = importlib.util.module_from_spec(_spec)
sys.modules["train_gbdt_gpu"] = tg
_spec.loader.exec_module(tg)


def test_explicit_number_is_passed_through():
    """숫자를 주면 자동 산정이 끼어들면 안 된다 — 재현 실행에서 값이 흔들린다."""
    assert tg.resolve_gpu_ram_part(0.4) == 0.4
    assert tg.resolve_gpu_ram_part("0.55") == 0.55


def test_auto_stays_inside_the_bounds(monkeypatch):
    """상한을 넘으면 데스크톱 몫까지 먹고, 하한 아래면 기존 고정값보다 손해다."""
    for free in (0, 1000, 6000, 12000, 999999):
        monkeypatch.setattr(tg.subprocess, "run", _fake_smi(free, 12227))
        part = tg.resolve_gpu_ram_part("auto")
        assert tg.GPU_RAM_PART_MIN <= part <= tg.GPU_RAM_PART_MAX


def test_auto_reserves_memory_for_the_desktop(monkeypatch):
    """예약분을 빼지 않으면 여유를 전부 잡아 스파이크에 죽는다."""
    total = 12227
    monkeypatch.setattr(tg.subprocess, "run", _fake_smi(total, total))
    part = tg.resolve_gpu_ram_part("auto")
    assert part * total <= total - tg.GPU_RESERVE_MIB + 1


def test_auto_falls_back_to_the_floor_when_smi_is_unavailable(monkeypatch):
    """조용히 크게 잡는 것보다 하한으로 떨어지는 쪽이 안전하다."""
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi 없음")
    monkeypatch.setattr(tg.subprocess, "run", boom)
    assert tg.resolve_gpu_ram_part("auto") == tg.GPU_RAM_PART_MIN


def test_auto_falls_back_on_unparsable_output(monkeypatch):
    monkeypatch.setattr(tg.subprocess, "run", _fake_raw("이건 숫자가 아니다"))
    assert tg.resolve_gpu_ram_part("auto") == tg.GPU_RAM_PART_MIN


def test_cli_default_is_auto():
    assert tg.build_parser().parse_args([]).gpu_ram_part == "auto"


def test_reserve_is_large_enough_for_a_browser_spike():
    """탭 하나가 1GB 를 넘게 잡는다. 예약이 그보다 넉넉해야 의미가 있다."""
    assert tg.GPU_RESERVE_MIB >= 2048


def _fake_smi(free: int, total: int):
    return _fake_raw(f"{free}, {total}")


def _fake_raw(text: str):
    class _Result:
        stdout = text + "\n"
    def run(*args, **kwargs):
        return _Result()
    return run
