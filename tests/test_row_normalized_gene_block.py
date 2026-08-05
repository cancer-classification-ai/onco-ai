"""행정규화 유전자 블록(`gecr`) 계약 검증.

이 블록의 존재 이유는 하나다 — test 가 train 보다 샘플당 변이 유전자가 2.21배 많아서
유전자별 원시 카운트가 통째로 부푸는 것을 약분해 없애는 것. 그러니 **밀도 배율에
불변인가**가 유일하게 중요한 성질이고, 그걸 테스트로 박는다.

`rollup`/`rollup16` 이 같은 parquet 을 읽으면서 캐시 키가 겹쳐 조용히 잘못된 배열을
재사용한 전례가 있다(`train_gbdt.SELECT_KIND` 주석). `gecr` 도 `gec` 와 같은 파일을
읽으므로 같은 함정이 있다 — 키가 갈리는지 확인한다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "train_gbdt_mod", PROJECT_ROOT / "scripts" / "train_gbdt.py"
)
tg = importlib.util.module_from_spec(_spec)
sys.modules["train_gbdt_mod"] = tg
_spec.loader.exec_module(tg)


def test_rows_sum_to_one():
    x = np.array([[1, 2, 1, 0], [5, 0, 0, 5]], dtype=np.float32)
    out = tg._row_normalize(x)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_zero_rows_stay_zero_and_finite():
    """변이가 하나도 없는 샘플은 0 으로 나누게 된다 — NaN/inf 가 나오면 안 된다."""
    x = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32)
    out = tg._row_normalize(x)
    assert np.isfinite(out).all()
    assert np.allclose(out[0], 0.0)


@pytest.mark.parametrize("factor", [1.5, 2.21, 3.0, 10.0])
def test_invariant_to_density_multiplier(factor):
    """블록의 존재 이유 그 자체. 행이 통째로 k배 부풀어도 결과가 같아야 한다."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=(20, 40)).astype(np.float32)
    assert np.allclose(tg._row_normalize(x * factor), tg._row_normalize(x), atol=1e-6)


def test_row_independent():
    """한 행만 따로 넣어도 같은 결과다 — 규정 판정 기준(행 단위 독립)이다."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, size=(12, 30)).astype(np.float32)
    full = tg._row_normalize(x)
    for i in range(len(x)):
        alone = tg._row_normalize(x[i : i + 1])
        assert np.allclose(alone[0], full[i], atol=1e-6)


def test_cache_key_differs_from_gec():
    """같은 parquet 을 읽으므로 키가 겹치면 정규화 안 된 배열이 조용히 재사용된다."""
    assert tg.block_cache_key("gecr") != tg.block_cache_key("gec")
    assert tg.BLOCK_SOURCES["gecr"] == tg.BLOCK_SOURCES["gec"]


def test_gecr_is_registered_everywhere():
    """네 표 중 하나라도 빠지면 KeyError 가 아니라 조용한 오작동으로 나타난다."""
    assert "gecr" in tg.GENE_BLOCKS
    assert "gecr" in tg.BLOCK_SOURCES
    assert "gecr" in tg.SELECT_KIND
    assert "gecr" in tg.BLOCK_DESC


def test_f16n_differs_from_f16_only_in_that_block():
    f16 = list(tg.CONFIGS["f16"]["blocks"])
    f16n = list(tg.CONFIGS["f16n"]["blocks"])
    assert len(f16) == len(f16n)
    diff = [(a, b) for a, b in zip(f16, f16n) if a != b]
    assert diff == [("gec", "gecr")], f"gec->gecr 외의 축이 같이 움직인다: {diff}"
