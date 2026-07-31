"""pytest 공용 설정 — `import cancer_hack` 가 되게 src 를 경로에 넣는다.

패키지를 설치하지 않고 `sys.path` 로 잡아 쓰는 방식이라(analysis/ 스크립트와 동일)
conftest 없이는 tests 에서 import 가 안 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RAW_DIR = PROJECT_ROOT / "data/raw"


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """전략표의 예시를 그대로 담은 작은 프레임.

    한 행이 한 샘플이고 유전자 컬럼은 3개다. 원본 csv 없이도 계약을 검증할 수
    있게 여기 값만으로 모든 유형이 최소 한 번씩 나오게 짰다.
    """
    return pd.DataFrame(
        {
            "ID": ["s1", "s2", "s3", "s4"],
            "SUBCLASS": ["BRCA", "ACC", "DLBC", "THYM"],
            "TP53": ["E412K R1800C", "WT", "Q369* I368N", "WT"],
            "KRAS": ["S622S G827R", "V600E V600E", "K16fs", "WT"],
            "EGFR": ["312_313QY>HH", "R649del", "P11_K12insP", "WT"],
        }
    )


def requires_raw(name: str) -> Path:
    """원본 csv 가 있어야만 도는 테스트용. 없으면 skip 한다.

    원본은 git 에 없으므로(용량·규정) CI 나 클론 직후 환경에서는 자연히 건너뛴다.
    """
    path = RAW_DIR / name
    if not path.exists():
        pytest.skip(f"{path} 없음 — 원본 csv 가 필요한 테스트")
    return path
