r"""최종 제출 노트북이 **실제로 제출한 파일과 같은 것**을 만드는지 지킨다.

`notebooks/09_final_submission.ipynb` 는 대회 제출물이다. 원본 csv 에서 시작해
`artifacts/submissions/submission_final.csv` 까지 가는데, 그게 실제로 LB 를 받은 파일과
한 행이라도 다르면 **제출한 것과 다른 코드를 낸 셈**이 된다.

노트북 실행은 10분쯤 걸려서 여기서 돌리지 않는다. 대신 노트북이 마지막으로 만든
산출물이 기준 파일과 같은지 본다 — 노트북을 고친 뒤 다시 돌리지 않으면 이 테스트가
낡은 산출물을 보고 통과할 수 있으므로, 노트북을 고쳤으면 반드시 재실행한다.

    .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=3600 notebooks\09_final_submission.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "09_final_submission.ipynb"
WORKSPACE = PROJECT_ROOT.parent

#: 노트북 산출물 -> 실제 제출한 기준 파일
PAIRS = {
    "1층 (짝 규칙 전)": (
        PROJECT_ROOT / "artifacts" / "submissions" / "submission_final_base.csv",
        WORKSPACE / "Models/Ensemble/v002_seed42_f16_group5_macroF1_0.5165/submission.csv",
    ),
    "2층 (최종)": (
        PROJECT_ROOT / "artifacts" / "submissions" / "submission_final.csv",
        WORKSPACE / "Models/pairrule_candidates/submission_ens_v002_pairrule_m3.csv",
    ),
}


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.name} 없음 — 노트북을 먼저 실행한다")
    return pd.read_csv(path)


@pytest.mark.parametrize("label", list(PAIRS))
def test_notebook_output_matches_the_submitted_file(label):
    produced, reference = (_require(p) for p in PAIRS[label])
    assert (produced["ID"].to_numpy() == reference["ID"].to_numpy()).all(), "ID 순서가 다르다"
    mismatched = int((produced["SUBCLASS"].to_numpy() != reference["SUBCLASS"].to_numpy()).sum())
    assert mismatched == 0, f"{label}: {mismatched}행이 제출본과 다르다"


@pytest.mark.parametrize("label", list(PAIRS))
def test_submission_schema(label):
    produced = _require(PAIRS[label][0])
    sample = _require(PROJECT_ROOT / "data" / "raw" / "sample_submission.csv")
    assert list(produced.columns) == ["ID", "SUBCLASS"]
    assert len(produced) == 2546
    assert (produced["ID"].to_numpy() == sample["ID"].to_numpy()).all()
    assert produced["SUBCLASS"].notna().all()


def test_pair_rule_changes_exactly_the_expected_rows():
    """1층과 2층의 차이가 정확히 짝 규칙 대상 행이어야 한다."""
    base, final = (_require(PAIRS[k][0]) for k in ("1층 (짝 규칙 전)", "2층 (최종)"))
    changed = base["SUBCLASS"].to_numpy() != final["SUBCLASS"].to_numpy()
    assert int(changed.sum()) == 214, f"바뀐 행이 {int(changed.sum())}개 (기대 214)"

    # 바뀐 방향이 전부 코호트 짝 안이어야 한다 (한 행은 SARC -> KIRC 로, 원래 예측이
    # 짝 4종이 아니었던 경우다)
    pair = {"KIPAN": "KIRC", "KIRC": "KIPAN", "GBMLGG": "LGG", "LGG": "GBMLGG"}
    after = final.loc[changed, "SUBCLASS"]
    assert set(after) <= set(pair), f"짝 4종 밖으로 간 라벨: {set(after) - set(pair)}"


def test_notebook_is_executable_and_uses_the_shared_code_path():
    """로직을 노트북에 복사해 넣으면 CLI 와 어긋난다 — import 해서 쓰는지 본다."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
    for entry in ("from train_gbdt import", "run_config", "from apply_pair_rule import",
                  "build_rule", "MacroF1LogitBias", "weighted_average"):
        assert entry in source, f"노트북이 {entry} 를 쓰지 않는다"


def test_notebook_states_it_does_not_upload():
    """제출은 사람이 한다 — 노트북에 그 문구가 남아 있어야 한다."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "DACON" in text and "직접" in text
