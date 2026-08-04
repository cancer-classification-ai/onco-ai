r"""제출 노트북이 **실제로 제출한 파일과 같은 것**을 만드는지 지킨다.

`notebooks/09_final_submission.ipynb` 와 `notebooks/10_seed_ensemble_submission.ipynb`
는 대회 제출물이다. 원본 csv(또는 캐시된 예측)에서 시작해 제출 csv 까지 가는데, 그게
실제로 LB 를 받은 파일과 한 행이라도 다르면 **제출한 것과 다른 코드를 낸 셈**이 된다.

노트북 실행은 오래 걸려서(09 는 약 10분, 10 은 캐시 재사용이면 몇 초) 여기서 돌리지
않는다. 대신 노트북이 마지막으로 만든 산출물이 기준 파일과 같은지 본다 — 노트북을 고친
뒤 다시 돌리지 않으면 이 테스트가 낡은 산출물을 보고 통과할 수 있으므로, 노트북을
고쳤으면 반드시 재실행한다.

    .\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=3600 notebooks\09_final_submission.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT_ROOT / "artifacts" / "submissions"
WORKSPACE = PROJECT_ROOT.parent
CANDIDATES = WORKSPACE / "Models" / "pairrule_candidates"

#: 노트북별 (산출물 -> 실제 제출한 기준 파일) 과 계약.
#:
#: 09 는 LB 최고점(0.4818)을 낸 v002 경로, 10 은 seed 앙상블 + cbopt10 경로(0.4725)다.
#: 10 은 기각된 구성이지만 재현은 되어야 한다 — 재현이 안 되면 그 판정 자체를 못 믿는다.
NOTEBOOKS: dict[str, dict] = {
    "09_final_submission.ipynb": {
        "pairs": {
            "1층 (짝 규칙 전)": (
                SUBMISSIONS / "submission_final_base.csv",
                WORKSPACE / "Models/Ensemble/v002_seed42_f16_group5_macroF1_0.5165/submission.csv",
            ),
            "2층 (최종)": (
                SUBMISSIONS / "submission_final.csv",
                CANDIDATES / "submission_ens_v002_pairrule_m3.csv",
            ),
        },
        "imports": (
            "from train_gbdt import",
            "run_config",
            "from cancer_hack.pair_rule import",
            "build_pair_rule",
            "crossfit_calibrated_blend",
            "weighted_average",
        ),
    },
    "10_seed_ensemble_submission.ipynb": {
        "pairs": {
            "1층 (짝 규칙 전)": (
                SUBMISSIONS / "submission_nb10_base.csv",
                WORKSPACE
                / "Models/Ensemble/v010_seed42_f16_cbopt10_group5_macroF1_0.5210/submission.csv",
            ),
            "2층 (최종)": (
                SUBMISSIONS / "submission_nb10_pairrule.csv",
                CANDIDATES / "submission_ens16_cbopt10_seed3_pairrule_m3.csv",
            ),
        },
        "imports": (
            "from train_gbdt import",
            "run_config",
            "from cancer_hack.pair_rule import",
            "build_pair_rule",
            "crossfit_calibrated_blend",
            "weighted_average",
            "PARAM_PRESETS",
        ),
    },
}

CASES = [(nb, label) for nb, spec in NOTEBOOKS.items() for label in spec["pairs"]]


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.name} 없음 — 노트북을 먼저 실행한다")
    return pd.read_csv(path)


def _code(notebook: str) -> str:
    cells = json.loads((PROJECT_ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"]) for cell in cells["cells"] if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize(("notebook", "label"), CASES)
def test_notebook_output_matches_the_submitted_file(notebook, label):
    produced, reference = (_require(p) for p in NOTEBOOKS[notebook]["pairs"][label])
    assert (produced["ID"].to_numpy() == reference["ID"].to_numpy()).all(), "ID 순서가 다르다"
    mismatched = int((produced["SUBCLASS"].to_numpy() != reference["SUBCLASS"].to_numpy()).sum())
    assert mismatched == 0, f"{notebook} {label}: {mismatched}행이 제출본과 다르다"


@pytest.mark.parametrize(("notebook", "label"), CASES)
def test_submission_schema(notebook, label):
    produced = _require(NOTEBOOKS[notebook]["pairs"][label][0])
    sample = _require(PROJECT_ROOT / "data" / "raw" / "sample_submission.csv")
    assert list(produced.columns) == ["ID", "SUBCLASS"]
    assert len(produced) == 2546
    assert (produced["ID"].to_numpy() == sample["ID"].to_numpy()).all()
    assert produced["SUBCLASS"].notna().all()


@pytest.mark.parametrize("notebook", list(NOTEBOOKS))
def test_pair_rule_changes_exactly_the_expected_rows(notebook):
    """1층과 2층의 차이가 정확히 짝 규칙 대상 행이어야 한다."""
    pairs = NOTEBOOKS[notebook]["pairs"]
    base, final = (_require(pairs[k][0]) for k in ("1층 (짝 규칙 전)", "2층 (최종)"))
    changed = base["SUBCLASS"].to_numpy() != final["SUBCLASS"].to_numpy()
    assert int(changed.sum()) == 214, f"{notebook}: 바뀐 행이 {int(changed.sum())}개 (기대 214)"

    # 바뀐 뒤 라벨은 전부 코호트 짝 4종이어야 한다. 바뀌기 **전** 라벨은 짝 4종이 아닐 수
    # 있다 — 09 의 한 행이 SARC 였다. 규칙은 train 프로파일 매칭으로 정해지지 원래 예측이
    # 무엇이었는지는 보지 않는다.
    pair = {"KIPAN": "KIRC", "KIRC": "KIPAN", "GBMLGG": "LGG", "LGG": "GBMLGG"}
    after = final.loc[changed, "SUBCLASS"]
    assert set(after) <= set(pair), f"짝 4종 밖으로 간 라벨: {set(after) - set(pair)}"


@pytest.mark.parametrize("notebook", list(NOTEBOOKS))
def test_notebook_uses_the_shared_code_path(notebook):
    """로직을 노트북에 복사해 넣으면 CLI 와 어긋난다 — import 해서 쓰는지 본다."""
    source = _code(notebook)
    for entry in NOTEBOOKS[notebook]["imports"]:
        assert entry in source, f"{notebook} 이 {entry} 를 쓰지 않는다"


@pytest.mark.parametrize("notebook", list(NOTEBOOKS))
def test_notebook_states_it_does_not_upload(notebook):
    """제출은 사람이 한다 — 노트북에 그 문구가 남아 있어야 한다."""
    cells = json.loads((PROJECT_ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in cells["cells"])
    assert "DACON" in text and "직접" in text


def test_seed_ensemble_notebook_does_not_hardcode_hyperparameters():
    """cbopt10 값을 노트북에 박아 두면 CLI 와 갈라진다 — 프리셋에서 가져와야 한다.

    이 파라미터가 코드 어디에도 없고 셸 명령줄에만 있어서 제출본을 재현할 수 없었던 게
    이 노트북을 만든 계기다. 같은 일이 반복되지 않게 여기서 막는다.
    """
    source = _code("10_seed_ensemble_submission.ipynb")
    assert "PARAM_PRESETS" in source
    assert "0.1002086028456688" not in source, "학습률을 노트북에 하드코딩했다"
    assert "1.0629966259002686" not in source, "l2_leaf_reg 를 노트북에 하드코딩했다"
