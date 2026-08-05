"""`train_gbdt.PARAM_PRESETS` 의 계약.

LB 를 받은 구성을 `--set` 나열로만 재현하면 그 명령줄이 셸 히스토리에서 사라지는 순간
재현이 불가능해진다. 실제로 `cbopt10`(EXP_041 제출본, LB 0.4725)의 CatBoost 파라미터가
저장소 어디에도 없어서 학습 로그에서 되찾아야 했다. 프리셋은 그 사고를 막는 장치이고,
아래 테스트는 프리셋이 **실제로 그 제출본이 쓴 값**인지를 지킨다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_gbdt import (  # noqa: E402
    MODEL_PARAMS,
    PARAM_PRESETS,
    build_parser,
    preset_params_for,
    resolve_model_params,
)

#: EXP_041 제출본을 만든 CatBoost 학습 로그. 프리셋의 근거다.
CBOPT10_LOG = (
    PROJECT_ROOT / "artifacts/logs/catboost_cbopt10_f16_group5_k500_sp1000m3p2"
    "_cm20drishamm83824c_lt64svdnmfl257c23c_gm24shac918bc_sg30sha71bc36_s42.json"
)


def _args(model: str, preset: str | None = None, overrides: dict | None = None):
    args = build_parser().parse_args(["--model", model])
    args.params_preset = preset
    args.override = overrides or {}
    return args


def test_every_preset_declares_its_model_and_verdict():
    for name, entry in PARAM_PRESETS.items():
        assert entry["model"] in MODEL_PARAMS, f"{name} 의 model 이 MODEL_PARAMS 에 없다"
        assert entry["params"], f"{name} 에 파라미터가 없다"
        # 채택인지 기각인지 적혀 있어야 한다. 이 저장소에서 튜닝은 기본적으로 기각됐다.
        assert entry.get("desc"), f"{name} 에 desc 가 없다 — 채택 여부를 적는다"


def test_no_preset_means_model_defaults():
    assert resolve_model_params(_args("catboost")) == MODEL_PARAMS["catboost"]


def test_preset_overrides_defaults():
    resolved = resolve_model_params(_args("catboost", "cbopt10"))
    assert resolved["iterations"] == 1600, "기본값 1000 을 덮어쓰지 못했다"
    assert resolved["depth"] == 6


def test_set_beats_preset():
    """`--set` 이 프리셋을 이긴다 — 프리셋 한 축만 바꿔 보는 게 흔한 사용이다."""
    resolved = resolve_model_params(_args("catboost", "cbopt10", {"iterations": 200}))
    assert resolved["iterations"] == 200
    assert resolved["learning_rate"] == PARAM_PRESETS["cbopt10"]["params"]["learning_rate"]


def test_preset_rejects_wrong_model():
    """CatBoost 파라미터를 XGBoost 에 넘기면 fold 한복판에서 터진다. 그 전에 죽인다."""
    with pytest.raises(SystemExit, match="cbopt10"):
        resolve_model_params(_args("xgb", "cbopt10"))


def test_unknown_preset_dies_early():
    with pytest.raises(SystemExit, match="PARAM_PRESETS"):
        preset_params_for("없는이름", "catboost")


def test_cbopt10_matches_the_submitted_run():
    """제출본이 실제로 쓴 값과 대조한다. 하나라도 다르면 재현이 깨진 것이다."""
    if not CBOPT10_LOG.exists():
        pytest.skip(f"{CBOPT10_LOG.name} 없음 — 학습 로그가 필요한 테스트")
    logged = json.loads(CBOPT10_LOG.read_text(encoding="utf-8"))["model_params"]
    resolved = resolve_model_params(_args("catboost", "cbopt10"))

    for key in ("iterations", "learning_rate", "depth", "l2_leaf_reg", "subsample", "rsm"):
        # 로그는 CatBoost 가 돌려준 값이라 전부 문자열이다.
        assert str(resolved[key]) == logged[key], f"{key} 가 제출본과 다르다"
