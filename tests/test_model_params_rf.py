"""`MODEL_PARAMS` 와 `RFModel.default_params()` 가 어긋나지 않는지 못 박는다.

`MODEL_PARAMS[model]` 은 "이번 실행이 실제로 쓴 값"을 보는 자리이고
`RFModel.default_params()` 는 그 값의 근거가 적힌 자리다. 둘이 드리프트하면
로그가 거짓말한다 — `train_gbdt.py:433-435` 의 xgb 주석이 경고하는 것과 같은 함정이다.

`model_params_for` 가 없는 모델 이름에 조용히 KeyError 를 던지면, `main()` 의
config 루프가 그걸 잡아 `*_ERROR.json` 만 남기고 다음 config 로 넘어간다 —
"모델 표에 항목이 없다"는 원인이 "config 실패"로 위장된다. 그래서 즉시 죽어야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cancer_hack.models_gbdt import RFModel  # noqa: E402
from train_gbdt import MODEL_PARAMS, _LOGGED_PARAMS, model_params_for  # noqa: E402


def test_rf_model_params_does_not_drift_from_default_params():
    """`MODEL_PARAMS["rf"]` 는 xgb 와 같은 패턴으로 일부 키만 명시한다 — 나머지는

    `create_model` 이 `default_params()` 에서 자연히 채운다(`BaseGBDT.__init__` 이
    `{**default_params(), **params}` 로 병합). 그래서 "모든 키가 있어야 한다"가 아니라
    "겹치는 키의 값이 서로 어긋나면 안 된다"가 실제 계약이다 — 값이 갈리면 이
    dict 가 "실제로 쓴 값"을 보여준다는 전제가 깨진다.
    """
    defaults = RFModel.default_params()
    for key, value in MODEL_PARAMS["rf"].items():
        assert key in defaults, f"{key} 가 RFModel.default_params() 에 없다 — 오타 의심"
        assert defaults[key] == value, f"{key} 값이 RFModel.default_params() 와 어긋났다"


def test_logged_params_cover_rf_defaults():
    """RF 기본값의 모든 키가 로그에 남아야 결과 JSON 만으로 설정을 복원할 수 있다."""
    missing = set(RFModel.default_params()) - set(_LOGGED_PARAMS)
    assert not missing, f"_LOGGED_PARAMS 에 없는 RF 파라미터: {missing}"


def test_model_params_for_unknown_model_exits():
    with pytest.raises(SystemExit):
        model_params_for("no-such-model")


def test_model_params_for_known_models_returns_a_copy():
    a = model_params_for("rf")
    b = model_params_for("rf")
    assert a == b
    a["n_estimators"] = -1
    assert MODEL_PARAMS["rf"]["n_estimators"] != -1
