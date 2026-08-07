"""난수원이 전부 고정돼 있는지.

왜 필요한가
-----------
이 대회는 **여러 사람이 각자 뽑은 OOF 를 한데 모아 블렌딩한다.** 같은 코드·같은 seed 가
같은 숫자를 내지 못하면 각자의 CV 는 멀쩡해 보이는데 합쳐 놓은 결과만 조용히 어긋난다.
그리고 제출 코드를 심사에서 다시 돌렸을 때 우리가 낸 점수가 안 나오면 그 자체로 문제다.

seed 를 인자로 받는 것만으로는 부족하다 — **기본값이 박혀 있어야** 아무나 코드만 받아
돌려도 같은 결과가 나온다. 그래서 기본값 자체를 여기서 못 박는다.

무엇이 재현되고 무엇이 안 되는지(GPU 커널 비결정성 등)는 `docs/reproducibility.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_gbdt  # noqa: E402
from cancer_hack.validation import SEED as FOLD_SEED  # noqa: E402


def _defaults(parser):
    return {action.dest: action.default for action in parser._actions}


def test_fold_split_seed_is_pinned():
    """fold 가 바뀌면 artifacts/oof 의 예측 100여 개가 전부 무효가 된다."""
    assert FOLD_SEED == 42


def test_seed_ensemble_is_pinned():
    """지금까지의 3-seed 산출물이 전부 이 값으로 나왔다.

    바꾸면 그 기록들과 비교가 끊기고, 코드만 받은 사람이 우리와 다른 숫자를 얻는다.
    늘릴 때는 **뒤에 덧붙인다** — 앞의 셋을 유지해야 기존 OOF 를 그대로 재사용한다.
    """
    from cancer_hack.validation import SEED_ENSEMBLE, SEED_SWEEP

    assert SEED_ENSEMBLE[:3] == (42, 7, 2024)
    assert len(SEED_ENSEMBLE) >= 3, "seed 앙상블은 최소 3개"
    assert len(set(SEED_ENSEMBLE)) == len(SEED_ENSEMBLE), "중복된 seed"
    # 스윕은 앙상블 셋을 앞에 두고 확장한다 — 그래야 앞 셋의 결과를 재사용할 수 있다.
    assert SEED_SWEEP[:len(SEED_ENSEMBLE)] == SEED_ENSEMBLE


def test_seed_lists_come_from_one_place():
    """seed 목록이 스크립트마다 흩어져 있으면 사람마다 다른 값으로 돌리게 된다."""
    import re

    scripts = PROJECT_ROOT / "scripts"
    literal = re.compile(r"\[\s*42\s*,\s*7\s*,\s*2024")
    offenders = [
        p.name for p in scripts.glob("*.py")
        if p.stat().st_size and literal.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"seed 목록을 직접 적은 스크립트: {offenders} — "
        "cancer_hack.validation.SEED_ENSEMBLE 을 쓴다"
    )


def test_pipeline_default_uses_the_shared_seed_ensemble():
    from cancer_hack.pipeline import PipelineConfig
    from cancer_hack.validation import SEED_ENSEMBLE

    config = PipelineConfig(run_tag="t")
    assert config.seeds == SEED_ENSEMBLE
    assert config.fold_seed == FOLD_SEED
    # 소스 가중은 모델 가중을 seed 수로 나눈 것 — 합이 1 이어야 한다.
    assert len(config.source_weights) == len(config.models) * len(config.seeds)
    assert sum(config.source_weights) == pytest.approx(1.0)


def test_pipeline_pins_gpu_ram_part():
    """`auto` 는 실행 시점 GPU 여유로 값을 정해 설정 자체가 매번 달라진다."""
    from cancer_hack.pipeline import PipelineConfig

    value = PipelineConfig(run_tag="t").gpu_ram_part
    assert isinstance(value, float) and 0 < value <= 1, f"숫자로 고정해야 한다: {value!r}"


def test_pipeline_runs_catboost_on_cpu():
    """CatBoost GPU 학습은 같은 seed·같은 설정으로도 간헐적으로 결과가 갈린다.

    실측(f2l · seed 42 · device·gpu_ram_part 둘 다 고정) — 3회 중 1회가 OOF 최대차
    7.9e-02 로 어긋났다. CPU 는 두 번 다 확률 행렬이 통째로 일치했다(0.0e+00).

    제출본을 만드는 자리에서는 재현이 속도보다 앞선다. 빠른 탐색이 필요하면 호출부에서
    `device_by_model={}` 로 비우되, 그렇게 뽑은 OOF 는 제출본의 근거로 삼지 않는다.
    `docs/reproducibility.md` §2-3.
    """
    from cancer_hack.pipeline import PipelineConfig

    config = PipelineConfig(run_tag="t")
    assert config.device_by_model.get("catboost") == "cpu"
    # XGBoost·RandomForest 는 GPU 에서도 비트 단위로 재현된다 — 굳이 내리지 않는다.
    assert "xgb" not in config.device_by_model
    assert "rf" not in config.device_by_model


def test_pipeline_applies_the_pair_rule_by_default():
    """짝 규칙은 LB +0.09 다. 기본으로 꺼져 있으면 누군가는 빼먹는다."""
    from cancer_hack.pipeline import PipelineConfig

    config = PipelineConfig(run_tag="t")
    assert config.apply_pair_rule is True
    assert config.min_mut == 3


def test_train_gbdt_seed_defaults():
    defaults = _defaults(train_gbdt.build_parser())
    assert defaults["seed"] == 42, "모델 시드 기본값"
    # 잠재(SVD/NMF)와 모듈(KMeans)은 자체 난수를 쓴다. 안 박으면 실행마다 기저가 달라진다.
    assert defaults["latent_random_state"] == 0
    assert defaults["module_random_state"] == 0


def test_every_randomised_option_has_a_default():
    """새로 추가된 난수 인자가 `default=None` 으로 들어오면 재현이 깨진다."""
    defaults = _defaults(train_gbdt.build_parser())
    randomised = {k: v for k, v in defaults.items()
                  if any(word in k for word in ("seed", "random_state"))}
    assert randomised, "난수 인자를 하나도 못 찾았다 — 이름 규칙이 바뀌었나"
    missing = [k for k, v in randomised.items() if v is None]
    assert not missing, f"기본값 없는 난수 인자: {missing}"


def test_greedy_blend_seed_defaults():
    import greedy_blend

    defaults = _defaults(greedy_blend.build_parser())
    # 그리디는 bagging 에 난수를 쓴다. random_state 가 없으면 같은 라이브러리에서도
    # 매번 다른 멤버를 고른다.
    assert defaults["random_state"] == 0
    assert defaults["n_rounds"] == 30
    assert defaults["bag_fraction"] == 0.6
    assert defaults["bag_rounds"] == 5


def test_model_seed_reaches_the_estimator():
    """`--seed` 가 실제 파라미터로 들어가는지. 안 들어가면 백엔드 기본값이 쓰인다."""
    args = train_gbdt.build_parser().parse_args(["--model", "catboost", "--seed", "7"])
    args.override = {}
    params = train_gbdt.resolve_model_params(args)
    # MODEL_PARAMS 에는 seed 가 없고 fit 경로에서 주입된다 — 그 계약을 확인한다.
    assert "random_seed" not in params and "random_state" not in params
    source = (PROJECT_ROOT / "scripts" / "train_gbdt.py").read_text(encoding="utf-8")
    assert "args.seed" in source, "seed 를 어디선가 모델에 넘겨야 한다"


def test_dl_seeds_every_source():
    """torch 만 seed 하면 부족하다 — python random·numpy·cuda·cuDNN 까지 봐야 한다."""
    source = (PROJECT_ROOT / "scripts" / "train_dl.py").read_text(encoding="utf-8")
    for marker in (
        "random.seed(seed)",
        "np.random.seed(seed)",
        "torch.manual_seed(seed)",
        "torch.cuda.manual_seed_all(seed)",
        # cuDNN 이 실행마다 알고리즘을 고르면 같은 seed 로도 커널이 달라진다.
        "torch.backends.cudnn.deterministic = True",
        "torch.backends.cudnn.benchmark = False",
    ):
        assert marker in source, f"train_dl.seed_everything 에 {marker} 가 없다"


def test_dl_dataloader_uses_a_seeded_generator():
    """DataLoader 셔플도 난수다. 제너레이터를 안 주면 실행마다 배치 순서가 달라진다."""
    source = (PROJECT_ROOT / "scripts" / "train_dl.py").read_text(encoding="utf-8")
    assert "Generator().manual_seed(" in source


@pytest.mark.parametrize("notebook", [
    "09_final_submission.ipynb",
    "10_seed_ensemble_submission.ipynb",
    "11_full_pipeline.ipynb",
    "12_greedy_blend_submission.ipynb",
])
def test_submission_notebooks_pin_their_seeds(notebook):
    """노트북이 seed 를 명시하지 않으면 파서 기본값에 기대게 된다 — 눈에 안 보인다."""
    import json

    path = PROJECT_ROOT / "notebooks" / notebook
    if not path.exists():
        pytest.skip(f"{notebook} 없음")
    cells = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in cells["cells"]
                       if c["cell_type"] == "code")
    assert "SEED" in source or "random_state" in source, f"{notebook} 에 seed 설정이 없다"
