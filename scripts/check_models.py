"""모델 세팅 확인 — pull 받은 뒤 한 번 돌려서 모델 4종이 준비됐는지 본다.

    python scripts/check_models.py                # device auto (GPU 되면 GPU)
    python scripts/check_models.py --device cpu   # CPU 경로만
    python scripts/check_models.py --device gpu   # GPU 경로만. 폴백하면 표에 cpu 로 찍힌다

원본 데이터 없이 합성 데이터로 몇 초 안에 끝난다. 모델 점검이 하나라도 실패하거나
예측을 바꾸는 라이브러리(CRITICAL)가 requirements.txt 핀과 어긋나면 종료코드 1.

`--device` 는 학습 스크립트(`train_gbdt.py`·`tune_optuna.py`·`run_seed_sweep.py`·
`tune_sweep.py`)와 같은 이름·같은 선택지를 쓴다. 여기서 통과하면 그쪽도 같은 경로로 돈다.
"""

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cancer_hack.models_gbdt import (  # noqa: E402
    BaseGBDT,
    available_models,
    balanced_sample_weight,
    create_model,
    gpu_available,
    registered_models,
)

N_CLASSES = 26

REQUIREMENTS = ROOT / "requirements.txt"

# 이 넷은 버전이 다르면 같은 seed 로도 트리가 달라진다. 어긋난 환경에서 뽑은 OOF 는
# 블렌딩에 넣으면 안 되므로 경고가 아니라 실패로 처리한다. 나머지는 알림만.
CRITICAL = {"scikit-learn", "xgboost", "lightgbm", "catboost"}

#: `--device` 문자열 -> `create_model(use_gpu=...)` 값. 학습 스크립트와 같은 규칙이다.
DEVICE_VALUE = {"gpu": True, "cpu": False, "auto": "auto"}

#: 이 환경에서 실제로 GPU 를 탈 수 있는 백엔드. lgbm 은 pip 기본 휠이 CPU 전용 빌드고
#: rf(sklearn) 는 GPU 백엔드 자체가 없다 — 둘이 cpu 로 찍히는 건 고장이 아니다.
GPU_CAPABLE = {"xgb", "catboost"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def pinned_versions() -> dict:
    """requirements.txt 에서 `pkg==ver` 줄만 걷는다. 주석·꼬리주석은 버린다."""
    pins = {}
    if not REQUIREMENTS.exists():
        return pins
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        pins[_normalize(name)] = version.strip()
    return pins


def version_drift() -> list:
    """핀과 다른 패키지만 (이름, 핀, 설치된 것) 으로 돌려준다."""
    drift = []
    for name, want in sorted(pinned_versions().items()):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "없음"
        if have != want:
            drift.append((name, want, have))
    return drift


def make_data():
    X, yi = make_classification(
        n_samples=520, n_features=40, n_informative=24, n_redundant=0,
        n_classes=N_CLASSES, n_clusters_per_class=1, random_state=0,
    )
    y = np.array([f"C{i:02d}" for i in yi])          # 문자열 라벨로 입력
    return sparse.csr_matrix(X.astype(np.float32)), y


def check(name: str, X, y, use_gpu: bool | str = "auto") -> dict:
    row = {"backend": name, "version": "-", "device": "-", "fit": "-",
           "proba": "-", "io": "-", "result": "FAIL", "error": ""}
    try:
        mod = __import__(
            {"xgb": "xgboost", "lgbm": "lightgbm", "catboost": "catboost", "rf": "sklearn"}[name]
        )
        row["version"] = getattr(mod, "__version__", "?")

        # 공통 이름 -> 백엔드 이름 자동 변환. use_gpu 는 학습 스크립트와 같은 값이 온다
        model = create_model(name, use_gpu=use_gpu, n_estimators=20)
        model.fit(X, y, sample_weight=balanced_sample_weight(y))
        row["device"] = "gpu" if model.use_gpu else "cpu"
        row["fit"] = "OK"

        proba = model.predict_proba(X)
        pred = model.predict(X)
        assert proba.shape == (X.shape[0], N_CLASSES), f"형상 {proba.shape}"
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4), "행 합 != 1"
        assert np.array_equal(pred, model.classes_[proba.argmax(axis=1)]), "predict/proba 불일치"
        assert list(model.classes_) == sorted(set(y)), "classes_ 정렬 불일치"
        row["proba"] = "OK"

        tmp = ROOT / "artifacts" / "_check" / f"{name}.joblib"
        model.save(tmp)
        assert np.allclose(BaseGBDT.load(tmp).predict_proba(X), proba), "save/load 불일치"
        tmp.unlink(missing_ok=True)
        row["io"] = "OK"

        row["result"] = "PASS"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                        help="학습 스크립트의 --device 와 같은 선택지")
    args = parser.parse_args()
    use_gpu = DEVICE_VALUE[args.device]

    print(f"Python  {sys.version.split()[0]}")
    print(f"venv    {Path(sys.prefix)}")
    print(f"GPU     {gpu_available()}  ·  요청 device = {args.device}")

    drift = version_drift()
    critical = [d for d in drift if d[0] in CRITICAL]
    if not drift:
        print("버전    requirements.txt 핀과 전부 일치")
    else:
        print(f"버전    {len(drift)}개가 핀과 다르다"
              f"{f' (그중 예측에 영향 {len(critical)}개)' if critical else ''}")
        for name, want, have in drift:
            mark = "!!" if name in CRITICAL else "  "
            print(f"        {mark} {name:<14} 핀 {want:<10} 지금 {have}")
        if critical:
            print("        !! 표시된 건 같은 seed 로도 트리가 달라진다."
                  " 이 환경에서 뽑은 OOF 는 블렌딩에 넣지 말 것.")

    installed = available_models()
    missing = sorted({c.split(":")[0] for c in ("xgb", "lgbm", "catboost", "rf")} - set(installed))
    print(f"등록    {registered_models()}")
    print(f"설치됨  {installed}")
    if missing:
        print(f"미설치  {missing}  ->  pip install -r requirements.txt")

    X, y = make_data()
    print(f"\n합성 데이터 {X.shape} · 클래스 {len(set(y))}개\n")

    rows = [check(n, X, y, use_gpu) for n in ("xgb", "lgbm", "catboost", "rf")]

    head = f"{'backend':<10}{'version':<10}{'device':<8}{'fit':<6}{'proba':<7}{'io':<6}결과"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['backend']:<10}{r['version']:<10}{r['device']:<8}"
              f"{r['fit']:<6}{r['proba']:<7}{r['io']:<6}{r['result']}")
        if r["error"]:
            print(f"           └ {r['error']}")

    if args.device == "gpu":
        fell_back = [r["backend"] for r in rows
                     if r["backend"] in GPU_CAPABLE and r["device"] == "cpu"]
        if fell_back:
            print(f"\nGPU 를 요청했는데 {fell_back} 가 CPU 로 되돌아왔다 — "
                  "드라이버·VRAM 여유를 본다 (다른 실험이 GPU 를 쥐고 있을 수 있다)")
        else:
            print(f"\nGPU 요청 정상 — {sorted(GPU_CAPABLE)} 가 gpu 로 돌았다")

    failed = [r["backend"] for r in rows if r["result"] != "PASS"]
    print()
    if failed:
        print(f"실패: {failed}")
        return 1
    if critical:
        print(f"버전 불일치: {[d[0] for d in critical]}  ->  pip install -r requirements.txt")
        return 1
    print("모델 4종 준비 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
