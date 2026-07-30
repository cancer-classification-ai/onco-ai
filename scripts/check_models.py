"""모델 세팅 확인 — pull 받은 뒤 한 번 돌려서 GBDT 3종이 준비됐는지 본다.

    python scripts/check_models.py

원본 데이터 없이 합성 데이터로 몇 초 안에 끝난다. 하나라도 실패하면 종료코드 1.
"""

import sys
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


def make_data():
    X, yi = make_classification(
        n_samples=520, n_features=40, n_informative=24, n_redundant=0,
        n_classes=N_CLASSES, n_clusters_per_class=1, random_state=0,
    )
    y = np.array([f"C{i:02d}" for i in yi])          # 문자열 라벨로 입력
    return sparse.csr_matrix(X.astype(np.float32)), y


def check(name: str, X, y) -> dict:
    row = {"backend": name, "version": "-", "device": "-", "fit": "-",
           "proba": "-", "io": "-", "result": "FAIL", "error": ""}
    try:
        mod = __import__({"xgb": "xgboost", "lgbm": "lightgbm", "catboost": "catboost"}[name])
        row["version"] = getattr(mod, "__version__", "?")

        model = create_model(name, n_estimators=20)   # 공통 이름 -> 백엔드 이름 자동 변환
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
    print(f"Python  {sys.version.split()[0]}")
    print(f"venv    {Path(sys.prefix)}")
    print(f"GPU     {gpu_available()}")

    installed = available_models()
    missing = sorted({c.split(":")[0] for c in ("xgb", "lgbm", "catboost")} - set(installed))
    print(f"등록    {registered_models()}")
    print(f"설치됨  {installed}")
    if missing:
        print(f"미설치  {missing}  ->  pip install -r requirements.txt")

    X, y = make_data()
    print(f"\n합성 데이터 {X.shape} · 클래스 {len(set(y))}개\n")

    rows = [check(n, X, y) for n in ("xgb", "lgbm", "catboost")]

    head = f"{'backend':<10}{'version':<10}{'device':<8}{'fit':<6}{'proba':<7}{'io':<6}결과"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['backend']:<10}{r['version']:<10}{r['device']:<8}"
              f"{r['fit']:<6}{r['proba']:<7}{r['io']:<6}{r['result']}")
        if r["error"]:
            print(f"           └ {r['error']}")

    failed = [r["backend"] for r in rows if r["result"] != "PASS"]
    print()
    if failed:
        print(f"실패: {failed}")
        return 1
    print("모델 3종 준비 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
