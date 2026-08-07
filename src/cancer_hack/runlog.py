"""실행 한 번을 한 파일에 남긴다.

왜 필요한가
-----------
지금까지 실행 기록이 `artifacts/logs/*.json` 100여 개에 흩어져 있었다. 파일 하나가
모델 하나의 fold 점수만 담고 있어서, "이 제출본이 어떤 파켓·어떤 fold·어떤 파라미터에서
나왔나"를 되짚으려면 파일 이름의 슬러그를 손으로 해독해야 했다. 실제로 `cbopt10`
파라미터를 로그에서 되찾아야 했던 일이 있다.

`RunLog` 는 실행 하나를 통째로 묶는다 — 설정, 환경, 입력 파켓의 지문, 단계별 소요 시간,
단계마다 남긴 수치, 만들어진 파일 목록. 산출물은 둘이다.

    run.json   기계가 읽는 전체 기록
    run.md     사람이 읽는 요약 (표 몇 개)

사용
----
    log = RunLog(run_dir, run_tag="nb11_s42", config={...})
    with log.step("피처 생성"):
        ...
        log.record("n_features", 5309)
    log.artifact("submission", path)
    log.save()
"""

from __future__ import annotations

import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

__all__ = ["RunLog"]


def _environment() -> dict[str, str]:
    """예측을 바꾸는 라이브러리 버전. 다르면 같은 seed 로도 결과가 달라진다."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "torch"):
        module_name = {"scikit-learn": "sklearn"}.get(name, name)
        try:
            module = __import__(module_name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            versions[name] = "없음"
    return versions


class RunLog:
    """실행 하나의 기록. `save()` 를 불러야 디스크에 남는다."""

    def __init__(self, run_dir: str | Path, *, run_tag: str, config: dict | None = None):
        self.run_dir = Path(run_dir)
        self.run_tag = run_tag
        self.config = dict(config or {})
        self.environment = _environment()
        self.steps: list[dict] = []
        self.values: dict[str, object] = {}
        self.artifacts: dict[str, str] = {}
        self._started = time.perf_counter()

    # ---------------------------------------------------------------- 기록
    @contextmanager
    def step(self, name: str):
        """단계 하나를 재고 화면에도 찍는다. 예외가 나면 실패로 남기고 다시 던진다."""
        print(f"[{name}] 시작", flush=True)
        started = time.perf_counter()
        entry: dict[str, object] = {"name": name}
        try:
            yield entry
        except Exception as error:  # noqa: BLE001 — 기록하고 그대로 올려보낸다
            entry["status"] = "실패"
            entry["error"] = f"{type(error).__name__}: {error}"
            raise
        else:
            entry["status"] = "완료"
        finally:
            entry["seconds"] = round(time.perf_counter() - started, 1)
            self.steps.append(entry)
            mark = entry.get("status")
            print(f"[{name}] {mark} · {entry['seconds']:.1f}초\n", flush=True)

    def record(self, key: str, value) -> None:
        """수치 하나를 남긴다. 나중에 run.md 표에 그대로 올라간다."""
        self.values[key] = value

    def artifact(self, label: str, path: str | Path) -> Path:
        """만들어진 파일을 등록한다."""
        path = Path(path)
        self.artifacts[label] = str(path)
        return path

    # ---------------------------------------------------------------- 저장
    def _payload(self) -> dict:
        return {
            "run_tag": self.run_tag,
            "elapsed_seconds": round(time.perf_counter() - self._started, 1),
            "config": self.config,
            "environment": self.environment,
            "steps": self.steps,
            "values": self.values,
            "artifacts": self.artifacts,
        }

    def _markdown(self) -> str:
        payload = self._payload()
        lines = [
            f"# 실행 기록 — `{self.run_tag}`",
            "",
            f"전체 {payload['elapsed_seconds'] / 60:.1f}분",
            "",
            "## 설정",
            "",
            "| 항목 | 값 |",
            "|---|---|",
        ]
        for key, value in self.config.items():
            lines.append(f"| `{key}` | {value} |")

        lines += ["", "## 단계", "", "| 단계 | 상태 | 초 |", "|---|---|---|"]
        for entry in self.steps:
            lines.append(f"| {entry['name']} | {entry.get('status','?')} | {entry.get('seconds','?')} |")

        if self.values:
            lines += ["", "## 수치", "", "| 항목 | 값 |", "|---|---|"]
            for key, value in self.values.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                lines.append(f"| `{key}` | {value} |")

        if self.artifacts:
            lines += ["", "## 산출물", "", "| 무엇 | 경로 |", "|---|---|"]
            for label, path in self.artifacts.items():
                lines.append(f"| {label} | `{path}` |")

        lines += [
            "",
            "## 환경",
            "",
            "| 라이브러리 | 버전 |",
            "|---|---|",
        ]
        for key, value in self.environment.items():
            lines.append(f"| {key} | {value} |")

        lines += ["", "---", "", "DACON 업로드는 사람이 직접 한다. 이 실행은 로컬 파일만 만든다.", ""]
        return "\n".join(lines)

    def save(self) -> dict[str, Path]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.run_dir / "run.json"
        md_path = self.run_dir / "run.md"
        json_path.write_text(
            json.dumps(self._payload(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        md_path.write_text(self._markdown(), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}

    def summary(self) -> str:
        """화면에 찍을 짧은 요약."""
        failed = [s["name"] for s in self.steps if s.get("status") == "실패"]
        total = sum(s.get("seconds", 0) for s in self.steps)
        head = f"{self.run_tag} · 단계 {len(self.steps)}개 · {total / 60:.1f}분"
        return head + (f"\n실패한 단계: {failed}" if failed else "\n전 단계 완료")


def _self_check() -> None:  # pragma: no cover - 손으로 돌려 보는 용도
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        log = RunLog(tmp, run_tag="check", config={"seed": 42})
        with log.step("무언가"):
            log.record("score", 0.5165)
        log.artifact("submission", Path(tmp) / "x.csv")
        print(log.save())
        print(log.summary())


if __name__ == "__main__":
    _self_check()
