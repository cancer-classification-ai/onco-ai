"""canonical class order 결정/기록 — `artifacts/metadata/class_order.json`.

`docs/specs/random_forest_stacking_model.md` §2.3 이 26개 SUBCLASS 알파벳 순서를
이미 확정해 뒀다("sklearn.LabelEncoder 정렬과 동일 관례, 저장소 전체에서
일관됨" — 승인 완료, 미변경). Ticket 1a 조사 시점에 저장소 안의 다른 canonical
산출물(GBDT OOF/로그 등)을 확인했으나 없었다(`artifacts/oof` 등은 `.gitkeep`
뿐이라 재사용할 기존 기록이 없다). 그래서 이 스펙의 리스트를 "이미 기록된
정본"으로 재사용하고, train.csv 에서 새로 계산하지 않는다 — Ticket 1a 는 실제
대회 데이터를 열지 않는다.

`resolve_class_order` 가 이 결정을 코드로 남긴다: 대상 경로에 파일이 이미
있으면 내용을 검증해 재사용하고, 없으면 새로 쓴다. 값이 다른데 이미 있으면
조용히 덮어쓰지 않고 예외를 낸다 — OOF/제출 파일의 확률 열 순서가 이 파일에
근거하므로, 순서가 바뀌는 건 사람이 검토해야 하는 일이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

__all__ = ["CANONICAL_CLASS_ORDER", "SOURCE_SPEC", "resolve_class_order", "load_class_order"]

CANONICAL_CLASS_ORDER: tuple[str, ...] = (
    "ACC", "BLCA", "BRCA", "CESC", "COAD", "DLBC", "GBMLGG", "HNSC", "KIPAN",
    "KIRC", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "PCPG",
    "PRAD", "SARC", "SKCM", "STES", "TGCT", "THCA", "THYM", "UCEC",
)

SOURCE_SPEC = "docs/specs/random_forest_stacking_model.md#2.3"


def resolve_class_order(
    path: str | Path,
    *,
    class_order: Sequence[str] = CANONICAL_CLASS_ORDER,
    source: str = SOURCE_SPEC,
) -> dict:
    """`path` 에 canonical class order 를 기록하거나, 있으면 검증 후 재사용한다.

    반환값(`action`: "reused"|"created", `path`, `source`, `class_order`)이
    결정 근거다. 파일 자체는 spec §9.1/tickets 계약대로 **단순 리스트**만
    담는다.
    """
    path = Path(path)
    wanted = list(class_order)

    if path.exists():
        existing = _read_list(path)
        if existing != wanted:
            raise ValueError(
                f"{path} 에 이미 다른 canonical class order 가 있다 "
                f"(기존 {len(existing)}개 vs 요청 {len(wanted)}개). "
                "값을 바꾸려면 사람이 검토 후 파일을 직접 갱신한다."
            )
        return {"action": "reused", "path": str(path), "source": "existing_artifact", "class_order": existing}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(wanted, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {"action": "created", "path": str(path), "source": source, "class_order": wanted}


def load_class_order(path: str | Path) -> list[str]:
    return _read_list(Path(path))


def _read_list(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} 가 단순 리스트가 아니다: {type(data).__name__}")
    return data
