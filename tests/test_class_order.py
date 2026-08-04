"""`artifacts/metadata/class_order.json` 결정 로직 — 기존 재사용 vs 신규 생성.

Ticket 1a 는 실제 대회 데이터를 쓰지 않으므로 train.csv 의 SUBCLASS 로부터
새로 계산하지 않는다. 대신 저장소에 이미 확정된 canonical 순서
(`docs/specs/random_forest_stacking_model.md` §2.3, 승인 완료)를 "이미 기록된
정본"으로 재사용한다 — 저장소 안에 다른 canonical 산출물(GBDT OOF/로그 등)이
실제로 있는지는 조사 시점에 확인했고(`artifacts/*` 는 `.gitkeep` 뿐), 없었다.

전부 tmp_path 로 돈다 — 실제 `artifacts/metadata/class_order.json` 을
테스트가 건드리지 않는다.
"""

from __future__ import annotations

import json

import pytest

from cancer_hack.class_order import (
    CANONICAL_CLASS_ORDER,
    load_class_order,
    resolve_class_order,
)


def test_canonical_class_order_has_26_alphabetical_entries():
    assert len(CANONICAL_CLASS_ORDER) == 26
    assert list(CANONICAL_CLASS_ORDER) == sorted(CANONICAL_CLASS_ORDER)


def test_resolve_creates_file_when_missing(tmp_path):
    path = tmp_path / "metadata" / "class_order.json"
    decision = resolve_class_order(path)

    assert decision["action"] == "created"
    assert decision["class_order"] == list(CANONICAL_CLASS_ORDER)
    assert path.exists()


def test_created_file_is_a_plain_list_on_disk(tmp_path):
    path = tmp_path / "class_order.json"
    resolve_class_order(path)

    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    assert isinstance(raw, list)
    assert raw == list(CANONICAL_CLASS_ORDER)


def test_resolve_reuses_existing_matching_file(tmp_path):
    path = tmp_path / "class_order.json"
    first = resolve_class_order(path)
    assert first["action"] == "created"

    second = resolve_class_order(path)
    assert second["action"] == "reused"
    assert second["class_order"] == list(CANONICAL_CLASS_ORDER)


def test_resolve_raises_on_conflicting_existing_file(tmp_path):
    path = tmp_path / "class_order.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(["ACC", "BRCA"], handle)

    with pytest.raises(ValueError, match="이미 다른"):
        resolve_class_order(path)


def test_resolve_raises_when_existing_file_is_not_a_list(tmp_path):
    path = tmp_path / "class_order.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"classes": ["ACC"]}, handle)

    with pytest.raises(ValueError, match="단순 리스트"):
        resolve_class_order(path)


def test_resolve_with_custom_class_order_for_synthetic_tests(tmp_path):
    path = tmp_path / "toy_class_order.json"
    toy = ["C00", "C01", "C02"]
    decision = resolve_class_order(path, class_order=toy)
    assert decision["class_order"] == toy
    assert load_class_order(path) == toy


def test_load_class_order_round_trips(tmp_path):
    path = tmp_path / "class_order.json"
    resolve_class_order(path)
    assert load_class_order(path) == list(CANONICAL_CLASS_ORDER)


def test_load_class_order_rejects_non_list(tmp_path):
    path = tmp_path / "bad.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"nope": True}, handle)
    with pytest.raises(ValueError, match="단순 리스트"):
        load_class_order(path)
