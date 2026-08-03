"""`scripts/apply_pair_rule.py` 의 계약 검증.

이 규칙은 로컬 CV 로 잴 수 없다(group CV 가 이 상황 자체를 못 만든다). 즉 구현이 조용히
틀리면 **LB 제출 한 번을 태우기 전까지 아무도 모른다.** 그래서 규칙의 전제를 테스트로
박아 둔다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import requires_raw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "apply_pair_rule", PROJECT_ROOT / "scripts" / "apply_pair_rule.py"
)
apply_pair_rule = importlib.util.module_from_spec(_spec)
sys.modules["apply_pair_rule"] = apply_pair_rule
_spec.loader.exec_module(apply_pair_rule)

PAIR = apply_pair_rule.PAIR


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join([",".join(header)] + [",".join(r) for r in rows]) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def toy_csvs(tmp_path: Path) -> tuple[Path, Path]:
    """train 3행 / test 4행짜리 최소 예제.

    유전자 5열이라 `n_mut` 이 최대 5다. 케이스별로 변이 수를 다르게 둬서
    `--min-mut` 경계가 실제로 동작하는지 본다.
    """
    genes = ["A", "B", "C", "D", "E"]
    train = [
        # 짝 없는 KIRC (변이 4개) — test 에 KIPAN 사본이 있다
        ["TR1", "KIRC", "m1", "m2", "m3", "m4", "WT"],
        # 짝 없는 LGG (변이 2개) — min_mut 3 이면 걸리지 않아야 한다
        ["TR2", "LGG", "m9", "m8", "WT", "WT", "WT"],
        # 짝 4종이 아닌 라벨 (변이 4개) — 규칙 대상이 아니다
        ["TR3", "BRCA", "z1", "z2", "z3", "z4", "WT"],
    ]
    test = [
        ["TE1", "m1", "m2", "m3", "m4", "WT"],   # TR1 과 완전 일치 -> KIPAN
        ["TE2", "m9", "m8", "WT", "WT", "WT"],   # TR2 와 일치하지만 변이 2개
        ["TE3", "z1", "z2", "z3", "z4", "WT"],   # TR3 과 일치하지만 BRCA
        ["TE4", "q1", "q2", "q3", "WT", "WT"],   # 매칭 없음
    ]
    train_csv, test_csv = tmp_path / "train.csv", tmp_path / "test.csv"
    _write(train_csv, ["ID", "SUBCLASS", *genes], train)
    _write(test_csv, ["ID", *genes], test)
    return train_csv, test_csv


def test_flips_to_pair_label_not_matched_label(toy_csvs):
    """매칭된 train 라벨이 아니라 **짝 라벨**로 바꾼다 — 평범한 1-NN 과 반대 방향이다."""
    train_csv, test_csv = toy_csvs
    mapping, _ = apply_pair_rule.build_rule(train_csv, test_csv, min_mut=3)
    assert mapping == {"TE1": "KIPAN"}, "KIRC 매칭은 KIRC 가 아니라 KIPAN 이 되어야 한다"


def test_min_mut_excludes_low_mutation_matches(toy_csvs):
    """변이 수가 적은 매칭은 우연 일치 위험이 커서 기본값 3 에서 빠진다."""
    train_csv, test_csv = toy_csvs
    assert "TE2" not in apply_pair_rule.build_rule(train_csv, test_csv, min_mut=3)[0]
    assert apply_pair_rule.build_rule(train_csv, test_csv, min_mut=2)[0]["TE2"] == "GBMLGG"


def test_off_pair_labels_are_reported_not_flipped(toy_csvs):
    """짝 4종이 아닌 라벨은 바꾸지 않고, 우연 일치 신호로 진단에 남긴다."""
    train_csv, test_csv = toy_csvs
    mapping, diagnostics = apply_pair_rule.build_rule(train_csv, test_csv, min_mut=3)
    assert "TE3" not in mapping
    assert diagnostics["test_matched_off_pair_labels"] == {"BRCA": 1}


def test_unmatched_rows_untouched(toy_csvs):
    train_csv, test_csv = toy_csvs
    assert "TE4" not in apply_pair_rule.build_rule(train_csv, test_csv, min_mut=3)[0]


def test_ambiguous_train_group_is_skipped(tmp_path: Path):
    """train 쪽 프로파일이 둘 이상이면(양쪽 라벨이 이미 train 에 있으면) 건드리지 않는다.

    그 경우 test 행은 세 번째 사본이라 어느 라벨인지 알 수 없다.
    """
    genes = ["A", "B", "C", "D"]
    _write(
        tmp_path / "train.csv",
        ["ID", "SUBCLASS", *genes],
        [
            ["TR1", "KIPAN", "m1", "m2", "m3", "WT"],
            ["TR2", "KIRC", "m1", "m2", "m3", "WT"],
        ],
    )
    _write(tmp_path / "test.csv", ["ID", *genes], [["TE1", "m1", "m2", "m3", "WT"]])
    mapping, _ = apply_pair_rule.build_rule(tmp_path / "train.csv", tmp_path / "test.csv", 3)
    assert mapping == {}


def test_pair_mapping_is_an_involution():
    """짝 매핑은 뒤집어도 자기 자신이어야 한다 — 방향을 한쪽만 고치는 실수를 막는다."""
    assert all(PAIR[PAIR[k]] == k for k in PAIR)
    assert set(PAIR) == {"KIPAN", "KIRC", "GBMLGG", "LGG"}


def test_real_data_premises_hold():
    """원본 csv 가 있으면 규칙의 전제를 실제 데이터에서 다시 확인한다.

    이 넷 중 하나라도 깨지면 규칙의 근거가 무너진 것이므로 제출하면 안 된다.
    """
    train_csv = requires_raw("train.csv")
    test_csv = requires_raw("test.csv")
    mapping, diagnostics = apply_pair_rule.build_rule(train_csv, test_csv, min_mut=3)

    # 1. 프로파일이 같으면 예외 없이 라벨이 갈리고, 예외 없이 코호트 짝이다.
    assert diagnostics["train_dup_groups_same_label"] == 0
    assert diagnostics["train_dup_groups_pair_label"] == 422

    # 2. train 의 짝 없는 KIRC·LGG 는 전부 그 사본이 test 에 있다 — 결정적 근거다.
    orphans = diagnostics["train_orphans_by_label"]
    matched = diagnostics["test_matched_by_train_label"]
    assert orphans["KIRC"] == matched["KIRC"] == 57
    assert orphans["LGG"] == matched["LGG"] == 50

    # 3. 변이 3개 이상 구간에서는 짝 4종이 아닌 매칭이 하나도 없다.
    assert diagnostics["test_matched_off_pair_labels"] == {}

    # 4. 규칙 대상은 214행이고 전부 짝 4종으로 간다.
    assert diagnostics["n_flipped"] == 214
    assert set(mapping.values()) <= set(PAIR)


def test_applied_submission_keeps_schema(tmp_path: Path):
    """규칙을 적용해도 제출 스키마가 유지된다 (ID 순서·행 수·컬럼)."""
    train_csv = requires_raw("train.csv")
    test_csv = requires_raw("test.csv")
    sample = pd.read_csv(requires_raw("sample_submission.csv"))

    mapping, _ = apply_pair_rule.build_rule(train_csv, test_csv, min_mut=3)
    out = sample.copy()
    out["SUBCLASS"] = [mapping.get(i, c) for i, c in zip(out["ID"], out["SUBCLASS"])]

    assert list(out.columns) == ["ID", "SUBCLASS"]
    assert len(out) == len(sample)
    assert (out["ID"].to_numpy() == sample["ID"].to_numpy()).all()
    assert out["SUBCLASS"].notna().all()
