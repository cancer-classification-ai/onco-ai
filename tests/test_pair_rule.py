"""`cancer_hack.pair_rule` 과 `scripts/apply_pair_rule.py` 의 계약 검증.

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

from cancer_hack.pair_rule import (
    PAIR,
    PairRule,
    _count_mutations,
    apply_to_submission,
    build_pair_rule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "apply_pair_rule", PROJECT_ROOT / "scripts" / "apply_pair_rule.py"
)
apply_pair_rule_cli = importlib.util.module_from_spec(_spec)
sys.modules["apply_pair_rule"] = apply_pair_rule_cli
_spec.loader.exec_module(apply_pair_rule_cli)


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
    rule = build_pair_rule(train_csv, test_csv, min_mut=3)
    assert rule.mapping == {"TE1": "KIPAN"}, "KIRC 매칭은 KIRC 가 아니라 KIPAN 이 되어야 한다"


def test_min_mut_excludes_low_mutation_matches(toy_csvs):
    """변이 수가 적은 매칭은 우연 일치 위험이 커서 기본값 3 에서 빠진다."""
    train_csv, test_csv = toy_csvs
    assert "TE2" not in build_pair_rule(train_csv, test_csv, min_mut=3).mapping
    assert build_pair_rule(train_csv, test_csv, min_mut=2).mapping["TE2"] == "GBMLGG"


def test_off_pair_labels_are_reported_not_flipped(toy_csvs):
    """짝 4종이 아닌 라벨은 바꾸지 않고, 우연 일치 신호로 진단에 남긴다."""
    train_csv, test_csv = toy_csvs
    rule = build_pair_rule(train_csv, test_csv, min_mut=3)
    assert "TE3" not in rule.mapping
    assert rule.diagnostics["test_matched_off_pair_labels"] == {"BRCA": 1}


def test_unmatched_rows_untouched(toy_csvs):
    train_csv, test_csv = toy_csvs
    assert "TE4" not in build_pair_rule(train_csv, test_csv, min_mut=3).mapping


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
    rule = build_pair_rule(tmp_path / "train.csv", tmp_path / "test.csv", 3)
    assert rule.mapping == {}


def test_pair_mapping_is_an_involution():
    """짝 매핑은 뒤집어도 자기 자신이어야 한다 — 방향을 한쪽만 고치는 실수를 막는다."""
    assert all(PAIR[PAIR[k]] == k for k in PAIR)
    assert set(PAIR) == {"KIPAN", "KIRC", "GBMLGG", "LGG"}


def test_count_mutations_matches_naive():
    """빠른 셈이 원래 정의(`WT` 도 빈 셀도 아닌 칸)와 같은 값을 낸다."""
    profiles = [
        b"WT,WT,WT",
        b"m1,WT,m2",
        b"WT,,WT",
        b",,",
        b"m1 m2,WT,m3",
        b"WT",
        b"",
    ]
    for profile in profiles:
        naive = sum(cell not in (b"WT", b"") for cell in profile.split(b","))
        assert _count_mutations(profile) == naive, profile


def test_relabel_leaves_unmatched_rows():
    rule = PairRule(mapping={"a": "KIPAN"})
    assert rule.relabel(["a", "b"], ["KIRC", "BRCA"]) == ["KIPAN", "BRCA"]


def test_verify_premises_flags_same_label_duplicates(tmp_path: Path):
    """프로파일이 같은데 라벨도 같으면 "중복 = 코호트 짝" 전제가 깨진 것이다."""
    genes = ["A", "B", "C", "D"]
    _write(
        tmp_path / "train.csv",
        ["ID", "SUBCLASS", *genes],
        [
            ["TR1", "BRCA", "m1", "m2", "m3", "WT"],
            ["TR2", "BRCA", "m1", "m2", "m3", "WT"],
        ],
    )
    _write(tmp_path / "test.csv", ["ID", *genes], [["TE1", "q1", "q2", "q3", "WT"]])
    rule = build_pair_rule(tmp_path / "train.csv", tmp_path / "test.csv", 3)
    reasons = rule.verify_premises()
    assert any("라벨도 같은 묶음" in r for r in reasons)


def test_apply_to_submission_rejects_wrong_columns(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"ID": ["a"], "label": ["BRCA"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="제출 파일 컬럼"):
        apply_to_submission(PairRule(), bad, tmp_path / "out.csv")


def test_apply_to_submission_writes_pair_labels(tmp_path: Path):
    source = tmp_path / "sub.csv"
    pd.DataFrame({"ID": ["a", "b"], "SUBCLASS": ["KIRC", "BRCA"]}).to_csv(source, index=False)
    rule = PairRule(mapping={"a": "KIPAN"})
    info = apply_to_submission(rule, source, tmp_path / "out.csv")

    written = pd.read_csv(tmp_path / "out.csv")
    assert written["SUBCLASS"].tolist() == ["KIPAN", "BRCA"]
    assert info["n_changed"] == 1
    # 바꾸기 전에 train 라벨(KIRC)을 복사하고 있었다 — 규칙이 노리는 상황이다.
    assert info["n_was_copying_train_label"] == 1


def test_cli_batch_output_names(tmp_path: Path):
    """입력을 여러 개 주면 `<원본>_pairrule_m<min-mut>.csv` 로 나간다."""
    args = apply_pair_rule_cli.argparse.Namespace(
        submission=[tmp_path / "a.csv", tmp_path / "b.csv"],
        out=None,
        out_dir=tmp_path / "out",
        min_mut=3,
    )
    outputs = apply_pair_rule_cli.resolve_outputs(args)
    assert [p.name for p in outputs] == ["a_pairrule_m3.csv", "b_pairrule_m3.csv"]
    assert all(p.parent == tmp_path / "out" for p in outputs)


def test_cli_rejects_single_out_for_many_inputs(tmp_path: Path):
    args = apply_pair_rule_cli.argparse.Namespace(
        submission=[tmp_path / "a.csv", tmp_path / "b.csv"],
        out=tmp_path / "one.csv",
        out_dir=None,
        min_mut=3,
    )
    with pytest.raises(SystemExit):
        apply_pair_rule_cli.resolve_outputs(args)


def test_real_data_premises_hold():
    """원본 csv 가 있으면 규칙의 전제를 실제 데이터에서 다시 확인한다.

    이 넷 중 하나라도 깨지면 규칙의 근거가 무너진 것이므로 제출하면 안 된다.
    """
    rule = build_pair_rule(requires_raw("train.csv"), requires_raw("test.csv"), min_mut=3)
    diagnostics = rule.diagnostics

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
    assert set(rule.mapping.values()) <= set(PAIR)

    # 5. 전제 검사가 실제 데이터에서 통과한다 — CLI 가 이걸 보고 멈출지 정한다.
    assert rule.verify_premises() == []


def test_applied_submission_keeps_schema(tmp_path: Path):
    """규칙을 적용해도 제출 스키마가 유지된다 (ID 순서·행 수·컬럼)."""
    sample_path = requires_raw("sample_submission.csv")
    sample = pd.read_csv(sample_path)
    rule = build_pair_rule(requires_raw("train.csv"), requires_raw("test.csv"), min_mut=3)

    apply_to_submission(rule, sample_path, tmp_path / "out.csv")
    out = pd.read_csv(tmp_path / "out.csv")

    assert list(out.columns) == ["ID", "SUBCLASS"]
    assert len(out) == len(sample)
    assert (out["ID"].to_numpy() == sample["ID"].to_numpy()).all()
    assert out["SUBCLASS"].notna().all()
