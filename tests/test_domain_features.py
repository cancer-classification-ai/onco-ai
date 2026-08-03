"""`cancer_hack.features_domain` 의 계약 검증.

두 가지를 고정한다.

1. **행 독립성** — 도메인 피처는 다른 행의 값에 영향받지 않는다. 대회 규정이
   금지하는 "train 통계로 test 를 처리" 경로가 구조적으로 없다는 뜻이다.
2. **train/test 표기 차이 흡수** — test 는 정지코돈을 `*` 대신 `X` 로 적는다.
   이걸 같은 유형으로 안 묶으면 `nonsense` 비율이 train 52.67% / test 26.87%
   로 벌어져 피처가 죽는다.
"""

from __future__ import annotations

import csv

import pytest

from cancer_hack.features_domain import (
    DOMAIN_PREFIXES,
    DRIVERS,
    build_pair_map,
    make_domain_features,
    parse_domain_token,
)


def _write_csv(path, rows, *, genes, has_label):
    header = ["ID"] + (["SUBCLASS"] if has_label else []) + list(genes)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


GENES = ("TP53", "EGFR", "B2M")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("R895H", ("missense", "R", "H")),
        ("R895R", ("silent", "R", "R")),
        ("Q369*", ("nonsense", "Q", "*")),
        # test 는 같은 사건을 X 로 재코딩한다 — 위와 같은 유형으로 나와야 한다.
        ("Q369X", ("nonsense", "Q", "*")),
        ("T1497Nfs", ("frameshift", "T", "N")),
        ("A630_P649del", ("inframe_indel", "A", "")),
        ("R376_A377delinsP", ("inframe_indel", "R", "P")),
        ("312_313QY>HH", ("range_sub", "QY", "HH")),
        ("무엇인가", ("unparsed", "", "")),
    ],
)
def test_token_classification(token, expected):
    assert parse_domain_token(token) == expected


def test_stop_codon_notations_agree():
    """`*` 와 `X` 가 갈리면 test 에서 nonsense 가 통째로 미스센스로 둔갑한다."""
    assert parse_domain_token("Q369*") == parse_domain_token("Q369X")


def test_pair_map_is_a_distribution():
    """SBS6 채널 가중치는 경로 수로 나눈 값이라 합이 1이어야 한다."""
    pair_map = build_pair_map()
    assert pair_map, "코돈 역추론 맵이 비었다"
    for (ref, alt), (channels, cpg, transition) in pair_map.items():
        assert sum(channels.values()) == pytest.approx(1.0), f"{ref}>{alt}"
        assert 0.0 <= cpg <= 1.0
        assert 0.0 <= transition <= 1.0


def test_block_prefixes_do_not_overlap():
    """`"A2_TP53_lof".startswith("A_")` 가 True 가 되면 블록 집계가 어긋난다."""
    assert not "A2_TP53_lof".startswith("A_")
    assert "A_TP53".startswith("A_")


def test_driver_panel_is_deduplicated():
    """문헌 목록에 NPM1·KMT2D 가 두 번 적혀 있다 — set 으로 접혀야 한다."""
    assert len(DRIVERS) == len(set(DRIVERS))
    assert "TP53" in DRIVERS


def test_features_are_row_independent(tmp_path):
    """한 행의 피처는 같은 파일에 어떤 행이 더 있든 똑같아야 한다.

    이게 깨지면 train 통계가 test 로 새는 경로가 생기고, 대회 규정상 수상 제외
    사유가 된다. 같은 행을 혼자 담은 파일과 다른 행들과 함께 담은 파일에서
    뽑아 비교한다.
    """
    # 셀 값은 변이 표기만 담는다 — 유전자 이름은 컬럼 헤더에 있다.
    target = ["s1", "R248W", "L858R", "WT"]
    crowd = [
        target,
        ["s2", "R248Q R249S", "WT", "L15Ffs"],
        ["s3", "WT", "WT", "WT"],
    ]

    alone_path = _write_csv(tmp_path / "alone.csv", [target], genes=GENES, has_label=False)
    crowd_path = _write_csv(tmp_path / "crowd.csv", crowd, genes=GENES, has_label=False)

    alone = make_domain_features(alone_path, has_label=False)
    together = make_domain_features(crowd_path, has_label=False)

    shared = [c for c in alone.columns if c in together.columns and c != "ID"]
    assert shared, "비교할 공통 열이 없다"
    row_alone = alone.loc[0, shared]
    row_together = together.loc[0, shared]
    assert (row_alone == row_together).all(), (
        "행 독립성이 깨졌다: "
        f"{[c for c in shared if row_alone[c] != row_together[c]][:5]}"
    )


def test_duplicate_tokens_in_a_cell_are_folded(tmp_path):
    """셀 안에서 `(유형, ref, alt)` 가 같은 토큰은 한 번만 센다.

    test 는 같은 변이를 여러 전사체 좌표로 중복 기재한다(토큰 39.8% 감소).
    접지 않으면 TMB 가 test 에서 1.7배 부푼다.
    """
    once = _write_csv(
        tmp_path / "once.csv", [["s1", "R248W", "WT", "WT"]], genes=GENES, has_label=False
    )
    twice = _write_csv(
        # 같은 치환을 두 번 적은 셀 — 이소폼 중복 기재를 흉내 낸다.
        tmp_path / "twice.csv",
        [["s1", "R248W R248W", "WT", "WT"]],
        genes=GENES,
        has_label=False,
    )
    a = make_domain_features(once, has_label=False)
    b = make_domain_features(twice, has_label=False)
    assert a.loc[0, "B_ntok"] == b.loc[0, "B_ntok"] == 1
    assert a.loc[0, "B_multihit"] == b.loc[0, "B_multihit"] == 0


def test_label_column_only_when_asked(tmp_path):
    labelled = _write_csv(
        tmp_path / "train.csv",
        [["s1", "BRCA", "R248W", "WT", "WT"]],
        genes=GENES,
        has_label=True,
    )
    frame = make_domain_features(labelled, has_label=True)
    assert frame.loc[0, "SUBCLASS"] == "BRCA"
    assert list(frame.columns)[0] == "ID"
    assert any(c.startswith(DOMAIN_PREFIXES) for c in frame.columns)
