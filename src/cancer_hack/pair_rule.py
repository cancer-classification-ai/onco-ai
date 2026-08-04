"""exact-match 짝 라벨 규칙 — 제출 csv 에서 코호트 짝이 뒤집힌 214행을 되돌린다.

무엇을 하는가
-------------
`train.csv` 안에서 유전자 프로파일(4,384열)이 바이트 단위로 완전히 같은 행 묶음은
**예외 없이 라벨이 갈리고, 예외 없이 TCGA 코호트 짝이다**(422/422, 같은 라벨 묶음 0개).

    KIPAN = KICH ∪ KIRC ∪ KIRP        GBMLGG = GBM ∪ LGG

같은 환자가 상위·하위 코호트 라벨로 두 번 들어가 있다는 뜻이다. 그 짝 중 한쪽만 train 에
있는 "고아" 프로파일의 반대편이 test 로 갔다. 근거는 고아 수와 test 매칭 수가 정확히 맞는
것이다 — train 의 짝 없는 KIRC 57개는 전부 KIPAN 사본이 test 에 있고, LGG 50개도 같다.

따라서 test 행이 train 의 **유일한** 프로파일과 완전히 같으면 정답은 매칭된 train 라벨이
아니라 그 **짝 라벨**이다. 규칙 없이 낸 제출은 이 구간에서 train 라벨을 99.5% 복사한다.

LB 로 확인됐다 (2026-08-04)
---------------------------
    v002_seed42_f16_group5              CV 0.5165 · LB 0.3896    규칙 없음
    ens16_cbopt10_seed3 + 짝 규칙       CV 0.5210 · LB 0.4725    규칙 있음

**+0.0829.** 두 점 사이에서 축이 셋 움직였지만(짝 규칙 · CatBoost 튜닝 · 3-seed 평균)
나머지 두 축의 CV 기여를 합치면 +0.0045 뿐이라 델타의 대부분이 이 214행이다.
자세한 근거와 추정식은 `docs/pair_rule.md`.

규정
----
규칙은 `train.csv` 에서만 유도되고, 적용에는 test 한 행이면 된다(프로파일을 train 테이블에
조회). test 통계도 test 라벨도 쓰지 않으므로 "test 한 행만 따로 넣어도 같은 결과가 나오는가"
판정을 만족한다. 사실상 짝 라벨로 변환하는 1-NN 이다.

**하지 않는 것: test 내부 중복 매칭.** test 두 행의 프로파일이 서로 같으면 구조상 한쪽은
KIPAN, 한쪽은 KIRC 여야 한다. 하지만 그건 다른 test 행을 봐야 정해지므로 행 단위 독립이
깨진다. 규정의 포괄 조항에 걸릴 소지가 있어 쓰지 않는다.

왜 CV 로 검증할 수 없는가
-------------------------
group CV 는 같은 프로파일을 한 fold 로 묶어 valid 프로파일의 짝을 train 에서 빼 버린다.
즉 이 상황 자체를 못 만든다. 예측 분포도 짝 안에서 스왑이라 총량이 보존돼 TVD 로도 안 보인다.
**로컬 점수로는 이 개선을 잴 수 없다.** 그래서 구현이 조용히 틀려도 제출하기 전까지는
아무도 모른다 — `verify_premises()` 가 매 실행마다 규칙의 전제 네 가지를 다시 재는 이유다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: 상위 ↔ 하위 코호트. 이 매핑은 사전에 고정하고 LB 결과에 맞춰 고치지 않는다.
PAIR: dict[str, str] = {
    "KIPAN": "KIRC",
    "KIRC": "KIPAN",
    "GBMLGG": "LGG",
    "LGG": "GBMLGG",
}

#: 매칭에 요구하는 최소 변이 수. 프로파일이 우연히 일치할 확률은 변이 수가 적을수록 크다.
#: 실측으로 짝 4종이 아닌 매칭이 나오는 구간은 `n_mut == 1` 하나뿐이라(LAML 1건) 여유를
#: 두고 3 으로 잡았다. 구간별 표는 `docs/pair_rule.md` 「왜 3인가」.
DEFAULT_MIN_MUT = 3

SUBMISSION_COLUMNS = ["ID", "SUBCLASS"]
SUBMISSION_ENCODING = "UTF-8-sig"


@dataclass(frozen=True)
class PairRule:
    """test ID → 짝 라벨 매핑과, 그 매핑을 만든 근거 수치."""

    mapping: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.mapping)

    def relabel(self, ids: Iterable[str], labels: Iterable[str]) -> list[str]:
        """예측 라벨 열에 규칙을 얹는다. 대상이 아닌 행은 그대로 둔다."""
        return [self.mapping.get(sample_id, label) for sample_id, label in zip(ids, labels)]

    def verify_premises(self) -> list[str]:
        """전제가 깨진 사유를 모아 낸다. 빈 리스트면 통과다.

        넷 다 `train.csv` 안에서만 확인되는 성질이라 test 라벨 없이도 잴 수 있다.
        하나라도 깨지면 규칙의 근거가 무너진 것이므로 그 제출은 올리면 안 된다.
        """
        d = self.diagnostics
        reasons: list[str] = []

        if d.get("train_dup_groups_same_label"):
            reasons.append(
                f"프로파일이 같은데 라벨도 같은 묶음이 {d['train_dup_groups_same_label']}개다 "
                "— 중복이 코호트 짝 때문이라는 전제가 깨졌다"
            )
        if not d.get("train_dup_groups_pair_label"):
            reasons.append("코호트 짝 묶음이 하나도 없다 — 규칙을 세울 근거가 없다")

        orphans = d.get("train_orphans_by_label") or {}
        matched = d.get("test_matched_by_train_label") or {}
        for label in ("KIRC", "LGG"):
            if orphans.get(label) != matched.get(label):
                reasons.append(
                    f"{label} 고아 {orphans.get(label)}개와 test 매칭 {matched.get(label)}개가 "
                    "어긋난다 — 고아의 짝이 test 에 있다는 근거가 이 일치였다"
                )

        off_pair = d.get("test_matched_off_pair_labels") or {}
        if off_pair:
            reasons.append(f"짝 4종이 아닌 매칭이 있다: {off_pair} — 우연 일치가 섞였다는 신호다")

        if not set(self.mapping.values()) <= set(PAIR):
            reasons.append("짝 4종 밖의 라벨로 바꾸려 한다")

        return reasons


def _iter_data_lines(path: Path) -> Iterator[bytes]:
    """헤더를 건너뛰고 개행을 뗀 데이터 줄을 흘린다.

    프로파일을 파싱하지 않고 **원문 바이트 그대로** 다룬다. pandas 로 읽으면 빈 셀이
    NaN 이 되고 dtype 추론이 끼어들어 "완전히 같은가" 판정이 흔들린다. 줄 단위로
    흘리므로 test 쪽은 35MB 를 통째로 들고 있지 않아도 된다.
    """
    with path.open("rb") as handle:
        handle.readline()  # 헤더
        for line in handle:
            line = line.rstrip(b"\r\n")
            if line:
                yield line


def _count_mutations(profile: bytes) -> int:
    """`WT` 도 빈 셀도 아닌 칸의 수.

    `list.count` 는 C 쪽에서 도는데 `sum(cell not in ... for cell in ...)` 는 4,384칸을
    파이썬 루프로 도는 차이라, 같은 값을 3배쯤 빨리 낸다. `WT` 와 빈 셀은 서로 겹치지
    않으므로 전체에서 둘을 빼면 된다.
    """
    cells = profile.split(b",")
    return len(cells) - cells.count(b"WT") - cells.count(b"")


def _digest(profile: bytes) -> bytes:
    return hashlib.blake2b(profile, digest_size=16).digest()


def _load_train_profiles(train_csv: Path) -> dict[bytes, list[tuple[str, bytes]]]:
    """프로파일 해시 → `[(라벨, 프로파일 원문), ...]`.

    해시는 후보를 좁히는 용도일 뿐이고 최종 판정은 항상 원문 비교다. 그래서 원문을
    같이 들고 있는다(train 82MB).
    """
    table: dict[bytes, list[tuple[str, bytes]]] = defaultdict(list)
    for line in _iter_data_lines(train_csv):
        id_end = line.find(b",")
        label_end = line.find(b",", id_end + 1)
        label = line[id_end + 1 : label_end].decode()
        profile = line[label_end + 1 :]
        table[_digest(profile)].append((label, profile))
    return table


def _train_diagnostics(
    table: dict[bytes, list[tuple[str, bytes]]], min_mut: int
) -> tuple[int, int, Counter]:
    """train 안의 짝 구조를 센다 — 같은 라벨 묶음 · 짝 라벨 묶음 · 라벨별 고아 수."""
    same_label = pair_label = 0
    orphans: Counter = Counter()

    for rows in table.values():
        if len(rows) == 1:
            label, profile = rows[0]
            if label in PAIR and _count_mutations(profile) >= min_mut:
                orphans[label] += 1
        elif _count_mutations(rows[0][1]) >= min_mut:
            if len({label for label, _ in rows}) == 1:
                same_label += 1
            elif len(rows) == 2 and PAIR.get(rows[0][0]) == rows[1][0]:
                pair_label += 1

    return same_label, pair_label, orphans


def build_pair_rule(
    train_csv: str | Path,
    test_csv: str | Path,
    min_mut: int = DEFAULT_MIN_MUT,
) -> PairRule:
    """train 을 한 번 훑어 test ID → 짝 라벨 매핑을 만든다.

    한 번 만들어 두면 제출 파일 여러 개에 그대로 쓸 수 있다 — 후보마다 다시 만들면
    82MB + 35MB 를 매번 다시 읽는 데다 `min_mut` 이 후보끼리 어긋날 여지가 생긴다.
    """
    train_csv, test_csv = Path(train_csv), Path(test_csv)
    table = _load_train_profiles(train_csv)
    same_label, pair_label, orphans = _train_diagnostics(table, min_mut)

    mapping: dict[str, str] = {}
    matched: Counter = Counter()
    off_pair: Counter = Counter()

    for line in _iter_data_lines(test_csv):
        id_end = line.find(b",")
        profile = line[id_end + 1 :]
        candidates = table.get(_digest(profile))
        if not candidates:
            continue
        hits = [label for label, known in candidates if known == profile]
        # 매칭이 둘 이상이면 양쪽 라벨이 이미 train 에 있다는 뜻이라 어느 쪽인지 모른다.
        if len(hits) != 1 or _count_mutations(profile) < min_mut:
            continue
        label = hits[0]
        if label in PAIR:
            mapping[line[:id_end].decode()] = PAIR[label]
            matched[label] += 1
        else:
            off_pair[label] += 1

    diagnostics: dict[str, object] = {
        "min_mut": min_mut,
        "train_dup_groups_same_label": same_label,
        "train_dup_groups_pair_label": pair_label,
        "train_orphans_by_label": dict(orphans),
        "test_matched_by_train_label": dict(matched),
        "test_matched_off_pair_labels": dict(off_pair),
        "n_flipped": len(mapping),
    }
    return PairRule(mapping=mapping, diagnostics=diagnostics)


def apply_to_submission(
    rule: PairRule, submission_csv: str | Path, output_csv: str | Path
) -> dict[str, object]:
    """제출 csv 에 규칙을 얹어 새 파일로 쓴다. 원본은 건드리지 않는다."""
    submission_csv, output_csv = Path(submission_csv), Path(output_csv)
    submission = pd.read_csv(submission_csv)
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"제출 파일 컬럼이 {SUBMISSION_COLUMNS} 가 아니다: {list(submission.columns)}"
        )

    before = submission["SUBCLASS"].to_numpy().copy()
    submission["SUBCLASS"] = rule.relabel(submission["ID"], submission["SUBCLASS"])
    after = submission["SUBCLASS"].to_numpy()

    # 규칙 대상인데 이미 짝 라벨을 예측하고 있었다면 바뀌는 행이 아니다. 반대로
    # "바꾸기 전에 train 라벨을 그대로 복사하고 있었다"가 규칙이 노리는 상황이다.
    copied = sum(
        1
        for sample_id, label in zip(submission["ID"], before)
        if sample_id in rule.mapping and PAIR.get(label) == rule.mapping[sample_id]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False, encoding=SUBMISSION_ENCODING)

    return {
        "submission": str(submission_csv),
        "output": str(output_csv),
        "rows": int(len(submission)),
        "n_changed": int((after != before).sum()),
        "n_was_copying_train_label": copied,
    }
