"""제출 csv 에 exact-match 짝 라벨 규칙을 적용한다.

무엇을 하는가
-------------
`train.csv` 안에서 유전자 프로파일(4,383열)이 바이트 단위로 완전히 같은 행 묶음을 찾으면
**예외 없이 라벨이 갈리고, 예외 없이 TCGA 코호트 짝이다**(422/422, 같은 라벨 묶음 0개).

    KIPAN = KICH ∪ KIRC ∪ KIRP        GBMLGG = GBM ∪ LGG

같은 환자가 상위·하위 코호트 라벨로 두 번 들어가 있다는 뜻이다. 그 짝 중 한쪽만 train 에
있는 "고아" 프로파일의 반대편이 test 로 갔다. 근거는 고아 수와 test 매칭 수가 정확히 맞는
것이다 — train 의 짝 없는 KIRC 57개는 전부 KIPAN 사본이 test 에 있고, LGG 50개도 같다.

따라서 test 행이 train 의 **유일한** 프로파일과 완전히 같으면 정답은 매칭된 train 라벨이
아니라 그 **짝 라벨**이다. 지금 모델들은 이 구간에서 train 라벨을 99.5% 복사한다.

규정
----
규칙은 `train.csv` 에서만 유도되고, 적용에는 test 한 행이면 된다(프로파일 해시를 train
테이블에 조회). test 통계·test 라벨을 전혀 쓰지 않으므로 "test 한 행만 따로 넣어도 같은
결과가 나오는가" 판정을 만족한다. 사실상 짝 라벨로 변환하는 1-NN 이다.

왜 CV 로 안 보이는가
--------------------
group CV 는 같은 프로파일을 한 fold 로 묶어 valid 프로파일의 짝을 train 에서 빼 버린다.
즉 이 상황 자체를 못 만든다. 예측 분포도 짝 안에서 스왑이라 총량이 보존돼 TVD 도 안 움직인다.
**로컬 점수로는 이 개선을 잴 수 없다. LB 로만 확인된다.**

`--min-mut` 기본값 3
--------------------
프로파일이 우연히 일치할 확률은 변이 수가 적을수록 높다. train 라벨이 짝 4종이 아닌 매칭이
나오는 구간이 실제로 `n_mut == 1` 하나뿐이고(LAML 1건) `n_mut >= 2` 부터는 0건이다.
안전하게 3 을 기본으로 둔다. 189건은 `n_mut >= 6` 이라 우연으로 보기 어렵다.

사용법
------
    python scripts/apply_pair_rule.py --submission <제출csv> --out <출력csv>
    python scripts/apply_pair_rule.py --submission <제출csv> --out <출력csv> --min-mut 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO / "data" / "raw" / "train.csv"
DEFAULT_TEST = REPO / "data" / "raw" / "test.csv"

#: 상위 ↔ 하위 코호트. 이 매핑은 사전에 고정하고 LB 결과에 맞춰 고치지 않는다.
PAIR = {"KIPAN": "KIRC", "KIRC": "KIPAN", "GBMLGG": "LGG", "LGG": "GBMLGG"}


def _rows(path: Path, *, labelled: bool) -> list[tuple[str | None, str, bytes]]:
    """`(label, id, profile_bytes)` 를 낸다.

    프로파일을 파싱하지 않고 **원문 바이트 그대로** 들고 있는다. pandas 로 읽으면
    빈 셀이 NaN 이 되고 dtype 추론이 끼어들어 "완전히 같은가" 판정이 흔들린다.
    """
    out = []
    with path.open("rb") as handle:
        handle.readline()  # 헤더
        for line in handle:
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            i = line.find(b",")
            if labelled:
                j = line.find(b",", i + 1)
                out.append((line[i + 1 : j].decode(), line[:i].decode(), line[j + 1 :]))
            else:
                out.append((None, line[:i].decode(), line[i + 1 :]))
    return out


def _n_mut(profile: bytes) -> int:
    return sum(cell not in (b"WT", b"") for cell in profile.split(b","))


def _digest(profile: bytes) -> bytes:
    return hashlib.blake2b(profile, digest_size=16).digest()


def build_rule(train_csv: Path, test_csv: Path, min_mut: int) -> tuple[dict[str, str], dict]:
    """test ID -> 짝 라벨 매핑과 진단 정보를 낸다."""
    train_rows = _rows(train_csv, labelled=True)
    test_rows = _rows(test_csv, labelled=False)

    by_hash: dict[bytes, list] = defaultdict(list)
    for label, sample_id, profile in train_rows:
        by_hash[_digest(profile)].append((sample_id, label, profile))

    # train 내부 짝 구조 — 규칙의 전제를 매번 다시 확인한다.
    same_label = pair_label = 0
    for rows in by_hash.values():
        if len(rows) > 1 and _n_mut(rows[0][2]) >= min_mut:
            if len({r[1] for r in rows}) == 1:
                same_label += 1
            elif len(rows) == 2 and PAIR.get(rows[0][1]) == rows[1][1]:
                pair_label += 1

    # 고아(짝이 train 에 없는 행) 수 — 이게 test 매칭 수와 맞는 게 결정적 근거다.
    orphan: Counter = Counter()
    for rows in by_hash.values():
        if len(rows) == 1:
            _, label, profile = rows[0]
            if label in PAIR and _n_mut(profile) >= min_mut:
                orphan[label] += 1

    mapping: dict[str, str] = {}
    matched: Counter = Counter()
    off_pair: Counter = Counter()
    for _, test_id, profile in test_rows:
        hits = [r for r in by_hash.get(_digest(profile), []) if r[2] == profile]
        if len(hits) != 1 or _n_mut(profile) < min_mut:
            continue
        label = hits[0][1]
        if label in PAIR:
            mapping[test_id] = PAIR[label]
            matched[label] += 1
        else:
            off_pair[label] += 1

    diagnostics = {
        "min_mut": min_mut,
        "train_dup_groups_same_label": same_label,
        "train_dup_groups_pair_label": pair_label,
        "train_orphans_by_label": dict(orphan),
        "test_matched_by_train_label": dict(matched),
        "test_matched_off_pair_labels": dict(off_pair),
        "n_flipped": len(mapping),
    }
    return mapping, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", type=Path, required=True, help="입력 제출 csv")
    parser.add_argument("--out", type=Path, required=True, help="출력 제출 csv")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--min-mut",
        type=int,
        default=3,
        help="매칭에 요구하는 최소 변이 수. 낮출수록 우연 일치가 섞인다 (기본 3)",
    )
    parser.add_argument("--log", type=Path, default=None, help="진단 json 을 남길 경로")
    args = parser.parse_args()

    mapping, diagnostics = build_rule(args.train, args.test, args.min_mut)

    submission = pd.read_csv(args.submission)
    if list(submission.columns) != ["ID", "SUBCLASS"]:
        raise ValueError(f"제출 파일 컬럼이 ['ID', 'SUBCLASS'] 가 아니다: {list(submission.columns)}")

    before = submission["SUBCLASS"].to_numpy().copy()
    submission["SUBCLASS"] = [mapping.get(i, c) for i, c in zip(submission["ID"], submission["SUBCLASS"])]
    changed = int((submission["SUBCLASS"].to_numpy() != before).sum())

    copied = sum(1 for i, c in zip(submission["ID"], before) if i in mapping and PAIR.get(c) == mapping[i])
    diagnostics["n_changed"] = changed
    diagnostics["n_was_copying_train_label"] = copied

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False, encoding="UTF-8-sig")

    print(f"train 중복 묶음  같은 라벨 {diagnostics['train_dup_groups_same_label']} / "
          f"짝 라벨 {diagnostics['train_dup_groups_pair_label']}")
    print(f"고아 수      {diagnostics['train_orphans_by_label']}")
    print(f"test 매칭    {diagnostics['test_matched_by_train_label']}")
    if diagnostics["test_matched_off_pair_labels"]:
        print(f"  ⚠ 짝 4종이 아닌 매칭: {diagnostics['test_matched_off_pair_labels']} — 우연 일치 신호다")
    print(f"규칙 대상 {diagnostics['n_flipped']}행 · 실제로 바뀐 행 {changed} · "
          f"바꾸기 전 train 라벨 복사였던 행 {copied}")
    print(f"→ {args.out}")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
