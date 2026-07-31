"""변이 문자열 파서 — 한 셀의 복합 변이 값을 토큰 단위로 분류한다.

한 셀에는 변이가 여러 개 들어갈 수 있고 공백으로 구분된다(`'Q369* I368N'`).
여기서는 셀 값을 토큰으로 쪼갠 뒤 각 토큰을 유형 하나로 분류하고, 셀 단위
집계값(유형별 존재 여부·개수, 토큰 수, 중복 여부)을 만든다.

분류 순서가 결과를 바꾸므로 아래 순서를 지킨다. 근거는 train/test 실측이다.

1. frameshift  `fs` 포함             (train 9,911 · test 25,832 토큰)
2. delins      `delins` 포함          (train 0 · test 545)
3. deletion    `del` 포함             (train 3 · test 2,583)
4. insertion   `ins` 또는 `dup` 포함   (train 0 · test 1,142)
5. nonsense    `^[A-Z]\\d+[*X]$`      (train `*` 13,289 · test `X` 14,477 + `*` 1,961)
6. 치환         `^[A-Z*]\\d+[A-Z*]$`   ref == alt 면 synonymous, 아니면 missense
7. complex     `_` 또는 `>` 포함       (train 239 · test 2,456)
8. other       위 어디에도 안 걸리는 것 (train 0 · test 2)

순서가 중요한 지점 세 곳:

- **frameshift가 deletion보다 먼저**여야 in-frame 여부가 갈린다. 이 데이터에서는
  `fs`와 `del|ins|dup`를 동시에 가진 토큰이 train·test 모두 0개라 두 집합이 애초에
  겹치지 않지만, 새 표기가 들어오면 이 순서가 판정을 결정한다.
- **deletion 계열이 complex보다 먼저**여야 한다. test에만 있는 `P11_K12insP`,
  `R376_A377delinsP` 같은 in-frame indel이 `_`를 쓰기 때문에, complex를 먼저 보면
  test 토큰 2,400여 개가 통째로 complex로 오분류된다.
- **nonsense가 치환보다 먼저**여야 한다. test는 정지코돈을 `*`가 아니라 `X`로
  재코딩해 뒀는데(`Q369X`), 치환 규칙 `^[A-Z]\\d+[A-Z]$`가 이걸 먼저 삼키면
  test에서 missense로 둔갑한다. 실측으로 test 토큰 14,477개가 뒤집힌다.

정지코돈 표기가 train `*` / test `X`로 갈리는 문제는 두 표기를 모두 nonsense로
받아 해결한다. 이 처리를 빼면 `has_nonsense`가 train 52.67% / test 26.87%로 벌어져
피처가 통째로 죽는다.

위치(숫자)는 분류에 쓰지 않는다. test가 같은 변이를 여러 전사체 좌표로 중복
기재해서 절대 위치가 train과 맞지 않기 때문이다.

## 유형은 배타적이다 — `has_mnv` 를 문자열 포함으로 보지 않는 이유

1차 전략표는 `has_mnv` 를 "`_` 또는 `>` 를 쓰는 표기"로, `has_indel` 을
"`del`·`ins`·`dup` 가 포함된 토큰"으로 적어 두었다. 이걸 글자 그대로 독립 플래그로
구현하면 `R376_A377delinsP` 가 indel 이면서 동시에 mnv 가 된다. 실측하면 이렇다.

                    train              test
  has_indel   포함 0.05% / 배타 0.05%   포함 28.79% / 배타 28.79%   (차이 없음)
  has_mnv     포함 2.94% / 배타 2.94%   포함 24.35% / 배타  1.30%   (587명 차이)

`has_indel` 은 어느 쪽으로 구현해도 결과가 같다. 반면 `has_mnv` 를 문자열 포함으로
두면 test 에서 24.35%로 뛰는데, 늘어난 2,416 토큰이 전부 `A630_P649del`,
`P11_K12insP` 같은 **범위 표기 indel** 이다. 다중 잔기 치환(`312_313QY>HH`)은
test 에 40토큰뿐이다. 즉 문자열 포함 정의로 가면 train 2.94% / test 24.35% 라는
8배 격차가 생기고, 그 격차의 정체는 생물학이 아니라 test 주석 파이프라인의
표기 습관이다. 그래서 배타 분류를 쓴다.

두 이름 모두 `CellMutation` 에 남아 있다. `has_indel`·`indel_count` 는 세분 유형
(deletion/insertion/delins)의 합집합이고, `has_mnv`·`mnv_count` 는 complex 와 같은
값이다. 세분 유형에서 유도하므로 서로 어긋날 수 없다.
"""


from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass

import pandas as pd
import re


# 변이 없음을 뜻하는 값들. 대문자로 바꿔 비교한다.
EMPTY_VALUES = frozenset({"", "WT", "0", "NA", "NAN", "NONE", "."})

# 유형 이름 — 피처 컬럼 접두사로 그대로 쓰인다.
MISSENSE = "missense"
SYNONYMOUS = "synonymous"
NONSENSE = "nonsense"
FRAMESHIFT = "frameshift"
DELETION = "deletion"
INSERTION = "insertion"
DELINS = "delins"
COMPLEX = "complex"
OTHER = "other"

# 셀 단위 피처를 만들 유형. other 는 진단용이라 따로 둔다.
MUTATION_KINDS: tuple[str, ...] = (
    MISSENSE,
    SYNONYMOUS,
    NONSENSE,
    FRAMESHIFT,
    DELETION,
    INSERTION,
    DELINS,
    COMPLEX,
)
ALL_KINDS: tuple[str, ...] = MUTATION_KINDS + (OTHER,)

# 단백질 서열이 바뀌는 유형. `functional_event_count` 의 정의다.
FUNCTIONAL_KINDS: tuple[str, ...] = (
    MISSENSE,
    NONSENSE,
    FRAMESHIFT,
    DELETION,
    INSERTION,
    DELINS,
    COMPLEX,
)

# 잔기가 실제로 빠지는 유형. `explicit_deletion_event_count` 의 정의다.
# delins 는 삭제 후 삽입이라 삭제 쪽에 포함한다.
DELETION_KINDS: tuple[str, ...] = (DELETION, DELINS)

# 1차 전략표의 `has_indel` — "del, ins, dup 중 하나가 포함된 토큰".
# 아래 세 유형의 합집합이 그 정의와 정확히 같다. train/test 전 행에서 문자열 포함
# 판정과 결과가 한 건도 다르지 않은 걸 확인했다(`fs` 와 `del|ins|dup` 를 동시에
# 가진 토큰이 데이터에 0개라서 배타 분류로도 손실이 없다).
INDEL_KINDS: tuple[str, ...] = (DELETION, INSERTION, DELINS)

# --- 토큰 분류 정규식 (여기가 유일한 정의다) --------------------------------
#
# 예전에는 같은 이름이 이 파일 안에서 두 번 정의돼 있었다. 파이썬은 나중 정의가
# 이기므로 아래쪽 것이 위쪽을 조용히 덮어썼고, 그 결과 `classify_token` 이
# `Q369*` 를 missense 로 분류했다(정지코돈 판정이 아예 안 걸렸다). 이름 하나당
# 정의 하나만 둔다.
_FRAMESHIFT_RE = re.compile(r"fs")
_DELINS_RE = re.compile(r"delins")
_DELETION_RE = re.compile(r"del")
_INSERTION_RE = re.compile(r"ins|dup")
# 정지코돈은 train `*` / test `X` 두 표기를 모두 받는다.
_NONSENSE_RE = re.compile(r"^[A-Z]\d+[*X]$")
# ref/alt 에 `*` 를 허용해 `*261*`(정지코돈 자리의 동의 변이)까지 잡는다.
_SUBSTITUTION_RE = re.compile(r"^([A-Z*])\d+([A-Z*])$")
_COMPLEX_RE = re.compile(r"[_>]")

# --- 분류에 쓰지 않는 보조 정규식 -------------------------------------------
# 유전자 3단계 인코딩(`features_basic.encode_mutation`)이 쓴다.
_SYNONYMOUS_RE = re.compile(r"^([A-Z])\d+\1$")
# `del` 로 끝나는 토큰만 센다(`R649del`, `E746_A750del`). 명시적 결실 집계용이라
# `_DELETION_RE`(부분일치)와 목적이 다르다.
_INDEL_RE = re.compile(r"del$", re.IGNORECASE)

#: `EMPTY_VALUES` 의 별칭. 값이 같은 집합을 두 벌 두면 한쪽만 고쳤을 때 어긋난다.
_MUTATION_EMPTY: frozenset[str] = EMPTY_VALUES


def split_tokens(value: object) -> list[str]:
    """셀 값을 변이 토큰 리스트로 쪼갠다. WT·결측이면 빈 리스트.

    >>> split_tokens("S622S G827R")
    ['S622S', 'G827R']
    >>> split_tokens("WT")
    []
    """
    if value is None:
        return []
    text = str(value).strip()
    if text.upper() in EMPTY_VALUES:
        return []
    return text.split()


def classify_token(token: str) -> str:
    """변이 토큰 하나를 유형 이름으로 분류한다.

    모듈 docstring의 순서를 그대로 따른다. 순서를 바꾸면 train/test 사이에서
    유형이 어긋난다.

    >>> classify_token("R649del")
    'deletion'
    >>> classify_token("R376_A377delinsP")
    'delins'
    >>> classify_token("468_469LG>F*")
    'complex'
    """
    if _FRAMESHIFT_RE.search(token):
        return FRAMESHIFT
    if _DELINS_RE.search(token):
        return DELINS
    if _DELETION_RE.search(token):
        return DELETION
    if _INSERTION_RE.search(token):
        return INSERTION
    if _NONSENSE_RE.match(token):
        return NONSENSE

    matched = _SUBSTITUTION_RE.match(token)
    if matched:
        ref, alt = matched.group(1), matched.group(2)
        return SYNONYMOUS if ref == alt else MISSENSE

    if _COMPLEX_RE.search(token):
        return COMPLEX
    return OTHER


# --- 서명(signature) — 잔기 번호를 지우고 유형과 잔기만 남긴다 ----------------
#
# test 는 같은 변이를 전사체마다 다른 좌표로 반복 기재한다(`M267I M206I`). 문자열
# 정확 일치로는 그게 중복으로 안 잡힌다 — 실측 잉여 토큰이 문자열 기준 test 218 개인데
# 서명 기준으로는 134,430 개다. 위치를 지우면 그 중복이 드러난다.
_SIG_DIGITS = re.compile(r"\d+")
_SIG_SUBSTITUTION = re.compile(r"^([A-Z*])\d+([A-Z*])$")
_SIG_NONSENSE = re.compile(r"^([A-Z])\d+[*X]$")
#: 첫 숫자 앞까지가 ref. `K16fs` -> `K`, `K437Rfs` -> `K`, `-287fs` -> `-`.
_SIG_FRAMESHIFT = re.compile(r"^(.*?)\d+[A-Z*]*fs")
_SIG_COMPLEX = re.compile(r"^\d+_\d+([A-Z*]+)>([A-Z*]+)$")


def token_signature(token: str, kind: str | None = None) -> str:
    """변이 토큰의 **위치 무시 서명**. 같은 사건이면 같은 문자열이 나온다.

    형식은 `{유형}|{본문}` 이고 유형은 `classify_token` 이 정한 값을 그대로 쓴다 —
    서명이 분류를 다시 하지 않는다. `kind` 를 넘기면 재분류를 건너뛴다.

    ## frameshift 는 왜 alt 잔기를 버리는가

    train 의 frameshift 9,911 토큰은 **전부** `K16fs` 형태로 alt 가 없고, test 는
    25,832 중 22,221 이 `K437Rfs` 형태로 alt 를 달고 있다. alt 를 서명에 남기면
    두 집합의 frameshift 어휘가 완전히 갈라져 이 축이 통째로 죽는다.

    ## 정지코돈 `*` 와 `X`

    nonsense 본문은 ref 만 남기므로 train `Q369*` 와 test `Q369X` 의 표기 차이가
    자동으로 사라진다. `X` 를 전역 치환하지 않는 이유는 `X541delinsX` 처럼 미상
    잔기로도 쓰이기 때문이다 — 전역 치환은 그쪽을 망가뜨린다.

    >>> token_signature("V600E")
    'missense|V>E'
    >>> token_signature("M267I") == token_signature("M206I")
    True
    >>> token_signature("Q369*") == token_signature("Q369X")
    True
    >>> token_signature("K16fs") == token_signature("K437Rfs")
    True
    >>> token_signature("312_313QY>HH")
    'complex|QY>HH'
    >>> token_signature("A630_P649del")
    'deletion|A_Pdel'
    """
    if kind is None:
        kind = classify_token(token)

    if kind == MISSENSE or kind == SYNONYMOUS:
        matched = _SIG_SUBSTITUTION.match(token)
        if matched:
            ref, alt = matched.group(1), matched.group(2)
            # 동의 변이는 ref == alt 라 alt 가 중복 정보다.
            return f"{kind}|{ref}" if kind == SYNONYMOUS else f"{kind}|{ref}>{alt}"
    elif kind == NONSENSE:
        matched = _SIG_NONSENSE.match(token)
        if matched:
            return f"{kind}|{matched.group(1)}"
    elif kind == FRAMESHIFT:
        matched = _SIG_FRAMESHIFT.match(token)
        if matched:
            return f"{kind}|{matched.group(1)}"
    elif kind == COMPLEX:
        matched = _SIG_COMPLEX.match(token)
        if matched:
            return f"{kind}|{matched.group(1)}>{matched.group(2)}"

    # deletion·insertion·delins·other, 그리고 위 정규식이 안 걸린 잔여. 숫자만
    # 지운다 — 새 표기가 들어와도 서명이 없어서 터지는 일은 없다.
    return f"{kind}|" + _SIG_DIGITS.sub("", token)


@dataclass(slots=True)
class CellMutation:
    """셀 하나에서 뽑은 복합 변이 피처.

    `*_count` 는 중복 토큰을 각각 세고, `unique_*_count` 는 셀 안에서 문자열이
    같은 토큰을 하나로 접은 뒤 센다. test 는 같은 변이를 여러 전사체 좌표로 적어
    두기 때문에 문자열 중복 제거만으로는 그 중복이 걸러지지 않는다 — 좌표가 다르면
    다른 문자열이다. 그래서 잉여 토큰 수를 두 기준으로 함께 센다.

    `duplicate_token_count`      문자열이 같은 잉여 토큰. train 6,100 · test 218
    `duplicate_signature_count`  `token_signature` 가 같은 잉여 토큰.
                                 train 8,072 · test 134,430

    test 쪽 두 숫자의 격차가 이소폼 중복 기재의 크기다. 서명 기준이 문자열 기준보다
    거친 판정이므로 `duplicate_signature_count >= duplicate_token_count` 가 항상
    성립한다.
    """

    has_missense: int = 0
    has_synonymous: int = 0
    has_nonsense: int = 0
    has_frameshift: int = 0
    has_deletion: int = 0
    has_insertion: int = 0
    has_delins: int = 0
    has_complex: int = 0
    # 1차 전략표 이름. has_indel 은 위 deletion/insertion/delins 의 합집합이고
    # has_mnv 는 has_complex 와 같은 값이다. 세분화된 유형만 두면 1차 표를 쓰는
    # 코드가 깨지므로 두 이름을 함께 유지한다.
    has_indel: int = 0
    has_mnv: int = 0

    mutation_token_count: int = 0
    unique_mutation_token_count: int = 0
    has_duplicate_token: int = 0
    duplicate_token_count: int = 0
    duplicate_signature_count: int = 0

    missense_count: int = 0
    synonymous_count: int = 0
    nonsense_count: int = 0
    frameshift_count: int = 0
    deletion_count: int = 0
    insertion_count: int = 0
    delins_count: int = 0
    complex_count: int = 0
    indel_count: int = 0
    mnv_count: int = 0

    unique_missense_count: int = 0
    unique_synonymous_count: int = 0
    unique_nonsense_count: int = 0
    unique_frameshift_count: int = 0
    unique_deletion_count: int = 0
    unique_insertion_count: int = 0
    unique_delins_count: int = 0
    unique_complex_count: int = 0
    unique_indel_count: int = 0
    unique_mnv_count: int = 0

    # 위 어느 유형에도 안 걸린 토큰. 파서가 새 표기를 놓치는지 보는 진단용이라
    # 0 이 아니면 표기 형식을 다시 봐야 한다.
    other_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def functional_count(self) -> int:
        """단백질 서열이 바뀌는 토큰 수 — 동의변이와 미분류를 뺀 값."""
        return sum(getattr(self, f"{kind}_count") for kind in FUNCTIONAL_KINDS)

    @property
    def explicit_deletion_count(self) -> int:
        """잔기가 빠지는 토큰 수 — `del` 과 `delins`."""
        return sum(getattr(self, f"{kind}_count") for kind in DELETION_KINDS)


CELL_FEATURE_COLUMNS: tuple[str, ...] = tuple(CellMutation().as_dict())


def parse_cell(value: object) -> CellMutation:
    """셀 값 하나를 `CellMutation` 으로 변환한다.

    >>> parse_cell("S622S G827R").has_synonymous
    1
    >>> parse_cell("V600E V600E").has_duplicate_token
    1
    >>> parse_cell("WT").mutation_token_count
    0
    >>> parse_cell("Q369* I368N").functional_count
    2
    >>> parse_cell("P11_K12insP").has_indel        # 1차 전략표 이름도 함께 채운다
    1
    >>> parse_cell("312_313QY>HH").has_mnv
    1
    >>> parse_cell("M267I M206I").duplicate_token_count      # 문자열은 서로 다르다
    0
    >>> parse_cell("M267I M206I").duplicate_signature_count  # 같은 사건의 이소폼
    1
    """
    tokens = split_tokens(value)
    cell = CellMutation()
    if not tokens:
        return cell

    unique_tokens = set(tokens)
    # 토큰마다 한 번만 분류한다. 아래 두 루프와 서명 계산이 이 사전을 공유한다.
    kind_by_token = {token: classify_token(token) for token in unique_tokens}
    signatures = {
        token_signature(token, kind) for token, kind in kind_by_token.items()
    }

    cell.mutation_token_count = len(tokens)
    cell.unique_mutation_token_count = len(unique_tokens)
    cell.has_duplicate_token = int(len(tokens) > len(unique_tokens))
    cell.duplicate_token_count = len(tokens) - len(unique_tokens)
    cell.duplicate_signature_count = len(tokens) - len(signatures)

    for token in tokens:
        kind = kind_by_token[token]
        setattr(cell, f"{kind}_count", getattr(cell, f"{kind}_count") + 1)

    for token in unique_tokens:
        kind = kind_by_token[token]
        if kind == OTHER:
            continue
        setattr(cell, f"unique_{kind}_count", getattr(cell, f"unique_{kind}_count") + 1)

    for kind in MUTATION_KINDS:
        setattr(cell, f"has_{kind}", int(getattr(cell, f"{kind}_count") > 0))

    # 1차 전략표 이름 채우기 — 세분 유형에서 유도하므로 서로 어긋날 수 없다.
    cell.indel_count = sum(getattr(cell, f"{k}_count") for k in INDEL_KINDS)
    cell.unique_indel_count = sum(
        getattr(cell, f"unique_{k}_count") for k in INDEL_KINDS
    )
    cell.has_indel = int(cell.indel_count > 0)

    cell.mnv_count = cell.complex_count
    cell.unique_mnv_count = cell.unique_complex_count
    cell.has_mnv = cell.has_complex

    return cell

#: 세분 유형(8종) -> `compute_*` 가 기대하는 이름(6종).
#: deletion/insertion/delins 를 `indel` 하나로 접고, 미분류는 `complex` 로 보낸다.
_COARSE_KIND: dict[str, str] = {
    MISSENSE: "missense",
    SYNONYMOUS: "synonymous",
    NONSENSE: "nonsense",
    FRAMESHIFT: "frameshift",
    DELETION: "indel",
    INSERTION: "indel",
    DELINS: "indel",
    COMPLEX: "complex",
    OTHER: "complex",
}


def _parse_mutation_tokens(value: object) -> list[str]:
    """`split_tokens` 의 별칭. 셀 값에서 유효한 변이 토큰만 뽑는다.

    예전에는 같은 일을 하는 함수가 둘이었다. 하나로 모으고 이름만 남긴다 —
    `features_basic` 의 `compute_*` 들이 이 이름으로 부르고 있다.
    """
    return split_tokens(value)


def _classify_token(token: str) -> str:
    """`classify_token` 의 결과를 6종 이름으로 접어 준다.

    분류 규칙은 `classify_token` 하나뿐이다. 예전에는 이 함수가 별도 규칙으로
    다시 판정했는데, 그쪽이 쓰던 `_NONSENSE_RE` 가 위쪽 정의를 덮어써서
    `classify_token` 까지 망가뜨렸다. 이제 판정은 한 군데서만 한다.

    >>> _classify_token("Q369*"), _classify_token("Q369X")
    ('nonsense', 'nonsense')
    >>> _classify_token("R649del"), _classify_token("P11_K12insP")
    ('indel', 'indel')
    """
    return _COARSE_KIND[classify_token(token)]

# 지정한 유전자 컬럼이 모두 존재하는지 확인
def _check_columns(df: pd.DataFrame, gene_columns: list[str]) -> None:
    missing = sorted(set(gene_columns).difference(df.columns))
    if missing:
        raise ValueError(f"Missing gene columns: {missing[:10]}")

# 한 샘플(row)의 모든 mutation token을 분류하여 유형별 개수를 계산
def _row_token_class_counts(row: pd.Series, gene_columns: list[str]) -> Counter:
    counts: Counter = Counter()
    for gene in gene_columns:
        for token in _parse_mutation_tokens(row[gene]):
            counts[_classify_token(token)] += 1
    return counts
