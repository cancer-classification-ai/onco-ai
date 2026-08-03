"""도메인 지식 기반 피처 블록 — 드라이버·TMB·유형조성·치환쌍·코돈 역추론.

`features_basic` 이 변이 문자열을 기계적으로 집계한다면 여기는 암유전체 도메인
지식을 넣는다. 블록 ablation 에서 이 539열이 단독으로 Macro F1 0.4272 를 냈다.

블록 구성 (접두사 = 블록 이름)

    A_    드라이버 유전자 이진               92    문헌 고정 목록
    A2_   드라이버 x 기능결과 (LoF/미스센스)  184    = 92 x 2
    B_    변이 부담(TMB) 집계                 7
    C_    변이 유형 조성비                     6
    D_    아미노산 치환 쌍 조성비            241    train 관측 쌍
    N_    코돈 역추론 SBS6 + CpG + Ti          9
    M_    MSI 표적 + 면역회피                  3    ablation 기여 -0.0004, 기본 제외

## 규정 준수

행마다 독립적으로 계산한다 — train 통계를 test 에 적용하는 부분이 없다. 유전자
목록(`DRIVERS`/`MSI_TARGETS`/`IMMUNE`)과 표준 코돈표는 교과서 상수이지 외부
데이터셋이 아니다. 코돈 사용빈도 같은 관측 통계는 쓰지 않고 단일 염기 치환
경로에 **균등 가중치**를 준다.

## 왜 위치를 안 쓰나

test 는 같은 변이를 여러 전사체 좌표로 중복 기재하고 정지코돈을 `*` 대신 `X` 로
적는다. 그래서 잔기 번호는 피처에 넣지 않고, 치환의 **종류**(ref -> alt 아미노산)
만 쓴다. `STOP` 정규식이 `[*X]` 를 둘 다 받는 이유도 같다.

## 열 순서

`pd.DataFrame(list[dict])` 의 열 순서는 키가 처음 등장한 순서로 정해진다.
XGBoost 의 `colsample_bytree` 가 열 순서에 반응하므로 이 순서가 재현의 일부다.
생성 경로를 바꾸지 말 것.
"""

from __future__ import annotations

import collections
import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

#: 기본으로 쓰는 블록. `M_` 은 ablation 기여가 -0.0004 라 뺀다.
DOMAIN_PREFIXES: tuple[str, ...] = ("A_", "A2_", "B_", "C_", "D_", "N_")
#: `M_` 까지 포함한 전체.
ALL_DOMAIN_PREFIXES: tuple[str, ...] = ("A_", "A2_", "B_", "C_", "D_", "M_", "N_")

# --------------------------------------------------------------------- 코돈표
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON2AA: dict[str, str] = {}
_i = 0
for _b1 in _BASES:
    for _b2 in _BASES:
        for _b3 in _BASES:
            CODON2AA[_b1 + _b2 + _b3] = _AAS[_i]
            _i += 1

AA2CODONS: dict[str, list[str]] = collections.defaultdict(list)
for _codon, _aa in CODON2AA.items():
    AA2CODONS[_aa].append(_codon)

_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
#: 피리미딘 기준 단일염기치환 6채널 (COSMIC SBS 관례).
SBS6 = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")


def pyrimidine_channel(ref: str, alt: str) -> str:
    """치환을 피리미딘(C/T) 기준으로 접는다. `G>A` 와 `C>T` 는 같은 사건이다."""
    return f"{ref}>{alt}" if ref in "CT" else f"{_COMP[ref]}>{_COMP[alt]}"


@lru_cache(maxsize=1)
def build_pair_map() -> dict[tuple[str, str], tuple[dict[str, float], float, float]]:
    """아미노산 쌍 -> (SBS6 분포, CpG 비율, Transition 비율).

    관측된 것은 단백질 수준 치환(`R895H`)뿐인데 돌연변이 서명은 염기 수준에서
    정의된다. 그래서 두 아미노산을 잇는 **단일 염기 차이 코돈쌍을 전수 열거**해
    가능한 염기 치환에 균등 가중치를 준다. 코돈 사용빈도를 쓰면 정확해지지만
    그건 외부 데이터라 규정상 못 쓴다.

    CpG 판정은 코돈 내부에서 확인 가능한 경우로 한정한다 — 코돈 경계를 넘는
    문맥은 알 수 없으므로 보수적으로 False 로 둔다(과소추정).

    >>> pair_map = build_pair_map()
    >>> round(sum(pair_map[("R", "H")][0].values()), 6)
    1.0
    """
    out: dict[tuple[str, str], tuple[dict[str, float], float, float]] = {}
    for ref_aa, ref_codons in AA2CODONS.items():
        for alt_aa, alt_codons in AA2CODONS.items():
            if ref_aa == alt_aa:
                continue
            sbs: collections.Counter[str] = collections.Counter()
            cpg = transition = 0.0
            paths = 0
            for ref_codon in ref_codons:
                for alt_codon in alt_codons:
                    diff = [k for k in range(3) if ref_codon[k] != alt_codon[k]]
                    if len(diff) != 1:
                        continue
                    k = diff[0]
                    base_from, base_to = ref_codon[k], alt_codon[k]
                    paths += 1
                    sbs[pyrimidine_channel(base_from, base_to)] += 1
                    if (base_from in "AG") == (base_to in "AG"):
                        transition += 1
                    if (base_from == "C" and k < 2 and ref_codon[k + 1] == "G") or (
                        base_from == "G" and k > 0 and ref_codon[k - 1] == "C"
                    ):
                        cpg += 1
            if paths:
                out[(ref_aa, alt_aa)] = (
                    {k: v / paths for k, v in sbs.items()},
                    cpg / paths,
                    transition / paths,
                )
    return out


# ----------------------------------------------------------------------- 파서
#: 정지코돈. train 은 `Q369*`, test 는 `Q369X` 로 적는다 — 둘 다 받는다.
_STOP = re.compile(r"^([A-Z])(\d+)[*X]$")
_SUB = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_FS = re.compile(r"^([A-Z]+)(\d+)([A-Z]*)fs$")
_INDEL = re.compile(r"^([A-Z])(\d+)(?:_([A-Z])(\d+))?(del|ins|delins|dup)([A-Z]*)$")
_RANGE = re.compile(r"^(\d+)_(\d+)([A-Z]+)>([A-Z*]+)$")

#: 유형 조성비(블록 C)의 축. 순서가 열 순서다.
KINDS: tuple[str, ...] = (
    "missense",
    "silent",
    "nonsense",
    "frameshift",
    "inframe_indel",
    "range_sub",
)
#: 기능상실(loss of function)로 묶는 유형.
LOF_KINDS = frozenset({"nonsense", "frameshift"})


def parse_domain_token(token: str) -> tuple[str, str, str]:
    """변이 토큰 -> (유형, ref 아미노산, alt 아미노산).

    `cancer_hack.parser.classify_token` 과 목적이 다르다. 저쪽은 8종 배타 분류로
    개수를 세고, 여기는 치환 쌍과 코돈 역추론에 쓸 **아미노산 두 글자**가 필요하다.

    >>> parse_domain_token("R895H")
    ('missense', 'R', 'H')
    >>> parse_domain_token("R895R")
    ('silent', 'R', 'R')
    >>> parse_domain_token("Q369X")
    ('nonsense', 'Q', '*')
    """
    m = _STOP.match(token)
    if m:
        return "nonsense", m.group(1), "*"
    m = _SUB.match(token)
    if m:
        ref, alt = m.group(1), m.group(3)
        return ("silent" if ref == alt else "missense"), ref, alt
    m = _FS.match(token)
    if m:
        return "frameshift", m.group(1), (m.group(3) or "")
    m = _INDEL.match(token)
    if m:
        return "inframe_indel", m.group(1), (m.group(6) or "")
    m = _RANGE.match(token)
    if m:
        return "range_sub", m.group(3), m.group(4)
    return "unparsed", "", ""


# ------------------------------------------------------------------ 유전자 목록
#: 범암종 드라이버. 문헌 고정 목록이라 fold 와 무관하다 — 누출 경로가 없다.
DRIVERS: tuple[str, ...] = tuple(
    sorted(
        set(
            """TP53 VHL APC PTEN RB1 NF1 ATRX CDKN2A NOTCH1 FBXW7 CREBBP KMT2D B2M CDH1 TSC1
TSC2 NF2 CASP8 CYLD RUNX1 CEBPA BRCA1 BRCA2 CHEK2 MLH1 MSH2 PMS2 ERCC2 RASA1 PTCH1 SMAD2
ACVR1B AXIN1 AXIN2 BRAF PIK3CA HRAS IDH1 IDH2 KIT EGFR CTNNB1 SPOP FGFR3 FGFR1 RHOA NFE2L2
MTOR AKT1 AKT2 GNAS ERBB2 ERBB3 MET RET JAK2 MYD88 CD79B EZH2 U2AF1 SRSF2 CALR NPM1 POLE
POLD1 RAC1 RIT1 CCND1 CDK4 MDM2 MYC MYCN PIM1 BTG1 BTG2 STAT3 ROS1 RPL22 HLA-A HLA-B HLA-C
TAP1 TAP2 NLRC5 CIITA JAK1 CD274 SDHA MAP3K1 GATA3 NPM1 ALB SOCS1 KMT2D""".split()
        )
    )
)

#: 마이크로새틀라이트 불안정(MSI) 종양에서 반복서열 frameshift 가 몰리는 표적.
MSI_TARGETS: tuple[str, ...] = tuple(
    "ACVR2A TGFBR2 BAX RPL22 TCF7L2 CASP5 RAD50 TTK EPHB2 MRE11 PRDM2 CHEK1 JAK1 B2M".split()
)
#: 항원제시·인터페론 경로 — LoF 가 면역회피 표현형의 대리 지표다.
IMMUNE: tuple[str, ...] = tuple(
    "HLA-A HLA-B HLA-C B2M TAP1 TAP2 NLRC5 CIITA JAK1 JAK2 CASP8 CD274".split()
)


# ------------------------------------------------------------------ 피처 생성
def make_domain_features(path: str | Path, *, has_label: bool) -> pd.DataFrame:
    """원본 csv -> 도메인 피처 프레임 (`ID`, 블록 열들, [`SUBCLASS`]).

    pandas 로 4,386열을 통째로 읽으면 메모리와 시간이 아깝다. `csv.reader` 로
    한 행씩 흘려 보내며 집계한다.

    한 유전자 셀 안에서 `(유형, ref, alt)` 가 같은 토큰은 **한 번만 센다.** test 가
    같은 변이를 여러 전사체 좌표로 중복 기재하기 때문이다(토큰 39.8% 감소).
    이걸 안 접으면 TMB 가 test 에서 1.7배 부푼다.

    열 순서는 첫 등장 순서로 정해지고 그게 재현의 일부다 — 모듈 docstring 참고.
    """
    pair_map = build_pair_map()
    drivers, msi_targets, immune = set(DRIVERS), set(MSI_TARGETS), set(IMMUNE)

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        gene_columns = [
            (i, c) for i, c in enumerate(header) if c not in ("ID", "SUBCLASS")
        ]
        id_index = header.index("ID")
        label_index = header.index("SUBCLASS") if has_label else None

        records: list[dict[str, float]] = []
        labels: list[str] = []
        ids: list[str] = []

        for row in reader:
            ids.append(row[id_index])
            if label_index is not None:
                labels.append(row[label_index])

            feature: dict[str, float] = {}
            pair_counts: collections.Counter[str] = collections.Counter()
            sbs: collections.Counter[str] = collections.Counter()
            cpg = transition = 0.0
            n_backmapped = 0
            kind_counts: collections.Counter[str] = collections.Counter()
            n_genes = n_tokens = n_multihit = 0
            msi_frameshift = immune_lof = 0

            for gene_index, gene in gene_columns:
                value = row[gene_index]
                # 빈 셀은 test 에만 237개 있다. 변이 없음으로 본다 — train/test 동일 규칙.
                if not value or value == "WT":
                    continue
                n_genes += 1
                seen: set[tuple[str, str, str]] = set()
                kinds_here: set[str] = set()

                for token in value.split():
                    kind, ref, alt = parse_domain_token(token)
                    key = (kind, ref, alt)
                    if key in seen:
                        continue
                    seen.add(key)
                    n_tokens += 1
                    kind_counts[kind] += 1
                    kinds_here.add(kind)
                    if kind == "missense":
                        pair_counts[f"{ref}>{alt}"] += 1
                    if kind in ("missense", "nonsense"):
                        entry = pair_map.get((ref, alt if kind == "missense" else "*"))
                        if entry:
                            n_backmapped += 1
                            for channel, weight in entry[0].items():
                                sbs[channel] += weight
                            cpg += entry[1]
                            transition += entry[2]

                if len(seen) >= 2:
                    n_multihit += 1
                has_lof = bool(kinds_here & LOF_KINDS)

                if gene in drivers:
                    feature[f"A_{gene}"] = 1
                    feature[f"A2_{gene}_lof"] = int(has_lof)
                    feature[f"A2_{gene}_mis"] = int("missense" in kinds_here)
                if gene in msi_targets and has_lof:
                    msi_frameshift += 1
                if gene in immune and has_lof:
                    immune_lof += 1

            # D — 치환 쌍 조성비. 카운트가 아니라 비율이라 TMB 시프트에 둔감하다.
            total_pairs = max(sum(pair_counts.values()), 1)
            for key, count in pair_counts.items():
                feature[f"D_{key}"] = count / total_pairs

            # N — 코돈 역추론. 블록 중 train/test 이동이 가장 작다(중앙 |log2| 0.07).
            denominator = max(n_backmapped, 1)
            for channel in SBS6:
                feature[f"N_{channel.replace('>', 'to')}"] = sbs[channel] / denominator
            feature["N_cpg"] = cpg / denominator
            feature["N_ti"] = transition / denominator
            feature["N_has"] = int(n_backmapped > 0)

            # B — TMB. 카운트형이라 단독으로 쓰면 위험하고, 조건 변수로 쓸 때 값이 있다.
            feature["B_ngene"] = n_genes
            feature["B_log_ngene"] = np.log1p(n_genes)
            feature["B_ntok"] = n_tokens
            feature["B_log_ntok"] = np.log1p(n_tokens)
            feature["B_tok_per_gene"] = n_tokens / max(n_genes, 1)
            feature["B_multihit"] = n_multihit
            feature["B_multihit_frac"] = n_multihit / max(n_genes, 1)

            # C — 유형 조성비.
            total_kinds = max(sum(kind_counts.values()), 1)
            for kind in KINDS:
                feature[f"C_{kind}"] = kind_counts[kind] / total_kinds

            # M — MSI / 면역회피. 기여 -0.0004 라 기본 블록에서 뺐지만 만들어는 둔다.
            feature["M_msi_fs"] = msi_frameshift
            feature["M_msi_fs_frac"] = msi_frameshift / max(n_genes, 1)
            feature["M_imm_lof"] = immune_lof

            records.append(feature)

    frame = pd.DataFrame(records).fillna(0.0)
    frame.insert(0, "ID", ids)
    if label_index is not None:
        frame["SUBCLASS"] = labels
    return frame


def domain_feature_columns(
    frame: pd.DataFrame, prefixes: tuple[str, ...] = DOMAIN_PREFIXES
) -> list[str]:
    """프레임에서 해당 블록 열만 원래 순서대로 고른다."""
    return [c for c in frame.columns if c.startswith(prefixes)]


def align_domain_columns(
    test_frame: pd.DataFrame, train_columns: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """test 를 train 열 구성에 맞춘다. -> (정렬된 프레임, 채운 열, 버린 열).

    `D_` 블록은 관측된 치환 쌍으로 열이 정해지므로 양쪽이 어긋난다. train 에만
    있는 쌍은 0(그 치환이 없었다는 뜻이라 옳다), test 에만 있는 쌍은 버린다.

    버려지는 쪽은 대부분 `D_X>*` 인데, test 가 정지코돈을 `X` 로 재코딩해서
    `X100L` 이 미스센스로 파싱된 표기 아티팩트다 — 버리는 게 맞다.

    **방향이 중요하다.** train 열 구성을 기준으로 삼아야 규정을 지킨다. test 에서
    본 열을 train 에 추가하면 test 를 피처 정의에 쓴 것이 된다.
    """
    present = set(test_frame.columns)
    filled = [c for c in train_columns if c not in present]
    dropped = [c for c in test_frame.columns if c.startswith(DOMAIN_PREFIXES) and c not in set(train_columns)]
    aligned = test_frame.reindex(columns=train_columns).fillna(0.0)
    return aligned, filled, dropped
