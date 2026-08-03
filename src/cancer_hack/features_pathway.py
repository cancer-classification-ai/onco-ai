"""수기 pathway 그룹 집계 — 유전자를 **문헌 기반 기능 경로**로 묶는다.

`features_latent.fit_gene_modules` 는 모듈을 데이터에서 뽑는다. 이쪽은 사람이 아는
경로를 그대로 쓴다. 둘의 차이가 "비지도 군집이 생물학을 찾아내는가"에 대한 대조다.

## 규정 — 왜 DRIVERS 안에서만 고르는가

KEGG·Reactome·STRING·Pfam 같은 pathway 데이터베이스는 **외부 데이터라 수상 제외
사유**다. 손으로 옮겨 적은 멤버십도 결국 같은 외부 지식이고,
`research/00_master_research_and_plan.md:923` 과 `research/model_and_feature_survey.md:303`
에 "주최측 확인이 필요한 회색지대"로 적혀 있다.

그래서 노출을 최소로 줄인다: **유전자는 `features_domain.DRIVERS` 92개 안에서만 고른다.**
그 목록은 이미 저장소에 있고 `A_`/`A2_` 도메인 블록(539열 중 276열)과
`features_graph.driver_pool_indices` 가 쓰고 있는 문헌 상수다. 그 안에서 묶기만 하는
건 저장소가 이미 받아들인 것 이상의 외부 지식을 더하지 않는다. 이 제약은 주석이 아니라
**import 시점에 검사**한다 (아래 `_validate_membership`).

경로는 서로 겹친다 — `CREBBP` 는 Notch 와 염색질 양쪽에 있고, `CHEK2` 는 p53 과 DNA
수선 양쪽에 있다. 하드 모듈이 배타적 분할인 것과 대비되는 지점이고, 실제 유전자가
여러 기능에 관여한다는 사실을 그대로 반영한다. 모든 DRIVERS 를 덮지도 않는다 —
경로가 분명한 것만 넣고 나머지는 어느 그룹에도 안 넣는다.

## 값 형태

`features_latent.aggregate_by_membership` 를 그대로 쓴다 — 7가지 값 방식의 정의와
시프트 배율 표는 거기 한 곳에만 있다. 기본값 `share` 는 행의 변이 유전자 수로 나눠
test 의 2.21배 희석에 내성이 있고, `fraction`/`wburden` 등은 대조군이다.

## 격리

`train_gbdt.py` 의 `f4rk` config 하나에만 붙는다. 팀이 빼기로 하면 `CONFIGS` 에서 한
줄 지우면 끝이고 다른 블록은 안 건드린다. **주최측 확인 전까지 제출 후보가 아니라
로컬 실험이다** — 확인은 사람이 한다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse

from .features_domain import DRIVERS
from .features_latent import _binarize, _gene_idf, aggregate_by_membership

#: 문헌 기반 기능 경로. **전부 `DRIVERS` 교집합이고 import 시점에 검사한다.**
#: 겹침을 허용한다 — 한 유전자가 여러 경로에 속하는 게 실제에 가깝다.
PATHWAYS: dict[str, tuple[str, ...]] = {
    "p53_cellcycle": ("TP53", "CDKN2A", "RB1", "MDM2", "CCND1", "CDK4", "CHEK2"),
    "pi3k_mtor": ("PIK3CA", "PTEN", "AKT1", "AKT2", "MTOR", "TSC1", "TSC2"),
    "rtk_ras_mapk": (
        "BRAF", "HRAS", "NF1", "RASA1", "RAC1", "RIT1", "MAP3K1",
        "EGFR", "ERBB2", "ERBB3", "KIT", "MET", "RET", "FGFR1", "FGFR3", "ROS1",
    ),
    "wnt_adhesion": ("APC", "CTNNB1", "AXIN1", "AXIN2", "CDH1"),
    "notch": ("NOTCH1", "FBXW7", "CREBBP"),
    "chromatin": ("KMT2D", "CREBBP", "EZH2", "SPOP", "ATRX"),
    "dna_repair_mmr": (
        "BRCA1", "BRCA2", "CHEK2", "MLH1", "MSH2", "PMS2", "ERCC2", "POLE", "POLD1",
    ),
    "metabolic_idh": ("IDH1", "IDH2", "SDHA", "VHL", "NFE2L2"),
    "antigen_immune": (
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "NLRC5", "CIITA",
        "JAK1", "JAK2", "CD274", "CASP8", "SOCS1",
    ),
    "tgfb": ("SMAD2", "ACVR1B"),
    "myeloid": ("RUNX1", "CEBPA", "NPM1", "U2AF1", "SRSF2", "CALR"),
    "lymphoid": ("MYD88", "CD79B", "BTG1", "BTG2", "PIM1"),
}


def _validate_membership() -> None:
    """모든 pathway 유전자가 `DRIVERS` 안에 있는지 import 시점에 검사한다.

    규정 근거가 "이미 저장소에 있는 문헌 상수 안에서만 묶는다"이므로, 그 경계를
    넘는 유전자가 하나라도 들어오면 근거가 무너진다. 주석으로 부탁하지 않고 막는다.
    """
    known = set(DRIVERS)
    strays = {
        name: sorted(set(genes) - known) for name, genes in PATHWAYS.items()
    }
    strays = {name: genes for name, genes in strays.items() if genes}
    if strays:
        raise ValueError(
            f"DRIVERS 밖의 유전자가 PATHWAYS 에 있다: {strays}. "
            "외부 pathway DB 를 끌어오는 것은 대회 규정상 수상 제외 사유다 — "
            "모듈 docstring 의 「규정」 절을 읽고 DRIVERS 안에서만 고른다."
        )


_validate_membership()

PATHWAY_NAMES: tuple[str, ...] = tuple(PATHWAYS)


def pathway_membership(
    gene_names: Sequence[str],
) -> tuple[list[str], sparse.csr_matrix, np.ndarray]:
    """`(경로 이름, 멤버십 희소행렬, 경로 크기)`. fold 와 무관한 상수다.

    멤버십은 `(len(gene_names), n_pathways)` 0/1 이고 한 유전자가 여러 열에 1을 가질
    수 있다. `gene_names` 에 없는 유전자는 조용히 빠진다 — 크기는 실제로 잡힌 수로 센다.
    """
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    rows: list[int] = []
    cols: list[int] = []
    for column, pathway in enumerate(PATHWAY_NAMES):
        for gene in PATHWAYS[pathway]:
            index = name_to_idx.get(gene)
            if index is not None:
                rows.append(index)
                cols.append(column)
    membership = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(gene_names), len(PATHWAY_NAMES)),
    )
    sizes = np.asarray(membership.sum(axis=0), dtype=np.int64).ravel()
    return list(PATHWAY_NAMES), membership, sizes


def transform_pathways(
    matrix: np.ndarray,
    membership: sparse.csr_matrix,
    sizes: np.ndarray,
    gene_weight: np.ndarray,
    *,
    mode: str = "mutated",
    value: str = "share",
) -> np.ndarray:
    """경로별 집계. 행 하나만 넣어도 그 행의 값이 같다.

    `gene_weight` 만 fold-train 통계(희귀도)이고 멤버십은 상수다.
    """
    binary = _binarize(matrix, mode)
    return aggregate_by_membership(
        binary,
        membership,
        group_sizes=sizes,
        gene_weight=gene_weight,
        n_genes_total=binary.shape[1],
        value=value,
    )


def build_fold_pathway_block(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    train_index: np.ndarray,
    y_train_fold: Sequence,
    *,
    gene_names: Sequence[str],
    prefix: str = "path_",
    value: str = "share",
    mode: str = "mutated",
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    """한 fold 분의 pathway 블록 -> (열 이름, train dense, test dense, 진단 목록).

    `features_latent.build_fold_module_block` 과 같은 계약이다. 멤버십이 상수라 fold
    안에서 fit 하는 건 희귀도 가중(`wshare`/`wburden` 용)뿐이지만, 나머지 블록과 계약을
    맞춰 두면 `train_gbdt.py` 의 fold 루프가 갈래를 하나 덜 갖는다.

    `y_train_fold` 는 진단에만 쓴다 — 경로 멤버십은 라벨을 안 본다.
    """
    pathways, membership, sizes = pathway_membership(gene_names)
    idf = _gene_idf(_binarize(train_matrix, mode)[np.asarray(train_index)])

    names = [f"{prefix}{value}__{p}" for p in pathways]
    train_out = transform_pathways(
        train_matrix, membership, sizes, idf, mode=mode, value=value
    )
    test_out = transform_pathways(
        test_matrix, membership, sizes, idf, mode=mode, value=value
    )

    fold_out = train_out[np.asarray(train_index)]
    test_arr = np.asarray(test_out)
    diagnostics = []
    for i, (name, pathway) in enumerate(zip(names, pathways)):
        column = fold_out[:, i].astype(np.float64)
        train_mean = float(np.abs(column).mean())
        test_mean = float(np.abs(test_arr[:, i]).mean())
        diagnostics.append(
            {
                "name": name,
                "pathway": pathway,
                "size": int(sizes[i]),
                "genes": [g for g in PATHWAYS[pathway] if g in set(gene_names)],
                "train_mean": train_mean,
                "test_mean": test_mean,
                "ratio": (test_mean / train_mean) if train_mean > 0 else None,
                "train_nonzero_rate": float((column > 0).mean()),
                "test_nonzero_rate": float((test_arr[:, i] > 0).mean()),
            }
        )
    return names, train_out, test_out, diagnostics
