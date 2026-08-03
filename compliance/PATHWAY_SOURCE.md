# 수기 pathway 목록의 출처와 규정 판단

`src/cancer_hack/features_pathway.py` 의 `PATHWAYS` 상수에 대한 근거 문서다.
`kpath` 블록(config `f4rk`)을 쓸지 말지 판단할 때 이 문서를 먼저 읽는다.

## 무엇인가

암 유전체에서 널리 알려진 기능 경로 12개에 유전자를 배정한 사전이다. 각 경로의
유전자가 함께 변이되는 경향을 한 열로 집계한다.

    p53_cellcycle  pi3k_mtor  rtk_ras_mapk  wnt_adhesion  notch  chromatin
    dna_repair_mmr  metabolic_idh  antigen_immune  tgfb  myeloid  lymphoid

## 규정 위험

대회 규정은 **외부 데이터 사용을 금지**하며 위반 시 수상 제외다. KEGG·Reactome·
STRING·BioGRID·Pfam 같은 pathway 데이터베이스를 내려받아 쓰는 것은 명백한 위반이다.

사람이 기억으로 적은 pathway 멤버십도 결국 같은 외부 지식이 코드에 들어온 것이다.
팀 조사 문서 두 곳이 이걸 회색지대로 표시해 뒀다:

- `research/00_master_research_and_plan.md:923`
- `research/model_and_feature_survey.md:303`

## 노출을 줄인 방법

**모든 pathway 유전자를 `features_domain.DRIVERS` 92개 안에서만 골랐다.**

그 목록은 이 블록보다 먼저 저장소에 있었고 이미 두 곳이 쓰고 있다:

- `features_domain` 의 `A_` 92열(드라이버 이진)과 `A2_` 184열(드라이버 x LoF/missense)
  — 합쳐서 도메인 블록 539열 중 276열이고, 기준선 `f4r` 에 이미 들어 있다.
- `features_graph.driver_pool_indices` — 공변이 쌍의 후보 풀 제한.

즉 이 블록이 더하는 건 "어떤 유전자가 중요한가"라는 새 지식이 아니라, **이미 쓰고 있는
92개를 어떻게 묶느냐**는 그룹 정보뿐이다. DRIVERS 밖의 유전자는 하나도 안 쓴다.

이 경계는 주석이 아니라 코드로 강제한다 — `features_pathway._validate_membership()` 이
import 시점에 돌고, DRIVERS 밖 유전자가 하나라도 있으면 `ValueError` 로 멈춘다.

경로는 92개 중 81개를 덮고 11개(`ALB`, `CYLD`, `GATA3`, `GNAS`, `MYC`, `MYCN`, `NF2`,
`PTCH1`, `RHOA`, `RPL22`, `STAT3`)는 어느 경로에도 안 넣었다. 경로가 분명한 것만
넣는 편이 억지로 채우는 것보다 낫다.

## 격리

`train_gbdt.py` 의 `f4rk` config 하나에만 붙는다. 팀이 빼기로 하면 `CONFIGS` 에서
한 줄을 지우면 끝이고 다른 블록·다른 config 는 안 건드린다.

## 판단이 필요한 것

**주최측 확인은 사람이 한다.** 확인 전까지 `f4rk` 는 제출 후보가 아니라 로컬 실험이다.

확인할 문장: "공개 문헌에 알려진 유전자-경로 대응을 코드에 상수로 적어 피처 그룹을
만드는 것이 외부 데이터 사용에 해당하는가?"

- 해당한다면 → `features_pathway.py` 를 지우고 `CONFIGS` 의 `f4rk` 를 지운다.
  나머지 모듈 블록(`lsvd`/`lnmf`/`gmod`)은 train.csv 에서만 뽑으므로 영향이 없다.
- 해당하지 않는다면 → 같은 논리로 `features_domain.DRIVERS` 도 안전하다(그쪽이 먼저
  들어와 있었으므로 판단 범위가 더 넓어진다).

## 비교 대상

같은 세션에 데이터에서만 뽑은 모듈 블록을 함께 만들었다. 규정 위험이 0 이고
`f4rk` 와 직접 비교된다.

| 블록 | 그룹 출처 | 외부 지식 | config |
|---|---|---|---|
| `lsvd` | fold-train 공변이 SVD | 없음 | `f4rl` |
| `lnmf` | fold-train 공변이 NMF | 없음 | `f4rn` |
| `gmod` | fold-train 공변이 KMeans | 없음 | `f4rm` |
| `kpath` | 수기 문헌 경로 | **DRIVERS 그룹핑** | `f4rk` |

`f4rk` 가 나머지 셋을 크게 이기지 못한다면 규정 위험을 감수할 이유 자체가 없다.
