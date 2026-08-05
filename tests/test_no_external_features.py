"""외부 데이터가 파이프라인에 들어오지 않았는지.

대회 규정이 외부 데이터 사용을 금지한다. 위반은 실격이라 되돌릴 방법이 없다.

완전한 증명은 불가능하다 — 사람이 유전자 목록을 손으로 적어 넣으면 코드만 봐서는
바깥에서 왔는지 알 수 없다. 여기서 잡을 수 있는 건 **기계적으로 확인 가능한 것들**이다.

  1. 피처 파켓의 행이 원본 csv 행 수와 같다 (다른 코호트를 붙이지 않았다)
  2. 유전자 열이 원본 유전자의 부분집합이다 (없던 유전자가 생기지 않았다)
  3. 소스 코드가 네트워크를 부르지 않는다
  4. 데이터를 읽는 경로가 `data/` 안이다

`compliance/DATA_POLICY.md` 가 사람이 지켜야 할 부분을 적는 자리다(현재 비어 있음).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from conftest import requires_raw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS = PROJECT_ROOT / "data" / "process"

N_TRAIN = 6201
N_TEST = 2546

#: 네트워크로 데이터를 끌어오는 흔한 호출. 주석·문서 문자열은 걸러 낸다.
NETWORK = re.compile(
    r"\b(requests\.(get|post)|urllib\.request|urlopen|wget|curl\s+http|"
    r"boto3|gcsfs|s3fs|kagglehub|datasets\.load_dataset|hf_hub_download)\b"
)

#: 대회 데이터가 아닌 곳을 읽는 흔한 경로.
OUTSIDE = re.compile(r"read_(csv|parquet|json)\(\s*[\"'](?:[A-Za-z]:|/|~|\.\./\.\./)")


def _source_files() -> list[Path]:
    out: list[Path] = []
    for directory in ("src", "scripts"):
        out += [p for p in (PROJECT_ROOT / directory).rglob("*.py") if p.stat().st_size]
    return sorted(out)


def _strip_comments(text: str) -> str:
    """주석과 문서 문자열을 지운다 — 설명에 URL 이 적힌 걸로 걸리면 안 된다."""
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    return re.sub(r"(?m)#.*$", "", text)


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_source_does_not_fetch_from_the_network(path: Path):
    code = _strip_comments(path.read_text(encoding="utf-8"))
    hit = NETWORK.search(code)
    assert hit is None, f"{path.name} 이 네트워크를 부른다: {hit.group(0)}"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_source_does_not_read_outside_the_repository(path: Path):
    code = _strip_comments(path.read_text(encoding="utf-8"))
    hit = OUTSIDE.search(code)
    assert hit is None, f"{path.name} 이 저장소 밖을 읽는다: {hit.group(0)}"


def _feature_parquets() -> list[Path]:
    if not PROCESS.exists():
        return []
    return sorted(p for p in PROCESS.glob("*.parquet") if "folds" not in p.name
                  and "group_keys" not in p.name)


def _has_id(path: Path) -> bool:
    import pyarrow.parquet as pq

    return "ID" in pq.ParquetFile(path).schema.names


@pytest.mark.parametrize("path", _feature_parquets(), ids=lambda p: p.name[:44])
def test_feature_ids_come_only_from_the_raw_split(path: Path):
    """샘플 ID 가 원본 밖에서 오면 다른 코호트를 붙인 것이다.

    행 수로는 못 잰다 — `mutation_events` 처럼 변이 1건당 1행인 long 포맷이 있어서
    6,201행이 아니라 249,133행이다. 불변식은 **ID 집합**이다.
    """
    if not _has_id(path):
        pytest.skip(f"{path.name} 에 ID 열이 없다")

    split = "train" if path.name.startswith("train_") else "test"
    raw_ids = set(pd.read_csv(requires_raw(f"{split}.csv"), usecols=["ID"], dtype=str)["ID"])
    ids = set(pd.read_parquet(path, columns=["ID"])["ID"].astype(str))

    outside = ids - raw_ids
    assert not outside, f"{path.name} 에 원본 밖 ID {len(outside)}개 (예: {sorted(outside)[:3]})"


@pytest.mark.parametrize("path", _feature_parquets(), ids=lambda p: p.name[:44])
def test_wide_feature_tables_have_one_row_per_sample(path: Path):
    """샘플당 1행인 표가 늘었다면 행을 덧붙인 것이다. long 포맷은 건너뛴다."""
    if not _has_id(path):
        pytest.skip(f"{path.name} 에 ID 열이 없다")
    ids = pd.read_parquet(path, columns=["ID"])["ID"]
    if not ids.is_unique:
        pytest.skip(f"{path.name} 은 long 포맷이다 ({len(ids):,}행)")
    expected = N_TRAIN if path.name.startswith("train_") else N_TEST
    assert len(ids) == expected, f"{path.name} 행 {len(ids)}, 기대 {expected}"


def test_gene_columns_are_a_subset_of_the_raw_genes():
    """원본에 없던 유전자가 피처에 생겼다면 바깥에서 온 것이다."""
    matrix = PROCESS / "train_gene_event_count_matrix.parquet"
    if not matrix.exists():
        pytest.skip("gene_event_count_matrix 없음")

    import pyarrow.parquet as pq

    raw_genes = set(pd.read_csv(requires_raw("train.csv"), nrows=1).columns[2:])
    columns = [c for c in pq.ParquetFile(matrix).schema.names if c not in ("ID", "SUBCLASS")]
    # 열 이름에 접두사가 붙는다 (`gene_event_count__TP53`). 마지막 조각을 유전자로 본다.
    genes = {c.split("__")[-1] for c in columns}
    unknown = genes - raw_genes
    assert not unknown, f"원본에 없는 유전자 {len(unknown)}개 (예: {sorted(unknown)[:5]})"


def test_pretrained_directory_has_no_unregistered_weights():
    """사전학습 모델은 허용되지만 출처를 적어야 한다. weights 만 덩그러니 있으면 안 된다."""
    pretrained = PROJECT_ROOT / "pretrained"
    if not pretrained.exists():
        pytest.skip("pretrained/ 없음")
    weights = [p for p in pretrained.rglob("*")
               if p.is_file() and p.suffix in {".pt", ".pth", ".bin", ".safetensors", ".h5"}]
    readme = pretrained / "README.md"
    if weights:
        assert readme.exists() and readme.stat().st_size > 0, (
            f"weights {len(weights)}개가 있는데 pretrained/README.md 가 비어 있다 — "
            "출처를 적어야 한다"
        )
