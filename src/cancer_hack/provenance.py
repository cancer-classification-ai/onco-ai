"""캐시된 산출물이 **같은 입력에서 나온 것인지** 확인하는 지문.

왜 필요한가
-----------
`artifacts/oof/` 에 쌓인 예측을 나중에 블렌딩하는데, 그 예측들은 각자 만들어질 때의
fold 분할에 묶여 있다. `data/process/train_folds.parquet` 를 다시 만들면 fold 경계가
바뀌고, 그러면 **예전 OOF 와 새 OOF 를 섞는 순간 교차적합 보정이 valid fold 를 보게 된다.**

이 사고는 조용하다. 파일 이름도 그대로고 행 수도 그대로라 점수만 살짝 좋아진다.
그래서 fold 를 읽는 자리마다 지문을 대조한다.

파일 바이트가 아니라 **내용**의 지문이다. parquet 은 같은 데이터라도 압축·메타데이터가
달라지면 바이트가 바뀌므로, 바이트 해시를 박아 두면 멀쩡한 재생성에도 깨진다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

#: 현재 `artifacts/` 의 모든 OOF·test 예측이 이 fold 분할에서 나왔다.
#: 2026-07-31 생성(seed 42 · n_splits 5 · 6,201행 · 5,636그룹) 이후 바뀐 적이 없다.
#: 값이 달라졌다면 fold 를 다시 만든 것이고, 그 순간 기존 예측과는 섞으면 안 된다.
EXPECTED_FOLD_FINGERPRINT = "997d89a20595cc23"

FOLD_COLUMNS = ("ID", "fold_skf5", "fold_group5")


def frame_fingerprint(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """열 몇 개의 내용을 16자 지문으로. 행 순서에 의존하지 않는다."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"지문에 필요한 열이 없다: {missing}")
    ordered = frame[list(columns)].astype(str).sort_values(list(columns), kind="stable")
    payload = "\n".join(",".join(row) for row in ordered.itertuples(index=False, name=None))
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def fold_fingerprint(folds_path: str | Path) -> str:
    return frame_fingerprint(pd.read_parquet(folds_path), FOLD_COLUMNS)


#: 피처 파켓의 내용 지문. `artifacts/` 의 예측이 전부 이 파켓들에서 나왔다.
#:
#: **`mutation_encoded` 는 지금 코드와 어긋나 있다.** develop 을 머지하면서
#: `features_basic.encode_mutation` 이 바뀌었는데(`*931*` 같은 동의 정지코돈을 2 가 아니라
#: 1 로 센다. 정정이 맞다) 이 파켓은 그 전 코드로 만들어졌다.
#:
#: 실측했다 — 지금 코드로 22개를 전부 새로 만들어 대조하니 **14개는 지문까지 같고 2개만
#: 다르다**(`train`/`test_mutation_encoded`). 차이는 train 기준 2,718만 셀 중 **68셀**이고
#: 전부 `2 -> 1` 이다(값 분포 1: 52,896 -> 52,964 / 2: 165,997 -> 165,929).
#:
#: 작아 보여도 enc3·comut·lsvd·lnmf·gmod 다섯 블록이 이 파일 하나에서 나온다. 재생성하면
#: 기존 OOF 100여 개와 새 예측을 섞을 수 없다 — **전부 다시 학습할 각오**로만 한다.
#: 새로 만들 때는 `paths.use_run_dirs()` 로 출력 위치를 갈라 이쪽을 남겨 둔다.
EXPECTED_FEATURE_FINGERPRINTS: dict[str, str] = {
    # enc3 · comut · lsvd · lnmf · gmod 가 전부 이 파일 하나에서 나온다
    "train_mutation_encoded.parquet": "fab6d20ab398beef",
    "test_mutation_encoded.parquet": "a7bf7121af68dd25",
    "train_gene_event_count_matrix.parquet": "3a22c27f3c5e9fb4",      # gec
    "test_gene_event_count_matrix.parquet": "9f4a0586adb1c3d8",
    "train_gene_mutation_type_matrix.parquet": "7812e2c6495642b9",    # gtype
    "test_gene_mutation_type_matrix.parquet": "586e4565727ee7d1",
    "train_sample_mutation_features_rollup.parquet": "5441ce0dea199447",  # rollup
    "test_sample_mutation_features_rollup.parquet": "465f32e548f62e7e",
    "train_domain_features.parquet": "443ca72d9f493d78",              # domain
    "test_domain_features.parquet": "b942a552d7efd62e",
    "train_mutation_parsed_features.parquet": "a1a2b4246a1a2ec0",     # parsed19
    "test_mutation_parsed_features.parquet": "af0a5bf0f3388168",
    "train_additional_burden_features.parquet": "30cd52f223b40445",   # burden8
    "test_additional_burden_features.parquet": "ac2a455d64a21f5d",
    "train_amino_acid_features.parquet": "0c2527d1ebd3e4c2",          # aa9
    "test_amino_acid_features.parquet": "039cc2f7243b9884",
    "train_signature_mutation_tokens.parquet": "e37fce51aef01bf6",    # sigtok
    "test_signature_mutation_tokens.parquet": "0d3669e01af42c70",
    "train_exact_mutation_tokens.parquet": "fd3f5976615f9b1a",        # exacttok
    "test_exact_mutation_tokens.parquet": "9b659a77460d903e",
    "train_parsed_mutation_tokens.parquet": "2aef7f5898393dde",       # ptok
    "test_parsed_mutation_tokens.parquet": "8c3fe2cad9a7e51e",
}

ID_COLUMNS = ("ID", "SUBCLASS")


def parquet_fingerprint(path: str | Path) -> str:
    """피처 파켓의 내용 지문. ID 로 정렬하고 라벨 열은 뺀다.

    파일 바이트가 아니라 값이라 압축·행 순서가 바뀌어도 안 흔들린다.
    """
    frame = pd.read_parquet(path)
    if "ID" in frame.columns:
        frame = frame.sort_values("ID", kind="stable")
    numeric = frame.drop(columns=[c for c in ID_COLUMNS if c in frame.columns])
    digest = hashlib.blake2b(digest_size=8)
    digest.update(",".join(map(str, numeric.columns)).encode())

    values = numeric.to_numpy()
    if values.dtype.kind in "OUS":
        # 토큰 문서처럼 문자열이 든 프레임이다. object 배열을 그대로 `tobytes()` 하면
        # 값이 아니라 **포인터 주소**가 해싱돼 실행할 때마다 지문이 달라진다.
        digest.update("\n".join(",".join(map(str, row)) for row in values).encode())
    else:
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def check_feature_fingerprints(
    process_dir: str | Path, expected: dict[str, str] | None = None
) -> dict[str, str]:
    """피처 파켓이 예측을 만들 때와 같은지 확인한다. 다르면 예외를 낸다."""
    expected = EXPECTED_FEATURE_FINGERPRINTS if expected is None else expected
    process_dir = Path(process_dir)
    actual, drifted = {}, []
    for name, want in expected.items():
        path = process_dir / name
        if not path.exists():
            raise FileNotFoundError(f"피처 파켓이 없다: {path}")
        got = parquet_fingerprint(path)
        actual[name] = got
        if got != want:
            drifted.append(f"{name} (기대 {want} · 현재 {got})")
    if drifted:
        raise ValueError(
            "피처 파켓이 예측을 만들 때와 다르다:\n  " + "\n  ".join(drifted) + "\n"
            "  기존 artifacts/oof 와 섞으면 안 된다. 되돌리거나 전부 다시 학습한다."
        )
    return actual


def check_fold_fingerprint(folds_path: str | Path, expected: str | None = None) -> str:
    """fold 파일이 기대한 분할인지 확인한다. 다르면 예외를 낸다.

    캐시된 예측을 재사용하기 전에 부른다. 새로 다 학습할 거라면 굳이 막을 이유는
    없지만, 그때도 기존 기록과 점수를 비교할 수 없다는 건 알아야 한다.
    """
    expected = expected or EXPECTED_FOLD_FINGERPRINT
    actual = fold_fingerprint(folds_path)
    if actual != expected:
        raise ValueError(
            f"fold 분할이 기대와 다르다 (기대 {expected} · 현재 {actual}).\n"
            f"  {folds_path}\n"
            "  기존 artifacts/oof 의 예측은 예전 분할에서 나왔으므로 섞으면 안 된다. "
            "전부 다시 학습하거나 fold 파일을 되돌린다."
        )
    return actual
