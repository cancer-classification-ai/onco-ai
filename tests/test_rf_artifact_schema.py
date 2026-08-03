"""RF artifact validator 계약 — spec §9.2.

정상 산출물이 통과하는지 뿐 아니라, **고의로 깨진** 합성 산출물을 넣었을 때
validator 가 실제로 잡아내는지까지 검증한다(Ticket 1a 필수 테스트 항목).
전부 합성 데이터·`tmp_path` 로 돈다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancer_hack.io import save_csv
from cancer_hack.metrics import build_prediction_frame
from cancer_hack.rf_artifact_validator import (
    macro_f1_with_labels,
    validate_oof_frame,
    validate_submission_frame,
    validate_test_probability_frame,
)

CLASS_ORDER = ["C00", "C01", "C02", "C03"]
TRAIN_IDS = [f"tr{i:02d}" for i in range(12)]
TEST_IDS = [f"te{i:02d}" for i in range(6)]
N_SPLITS = 3
RNG = np.random.default_rng(0)


def _random_proba(n_rows: int, n_classes: int) -> np.ndarray:
    raw = RNG.random((n_rows, n_classes)) + 0.01
    return raw / raw.sum(axis=1, keepdims=True)


def make_valid_oof() -> pd.DataFrame:
    proba = _random_proba(len(TRAIN_IDS), len(CLASS_ORDER))
    y_true = [CLASS_ORDER[i % len(CLASS_ORDER)] for i in range(len(TRAIN_IDS))]
    frame = build_prediction_frame(TRAIN_IDS, proba, CLASS_ORDER, y_true=y_true)
    frame.insert(1, "fold", [i % N_SPLITS for i in range(len(TRAIN_IDS))])
    return frame[["ID", "fold", "y_true", "y_pred", *[f"p_{c}" for c in CLASS_ORDER]]]


def make_valid_test_proba() -> pd.DataFrame:
    """spec §8.2 스키마 — `ID` + `p_{class}` 만. `y_pred` 는 넣지 않는다.

    `build_prediction_frame` 은 편의상 `y_pred` 를 항상 붙이지만, 저장용 test
    확률 파일에는 그 열이 있으면 안 된다(`scripts/train_rf.py` 가 저장 전
    잘라낸다) — 그래서 여기서도 드롭해 실제 저장될 스키마와 맞춘다.
    """
    proba = _random_proba(len(TEST_IDS), len(CLASS_ORDER))
    return build_prediction_frame(TEST_IDS, proba, CLASS_ORDER).drop(columns=["y_pred"])


def make_valid_submission(test_proba: pd.DataFrame) -> pd.DataFrame:
    proba_columns = [f"p_{c}" for c in CLASS_ORDER]
    classes = np.asarray(CLASS_ORDER)
    argmax = classes[test_proba[proba_columns].to_numpy().argmax(axis=1)]
    return pd.DataFrame({"ID": test_proba["ID"], "SUBCLASS": argmax})


# ---------------------------------------------------------------- OOF: 정상


def test_valid_oof_frame_passes():
    validate_oof_frame(
        make_valid_oof(), class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS
    )


def test_valid_oof_frame_with_matching_macro_f1_passes():
    frame = make_valid_oof()
    macro_f1 = macro_f1_with_labels(frame["y_true"], frame["y_pred"], CLASS_ORDER)
    validate_oof_frame(
        frame,
        class_order=CLASS_ORDER,
        train_ids=TRAIN_IDS,
        n_splits=N_SPLITS,
        expected_macro_f1=macro_f1,
    )


def test_oof_group_leakage_check_passes_when_groups_stay_in_one_fold():
    frame = make_valid_oof()
    # 그룹 = fold 와 동일하게 둔다(그룹이 fold 를 절대 넘지 않는 자명한 경우).
    group_key_by_id = dict(zip(frame["ID"], frame["fold"]))
    validate_oof_frame(
        frame,
        class_order=CLASS_ORDER,
        train_ids=TRAIN_IDS,
        n_splits=N_SPLITS,
        group_key_by_id=group_key_by_id,
    )


# ---------------------------------------------------------------- OOF: 깨진 산출물


def test_oof_duplicate_id_is_rejected():
    frame = make_valid_oof()
    frame.loc[1, "ID"] = frame.loc[0, "ID"]
    with pytest.raises(ValueError, match="중복"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_missing_row_is_rejected():
    frame = make_valid_oof().iloc[:-1]
    with pytest.raises(ValueError, match="행 수"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_unknown_id_is_rejected():
    frame = make_valid_oof()
    frame.loc[0, "ID"] = "not-a-train-id"
    with pytest.raises(ValueError, match="ID 집합"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_fold_out_of_range_is_rejected():
    frame = make_valid_oof()
    frame.loc[0, "fold"] = N_SPLITS  # 0..N_SPLITS-1 밖
    with pytest.raises(ValueError, match="fold 값 범위"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_missing_probability_column_is_rejected():
    frame = make_valid_oof().drop(columns=[f"p_{CLASS_ORDER[-1]}"])
    with pytest.raises(ValueError, match="없는 컬럼"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_extra_column_is_rejected():
    frame = make_valid_oof()
    frame["Unnamed: 0"] = range(len(frame))
    with pytest.raises(ValueError, match="예상 밖 컬럼"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_nan_probability_is_rejected():
    frame = make_valid_oof()
    frame.loc[0, f"p_{CLASS_ORDER[0]}"] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_row_sum_not_one_is_rejected():
    frame = make_valid_oof()
    frame.loc[0, f"p_{CLASS_ORDER[0]}"] = 0.05  # [0,1] 안이지만 행 합은 깨진다
    with pytest.raises(ValueError, match="행 합"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_y_pred_not_argmax_is_rejected():
    frame = make_valid_oof()
    other = [c for c in CLASS_ORDER if c != frame.loc[0, "y_pred"]][0]
    frame.loc[0, "y_pred"] = other
    with pytest.raises(ValueError, match="argmax"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_y_pred_5e7_below_max_is_rejected():
    """예전에 있던 atol=1e-6 근접동점 허용을 되돌린다 — 이제는 정확히 일치해야 한다.

    최댓값보다 5e-7 낮은(=예전 atol 1e-6 보다 작은 차이) 클래스를 골라도 지금은
    반드시 실패해야 한다.
    """
    frame = make_valid_oof()
    proba_cols = [f"p_{c}" for c in CLASS_ORDER]
    custom = np.array([0.5 - 5e-7, 0.5, 5e-7, 0.0])
    assert abs(custom.sum() - 1.0) < 1e-12
    frame.loc[0, proba_cols] = custom
    frame.loc[0, "y_pred"] = CLASS_ORDER[0]  # 진짜 최댓값(CLASS_ORDER[1])이 아니다
    with pytest.raises(ValueError, match="argmax"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_exact_tie_must_pick_first_canonical_class():
    """정확한 동점에서는 `np.argmax` 규칙대로 canonical 순서상 첫 클래스만 유효하다."""
    frame = make_valid_oof()
    proba_cols = [f"p_{c}" for c in CLASS_ORDER]
    frame.loc[0, proba_cols] = [0.5, 0.5, 0.0, 0.0]  # CLASS_ORDER[0]/[1] 정확한 동점
    frame.loc[0, "y_pred"] = CLASS_ORDER[1]  # 동점의 두 번째 클래스 — np.argmax 규칙 위반
    with pytest.raises(ValueError, match="argmax"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_exact_tie_picking_first_canonical_class_passes():
    frame = make_valid_oof()
    proba_cols = [f"p_{c}" for c in CLASS_ORDER]
    frame.loc[0, proba_cols] = [0.5, 0.5, 0.0, 0.0]
    frame.loc[0, "y_pred"] = CLASS_ORDER[0]  # 동점의 첫 클래스 — np.argmax 규칙과 일치
    validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_unexpected_label_is_rejected():
    frame = make_valid_oof()
    frame.loc[0, "y_true"] = "NOT_CANONICAL"
    with pytest.raises(ValueError, match="canonical 클래스 밖"):
        validate_oof_frame(frame, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS)


def test_oof_group_crossing_fold_is_rejected():
    frame = make_valid_oof()
    # 그룹 g0 을 만들고 그중 한 행만 다른 fold 에 두어 교차를 강제한다.
    group_key_by_id = {id_: 0 for id_ in frame["ID"]}
    frame.loc[0, "fold"] = 0
    frame.loc[1, "fold"] = 1
    with pytest.raises(ValueError, match="fold 를 넘는"):
        validate_oof_frame(
            frame,
            class_order=CLASS_ORDER,
            train_ids=TRAIN_IDS,
            n_splits=N_SPLITS,
            group_key_by_id=group_key_by_id,
        )


def test_oof_wrong_stored_macro_f1_is_rejected():
    frame = make_valid_oof()
    with pytest.raises(ValueError, match="Macro F1"):
        validate_oof_frame(
            frame,
            class_order=CLASS_ORDER,
            train_ids=TRAIN_IDS,
            n_splits=N_SPLITS,
            expected_macro_f1=-1.0,
        )


# ---------------------------------------------------------------- test 확률


def test_valid_test_probability_frame_passes():
    validate_test_probability_frame(
        make_valid_test_proba(), class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
    )


def test_test_probability_frame_with_y_pred_column_is_rejected():
    """spec §8.2 스키마는 `ID` + `p_{class}` 뿐이다 — `y_pred` 가 섞이면 실패해야 한다.

    `build_prediction_frame` 을 그대로 쓰면 `y_pred` 가 편의상 따라오는데,
    저장 전에 잘라내지 않으면 스키마가 어긋난다. index/Unnamed 열도 같은
    이유로 실패해야 한다(§9.2).
    """
    frame = make_valid_test_proba()
    frame["y_pred"] = CLASS_ORDER[0]
    with pytest.raises(ValueError, match="예상 밖 컬럼"):
        validate_test_probability_frame(
            frame, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


def test_test_probability_frame_with_unnamed_index_column_is_rejected():
    frame = make_valid_test_proba()
    frame["Unnamed: 0"] = range(len(frame))
    with pytest.raises(ValueError, match="예상 밖 컬럼"):
        validate_test_probability_frame(
            frame, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


def test_test_probability_wrong_order_is_rejected():
    frame = make_valid_test_proba().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="순서"):
        validate_test_probability_frame(
            frame, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


def test_test_probability_duplicate_id_is_rejected():
    frame = make_valid_test_proba()
    frame.loc[1, "ID"] = frame.loc[0, "ID"]
    with pytest.raises(ValueError, match="중복"):
        validate_test_probability_frame(
            frame, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


def test_test_probability_row_sum_not_one_is_rejected():
    frame = make_valid_test_proba()
    frame.loc[0, f"p_{CLASS_ORDER[0]}"] = 0.05  # [0,1] 안이지만 행 합은 깨진다
    with pytest.raises(ValueError, match="행 합"):
        validate_test_probability_frame(
            frame, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


# ---------------------------------------------------------------- submission


def test_valid_submission_frame_passes():
    test_proba = make_valid_test_proba()
    submission = make_valid_submission(test_proba)
    validate_submission_frame(
        submission,
        class_order=CLASS_ORDER,
        sample_submission_ids=TEST_IDS,
        test_proba_frame=test_proba,
    )


def test_submission_wrong_columns_is_rejected():
    test_proba = make_valid_test_proba()
    submission = make_valid_submission(test_proba)
    submission["index"] = range(len(submission))
    with pytest.raises(ValueError, match="ID,SUBCLASS"):
        validate_submission_frame(
            submission, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


def test_submission_wrong_order_is_rejected():
    test_proba = make_valid_test_proba()
    submission = make_valid_submission(test_proba).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="순서"):
        validate_submission_frame(
            submission, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


def test_submission_non_canonical_subclass_is_rejected():
    test_proba = make_valid_test_proba()
    submission = make_valid_submission(test_proba)
    submission.loc[0, "SUBCLASS"] = "NOT_CANONICAL"
    with pytest.raises(ValueError, match="canonical 클래스 밖"):
        validate_submission_frame(
            submission, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


def test_submission_mismatched_argmax_is_rejected():
    test_proba = make_valid_test_proba()
    submission = make_valid_submission(test_proba)
    other = [c for c in CLASS_ORDER if c != submission.loc[0, "SUBCLASS"]][0]
    submission.loc[0, "SUBCLASS"] = other
    with pytest.raises(ValueError, match="argmax"):
        validate_submission_frame(
            submission,
            class_order=CLASS_ORDER,
            sample_submission_ids=TEST_IDS,
            test_proba_frame=test_proba,
        )


def test_submission_5e7_below_max_is_rejected():
    """OOF 와 동일한 엄격성 — submission 쪽 argmax 교차검증도 근접동점을 허용하지 않는다."""
    test_proba = make_valid_test_proba()
    proba_cols = [f"p_{c}" for c in CLASS_ORDER]
    test_proba.loc[0, proba_cols] = [0.5 - 5e-7, 0.5, 5e-7, 0.0]
    submission = make_valid_submission(test_proba)
    submission.loc[0, "SUBCLASS"] = CLASS_ORDER[0]  # 진짜 최댓값(CLASS_ORDER[1])이 아니다
    with pytest.raises(ValueError, match="argmax"):
        validate_submission_frame(
            submission,
            class_order=CLASS_ORDER,
            sample_submission_ids=TEST_IDS,
            test_proba_frame=test_proba,
        )


def test_submission_missing_id_is_rejected():
    test_proba = make_valid_test_proba()
    submission = make_valid_submission(test_proba).iloc[:-1]
    with pytest.raises(ValueError, match="행 수"):
        validate_submission_frame(
            submission, class_order=CLASS_ORDER, sample_submission_ids=TEST_IDS
        )


# ---------------------------------------------------------------- CSV 저장·재읽기


def test_submission_survives_csv_round_trip(tmp_path):
    test_proba = make_valid_test_proba()
    submission = make_valid_submission(test_proba)
    path = save_csv(submission, tmp_path / "submission.csv", encoding="UTF-8-sig")

    reloaded = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    validate_submission_frame(
        reloaded,
        class_order=CLASS_ORDER,
        sample_submission_ids=TEST_IDS,
        test_proba_frame=test_proba,
    )


def test_oof_survives_csv_round_trip(tmp_path):
    frame = make_valid_oof()
    path = save_csv(frame, tmp_path / "oof.csv")

    reloaded = pd.read_csv(path, encoding="utf-8")
    validate_oof_frame(
        reloaded, class_order=CLASS_ORDER, train_ids=TRAIN_IDS, n_splits=N_SPLITS
    )
