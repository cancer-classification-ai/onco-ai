from __future__ import annotations

import numpy as np
import pytest

from cancer_hack.metrics import (
    build_prediction_frame,
    probability_columns,
    validate_prediction_frame_schema,
)


def test_dl_oof_and_test_match_shared_prediction_schema() -> None:
    ids = np.asarray(["s1", "s2"])
    classes = np.asarray(["ACC", "BRCA", "DLBC"])
    probability = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
    oof = build_prediction_frame(
        ids, probability, classes, y_true=["ACC", "BRCA"]
    )
    test = build_prediction_frame(ids, probability, classes)

    validate_prediction_frame_schema(oof, ids, classes, oof=True)
    validate_prediction_frame_schema(test, ids, classes, oof=False)
    assert list(oof.columns) == [
        "ID",
        *probability_columns(classes),
        "y_true",
        "y_pred",
    ]


def test_dl_schema_rejects_probability_column_drift() -> None:
    ids = np.asarray(["s1"])
    classes = np.asarray(["ACC", "BRCA"])
    frame = build_prediction_frame(ids, [[0.5, 0.5]], classes)
    frame = frame.rename(columns={"p_ACC": "ACC_probability"})

    with pytest.raises(ValueError, match="columns differ"):
        validate_prediction_frame_schema(frame, ids, classes, oof=False)
