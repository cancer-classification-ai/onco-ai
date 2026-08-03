import numpy as np

from cancer_hack.features_basic import encode_mutation
from cancer_hack.features_sparse import build_fold_parsed_token_block


def test_encode_mutation_treats_empty_cell_as_wt():
    for value in ("", "   ", "WT", "NA", "NAN", "NONE", ".", "0", None, np.nan):
        assert encode_mutation(value) == 0


def test_encode_mutation_treats_stop_to_stop_as_synonymous():
    assert encode_mutation("*261*") == 1
    assert encode_mutation("X541X") == 1


def test_parsed_token_fold_builder_does_not_fit_valid_or_test_vocabulary():
    train_documents = np.array(
        [
            "GENE__A TYPE__MISSENSE",
            "GENE__A TYPE__MISSENSE",
            "GENE__B TYPE__NONSENSE",
            "GENE__B TYPE__NONSENSE",
            "VALID_ONLY_TOKEN",
        ],
        dtype=object,
    )
    test_documents = np.array(["TEST_ONLY_TOKEN"], dtype=object)
    train_index = np.array([0, 1, 2, 3])
    y_train = np.array(["X", "X", "Y", "Y"])

    names, train_matrix, test_matrix = build_fold_parsed_token_block(
        train_documents,
        test_documents,
        train_index,
        y_train,
        topk=10,
        min_df=1,
    )

    assert not any("VALID_ONLY_TOKEN" in name for name in names)
    assert not any("TEST_ONLY_TOKEN" in name for name in names)
    assert train_matrix.shape == (5, len(names))
    assert test_matrix.shape == (1, len(names))
    assert train_matrix[4].sum() == 0
    assert test_matrix[0].sum() == 0
