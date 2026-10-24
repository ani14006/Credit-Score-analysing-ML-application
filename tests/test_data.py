from ml.data import EXPECTED_ROWS, load_dataset
from ml.features import CATEGORY_CODES, MODEL_FEATURES, TARGET


def test_dataset_loads_with_expected_shape_and_labels():
    df = load_dataset()
    assert len(df) == EXPECTED_ROWS
    assert set(MODEL_FEATURES) <= set(df.columns)
    # german.doc: 700 good (label 1 -> 0) and 300 bad (label 2 -> 1)
    assert df[TARGET].value_counts().to_dict() == {0: 700, 1: 300}


def test_categorical_codes_are_fully_decoded():
    df = load_dataset()
    for column, codes in CATEGORY_CODES.items():
        assert set(df[column].unique()) <= set(codes.values())
