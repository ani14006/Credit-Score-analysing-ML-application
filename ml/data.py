"""Load and decode the UCI Statlog (German Credit Data) dataset."""

import os
from pathlib import Path

import pandas as pd

from ml.features import CATEGORY_CODES, RAW_COLUMNS, TARGET, TARGET_MAPPING

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german.data"
EXPECTED_ROWS = 1000


def load_dataset(path=None):
    path = Path(path or os.environ.get("CREDIT_DATA_PATH", DEFAULT_DATA_PATH))
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for the download source."
        )

    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")

    for column, codes in CATEGORY_CODES.items():
        unknown = set(df[column].unique()) - set(codes)
        if unknown:
            raise ValueError(f"Unexpected codes in {column}: {sorted(unknown)}")
        df[column] = df[column].map(codes)

    unknown_targets = set(df["target_raw"].unique()) - set(TARGET_MAPPING)
    if unknown_targets:
        raise ValueError(f"Unexpected target values: {sorted(unknown_targets)}")
    df[TARGET] = df["target_raw"].map(TARGET_MAPPING).astype(int)
    return df.drop(columns=["target_raw"])
