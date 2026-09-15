"""Loading and validation helpers for the mushroom classification dataset.

The raw CSV (`data/mushrooms.csv`) holds 23 single-letter categorical columns:
`class` (edible `e` / poisonous `p`) plus 22 feature columns. See the UCI
"agaricus-lepiota" codebook for what each letter means.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mushrooms.csv"

TARGET_COLUMN = "class"
MISSING_MARKER = "?"
MISSING_LABEL = "missing"
CONSTANT_COLUMN = "veil-type"

# Column names as they appear in the raw CSV header.
EXPECTED_COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV with no NA interpretation.

    `keep_default_na=False` keeps every cell as the literal string it is in
    the file (in particular `"?"` stays `"?"` instead of becoming NaN), so
    missingness is handled explicitly rather than silently by pandas.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_raw(df: pd.DataFrame) -> None:
    """Sanity-check the raw dataframe before any cleaning happens.

    Raises AssertionError with a clear message if an expectation is violated.
    """
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Unexpected columns: {list(df.columns)}"
    )
    assert set(df[TARGET_COLUMN].unique()) <= {"e", "p"}, (
        f"Unexpected class values: {sorted(df[TARGET_COLUMN].unique())}"
    )
    assert df.duplicated().sum() == 0, "Found duplicate rows in raw data"
    # '?' is only ever expected in stalk-root; flag it anywhere else.
    for col in df.columns:
        if col == "stalk-root":
            continue
        assert (df[col] == MISSING_MARKER).sum() == 0, (
            f"Unexpected '?' marker found in column '{col}'"
        )


def constant_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns that take on a single value in `df`."""
    return [c for c in df.columns if c != TARGET_COLUMN and df[c].nunique() == 1]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed cleaning steps to the raw dataframe.

    - Relabel stalk-root '?' as an explicit "missing" category, since
      missingness there is strongly associated with the target and should
      stay visible to any model rather than being imputed away.
    - Drop veil-type, which is constant across all rows and carries no
      information.
    """
    df = df.copy()
    df["stalk-root"] = df["stalk-root"].replace(MISSING_MARKER, MISSING_LABEL)
    df = df.drop(columns=CONSTANT_COLUMN)
    return df


def load_mushrooms(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load, validate, and clean the dataset, returning (X, y).

    `y` is 1 for poisonous and 0 for edible. `X` holds the remaining
    (cleaned) feature columns as strings, ready for encoding downstream.
    """
    raw = load_raw(path)
    validate_raw(raw)
    df = clean(raw)

    y = (df[TARGET_COLUMN] == "p").astype(int).rename("poisonous")
    X = df.drop(columns=TARGET_COLUMN)
    return X, y
