"""Train/test splitting and encoding setup for the mushroom dataset.

Splitting and building the encoder are kept here since they are reused as-is;
fitting the encoder happens later, inside each model's pipeline, so it is not
included here.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42
TEST_SIZE = 0.2


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 train/test split with a fixed random seed.

    Stratifying on `y` keeps the edible/poisonous proportions the same in
    both splits, and the fixed seed makes the split reproducible.
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )


def build_encoder() -> OneHotEncoder:
    """A one-hot encoder for the categorical feature columns.

    `handle_unknown="ignore"` lets the encoder handle a category at
    prediction time that it never saw while fitting (encoded as all zeros)
    instead of raising an error. This encoder should be fit only on training
    data, inside each model's pipeline - never on the full dataset.
    """
    return OneHotEncoder(handle_unknown="ignore")
