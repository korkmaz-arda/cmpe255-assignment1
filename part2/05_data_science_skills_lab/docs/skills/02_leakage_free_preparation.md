# Leakage-free preparation

## The rule

**Fit every cleaning statistic on the training split only.** An imputer's median, a scaler's
mean and standard deviation, and an encoder's category vocabulary are all *parameters*
estimated from data. Estimating them from rows you will later score on lets those rows
influence their own predictions.

## Why it matters more than it looks

The damage is invisible after the fact: the score you report is inflated by an amount you
cannot measure without redoing the experiment correctly. There is no diagnostic in the output
that tells you it happened.

## How this lab enforces it

`skills_lab/benchmarks/common.py` builds one `ColumnTransformer`:

- numeric block: median imputation, then standardisation;
- categorical block: most-frequent imputation, then one-hot encoding with
  `handle_unknown="ignore"`.

That transformer is placed *inside* a `Pipeline` with the estimator, so `fit(X_train, y_train)`
can only ever see training rows, and `predict` applies the fitted parameters unchanged. The
ordering is structural rather than a habit someone has to remember.

`tests/test_leakage.py` asserts the fitted medians equal the training-split medians *and*
that they differ from the full-data medians, so the test cannot pass vacuously.

## The same rule for thresholds

Preprocessing is not the only thing that can leak. A decision threshold chosen by maximising
F1 on the test set is a parameter fitted to the test set. The fraud workspace therefore splits
three ways and selects its cutoff on validation - see
[04_imbalanced_classification.md](04_imbalanced_classification.md).

## Pitfalls

- `df.fillna(df.median())` before `train_test_split`.
- Scaling the full matrix, then splitting.
- Refitting an encoder on the test partition, which silently changes the column space.
- Computing a feature one way in training and another way at serving time.
