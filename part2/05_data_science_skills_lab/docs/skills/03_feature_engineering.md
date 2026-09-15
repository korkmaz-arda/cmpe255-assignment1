# Feature engineering and target transforms

## Derived features

A derived feature is a hand-built basis function: it earns its place when it encodes
structure the model would otherwise need many splits to approximate.

This lab derives two, in `skills_lab/data/titanic.py`:

```python
family_size = sibsp + parch + 1
is_alone    = "alone" if family_size == 1 else "with family"
```

The same function computes them for the training table and for the live predictor, so a
profile entered in the UI is transformed exactly as the training rows were. `tests/`
asserts that equivalence directly - a serving/training skew here is a classic silent bug.

## Target transforms

House prices are right-skewed, so squared error on the raw scale is dominated by expensive
houses. The regression workspace fits on `log1p(SalePrice)` and inverts with `expm1`
**before** computing any metric, so RMSE, MAE and R-squared are all in dollars.

Two things to keep straight:

- `expm1` of a mean log is not a mean price. Back-transforming point predictions is fine for
  ranking and for error metrics; it is not an unbiased estimate of the conditional mean.
- Report the transform. A reader cannot interpret an R-squared without knowing which scale
  it was computed on. The app states it in the banner and in the result payload.

## Pitfalls

- Deriving a feature from the target. That is leakage wearing a disguise.
- Adding correlated variants and then reading split importances as causal effects. In the
  Ames data, living area and basement area correlate at about 0.45, so their importances
  partly substitute for one another - the app says so on screen.
