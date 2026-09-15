# Model evaluation

## Match the metric to the decision

| Question | Metric | Why |
|---|---|---|
| Are the flagged cases worth reviewing? | Precision | Denominator is the alerts you raise |
| Are we catching the cases that matter? | Recall | Denominator is the true positives |
| Both at once, one number? | F1 | Harmonic mean; high only when both are |
| How good is the ranking, rare positives? | Average precision | Precision-weighted across all cutoffs |
| How wrong are the predicted prices? | RMSE / MAE in dollars | Units a stakeholder can act on |
| How much variance is explained? | R-squared | Relative to always predicting the mean |

## Fair comparison

When two models are compared in one table they must receive the same treatment: same data,
same split, same cutoff, same metric definitions. In this lab the only difference between the
two fraud models is `class_weight`, and the comparison table states the cutoff it was measured
at. Anything else - different thresholds, different features, one model tuned and the other
not - makes the comparison meaningless.

## Report what the number is conditional on

Every score in this lab is reported with its split, its seed and its cutoff. The app shows
them; `skills_lab/config.py` holds them. A metric without its configuration is not
reproducible, and reproducibility is the claim being made.

## Do not chase recorded numbers

The specification this project reimplements quotes historical figures from its source. They
are history, not targets. Data, library versions and splits all differ here, so the correct
response is to preserve the metric *definitions*, recompute, and report what came out. Tuning
until a number matches a remembered one is how fabricated results happen.

## Pitfalls

- Comparing models at different cutoffs.
- Quoting a threshold-free metric to justify a thresholded decision.
- Averaging metrics across folds without reporting the spread.
- Reading feature importance as causal effect.
