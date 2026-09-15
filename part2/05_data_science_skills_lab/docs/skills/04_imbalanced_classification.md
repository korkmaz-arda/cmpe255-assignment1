# Imbalanced classification

The credit-card fraud table has 492 frauds in 284,807 transactions - a 0.172% base rate.
Predicting "legitimate" for everything scores 99.83% accuracy and catches nothing.

## Metric choice comes first

ROC-AUC integrates over the false-positive rate, whose denominator is enormous when negatives
dominate, so it stays flattering even for a weak detector. **Average precision** (the area
under the precision-recall curve) uses precision, whose denominator is the alerts you actually
raise. This lab reports average precision as the primary metric and ROC-AUC as secondary
context, and labels them that way in the UI.

## Class weighting is only half the job

Two logistic regressions are fitted on identical data, differing only in `class_weight`.
Weighting rescales the loss so minority errors cost more - and in doing so it deliberately
distorts the probability scale. At the default cutoff the weighted model's recall rises
sharply while accuracy barely moves, but its precision collapses: it floods the review queue.

The teaching point is that reweighting changes what the model optimises; it does not choose
an operating point for you.

## Choosing the operating point honestly

1. Split three ways: train / validation / test, stratified.
2. Fit on train.
3. Consider **every** threshold on the validation precision-recall curve. A fixed grid such
   as "30 points from 0.10 to 0.90" cannot find the optimum here: after class weighting, the
   F1-optimal cutoff sits far outside that window.
4. Take the F1 maximum on validation as the production rule.
5. Apply that fixed rule **once** to the test partition and report those numbers separately.

The gap between validation F1 and test F1 is displayed in the app. That gap is precisely the
optimism that selecting the cutoff on the test set would have concealed.

## Working on the score scale

After weighting, thousands of predicted probabilities round to exactly 1.0 in float64, leaving
no resolution near the top of the range. All threshold logic therefore runs on the model's
decision function (log-odds), where every distinct decision is representable. Log-odds 0 is
exactly the familiar probability 0.5.

The interactive workbench exposes the cutoff as a **review budget** - the share of
transactions sent for manual review - which is a monotone relabelling of the same threshold
and reads directly as an operational capacity decision. It measures on validation, so
exploring it cannot contaminate the final held-out report.

## Pitfalls

- Reporting accuracy as the headline on skewed data.
- Assuming 0.5 remains a sensible cutoff after reweighting.
- Tuning the cutoff on the data you then report.
- Buying recall without pricing the false positives it costs.
