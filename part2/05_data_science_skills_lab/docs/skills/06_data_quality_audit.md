# Data-quality audit

## Score with rates, never counts

"More than one issue means CRITICAL" is meaningless on a million-row table: any real table
will breach it, and a tiny table will pass while being far worse. Every dimension here is a
normalised 0-100 rate:

| Dimension | Definition | Weight |
|---|---|---|
| Completeness | share of cells populated | 0.35 |
| Validity | share of populated cells satisfying their column's typed rule | 0.30 |
| Uniqueness | share of rows that are not exact duplicates | 0.20 |
| Consistency | share of text values already in canonical form | 0.15 |

The composite is their weighted sum, so it stays on a 0-100 scale and the weights are the only
judgement call - written down in `skills_lab/config.py` and shown in the UI.

Health states come from score and severity thresholds: PASS at 90+, WARNING at 70+, CRITICAL
below, with a severe breach (over 20% null or invalid) forcing CRITICAL regardless of score.
Raw counts stay visible as diagnostics, because a rate alone does not tell you how much work
a fix is.

## Business events are not defects

Online Retail II contains genuine commercial records that a naive auditor flags as corruption:

- credit notes (invoice numbers beginning with `C`);
- negative quantities, which are returns and cancellations;
- non-product stock codes such as `POST`, `M` and `BANK CHARGES`.

These are classified as **business events** and reported in their own panel. Statistical
outliers get the same treatment as **review flags**: a £15,000 order line is unusual, not
invalid, and deleting it would change the meaning of every total computed afterwards.

What *is* a defect: missing values, a non-positive price, a placeholder description such as
`?`, a country recorded as `Unspecified`, a non-integer customer id, exact duplicate rows, and
the same value written two ways (`MUG` and `mug`).

## Definitions must be stated, not implied

Two quantities in this lab are ambiguous unless defined, so both are defined on screen:

- **Revenue** - net of returns (negative lines included), with gross sales shown alongside.
- **Retention population** - customers with a non-null id whose invoice is a product line,
  is not a credit note, and has positive quantity and price.

An unstated definition is how two dashboards end up disagreeing about the same month.
