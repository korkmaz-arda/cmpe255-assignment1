# Exploratory data analysis

## Purpose

Describe a table before you model it: shape, dtypes, missingness, class balance, and the
distribution of each candidate feature.

## The statistical idea

EDA estimates marginal distributions and their joint behaviour with the target. Every
statement it produces is a claim about your sample, not the population, until you attach
uncertainty to it.

## How this lab does it

`skills_lab/data/*.py` each expose a `profile()` function returning only facts that were
measured: row counts, positive rate, missingness rate, correlations. The app prints those
values in each workspace banner. For example the Titanic table really is about 20% missing in
`age`, and that rate is displayed rather than assumed.

## Rules worth keeping

1. Never look at the evaluation partition while exploring. Leakage by eye is still leakage.
2. Read a mean with its spread and a rate with its denominator.
3. Treat missingness as a pattern to explain, not a nuisance to fill. 20% missing ages is a
   fact about who was recorded, not a random accident.
4. Drop nothing silently: removing rows changes the denominator of every later statistic.

## Pitfall this lab specifically avoids

The Titanic table ships `boat` and `body` columns, which record what happened *after* the
sinking. Including them would produce a near-perfect classifier that has learned the answer.
`skills_lab/data/titanic.py` excludes them explicitly, with a comment saying why.
