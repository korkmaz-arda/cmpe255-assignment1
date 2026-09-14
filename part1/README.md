# Mushroom Classification

## Problem

Predict whether a mushroom is **edible** or **poisonous** from its observable
physical characteristics (cap shape, odor, gill color, habitat, etc.). This is
binary classification where a false negative (a poisonous mushroom predicted
edible) is the costly mistake, so the analysis pays close attention to
recall and false negatives, not just accuracy.

## Dataset

[UCI Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- 8,124 samples, 22 categorical features, all single-letter codes
- Target `class`: edible (`e`) / poisonous (`p`), fairly balanced (~52% / 48%)
- `veil-type` is constant (dropped); `stalk-root` has ~30% missing values
  (marked `?`, kept as an explicit `"missing"` category rather than imputed,
  since missingness is itself associated with the target)

## Analysis workflow

1. **Data loading & validation** - load the raw CSV, check expected columns,
   duplicates, and missing-value markers.
2. **EDA** - Cramer's V and chi-square tests to rank each feature's
   association with the target.
3. **Preprocessing** - stratified 80/20 train/test split; one-hot encoding
   fit only on training data, inside each model's pipeline.
4. **Modeling** - six models fit on the training set.
5. **Evaluation** - stratified 5-fold CV and held-out test set: accuracy,
   precision, recall, F1, ROC-AUC, confusion matrices, false negatives.
6. **Interpretation & ablations** - coefficients, tree rules, feature
   importances, and ablations removing the top features.

## Models

| Model | Purpose |
|---|---|
| Dummy (majority class) | Floor baseline |
| Odor-only (Logistic Regression) | How far the single strongest feature gets alone |
| Categorical Naive Bayes | Native categorical modeling |
| Logistic Regression | Interpretable linear baseline |
| Decision Tree | Readable if/then rules |
| Random Forest | Ensemble, best-performing model |

## Key findings

- **`odor` and `spore-print-color`** are the only two features every model
  ranks as important; `odor` alone reaches 98.6% test accuracy (23 false
  negatives).
- **Random Forest achieves 100% test accuracy with zero false negatives**,
  and stays perfect even after removing `odor` and `spore-print-color` -
  the dataset has enough redundant signal across features that no single
  one is strictly necessary.
- **Categorical Naive Bayes underperforms** (~94.6% accuracy, 81 false
  negatives) because its feature-independence assumption doesn't hold - many
  features are correlated with each other.
- Cross-validation and test-set results agree closely for every model, with
  no sign of overfitting to the split.

## Project structure

```
part1/
├── README.md
├── mushroom_classification.ipynb   # main analysis notebook
├── data/
│   └── mushrooms.csv                # raw dataset (untouched)
├── src/
│   ├── data.py                      # loading, cleaning, validation
│   ├── eda.py                       # Cramer's V, association ranking plot
│   ├── preprocessing.py             # train/test split, encoder setup
│   ├── modeling.py                  # model pipelines
│   ├── evaluation.py                # CV, test evaluation, confusion matrices
│   └── interpretation.py            # coefficients, tree rules, importances
└── figures/                          # saved plots
```

## Running the notebook

```bash
conda activate cmpe255          # or any env with the packages below
jupyter notebook mushroom_classification.ipynb
```

Requires: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`. Run all
cells top to bottom; each section depends on variables defined earlier in
the notebook (`X`/`y`, `X_train`/`X_test`, `models`, etc.).

---
<!--
**Claude Code transcript:** `TODO.html`

**YouTube walkthrough:** `TODO`
-->