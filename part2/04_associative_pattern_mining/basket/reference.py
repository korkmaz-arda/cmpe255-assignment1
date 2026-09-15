"""mlxtend as an external reference implementation.

The specification's benchmark carried a hardcoded "Kaggle Grandmaster SOTA" row —
a constant with no provenance, never computed and not reproducible. It is
replaced with a maintained third-party library, `mlxtend`, run on the *same*
corpus at the *same* thresholds so the comparison means something.

It is a reference implementation, not a fourth algorithm family and not a state
of the art: its job is to independently confirm that the three implementations in
`mining.py` find the right itemsets and rules. It is excluded from the runtime
ranking and from the production-champion badge, because its timing reflects a
different engineering approach (vectorised pandas) rather than a different
paradigm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules as mlxtend_rules

from .data import Corpus
from .mining import Itemset

LIBRARY = "mlxtend"
ROW_LABEL = "External Reference"


@dataclass
class ReferenceResult:
    library: str
    version: str
    function: str
    itemsets: dict[Itemset, float]
    n_rules: int
    top_lift: float
    mean_confidence: float
    mean_lift: float
    elapsed_seconds: float
    params: dict

    @property
    def n_itemsets(self) -> int:
        return len(self.itemsets)


def _encode(corpus: Corpus) -> pd.DataFrame:
    """One-hot transaction matrix in the layout mlxtend expects."""
    items = sorted({item for transaction in corpus.transactions for item in transaction})
    index = {item: position for position, item in enumerate(items)}
    rows = []
    for transaction in corpus.transactions:
        row = [False] * len(items)
        for item in transaction:
            row[index[item]] = True
        rows.append(row)
    return pd.DataFrame(rows, columns=items, dtype=bool)


def run(
    corpus: Corpus,
    min_support: float,
    max_len: int,
    min_confidence: float,
    min_lift: float,
) -> ReferenceResult:
    """Run mlxtend's apriori + association_rules on the same corpus and thresholds."""
    import mlxtend

    matrix = _encode(corpus)

    # Encoding is preparation, not mining, so it sits outside the timed region —
    # the same courtesy the three local implementations get.
    start = time.perf_counter()
    frequent = mlxtend_apriori(
        matrix, min_support=min_support, use_colnames=True, max_len=max_len
    )
    if len(frequent):
        rules = mlxtend_rules(
            frequent,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(matrix),
        )
        rules = rules[rules["lift"] >= min_lift]
    else:
        rules = pd.DataFrame(columns=["lift", "confidence"])
    elapsed = time.perf_counter() - start

    itemsets = {
        tuple(sorted(int(i) for i in itemset)): float(support)
        for itemset, support in zip(frequent["itemsets"], frequent["support"])
    } if len(frequent) else {}

    return ReferenceResult(
        library=LIBRARY,
        version=mlxtend.__version__,
        function="apriori + association_rules",
        itemsets=itemsets,
        n_rules=int(len(rules)),
        top_lift=float(rules["lift"].max()) if len(rules) else 0.0,
        mean_confidence=float(rules["confidence"].mean()) if len(rules) else 0.0,
        mean_lift=float(rules["lift"].mean()) if len(rules) else 0.0,
        elapsed_seconds=elapsed,
        params={
            "min_support": min_support,
            "max_len": max_len,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
        },
    )


def agreement(local: dict[Itemset, float], external: dict[Itemset, float], tol: float = 1e-9) -> dict:
    """Compare a local itemset table against the reference implementation's."""
    only_local = sorted(set(local) - set(external))
    only_external = sorted(set(external) - set(local))
    mismatched = [
        itemset
        for itemset in set(local) & set(external)
        if abs(local[itemset] - external[itemset]) > tol
    ]
    return {
        "agrees": not (only_local or only_external or mismatched),
        "n_local": len(local),
        "n_external": len(external),
        "only_local": only_local[:10],
        "only_external": only_external[:10],
        "mismatched_support": mismatched[:10],
    }
