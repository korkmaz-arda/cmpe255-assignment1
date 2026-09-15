"""Algorithm leaderboard.

All entrants are mined from the same corpus at the same thresholds, and every
runtime is the measured wall-clock time of the run that produced the numbers on
its own row. That discipline is the only reason the table can be read at all.

The three local implementations are ranked against each other by measured
runtime, and the fastest is badged production champion. The mlxtend row is
included as a clearly labelled external reference: it confirms the itemsets and
rules are right, but it is not ranked and cannot take the badge.
"""

from __future__ import annotations

from . import config, reference, rules as rules_module
from .data import Corpus
from .mining import MiningResult, mine_all


def _row(result: MiningResult, rule_list: list[rules_module.Rule]) -> dict:
    return {
        "algorithm": result.algorithm,
        "kind": "implemented",
        "paradigm": result.paradigm,
        "memory_note": result.memory_note,
        "n_itemsets": result.n_itemsets,
        "itemsets_by_length": result.by_length(),
        "n_rules": len(rule_list),
        "top_lift": rules_module.top_lift(rule_list),
        "mean_lift": rules_module.mean_lift(rule_list),
        "mean_confidence": rules_module.mean_confidence(rule_list),
        "elapsed_seconds": result.elapsed_seconds,
        "is_reference": False,
    }


def _reference_row(result: reference.ReferenceResult) -> dict:
    return {
        "algorithm": f"{result.library} {result.function}",
        "kind": reference.ROW_LABEL,
        "paradigm": (
            "Maintained third-party library, run on the same corpus and thresholds. "
            "Included to independently confirm the itemsets and rules found above."
        ),
        "memory_note": "Vectorised pandas; memory scales with the one-hot transaction matrix.",
        "n_itemsets": result.n_itemsets,
        "itemsets_by_length": {},
        "n_rules": result.n_rules,
        "top_lift": result.top_lift,
        "mean_lift": result.mean_lift,
        "mean_confidence": result.mean_confidence,
        "elapsed_seconds": result.elapsed_seconds,
        "is_reference": True,
        "library_version": result.version,
        "comparability_note": (
            "Runtime reflects a different engineering approach (vectorised pandas), "
            "not a different mining paradigm, so it is not ranked against the "
            "implementations above and cannot be the production champion."
        ),
    }


def run(
    corpus: Corpus,
    min_support: float,
    max_len: int,
    min_confidence: float,
    min_lift: float,
    include_reference: bool = True,
) -> dict:
    """Mine the corpus with all three backbones plus the reference implementation."""
    results = mine_all(corpus, min_support, max_len)
    rows = []
    for result in results:
        rule_list = rules_module.generate(result, min_confidence, min_lift)
        rows.append(_row(result, rule_list))

    # Ascending measured runtime; the fastest implemented backbone is champion.
    rows.sort(key=lambda r: r["elapsed_seconds"])
    champion = rows[0]["algorithm"] if rows else None
    for row in rows:
        row["is_champion"] = row["algorithm"] == champion

    agreement = _agreement(results)

    leaderboard = list(rows)
    reference_result = None
    if include_reference:
        reference_result = reference.run(corpus, min_support, max_len, min_confidence, min_lift)
        reference_row = _reference_row(reference_result)
        reference_row["is_champion"] = False
        reference_row["agrees_with_implementations"] = reference.agreement(
            results[0].itemsets, reference_result.itemsets
        )["agrees"]
        leaderboard.append(reference_row)   # always last: it is context, not a contender

    return {
        "params": {
            "min_support": min_support,
            "max_len": max_len,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
        },
        "n_transactions": corpus.n_transactions,
        "n_transactions_with_two_or_more": corpus.n_minable,
        "champion": champion,
        "leaderboard": leaderboard,
        "implementations_agree": agreement,
    }


def _agreement(results: list[MiningResult]) -> dict:
    """The three exact algorithms must find the same itemsets at the same supports."""
    if not results:
        return {"agrees": True, "detail": "no results"}
    baseline = results[0]
    disagreements = []
    for other in results[1:]:
        if set(baseline.itemsets) != set(other.itemsets):
            disagreements.append(f"{baseline.algorithm} vs {other.algorithm}: itemset sets differ")
            continue
        mismatched = [
            itemset
            for itemset in baseline.itemsets
            if abs(baseline.itemsets[itemset] - other.itemsets[itemset]) > 1e-12
        ]
        if mismatched:
            disagreements.append(
                f"{baseline.algorithm} vs {other.algorithm}: {len(mismatched)} supports differ"
            )
    return {"agrees": not disagreements, "detail": disagreements}
