"""Basket cross-sell recommendation: subset matching over the mined rule set.

A rule fires when its whole antecedent is already in the basket. Every consequent
product that is not yet in the basket becomes a candidate, ranked by
``lift x confidence`` taken as a maximum over the rules that fired for it.

The source project reported a candidate's lift and confidence as independent
maxima, and named as the "supporting rule" whichever firing rule came first in
lift order — so the number on the badge could come from a rule other than the one
named, and other than the one that produced the ranking. Here the ranking rule is
the rule that produced the score, and the lift and confidence shown are that
rule's own. The remaining supporting rules are listed beneath it.

Nothing is fitted here. This is a dictionary lookup plus arithmetic over
precomputed metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .rules import Rule


@dataclass
class Suggestion:
    product_id: int
    product_name: str
    department: str
    price_illustrative: float
    score: float
    lift: float
    confidence: float
    support: float
    reason: str
    headline_rule: str
    supporting_rules: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        record = self.__dict__.copy()
        return record


@dataclass
class Recommendation:
    suggestions: list[Suggestion]
    basket_size: int
    basket_value_illustrative: float
    addon_value_illustrative: float
    n_rules_fired: int
    elapsed_ms: float

    @property
    def is_empty(self) -> bool:
        return not self.suggestions


def _reason(rule: Rule, names: dict[int, str]) -> str:
    """Plain-language statement of what the rule's confidence actually means."""
    antecedent = " and ".join(names.get(i, str(i)) for i in rule.antecedent)
    consequent = " and ".join(names.get(i, str(i)) for i in rule.consequent)
    return (
        f"Of the baskets that already contain {antecedent}, "
        f"{rule.confidence:.0%} also contain {consequent}."
    )


def recommend(
    basket: list[int] | set[int],
    rules: list[Rule],
    catalog: dict[int, dict],
    limit: int = config.RECOMMENDATION_COUNT,
) -> Recommendation:
    start = time.perf_counter()
    basket_set = {int(item) for item in basket}
    names = {pid: meta.get("product_name", str(pid)) for pid, meta in catalog.items()}

    best: dict[int, Rule] = {}          # candidate -> the rule that produced its score
    best_score: dict[int, float] = {}
    supporting: dict[int, list[Rule]] = {}
    n_fired = 0

    for rule in rules:
        if not set(rule.antecedent) <= basket_set:
            continue
        n_fired += 1
        score = rule.lift * rule.confidence
        for product_id in rule.consequent:
            if product_id in basket_set:
                continue
            supporting.setdefault(product_id, []).append(rule)
            if score > best_score.get(product_id, -1.0):
                best_score[product_id] = score
                best[product_id] = rule

    ranked = sorted(best_score, key=lambda pid: (-best_score[pid], pid))[:limit]

    suggestions: list[Suggestion] = []
    for product_id in ranked:
        rule = best[product_id]
        meta = catalog.get(product_id, {})
        others = sorted(
            supporting[product_id],
            key=lambda r: (-(r.lift * r.confidence), r.antecedent, r.consequent),
        )[:3]
        suggestions.append(
            Suggestion(
                product_id=product_id,
                product_name=meta.get("product_name", str(product_id)),
                department=meta.get("department", "unknown"),
                price_illustrative=float(meta.get("price_illustrative", 0.0)),
                score=best_score[product_id],
                lift=rule.lift,
                confidence=rule.confidence,
                support=rule.support,
                reason=_reason(rule, names),
                headline_rule=(
                    f"{' + '.join(names.get(i, str(i)) for i in rule.antecedent)} ➔ "
                    f"{' + '.join(names.get(i, str(i)) for i in rule.consequent)}"
                ),
                supporting_rules=[
                    {
                        "text": (
                            f"{' + '.join(names.get(i, str(i)) for i in r.antecedent)} ➔ "
                            f"{' + '.join(names.get(i, str(i)) for i in r.consequent)}"
                        ),
                        "lift": r.lift,
                        "confidence": r.confidence,
                        "support": r.support,
                    }
                    for r in others
                ],
            )
        )

    # Both monetary figures are sums of illustrative prices, so they are
    # descriptive of the basket's composition, not a measured or forecast revenue
    # effect. The UI labels them accordingly.
    basket_value = sum(
        float(catalog.get(pid, {}).get("price_illustrative", 0.0)) for pid in basket_set
    )
    addon_value = sum(s.price_illustrative for s in suggestions[: config.UPLIFT_TOP_N])

    return Recommendation(
        suggestions=suggestions,
        basket_size=len(basket_set),
        basket_value_illustrative=round(basket_value, 2),
        addon_value_illustrative=round(addon_value, 2),
        n_rules_fired=n_fired,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )
