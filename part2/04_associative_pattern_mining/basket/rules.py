"""Association-rule generation and the five interestingness metrics.

Every frequent itemset of size >= 2 is split into all non-empty
antecedent/consequent partitions, each partition is scored, and the survivors are
filtered and ranked.

Metric definitions (X = antecedent, Y = consequent, both sets of product_ids):

    support(X -> Y)    = support(X u Y)
                         share of all transactions containing everything in the rule
    confidence(X -> Y) = support(X u Y) / support(X)
                         *conditional* rate: of the baskets containing X, the share
                         that also contain Y
    lift(X -> Y)       = support(X u Y) / (support(X) * support(Y))
                         how much more often X and Y co-occur than they would if the
                         two were statistically independent. Association, not causation.
    leverage(X -> Y)   = support(X u Y) - support(X) * support(Y)
                         the same comparison as a difference rather than a ratio
    conviction(X -> Y) = (1 - support(Y)) / (1 - confidence)
                         how much more often the rule would be wrong if X and Y were
                         independent; the denominator is floored so confidence == 1
                         yields a large finite number instead of a division by zero
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

from . import config
from .mining import Itemset, MiningResult


@dataclass(frozen=True)
class Rule:
    antecedent: tuple[int, ...]
    consequent: tuple[int, ...]
    support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float
    antecedent_support: float
    consequent_support: float

    @property
    def size(self) -> int:
        return len(self.antecedent) + len(self.consequent)

    def as_dict(self) -> dict:
        record = asdict(self)
        record["antecedent"] = list(self.antecedent)
        record["consequent"] = list(self.consequent)
        return record


def conviction(consequent_support: float, confidence: float) -> float:
    """(1 - support(Y)) / (1 - confidence), with the denominator floored."""
    denominator = max(1.0 - confidence, config.EPSILON)
    return (1.0 - consequent_support) / denominator


def generate(
    result: MiningResult,
    min_confidence: float,
    min_lift: float,
) -> list[Rule]:
    """All qualifying rules from a mining result, ranked lift desc then confidence desc."""
    itemsets = result.itemsets
    rules: list[Rule] = []

    for itemset, union_support in itemsets.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                consequent: Itemset = tuple(i for i in itemset if i not in antecedent)

                antecedent_support = itemsets.get(antecedent)
                consequent_support = itemsets.get(consequent)
                # A partition whose either side is not itself frequent has no
                # marginal to divide by, so it is skipped rather than guessed at.
                if not antecedent_support or not consequent_support:
                    continue

                confidence = union_support / antecedent_support
                if confidence < min_confidence:
                    continue
                lift = union_support / (antecedent_support * consequent_support)
                if lift < min_lift:
                    continue

                rules.append(
                    Rule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=union_support,
                        confidence=confidence,
                        lift=lift,
                        leverage=union_support - antecedent_support * consequent_support,
                        conviction=conviction(consequent_support, confidence),
                        antecedent_support=antecedent_support,
                        consequent_support=consequent_support,
                    )
                )

    return sort_rules(rules)


def sort_rules(rules: list[Rule]) -> list[Rule]:
    """Lift descending, then confidence descending; ids break ties deterministically."""
    return sorted(
        rules,
        key=lambda r: (-r.lift, -r.confidence, r.antecedent, r.consequent),
    )


def mean_lift(rules: list[Rule]) -> float:
    """Objective the parameter search optimises. Zero for an empty rule set."""
    return float(sum(r.lift for r in rules) / len(rules)) if rules else 0.0


def mean_confidence(rules: list[Rule]) -> float:
    return float(sum(r.confidence for r in rules) / len(rules)) if rules else 0.0


def top_lift(rules: list[Rule]) -> float:
    return float(max((r.lift for r in rules), default=0.0))


# --------------------------------------------------------------------------- #
# Display rendering
#
# Names are display metadata, joined here rather than carried through mining.
# --------------------------------------------------------------------------- #

def render(rule: Rule, names: dict[int, str]) -> dict:
    """A rule as a serialisable record with both ids and human-readable names."""
    antecedent_names = [names.get(i, str(i)) for i in rule.antecedent]
    consequent_names = [names.get(i, str(i)) for i in rule.consequent]
    record = rule.as_dict()
    record["antecedent_names"] = antecedent_names
    record["consequent_names"] = consequent_names
    record["text"] = f"{' + '.join(antecedent_names)} ➔ {' + '.join(consequent_names)}"
    return record


def render_all(rules: list[Rule], names: dict[int, str]) -> list[dict]:
    return [render(rule, names) for rule in rules]


def from_record(record: dict) -> Rule:
    """Rebuild a Rule from a persisted artifact record."""
    return Rule(
        antecedent=tuple(record["antecedent"]),
        consequent=tuple(record["consequent"]),
        support=record["support"],
        confidence=record["confidence"],
        lift=record["lift"],
        leverage=record["leverage"],
        conviction=record["conviction"],
        antecedent_support=record["antecedent_support"],
        consequent_support=record["consequent_support"],
    )
