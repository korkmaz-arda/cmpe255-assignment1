"""Plain-language definitions for every term the dashboard uses.

One definition per term, written for someone who has not done market-basket
analysis before, and used both for the on-screen glossary and for the `help=`
tooltips. The wording has to be exactly right on two points:

* confidence is a **conditional** rate — "of the baskets containing X, this share
  also contain Y" — not the share of all baskets, which is support;
* lift measures **association**, not causation.
"""

from __future__ import annotations

TERMS: dict[str, str] = {
    "basket": "One customer order: the set of products bought together in a single trip.",
    "itemset": "A combination of products that shows up in the same basket.",
    "frequent itemset": (
        "A combination that appears in at least a set share of all baskets. "
        "Everything else is discarded before any rules are built."
    ),
    "association rule": (
        "A statement of the form 'baskets with these products also tend to contain "
        "those products'. The left-hand side is the antecedent, the right-hand side "
        "the consequent."
    ),
    "antecedent": "The products already in the basket — the 'if' side of a rule.",
    "consequent": "The products the rule predicts — the 'then' side.",
    "support": (
        "How common the whole combination is: the share of all baskets that contain "
        "every product in the rule. Support 0.4% means 4 baskets in every 1,000."
    ),
    "confidence": (
        "A conditional rate: of the baskets that already contain the antecedent, the "
        "share that also contain the consequent. Confidence 30% means 'among baskets "
        "with the first products, 30% also had the second' — not '30% of all baskets'."
    ),
    "lift": (
        "How much more often the products appear together than they would if they "
        "were unrelated. Lift 3x means three times more often than chance. Lift 1 "
        "means no association at all. It measures association, not cause: it does "
        "not say buying one makes anyone buy the other."
    ),
    "leverage": (
        "The same comparison as lift, but as a difference rather than a ratio: how "
        "many more baskets contain the combination than independence would predict. "
        "It favours common combinations where lift favours rare ones."
    ),
    "conviction": (
        "How much more often the rule would be wrong if the two sides were unrelated. "
        "Higher means the rule breaks less often than chance would explain; 1 means no "
        "association."
    ),
    "Apriori": (
        "Builds combinations one product at a time: find single products that are "
        "common enough, pair them up, check which pairs survive, and so on. Simple and "
        "memory-light, but it re-reads the baskets once per round."
    ),
    "FP-Growth": (
        "Compresses all the baskets into a shared tree, then mines the tree directly. "
        "Never builds candidate combinations, but holds the whole tree in memory."
    ),
    "ECLAT": (
        "Flips the data around: for each product, store the list of baskets containing "
        "it. A combination's frequency is then just the size of the overlap between "
        "those lists."
    ),
    "tidset": (
        "The set of basket ids containing a given product — the sideways view of the "
        "data that ECLAT works on."
    ),
    "prefix tree": (
        "The shared-prefix structure FP-Growth builds: baskets that start with the "
        "same products share a branch, so common patterns become short paths."
    ),
    "conditional pattern base": (
        "For one product, the collection of tree branches leading to it. FP-Growth "
        "mines that smaller collection recursively to find what co-occurs with it."
    ),
    "mean lift": (
        "The average lift across every rule that survived the filters. The parameter "
        "search uses it as its objective."
    ),
    "external reference": (
        "A maintained third-party library run on the same data and settings, used to "
        "confirm this project's results are right. It is not a competitor and not a "
        "state-of-the-art claim."
    ),
    "illustrative price": (
        "A made-up unit price. Instacart publishes no prices, so these are generated "
        "from a fixed seed purely so the basket panel has something to add up. They "
        "are not real prices and never affect the mining or the recommendations."
    ),
}


def define(term: str) -> str:
    return TERMS.get(term, "")


def tooltip(term: str) -> str:
    """Tooltip text for `help=`, prefixed with the term itself."""
    body = define(term)
    return f"**{term}** — {body}" if body else ""


METRIC_TERMS = ("support", "confidence", "lift", "leverage", "conviction")
ALGORITHM_TERMS = ("Apriori", "FP-Growth", "ECLAT")
