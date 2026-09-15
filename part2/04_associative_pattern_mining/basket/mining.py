"""Three genuine frequent-itemset algorithms over the same transaction database.

Each of the three computes its own itemsets from the corpus and reports its own
measured wall-clock time. None of them delegates to another, and no runtime is
scaled, floored or otherwise adjusted: the benchmark is only interpretable if the
reported time is the time the reported result actually took.

All three are exact, so under identical thresholds they must agree on both the
set of frequent itemsets and their supports. That agreement is asserted by the
tests and cross-checked against mlxtend.

Items are ``product_id`` integers throughout.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from .data import Corpus

Itemset = tuple[int, ...]


@dataclass
class MiningResult:
    """Frequent itemsets plus how long this algorithm took to find them."""

    algorithm: str
    paradigm: str
    memory_note: str
    itemsets: dict[Itemset, float]           # sorted tuple -> support (fraction of transactions)
    elapsed_seconds: float
    n_transactions: int
    params: dict = field(default_factory=dict)

    @property
    def n_itemsets(self) -> int:
        return len(self.itemsets)

    def support(self, items) -> float:
        return self.itemsets.get(tuple(sorted(items)), 0.0)

    def by_length(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for itemset in self.itemsets:
            counts[len(itemset)] += 1
        return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# Apriori — level-wise candidate generation
# --------------------------------------------------------------------------- #

def _apriori_candidates(prev: list[Itemset], k: int) -> set[Itemset]:
    """Join (k-1)-itemsets sharing a prefix, then prune by downward closure."""
    prev_set = set(prev)
    candidates: set[Itemset] = set()
    for i in range(len(prev)):
        a = prev[i]
        for j in range(i + 1, len(prev)):
            b = prev[j]
            if a[:-1] != b[:-1]:
                break                      # sorted input: no further match shares this prefix
            candidate = a + (b[-1],)
            # Downward closure: every (k-1)-subset must itself be frequent.
            if all(sub in prev_set for sub in combinations(candidate, k - 1)):
                candidates.add(candidate)
    return candidates


def apriori(corpus: Corpus, min_support: float, max_len: int) -> MiningResult:
    start = time.perf_counter()
    n = corpus.n_transactions
    min_count = min_support * n

    counts: dict[int, int] = defaultdict(int)
    for transaction in corpus.transactions:
        for item in transaction:
            counts[item] += 1

    frequent: dict[Itemset, float] = {}
    current: list[Itemset] = []
    for item, count in counts.items():
        if count >= min_count:
            current.append((item,))
            frequent[(item,)] = count / n
    current.sort()

    # Restrict each transaction to frequent items once; infrequent items can
    # never appear in a frequent itemset.
    frequent_items = {itemset[0] for itemset in current}
    projected = [
        tuple(sorted(transaction & frequent_items)) for transaction in corpus.transactions
    ]

    k = 2
    while current and k <= max_len:
        candidates = _apriori_candidates(current, k)
        if not candidates:
            break
        candidate_counts: dict[Itemset, int] = dict.fromkeys(candidates, 0)
        for transaction in projected:
            if len(transaction) < k:
                continue
            for combo in combinations(transaction, k):
                if combo in candidate_counts:
                    candidate_counts[combo] += 1
        current = sorted(c for c, count in candidate_counts.items() if count >= min_count)
        for itemset in current:
            frequent[itemset] = candidate_counts[itemset] / n
        k += 1

    elapsed = time.perf_counter() - start
    return MiningResult(
        algorithm="Apriori",
        paradigm="Level-wise candidate generation: build k-item candidates from frequent "
                 "(k-1)-itemsets, then count them with a pass over the transactions.",
        memory_note="Low memory, but one full pass over the data per itemset length.",
        itemsets=frequent,
        elapsed_seconds=elapsed,
        n_transactions=n,
        params={"min_support": min_support, "max_len": max_len},
    )


# --------------------------------------------------------------------------- #
# FP-Growth — prefix-tree mining
# --------------------------------------------------------------------------- #

class _FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item: int | None, parent: "_FPNode | None"):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[int, _FPNode] = {}
        self.link: _FPNode | None = None


class _FPTree:
    def __init__(self) -> None:
        self.root = _FPNode(None, None)
        self.heads: dict[int, _FPNode] = {}
        self.tails: dict[int, _FPNode] = {}

    def add(self, items: list[int], count: int) -> None:
        node = self.root
        for item in items:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, node)
                node.children[item] = child
                if item in self.tails:
                    self.tails[item].link = child
                else:
                    self.heads[item] = child
                self.tails[item] = child
            child.count += count
            node = child

    def nodes(self, item: int):
        node = self.heads.get(item)
        while node is not None:
            yield node
            node = node.link

    def single_path(self) -> list[_FPNode] | None:
        """The path, if the tree is a single branch; otherwise None."""
        path: list[_FPNode] = []
        node = self.root
        while node.children:
            if len(node.children) > 1:
                return None
            node = next(iter(node.children.values()))
            path.append(node)
        return path


def _fp_build(transactions: list[tuple[int, int]], counts: dict[int, int], min_count: float) -> _FPTree:
    """Insert transactions ordered by descending item frequency."""
    tree = _FPTree()
    for items, count in transactions:
        filtered = [i for i in items if counts.get(i, 0) >= min_count]
        if not filtered:
            continue
        filtered.sort(key=lambda i: (-counts[i], i))
        tree.add(filtered, count)
    return tree


def _fp_mine(
    tree: _FPTree,
    counts: dict[int, int],
    min_count: float,
    suffix: tuple[int, ...],
    max_len: int,
    out: dict[Itemset, int],
) -> None:
    if len(suffix) >= max_len:
        return

    path = tree.single_path()
    if path is not None:
        # A single branch: every subset of it is frequent, with the support of
        # its deepest node. Enumerating them directly avoids building further
        # conditional trees.
        room = max_len - len(suffix)
        for size in range(1, min(len(path), room) + 1):
            for combo in combinations(path, size):
                count = min(node.count for node in combo)
                if count < min_count:
                    continue
                itemset = tuple(sorted(suffix + tuple(node.item for node in combo)))
                out[itemset] = max(out.get(itemset, 0), count)
        return

    # Least frequent item first, so conditional trees stay small.
    for item in sorted(tree.heads, key=lambda i: (counts.get(i, 0), -i)):
        total = sum(node.count for node in tree.nodes(item))
        if total < min_count:
            continue
        itemset = tuple(sorted(suffix + (item,)))
        out[itemset] = max(out.get(itemset, 0), total)

        if len(itemset) >= max_len:
            continue

        # Conditional pattern base: the prefix paths of every node for this item.
        base: list[tuple[int, int]] = []
        conditional_counts: dict[int, int] = defaultdict(int)
        for node in tree.nodes(item):
            prefix: list[int] = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                prefix.append(parent.item)
                parent = parent.parent
            if prefix:
                prefix.reverse()
                base.append((prefix, node.count))
                for prefix_item in prefix:
                    conditional_counts[prefix_item] += node.count
        if not base:
            continue
        conditional_tree = _fp_build(base, conditional_counts, min_count)
        if conditional_tree.heads:
            _fp_mine(conditional_tree, conditional_counts, min_count, itemset, max_len, out)


def fpgrowth(corpus: Corpus, min_support: float, max_len: int) -> MiningResult:
    start = time.perf_counter()
    n = corpus.n_transactions
    min_count = min_support * n

    counts: dict[int, int] = defaultdict(int)
    for transaction in corpus.transactions:
        for item in transaction:
            counts[item] += 1

    base = [(list(transaction), 1) for transaction in corpus.transactions if transaction]
    tree = _fp_build(base, counts, min_count)

    raw: dict[Itemset, int] = {}
    _fp_mine(tree, counts, min_count, (), max_len, raw)

    frequent = {itemset: count / n for itemset, count in raw.items() if count >= min_count}
    elapsed = time.perf_counter() - start
    return MiningResult(
        algorithm="FP-Growth",
        paradigm="Prefix-tree mining: compress the transactions into a frequency-ordered tree, "
                 "then mine it recursively through conditional pattern bases — no candidates.",
        memory_note="Holds the whole tree in memory, but needs only two passes over the data.",
        itemsets=frequent,
        elapsed_seconds=elapsed,
        n_transactions=n,
        params={"min_support": min_support, "max_len": max_len},
    )


# --------------------------------------------------------------------------- #
# ECLAT — vertical tidset mining
# --------------------------------------------------------------------------- #

def eclat(corpus: Corpus, min_support: float, max_len: int) -> MiningResult:
    start = time.perf_counter()
    n = corpus.n_transactions
    min_count = min_support * n

    # Vertical layout: one integer bitmask per item, bit t set when transaction t
    # contains it. Support is then a popcount of the intersection.
    tidsets: dict[int, int] = defaultdict(int)
    for index, transaction in enumerate(corpus.transactions):
        bit = 1 << index
        for item in transaction:
            tidsets[item] |= bit

    frequent: dict[Itemset, float] = {}
    roots: list[tuple[Itemset, int, int]] = []
    for item, bits in tidsets.items():
        count = bits.bit_count()
        if count >= min_count:
            frequent[(item,)] = count / n
            roots.append(((item,), bits, count))
    roots.sort()

    def extend(prefix_items: Itemset, siblings: list[tuple[Itemset, int, int]]) -> None:
        for i, (items, bits, _count) in enumerate(siblings):
            if len(items) >= max_len:
                continue
            children: list[tuple[Itemset, int, int]] = []
            for j in range(i + 1, len(siblings)):
                other_items, other_bits, _ = siblings[j]
                merged_bits = bits & other_bits
                merged_count = merged_bits.bit_count()
                # Threshold checked here, before the itemset is recorded, rather
                # than relying on the parent filter to have implied it.
                if merged_count < min_count:
                    continue
                merged = tuple(sorted(set(items) | set(other_items)))
                if len(merged) != len(items) + 1:
                    continue
                frequent[merged] = merged_count / n
                children.append((merged, merged_bits, merged_count))
            if children:
                extend(items, children)

    extend((), roots)

    elapsed = time.perf_counter() - start
    return MiningResult(
        algorithm="ECLAT",
        paradigm="Vertical tidset mining: store the set of transactions each product appears in, "
                 "then intersect those sets depth-first — support is the size of the intersection.",
        memory_note="Tidsets can be large, but support needs no extra pass over the data.",
        itemsets=frequent,
        elapsed_seconds=elapsed,
        n_transactions=n,
        params={"min_support": min_support, "max_len": max_len},
    )


ALGORITHMS = {
    "Apriori": apriori,
    "FP-Growth": fpgrowth,
    "ECLAT": eclat,
}


def mine(algorithm: str, corpus: Corpus, min_support: float, max_len: int) -> MiningResult:
    try:
        function = ALGORITHMS[algorithm]
    except KeyError:
        raise ValueError(f"Unknown algorithm {algorithm!r}; expected one of {sorted(ALGORITHMS)}")
    return function(corpus, min_support, max_len)


def mine_all(corpus: Corpus, min_support: float, max_len: int) -> list[MiningResult]:
    return [mine(name, corpus, min_support, max_len) for name in ALGORITHMS]
