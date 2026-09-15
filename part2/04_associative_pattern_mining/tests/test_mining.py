"""The three backbones must each mine, and must agree."""

from __future__ import annotations

import pytest

from basket import mining, reference
from basket.data import Corpus


def test_all_three_find_the_same_itemsets_and_supports(corpus):
    """The invariant the leaderboard rests on: same corpus, same thresholds, same answer."""
    results = mining.mine_all(corpus, min_support=0.05, max_len=4)
    baseline = results[0].itemsets
    for other in results[1:]:
        assert other.itemsets.keys() == baseline.keys(), other.algorithm
        for itemset, support in baseline.items():
            assert other.itemsets[itemset] == pytest.approx(support), (other.algorithm, itemset)


def test_supports_are_hand_checkable(corpus):
    result = mining.apriori(corpus, min_support=0.05, max_len=4)
    assert result.support([1]) == pytest.approx(65 / 105)
    assert result.support([2]) == pytest.approx(55 / 105)
    assert result.support([1, 2]) == pytest.approx(45 / 105)
    assert result.support([1, 2, 3]) == pytest.approx(15 / 105)


def test_max_len_is_respected(corpus):
    for max_len in (2, 3, 4):
        result = mining.eclat(corpus, min_support=0.05, max_len=max_len)
        assert max(len(i) for i in result.itemsets) == max_len


def test_longer_itemsets_are_reachable_above_length_four(corpus):
    """Phase 4 of the search needs length-5 itemsets to build a 4-product antecedent."""
    result = mining.eclat(corpus, min_support=0.10, max_len=5)
    assert (6, 7, 8, 9, 10) in result.itemsets
    assert result.support([6, 7, 8, 9, 10]) == pytest.approx(15 / 105)


def test_fpgrowth_mines_through_conditional_trees(corpus):
    """Guard against the source's defect, where the tree was built and then discarded.

    Exercised directly: the conditional pattern base of item 3 must yield {1,2,3},
    which can only come from projecting the tree, not from any single pass.
    """
    result = mining.fpgrowth(corpus, min_support=0.05, max_len=4)
    assert (1, 2, 3) in result.itemsets
    assert result.support([1, 2, 3]) == pytest.approx(15 / 105)

    # The tree is a real shared-prefix structure, not a flat list.
    counts = {}
    for transaction in corpus.transactions:
        for item in transaction:
            counts[item] = counts.get(item, 0) + 1
    tree = mining._fp_build([(list(t), 1) for t in corpus.transactions if t], counts, 0.0)
    root_children = tree.root.children
    assert len(root_children) < sum(1 for t in corpus.transactions if t)


def test_eclat_applies_the_threshold_at_emission(corpus):
    """Every emitted itemset clears the floor, not merely its parent."""
    min_support = 0.12
    result = mining.eclat(corpus, min_support=min_support, max_len=4)
    assert result.itemsets
    for itemset, support in result.itemsets.items():
        assert support >= min_support, itemset


def test_runtimes_are_measured_not_synthesised(corpus):
    """Each algorithm reports its own elapsed time; nothing is scaled or floored."""
    results = mining.mine_all(corpus, min_support=0.05, max_len=4)
    for result in results:
        assert result.elapsed_seconds > 0.0
    # Two runs of the same algorithm differ, because the number is a measurement.
    first = mining.eclat(corpus, 0.05, 4).elapsed_seconds
    second = mining.eclat(corpus, 0.05, 4).elapsed_seconds
    assert first != second


def test_empty_corpus_degrades_gracefully():
    empty = Corpus(transactions=[frozenset(), frozenset()], order_ids=[1, 2])
    for name in mining.ALGORITHMS:
        result = mining.mine(name, empty, 0.1, 4)
        assert result.itemsets == {}


def test_agrees_with_the_external_reference_implementation(corpus):
    """mlxtend, run on the same corpus and thresholds, must find the same itemsets."""
    local = mining.apriori(corpus, min_support=0.05, max_len=4)
    external = reference.run(corpus, 0.05, 4, min_confidence=0.2, min_lift=1.0)
    report = reference.agreement(local.itemsets, external.itemsets)
    assert report["agrees"], report
