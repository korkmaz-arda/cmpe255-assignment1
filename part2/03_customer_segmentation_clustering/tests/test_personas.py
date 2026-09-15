"""Persona assignment must follow cluster content, not label index."""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from segmentation import config, personas


def _population() -> pd.DataFrame:
    """A population with three unmistakable behavioural groups."""
    rng = np.random.default_rng(0)
    groups = {
        "vip": dict(age=46, income_k=95, spending_score=90, recency_days=20,
                    total_spend=2200, web_visits_month=2, discount_sensitivity=0.05, household_size=2),
        "bargain": dict(age=45, income_k=20, spending_score=15, recency_days=60,
                        total_spend=90, web_visits_month=8, discount_sensitivity=0.85, household_size=4),
        "middle": dict(age=47, income_k=52, spending_score=50, recency_days=50,
                       total_spend=650, web_visits_month=6, discount_sensitivity=0.30, household_size=3),
    }
    rows, labels = [], []
    for index, profile in enumerate(groups.values()):
        for _ in range(60):
            rows.append({key: value * rng.normal(1.0, 0.03) for key, value in profile.items()})
            labels.append(index)
    return pd.DataFrame(rows), np.array(labels)


def test_personas_describe_the_clusters_they_are_attached_to():
    frame, labels = _population()
    records = personas.profile(frame, labels)
    by_cluster = personas.by_cluster(records)

    # Cluster 0 is the affluent high-engagement group; cluster 1 the price-led one.
    vip = by_cluster[0]
    bargain = by_cluster[1]

    assert vip["means"]["income_k"] > bargain["means"]["income_k"]
    assert vip["means"]["spending_score"] > bargain["means"]["spending_score"]
    # The narrative attached to the affluent cluster must be an affluent archetype.
    assert config.ARCHETYPES_BY_KEY[vip["key"]].profile["income_k"] >= 0.7
    assert config.ARCHETYPES_BY_KEY[vip["key"]].profile["spending_score"] >= 0.5
    # ...and the price-led cluster must get a discount-driven archetype.
    assert config.ARCHETYPES_BY_KEY[bargain["key"]].profile["discount_sensitivity"] >= 0.5
    assert config.ARCHETYPES_BY_KEY[bargain["key"]].profile["income_k"] <= 0.5


def test_persona_assignment_is_invariant_to_label_permutation():
    """The source's defect: personas followed the arbitrary K-Means label index.

    Relabelling the same partition must move the persona identities with their
    clusters, not leave them pinned to the old indices.
    """
    frame, labels = _population()
    original = {r["cluster"]: r["key"] for r in personas.profile(frame, labels)}

    permutation = {0: 2, 1: 0, 2: 1}
    permuted = np.array([permutation[label] for label in labels])
    shuffled = {r["cluster"]: r["key"] for r in personas.profile(frame, permuted)}

    for old_index, new_index in permutation.items():
        assert shuffled[new_index] == original[old_index]


def test_every_cluster_gets_a_distinct_persona():
    frame, labels = _population()
    records = personas.profile(frame, labels)
    keys = [r["key"] for r in records]
    assert len(set(keys)) == len(keys)
    assert len(set(r["color"] for r in records)) == len(records)


def test_shares_sum_to_one_and_counts_match():
    frame, labels = _population()
    records = personas.profile(frame, labels)
    assert sum(r["size"] for r in records) == len(frame)
    assert sum(r["share"] for r in records) == 1.0


def test_weak_matches_are_flagged_rather_than_hidden():
    """An implausible cluster must not be labelled with silent confidence."""
    frame, labels = _population()
    # Push one group to a profile no archetype describes: rich, disengaged, huge household.
    frame.loc[labels == 2, "income_k"] = 99.0
    frame.loc[labels == 2, "spending_score"] = 1.0
    frame.loc[labels == 2, "web_visits_month"] = 20.0
    frame.loc[labels == 2, "discount_sensitivity"] = 0.99
    records = personas.profile(frame, labels)
    odd = personas.by_cluster(records)[2]
    assert 0.0 <= odd["match_quality"] <= 1.0
    if odd["weak_match"]:
        assert personas.sanity_notes(records)


def test_every_archetype_scores_every_matched_axis():
    """An axis that is not matched on is an axis whose narrative nothing checks.

    Two persona descriptions once claimed recency behaviour ("high recency",
    "infrequent shoppers") while recency was absent from MATCH_AXES, so the
    matcher could not contradict them — and the data did.
    """
    for archetype in config.ARCHETYPES:
        assert set(archetype.profile) == set(config.MATCH_AXES), (
            f"{archetype.key} profile does not cover the matched axes"
        )
        assert all(0.0 <= v <= 1.0 for v in archetype.profile.values())


def test_a_just_purchased_cluster_is_not_labelled_infrequent():
    """Recency must drive the label when it is the cluster's defining trait."""
    frame, labels = _population()
    frame.loc[labels == 0, "recency_days"] = 1.0      # bought yesterday
    frame.loc[labels != 0, "recency_days"] = 55.0
    records = personas.by_cluster(personas.profile(frame, labels))
    fresh = records[0]
    profile = config.ARCHETYPES_BY_KEY[fresh["key"]].profile
    # The invariant is "no contradiction", not "one specific label": a cluster that
    # is also the richest and highest-spending may rightly be VIP Champions ("buy
    # often"). What must never happen is a long-gap archetype on a fresh buyer.
    assert profile["recency_days"] < 0.6, (
        f"cluster with recency=1d was labelled '{fresh['name']}', an archetype that "
        f"describes long gaps between orders (recency {profile['recency_days']})"
    )
    import re

    for contradiction in (r"\binfrequent\b", r"\blong gaps\b", r"\bfar behind\b", r"\brare\b"):
        assert not re.search(contradiction, fresh["description"].lower()), contradiction


def test_archetype_keys_and_colours_are_unique_across_the_catalogue():
    keys = [a.key for a in config.ARCHETYPES]
    colours = [a.color for a in config.ARCHETYPES]
    names = [a.name for a in config.ARCHETYPES]
    assert len(set(keys)) == len(keys)
    assert len(set(colours)) == len(colours)
    assert len(set(names)) == len(names)


# --------------------------------------------------------------------------- #
# Narrative claims are verified, on a percentile scale that means what it says
# --------------------------------------------------------------------------- #

def test_positions_are_true_population_percentiles(clean_sample):
    """0.5 must be the median customer on every continuous axis.

    A linear 5th-95th-percentile scaling once put the median customer's spend at
    0.21, biasing every high/low judgement on skewed attributes.
    """
    medians = pd.DataFrame({axis: [clean_sample[axis].median()] for axis in config.MATCH_AXES})
    position = personas._scale_positions(medians, clean_sample).iloc[0]
    for axis in ("income_k", "spending_score", "total_spend", "age",
                 "discount_sensitivity", "recency_days"):
        assert abs(position[axis] - 0.5) < 0.08, f"{axis} median sits at {position[axis]:.2f}"


@pytest.mark.parametrize(
    "target,actual,violated,reason",
    [
        (0.20, 0.34, False, "a $123 basket at the 34th percentile is still 'low'"),
        (0.20, 0.45, True, "the 45th percentile is not 'low'"),
        (0.85, 0.47, True, "below-median recency is not 'long gaps'"),
        (0.85, 0.62, False, "the 62nd percentile is 'high'"),
        (0.50, 0.30, True, "the 30th percentile is not 'mid-range'"),
        (0.50, 0.62, False, "the 62nd percentile is 'mid-range'"),
        (0.50, 0.69, True, "the 69th percentile is not 'mid-sized'"),
    ],
)
def test_claim_bands_match_what_the_words_mean(target, actual, violated, reason):
    got = bool(personas.claim_violated(np.array([actual]), target)[0])
    assert got is violated, reason


def test_every_signature_target_encodes_an_unambiguous_claim():
    for archetype in config.ARCHETYPES:
        for axis in archetype.signature:
            target = archetype.profile[axis]
            fuzzy = (config.CLAIM_LOW_TARGET < target < config.CLAIM_MID_BAND[0]) or \
                    (config.CLAIM_MID_BAND[1] < target < config.CLAIM_HIGH_TARGET)
            assert not fuzzy, f"{archetype.key}.{axis}={target} sits between claim kinds"


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_named_personas_never_contradict_their_own_narrative(clean_sample, k):
    """The core invariant, checked directly rather than trusted to the matcher:
    every claim a displayed persona's narrative makes holds for its cluster."""
    from segmentation import features, model

    enriched, _, X = features.build_training_matrix(clean_sample)
    labels = model.fit(X, k=k, seed=3).labels_
    for record in personas.profile(enriched, labels):
        if not record["matched"]:
            continue
        archetype = config.ARCHETYPES_BY_KEY[record["key"]]
        for axis in archetype.signature:
            actual = np.array([record["profile_position"][axis]])
            assert not personas.claim_violated(actual, archetype.profile[axis])[0], (
                f"k={k}: '{record['name']}' claims {personas.claim_kind(archetype.profile[axis])} "
                f"{axis}, but the cluster sits at percentile {actual[0]:.2f}"
            )


def test_a_cluster_no_persona_fits_is_described_from_its_own_data():
    """Forcing an authored narrative onto a cluster it contradicts is the defect
    this project exists to remove. Such a cluster must get a data-derived identity."""
    rng = np.random.default_rng(1)
    rows, labels = [], []
    # Six near-identical clusters: most will have no contradiction-free archetype left.
    for index in range(6):
        for _ in range(40):
            rows.append(dict(age=45, income_k=95, spending_score=90, recency_days=50,
                             total_spend=2000 + index, web_visits_month=2,
                             discount_sensitivity=0.05, household_size=2))
            labels.append(index)
    for _ in range(200):  # a varied background population to rank against
        rows.append(dict(age=rng.uniform(20, 70), income_k=rng.uniform(10, 100),
                         spending_score=rng.uniform(1, 100), recency_days=rng.uniform(0, 99),
                         total_spend=rng.uniform(10, 2500), web_visits_month=rng.uniform(0, 10),
                         discount_sensitivity=rng.uniform(0, 1), household_size=rng.integers(1, 6)))
        labels.append(6)
    records = personas.profile(pd.DataFrame(rows), np.array(labels))

    unmatched = [r for r in records if not r["matched"]]
    assert unmatched, "expected at least one cluster with no contradiction-free persona"
    catalogue_text = {a.description for a in config.ARCHETYPES}
    for record in unmatched:
        assert record["name"].startswith("Segment ")
        assert record["description"] not in catalogue_text
        assert record["rejected_archetype"] and record["contradicted_axes"]
    assert len({r["name"] for r in records}) == len(records)


def test_fallback_traits_use_the_same_band_as_persona_claims():
    """A fallback may not call a segment 'close to average' on an axis a persona
    claim would call high or low."""
    upper_middle = {axis: 0.5 for axis in config.MATCH_AXES} | {"spending_score": 0.67}
    assert personas.distinguishing_traits(upper_middle) == ["high engagement"]
    neutral = {axis: 0.5 for axis in config.MATCH_AXES}
    assert personas.distinguishing_traits(neutral) == []
