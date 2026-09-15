"""Cluster profiling and persona assignment (F07).

The source bound persona metadata *positionally*: persona n in a fixed table was
attached to cluster index n emitted by K-Means. K-Means label indices are
arbitrary, so the committed personas were visibly wrong — the cluster labelled
"Prudent Affluents" was in fact the bargain-hunter group, and so on.

Here personas are bound to cluster **content**. Each cluster centroid is placed
on the same 0–1 scale as the archetype profiles and aligned one-to-one with the
archetypes by Hungarian assignment. That is an optimal matching under a chosen
distance, not a proof of semantic correctness: if no cluster resembles an
archetype, the matching still assigns one. So every persona also carries a
``match_quality`` score which the dashboard shows, and the pipeline warns when a
match is weak, rather than hiding a poor fit behind a confident label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import config

# A match this poor means the cluster does not really look like any archetype.
WEAK_MATCH_THRESHOLD = 0.55

# Large enough that a contradicted pairing is chosen only when every alternative
# assignment is also contradicted.
CONTRADICTION_PENALTY = 1_000.0


def _scale_positions(profiles: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Place each cluster mean at its percentile within the whole customer base.

    0.5 is the population median on every axis by construction, so "high",
    "low" and "average" mean the same thing on every attribute. An earlier linear
    5th-95th-percentile scaling broke that on skewed and discrete axes: the
    median customer's spend landed at 0.21 and web visits at 0.71, so a cluster
    at the 78th percentile of spend read as "average" and every high/low judgement
    on those axes was biased.

    Ties use the mid-rank, which matters for discrete attributes such as web
    visits and household size where many customers share a value.
    """
    positions = {}
    for axis in config.MATCH_AXES:
        population = np.sort(reference[axis].to_numpy(dtype=float))
        n = len(population)
        values = profiles[axis].to_numpy(dtype=float)
        below = np.searchsorted(population, values, side="left")
        at_or_below = np.searchsorted(population, values, side="right")
        positions[axis] = (below + 0.5 * (at_or_below - below)) / n
    return pd.DataFrame(positions, index=profiles.index)


def claim_kind(target: float) -> str:
    """Which claim a signature target encodes: "high", "low" or "average"."""
    if target >= config.CLAIM_HIGH_TARGET:
        return "high"
    if target <= config.CLAIM_LOW_TARGET:
        return "low"
    return "average"


def claim_violated(actual: np.ndarray, target: float) -> np.ndarray:
    """Whether each cluster position contradicts the claim the target encodes."""
    kind = claim_kind(target)
    if kind == "high":
        return actual < config.CLAIM_HIGH_MIN
    if kind == "low":
        return actual > config.CLAIM_LOW_MAX
    low, high = config.CLAIM_MID_BAND
    return (actual < low) | (actual > high)


def assign_archetypes(cluster_means: pd.DataFrame, reference: pd.DataFrame) -> dict[int, dict]:
    """Align clusters to archetypes one-to-one, minimising total profile distance."""
    positions = _scale_positions(cluster_means, reference)

    # The matcher chooses the best-fitting archetypes from the *whole* catalogue
    # rather than being forced onto a fixed five. Constraining it to a prefix
    # would recreate the source's defect in a subtler form: with no young or
    # low-engagement-affluent segment in this population, a fixed five would push
    # those names onto clusters that contradict them.
    archetypes = list(config.ARCHETYPES)
    if len(archetypes) < len(cluster_means):  # pragma: no cover - k <= len(ARCHETYPES)
        raise ValueError(
            f"Only {len(archetypes)} archetypes available for {len(cluster_means)} clusters."
        )

    target = np.array([[a.profile[axis] for axis in config.MATCH_AXES] for a in archetypes])
    actual = positions[config.MATCH_AXES].to_numpy(dtype=float)

    # Normalised Euclidean distance in the 0–1 profile space.
    distance = np.linalg.norm(actual[:, None, :] - target[None, :, :], axis=2)
    distance /= np.sqrt(len(config.MATCH_AXES))  # max possible distance -> 1.0

    # Hard constraint: an averaged distance lets many agreeing axes outvote the
    # one a persona is named for (a "Lapsing Browsers" label on a cluster with
    # below-average recency, say). Any pair that contradicts the archetype on one
    # of its signature axes is priced out of the assignment, so it is only ever
    # chosen when no contradiction-free assignment exists at all — and is then
    # flagged rather than presented as a confident match.
    axis_index = {axis: i for i, axis in enumerate(config.MATCH_AXES)}
    contradictions: dict[tuple[int, int], list[str]] = {}
    penalty = np.zeros_like(distance)
    for col, archetype in enumerate(archetypes):
        for axis in archetype.signature:
            violated = claim_violated(actual[:, axis_index[axis]], archetype.profile[axis])
            for row in np.flatnonzero(violated):
                contradictions.setdefault((row, col), []).append(axis)
                penalty[row, col] = CONTRADICTION_PENALTY

    rows, cols = linear_sum_assignment(distance + penalty)

    assignment = {}
    for row, col in zip(rows, cols):
        cluster = int(cluster_means.index[row])
        archetype = archetypes[col]
        assignment[cluster] = {
            "archetype": archetype,
            "match_quality": float(1.0 - distance[row, col]),
            "contradicted_axes": contradictions.get((row, col), []),
            "profile_position": {axis: float(positions.iloc[row][axis]) for axis in config.MATCH_AXES},
        }
    return assignment


def profile(frame: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    """Build the full persona records: statistics plus matched identity."""
    working = frame.copy()
    working["cluster"] = np.asarray(labels)

    cluster_means = working.groupby("cluster")[config.BASE_FEATURES].mean()
    counts = working.groupby("cluster").size()
    total = int(len(working))

    assignment = assign_archetypes(cluster_means, working)

    records = []
    unmatched_index = 0
    for cluster in cluster_means.index:
        matched = assignment[int(cluster)]
        if matched["contradicted_axes"]:
            # No catalogued persona fits without contradicting this cluster's data.
            # Forcing one — even with a warning — would put a marketing narrative on
            # screen that the numbers disprove. Describe the cluster from its own
            # profile instead.
            identity = describe_unmatched(
                int(cluster), matched["profile_position"], unmatched_index
            )
            unmatched_index += 1
        else:
            archetype = matched["archetype"]
            identity = {
                "name": archetype.name,
                "key": archetype.key,
                "tagline": archetype.tagline,
                "badge": archetype.badge,
                "color": archetype.color,
                "description": archetype.description,
                "strategy": archetype.strategy,
                "matched": True,
            }
        records.append(
            {
                "cluster": int(cluster),
                **identity,
                "size": int(counts.loc[cluster]),
                "share": float(counts.loc[cluster] / total),
                "means": {col: float(cluster_means.loc[cluster, col]) for col in config.BASE_FEATURES},
                "match_quality": matched["match_quality"],
                "rejected_archetype": (
                    matched["archetype"].name if matched["contradicted_axes"] else None
                ),
                "contradicted_axes": matched["contradicted_axes"],
                "weak_match": bool(
                    matched["match_quality"] < WEAK_MATCH_THRESHOLD or matched["contradicted_axes"]
                ),
                "profile_position": matched["profile_position"],
            }
        )

    records.sort(key=lambda r: r["cluster"])
    return records


def distinguishing_traits(position: dict[str, float], limit: int = 3) -> list[str]:
    """The cluster's most pronounced traits, most extreme first, in plain words.

    A trait counts as distinguishing exactly when it falls outside the band the
    persona claims treat as "average", so a fallback description can never call a
    segment "close to average" on an axis a persona claim would call high or low.
    """
    low_edge, high_edge = config.CLAIM_MID_BAND
    ranked = sorted(position.items(), key=lambda item: abs(item[1] - 0.5), reverse=True)
    traits = []
    for axis, value in ranked:
        if low_edge < value < high_edge:
            break
        low, high = config.TRAIT_WORDS[axis]
        traits.append(high if value > 0.5 else low)
        if len(traits) == limit:
            break
    return traits


def describe_unmatched(cluster: int, position: dict[str, float], index: int) -> dict:
    """An honest, data-derived identity for a cluster no persona fits."""
    traits = distinguishing_traits(position)
    summary = ", ".join(traits) if traits else "close to average on every axis"
    return {
        "name": f"Segment {cluster}",
        "key": f"unmatched_{cluster}",
        "tagline": summary[:1].upper() + summary[1:],
        "badge": "◇",
        "color": config.UNMATCHED_COLORS[index % len(config.UNMATCHED_COLORS)],
        "description": (
            f"No catalogued persona describes this segment without contradicting its data, so it "
            f"is shown by its own profile rather than given a narrative the numbers do not "
            f"support. Distinguishing traits: {summary}."
        ),
        "strategy": (
            "No validated marketing playbook exists for this profile. Review the segment's "
            "attribute means before targeting it, rather than applying a persona that does not fit."
        ),
        "matched": False,
    }


def by_cluster(records: list[dict]) -> dict[int, dict]:
    return {r["cluster"]: r for r in records}


def sanity_notes(records: list[dict]) -> list[str]:
    """Human-readable warnings about weak or surprising persona matches."""
    notes = []
    for record in records:
        if record["contradicted_axes"]:
            axes = ", ".join(config.ATTRIBUTE_LABELS[a] for a in record["contradicted_axes"])
            notes.append(
                f"Cluster {record['cluster']} fits no catalogued persona: the closest, "
                f"“{record['rejected_archetype']}”, contradicts its data on {axes}. It is shown "
                f"as “{record['name']}” with a data-derived description instead."
            )
        elif record["weak_match"]:
            notes.append(
                f"Cluster {record['cluster']} was matched to “{record['name']}” at only "
                f"{record['match_quality']:.0%} profile agreement — the segment does not "
                f"closely resemble any catalogued archetype."
            )
    return notes
