"""Autonomous "AutoResearch" hill-climbing optimizer (F10).

A greedy, stateful search for a better clustering configuration, in four
sequential phases. Every trial is recorded as an inspectable step carrying a
hypothesis, the transformation applied, before/after silhouette, the delta, the
accept/reject decision and a written reflection.

Two properties matter and are preserved:

* the search is *stateful* — an accepted feature mutates the working matrix for
  every later step, so the trajectory is order-dependent and phases 3 and 4 are
  evaluated against an evolved feature set, not the original one;
* the acceptance gate is a real gate — a candidate is kept only if silhouette
  improves by more than ``AUTORESEARCH_GATE``.

Unlike the source, phase 4 actually computes and scores the consensus ensemble
rather than logging a fixed narrative with a fabricated co-assignment figure.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer, StandardScaler

from . import config, data, tournament

PHASES = {
    "backbone": "Backbone battle",
    "features": "Feature evolution",
    "hyperparameters": "Hyperparameter tuning",
    "consensus": "Consensus ensembling",
}


# --------------------------------------------------------------------------- #
# Candidate feature mutations (phase 2)
# --------------------------------------------------------------------------- #
# Deliberately started from the eight base attributes only: the engineered
# feature set the production pipeline uses is part of what this search is meant
# to rediscover on its own.

FEATURE_MUTATIONS = [
    (
        "monetary_velocity",
        "total_spend / (recency_days + 1)",
        "Spend per day since the last order should separate actively-converting customers from dormant ones.",
        lambda f: f["total_spend"] / (f["recency_days"] + 1.0),
    ),
    (
        "discretionary_ratio",
        "income_k / (spending_score + 1)",
        "Earning power relative to engagement should isolate high-capacity, low-activity households.",
        lambda f: f["income_k"] / (f["spending_score"] + 1.0),
    ),
    (
        "digital_engagement",
        "web_visits_month * (spending_score / 100)",
        "Browsing weighted by conversion should split genuine researchers from idle traffic.",
        lambda f: f["web_visits_month"] * (f["spending_score"] / 100.0),
    ),
    (
        "deal_affinity",
        "discount_sensitivity * (1 - spending_score / 100)",
        "Discount reliance concentrated among the least engaged should sharpen the price-led segment.",
        lambda f: f["discount_sensitivity"] * (1.0 - f["spending_score"] / 100.0),
    ),
    (
        "log_total_spend",
        "log1p(total_spend)",
        "Spend is heavily right-skewed; a log transform should stop the top decile dominating Euclidean distance.",
        lambda f: np.log1p(f["total_spend"]),
    ),
    (
        "power_income",
        "PowerTransformer(yeo-johnson).fit_transform(income_k)",
        "Gaussianizing income should make the income axis behave under a distance metric that assumes symmetry.",
        lambda f: pd.Series(
            PowerTransformer(method="yeo-johnson")
            .fit_transform(f[["income_k"]])
            .ravel(),
            index=f.index,
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #

def _fit_labels(spec: dict, X: np.ndarray, seed: int) -> np.ndarray:
    """Fit one configuration and return labels."""
    algorithm = spec["algorithm"]
    if algorithm == "kmeans":
        return KMeans(
            n_clusters=spec.get("k", config.DEFAULT_K),
            init="k-means++",
            n_init=spec.get("n_init", 10),
            random_state=seed,
        ).fit_predict(X)
    if algorithm == "gmm":
        return GaussianMixture(
            n_components=spec.get("k", config.DEFAULT_K),
            covariance_type=spec.get("covariance_type", "full"),
            random_state=seed,
        ).fit_predict(X)
    if algorithm == "agglomerative":
        return AgglomerativeClustering(
            n_clusters=spec.get("k", config.DEFAULT_K),
            linkage=spec.get("linkage", "ward"),
        ).fit_predict(X)
    if algorithm == "spectral":
        from sklearn.cluster import SpectralClustering

        return SpectralClustering(
            n_clusters=spec.get("k", config.DEFAULT_K),
            affinity="nearest_neighbors",
            random_state=seed,
            assign_labels="kmeans",
        ).fit_predict(X)
    if algorithm == "dbscan":
        from sklearn.cluster import DBSCAN

        min_samples = tournament.dbscan_min_samples(X.shape[1])
        return DBSCAN(
            eps=tournament.dbscan_epsilon(X, min_samples), min_samples=min_samples
        ).fit_predict(X)
    raise ValueError(f"Unknown algorithm: {algorithm}")  # pragma: no cover


ALGORITHM_FROM_NAME = {
    "K-Means": "kmeans",
    "Gaussian Mixture": "gmm",
    "Agglomerative": "agglomerative",
    "DBSCAN": "dbscan",
    "Spectral": "spectral",
}


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

class AutoResearch:
    """Greedy hill climb over feature mutations and hyperparameters."""

    def __init__(self, frame: pd.DataFrame, k: int = config.DEFAULT_K,
                 seed: int = config.DEFAULT_SEED):
        self.frame = frame.reset_index(drop=True)
        self.k = k
        self.seed = seed
        self.columns = list(config.BASE_FEATURES)
        self.working = self.frame[config.BASE_FEATURES].copy()
        self.steps: list[dict] = []
        self.incumbent: dict = {"algorithm": "kmeans", "k": k, "n_init": 10}
        self.incumbent_name = "K-Means"
        self.best_silhouette = -1.0
        self.best_config: dict = {}
        self.start_silhouette = -1.0
        self.backbone: list[dict] = []

    # -- helpers ----------------------------------------------------------- #

    def _matrix(self, extra: pd.Series | None = None, name: str | None = None) -> np.ndarray:
        frame = self.working
        if extra is not None and name is not None:
            frame = frame.assign(**{name: extra})
        return StandardScaler().fit_transform(frame.to_numpy(dtype=float))

    def _score(self, spec: dict, X: np.ndarray) -> tuple[float, dict]:
        return self._evaluate(_fit_labels(spec, X, self.seed), X)

    def _evaluate(self, labels: np.ndarray, X: np.ndarray) -> tuple[float, dict]:
        """Silhouette plus the partition-shape diagnostics the gate needs."""
        labels = np.asarray(labels)
        values, counts = np.unique(labels[labels != tournament.NOISE_LABEL], return_counts=True)
        shares = counts / max(len(labels), 1)
        return (
            tournament.silhouette_of(X, labels),
            {
                "sizes": {int(v): int(c) for v, c in zip(values, counts)},
                "smallest_share": float(shares.min()) if len(shares) else 0.0,
                "n_clusters": int(len(values)),
            },
        )

    @staticmethod
    def _describe_sizes(shape: dict) -> str:
        return "/".join(str(v) for v in sorted(shape.get("sizes", {}).values(), reverse=True))

    def _log(self, phase: str, component: str, hypothesis: str, transformation: str,
             before: float, after: float, decision: str, reflection: str, params: dict,
             shape: dict | None = None) -> None:
        self.steps.append(
            {
                "round": len(self.steps),
                "phase": phase,
                "phase_label": PHASES[phase],
                "component": component,
                "hypothesis": hypothesis,
                "transformation": transformation,
                "silhouette_before": float(before),
                "silhouette_after": float(after),
                "delta": float(after - before),
                "decision": decision,
                "reflection": reflection,
                "params": params,
                "partition": shape or {},
                "active_features": list(self.columns),
            }
        )

    def _consider(self, after: float, before: float, shape: dict) -> tuple[bool, str]:
        """The acceptance gate: a real silhouette gain *and* an actionable partition.

        Silhouette alone is gameable — isolating a few outliers into singleton
        clusters scores well and segments nothing useful. So a candidate must also
        leave every cluster holding at least ``MIN_CLUSTER_SHARE`` of the
        customers. The constraint is applied identically to every candidate in
        every phase, so it favours no particular algorithm; it encodes the
        business requirement that a segment has to be large enough to act on.
        """
        if shape["smallest_share"] < config.MIN_CLUSTER_SHARE:
            return False, "degenerate"
        if (after - before) > config.AUTORESEARCH_GATE:
            return True, "improved"
        return False, "gate"

    # -- phases ------------------------------------------------------------ #

    def phase_backbone(self) -> None:
        """Evaluate the five families on the base-attribute matrix."""
        X = self._matrix()
        self.backbone = tournament.run(X, k=self.k, seed=self.seed)
        best = tournament.best_entry(self.backbone)
        if best is None:  # pragma: no cover - defensive
            raise RuntimeError("No clustering family produced a valid partition.")

        _, baseline_shape = self._score(
            {"algorithm": ALGORITHM_FROM_NAME[best["algorithm"]], "k": self.k, "n_init": 10}, X
        )
        self.incumbent_name = best["algorithm"]
        self.incumbent = {"algorithm": ALGORITHM_FROM_NAME[best["algorithm"]], "k": self.k, "n_init": 10}
        self.start_silhouette = float(best["silhouette"])
        self.best_silhouette = self.start_silhouette
        self.best_config = {
            "stage": "backbone",
            "algorithm": self.incumbent_name,
            "features": list(self.columns),
            **self.incumbent,
        }

        self._log(
            phase="backbone",
            component=best["algorithm"],
            hypothesis=(
                "Before engineering anything, establish which clustering family best separates "
                "customers on the eight raw behavioural attributes."
            ),
            transformation=f"StandardScaler().fit_transform(frame[{len(self.columns)} base attributes])",
            before=self.start_silhouette,
            after=self.start_silhouette,
            decision="baseline",
            reflection=(
                f"{best['algorithm']} leads the backbone tournament at silhouette "
                f"{self.start_silhouette:.5f} on {len(self.columns)} raw attributes. It becomes the "
                f"incumbent; every later step is measured against this configuration."
            ),
            params=dict(self.incumbent),
            shape=baseline_shape,
        )

    def phase_features(self) -> None:
        """Propose interaction features one at a time, keeping only improvements."""
        for name, formula, hypothesis, derive in FEATURE_MUTATIONS:
            before = self.best_silhouette
            candidate = derive(self.frame)
            candidate = pd.Series(np.asarray(candidate, dtype=float), index=self.working.index)
            if not np.all(np.isfinite(candidate)):  # pragma: no cover - defensive
                candidate = candidate.replace([np.inf, -np.inf], np.nan).fillna(candidate.median())

            after, shape = self._score(self.incumbent, self._matrix(candidate, name))
            accepted, reason = self._consider(after, before, shape)

            if accepted:
                self.working[name] = candidate
                self.columns.append(name)
                self.best_silhouette = after
                self.best_config = {
                    "stage": "features",
                    "algorithm": self.incumbent_name,
                    "features": list(self.columns),
                    **self.incumbent,
                }
                reflection = (
                    f"Adding {name} lifted silhouette {after - before:+.5f} to {after:.5f}, clearing "
                    f"the {config.AUTORESEARCH_GATE:+.4f} gate. The feature is kept and every "
                    f"subsequent step is evaluated on the evolved {len(self.columns)}-column matrix."
                )
            elif reason == "degenerate":
                reflection = (
                    f"{name} reached silhouette {after:.5f}, but the resulting partition leaves its "
                    f"smallest cluster at {shape['smallest_share']:.1%} of customers — below the "
                    f"{config.MIN_CLUSTER_SHARE:.0%} actionability floor. The score comes from "
                    f"isolating outliers rather than segmenting the base, so the feature is rejected."
                )
            else:
                reflection = (
                    f"{name} moved silhouette {after - before:+.5f}, short of the "
                    f"{config.AUTORESEARCH_GATE:+.4f} gate. The working matrix is left unchanged — "
                    f"this signal is already carried by the columns in play."
                )

            self._log(
                phase="features",
                component=name,
                hypothesis=hypothesis,
                transformation=f"working['{name}'] = {formula}",
                before=before,
                after=after,
                decision="accepted" if accepted else "rejected",
                reflection=reflection,
                params={**self.incumbent, "n_features": len(self.columns)},
                shape=shape,
            )

    def phase_hyperparameters(self) -> None:
        """Try alternative configurations against the evolved matrix."""
        X = self._matrix()
        trials = [
            ("K-Means, 50 restarts", {"algorithm": "kmeans", "k": self.k, "n_init": 50},
             "More restarts should escape poor k-means++ initialisations on the evolved matrix."),
            (f"K-Means, k={self.k + 1}", {"algorithm": "kmeans", "k": self.k + 1, "n_init": 10},
             "The engineered axes may have split an existing segment in two."),
            (f"K-Means, k={self.k - 1}", {"algorithm": "kmeans", "k": self.k - 1, "n_init": 10},
             "Fewer, broader clusters may separate more cleanly than the current partition."),
            ("Agglomerative, average linkage", {"algorithm": "agglomerative", "k": self.k, "linkage": "average"},
             "Average linkage tolerates elongated clusters that Ward's variance criterion fractures."),
            ("Gaussian mixture, tied covariance", {"algorithm": "gmm", "k": self.k, "covariance_type": "tied"},
             "A shared covariance is far cheaper in parameters and may generalise better than full."),
        ]

        for label, spec, hypothesis in trials:
            before = self.best_silhouette
            after, shape = self._score(spec, X)
            accepted, reason = self._consider(after, before, shape)

            if accepted:
                self.incumbent = dict(spec)
                self.incumbent_name = label
                self.best_silhouette = after
                self.best_config = {
                    **spec,
                    "stage": "hyperparameters",
                    "algorithm": label,
                    "features": list(self.columns),
                }
                reflection = (
                    f"{label} reached {after:.5f} ({after - before:+.5f}) and becomes the new "
                    f"incumbent configuration."
                )
            elif reason == "degenerate":
                reflection = (
                    f"{label} scored {after:.5f}, nominally the best figure seen, but it splits the "
                    f"base {self._describe_sizes(shape)} — its smallest cluster holds "
                    f"{shape['smallest_share']:.1%} of customers, under the "
                    f"{config.MIN_CLUSTER_SHARE:.0%} actionability floor. This is the classic "
                    f"silhouette artefact of quarantining outliers into singleton clusters: the "
                    f"metric improves while the segmentation becomes useless. Rejected."
                )
            else:
                reflection = (
                    f"{label} scored {after:.5f} ({after - before:+.5f}), below the gate. The "
                    f"incumbent configuration is retained."
                )

            self._log(
                phase="hyperparameters",
                component=label,
                hypothesis=hypothesis,
                transformation=f"fit({spec}) on the {len(self.columns)}-column evolved matrix",
                before=before,
                after=after,
                decision="accepted" if accepted else "rejected",
                reflection=reflection,
                params=spec,
                shape=shape,
            )

    def phase_consensus(self) -> None:
        """Build and score a real co-association consensus ensemble.

        The source fitted two models here but never computed a consensus, logging
        the step as accepted with a zero delta and a fixed narrative quoting a
        ">91.4% co-assignment" figure that nothing produced. Here the
        co-association matrix is genuinely built from the two partitions, the
        consensus partition is extracted by average-linkage agglomerative
        clustering over ``1 - co-association``, and both the resulting silhouette
        and the measured co-assignment rate are computed from the data.
        """
        before = self.best_silhouette
        X = self._matrix()

        kmeans_labels = _fit_labels({"algorithm": "kmeans", "k": self.k, "n_init": 20}, X, self.seed)
        gmm_labels = _fit_labels({"algorithm": "gmm", "k": self.k, "covariance_type": "full"}, X, self.seed)

        # Co-association: for each pair, the fraction of the two partitions that
        # place them in the same cluster. Vectorised as two boolean adjacency
        # matrices averaged together.
        same_kmeans = kmeans_labels[:, None] == kmeans_labels[None, :]
        same_gmm = gmm_labels[:, None] == gmm_labels[None, :]
        co_association = (same_kmeans.astype(float) + same_gmm.astype(float)) / 2.0

        # Measured agreement between the two partitions, off-diagonal only.
        off_diagonal = ~np.eye(len(X), dtype=bool)
        co_assignment_rate = float((same_kmeans == same_gmm)[off_diagonal].mean())

        consensus_labels = AgglomerativeClustering(
            n_clusters=self.k, metric="precomputed", linkage="average"
        ).fit_predict(1.0 - co_association)

        after, shape = self._evaluate(consensus_labels, X)
        accepted, reason = self._consider(after, before, shape)

        if accepted:
            self.best_silhouette = after
            self.best_config = {
                "stage": "consensus",
                "algorithm": "Co-association consensus (K-Means + Gaussian mixture)",
                "features": list(self.columns),
                "k": self.k,
            }
            reflection = (
                f"The consensus partition scored {after:.5f} ({after - before:+.5f}), beating the "
                f"incumbent. The two base partitions agree on {co_assignment_rate:.1%} of customer "
                f"pairs, and resolving the remaining disagreement through the co-association matrix "
                f"produced a cleaner partition than either model alone."
            )
        elif reason == "degenerate":
            reflection = (
                f"The consensus partition scored {after:.5f}, but its smallest cluster holds only "
                f"{shape['smallest_share']:.1%} of customers, below the "
                f"{config.MIN_CLUSTER_SHARE:.0%} actionability floor. The two base partitions agree "
                f"on {co_assignment_rate:.1%} of customer pairs; averaging them pushed the "
                f"contested minority into tiny clusters. Rejected."
            )
        else:
            reflection = (
                f"The consensus partition scored {after:.5f} ({after - before:+.5f}), which does not "
                f"clear the gate. The two base partitions already agree on {co_assignment_rate:.1%} "
                f"of customer pairs, so there is little disagreement for an ensemble to resolve — "
                f"consensus mostly reproduces what the incumbent already found. Rejected."
            )

        self._log(
            phase="consensus",
            component="Co-association consensus (K-Means + Gaussian mixture)",
            hypothesis=(
                "Averaging two independent partitions into a co-association matrix should smooth "
                "away the boundary customers each model gets wrong on its own."
            ),
            transformation=(
                "C = (1[kmeans_i == kmeans_j] + 1[gmm_i == gmm_j]) / 2; "
                "AgglomerativeClustering(metric='precomputed', linkage='average').fit_predict(1 - C)"
            ),
            before=before,
            after=after,
            decision="accepted" if accepted else "rejected",
            reflection=reflection,
            params={
                "k": self.k,
                "members": ["K-Means (n_init=20)", "Gaussian mixture (full covariance)"],
                "measured_co_assignment_rate": co_assignment_rate,
                "linkage": "average over 1 - co-association",
            },
            shape=shape,
        )

    # -- driver ------------------------------------------------------------ #

    def run(self) -> dict:
        started = time.perf_counter()
        self.phase_backbone()
        self.phase_features()
        self.phase_hyperparameters()
        self.phase_consensus()
        elapsed = time.perf_counter() - started

        accepted = [s for s in self.steps if s["decision"] == "accepted"]
        rejected = [s for s in self.steps if s["decision"] == "rejected"]
        improvement = (
            (self.best_silhouette - self.start_silhouette) / abs(self.start_silhouette) * 100.0
            if self.start_silhouette
            else 0.0
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": self.seed,
            "k": self.k,
            "n_rows": int(len(self.frame)),
            "elapsed_seconds": float(elapsed),
            "start_silhouette": float(self.start_silhouette),
            "best_silhouette": float(self.best_silhouette),
            "improvement_pct": float(improvement),
            "improvement_basis": (
                "Percentage change against this run's own phase-1 starting score "
                "(the best backbone family on the eight raw attributes). It is not a gain "
                "over any external benchmark, and it does not describe the deployed model."
            ),
            "acceptance_gate": (
                f"A candidate is accepted only if silhouette improves by more than "
                f"{config.AUTORESEARCH_GATE:+.4f} AND every cluster retains at least "
                f"{config.MIN_CLUSTER_SHARE:.0%} of customers. The second condition is applied "
                f"identically to every candidate and exists because silhouette alone rewards "
                f"isolating outliers into singleton clusters, which scores well and segments nothing."
            ),
            "total_steps": len(self.steps),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "backbone": self.backbone,
            "active_features": list(self.columns),
            "best_config": self.best_config,
            "steps": self.steps,
            "promoted_to_production": False,
            "promotion_note": (
                "AutoResearch is exploratory. Its best configuration is not fed back into the "
                "served model: the production pipeline keeps its own fixed feature set and k so "
                "that persona identities stay stable and the served metrics describe exactly the "
                "model being served. Its silhouette is measured on a different sample and feature "
                "set, so it is not directly comparable to the served model's silhouette."
            ),
        }


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def execute(n_rows: int | None = config.AUTORESEARCH_SAMPLE, k: int = config.DEFAULT_K,
            seed: int = config.DEFAULT_SEED, frame: pd.DataFrame | None = None) -> dict:
    """Run a full search and return the run record."""
    frame = data.load_clean() if frame is None else frame
    subset = data.sample(frame, n_rows, seed=seed)
    return AutoResearch(subset, k=k, seed=seed).run()


def save(record: dict, path=None) -> None:
    path = path or config.AUTORESEARCH_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def load(path=None) -> dict | None:
    path = path or config.AUTORESEARCH_JSON
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoResearch hill-climbing search.")
    parser.add_argument("--rows", type=int, default=config.AUTORESEARCH_SAMPLE)
    parser.add_argument("--k", type=int, default=config.DEFAULT_K)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args()

    record = execute(n_rows=args.rows, k=args.k, seed=args.seed)
    save(record)
    print(f"Rows:        {record['n_rows']}   k={record['k']}   seed={record['seed']}")
    print(f"Start:       {record['start_silhouette']:.5f}")
    print(f"Best:        {record['best_silhouette']:.5f}  ({record['improvement_pct']:+.1f}% vs its own start)")
    print(f"Steps:       {record['total_steps']} ({record['accepted']} accepted / {record['rejected']} rejected)")
    print(f"Features:    {', '.join(record['active_features'])}")
    print(f"Written to:  {config.AUTORESEARCH_JSON}")


if __name__ == "__main__":
    main()
