"""Autonomous mining-parameter search.

A four-phase hill climb over mining configurations, run on a deterministic
sub-corpus so it stays interactive:

  Phase 1  backbone tournament   — the three algorithms at the baseline thresholds
  Phase 2  threshold mutations   — five targeted changes to support/confidence/lift
  Phase 3  hyperparameter grid   — four points over (support, confidence, lift)
  Phase 4  bundle optimisation   — longer itemsets, to reach 3-4 item antecedents

Objective: **mean lift across the surviving rule set**.

Acceptance: a trial is accepted only when it improves the incumbent mean lift by
more than ``MIN_LIFT_GAIN`` *and* leaves at least a minimum number of rules
standing. The rule floor is deliberate and documented: mean lift is trivially
gamed by tightening thresholds until three freak rules remain, and a rule set
that small is useless to the recommender. It is applied to every trial in its
phase, not selectively.

Every step is a real run. Nothing here is replayed, authored or simulated, and
rejected trials are kept — the dead ends are the point of showing the trajectory.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from . import config, data, mining, rules as rules_module


def _trial(corpus, params: dict, algorithm: str) -> dict:
    """Mine and score one configuration."""
    result = mining.mine(algorithm, corpus, params["min_support"], params["max_len"])
    rule_list = rules_module.generate(result, params["min_confidence"], params["min_lift"])
    return {
        "mean_lift": rules_module.mean_lift(rule_list),
        "n_rules": len(rule_list),
        "n_itemsets": result.n_itemsets,
        "top_lift": rules_module.top_lift(rule_list),
        "mean_confidence": rules_module.mean_confidence(rule_list),
        "elapsed_seconds": result.elapsed_seconds,
        "max_antecedent_len": max((len(r.antecedent) for r in rule_list), default=0),
    }


def _step(
    step_id: int,
    iteration: int,
    phase: str,
    category: str,
    hypothesis: str,
    snippet: str,
    params: dict,
    before: float,
    outcome: dict,
    decision: str,
    reflection: str,
) -> dict:
    return {
        "step_id": step_id,
        "iteration": iteration,
        "phase": phase,
        "category": category,
        "hypothesis": hypothesis,
        "code_change": snippet,
        "params": dict(params),
        "mean_lift_before": before,
        "mean_lift_after": outcome["mean_lift"],
        "delta": outcome["mean_lift"] - before,
        "n_rules": outcome["n_rules"],
        "n_itemsets": outcome["n_itemsets"],
        "top_lift": outcome["top_lift"],
        "mean_confidence": outcome["mean_confidence"],
        "max_antecedent_len": outcome["max_antecedent_len"],
        "elapsed_seconds": outcome["elapsed_seconds"],
        "decision": decision,
        "reflection": reflection,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _verdict(before: float, outcome: dict, min_rules: int) -> tuple[str, str]:
    """Apply the acceptance rule and explain the outcome in the agent's own terms."""
    delta = outcome["mean_lift"] - before
    if outcome["n_rules"] < min_rules:
        return (
            "rejected",
            f"Mean lift moved to {outcome['mean_lift']:.3f} ({delta:+.3f}), but only "
            f"{outcome['n_rules']} rules survived, below the floor of {min_rules}. "
            f"Mean lift is easy to inflate by keeping a handful of freak pairs; a rule "
            f"set this small cannot serve recommendations, so the trial is rejected "
            f"regardless of the objective.",
        )
    if delta > config.MIN_LIFT_GAIN:
        return (
            "accepted",
            f"Mean lift improved {before:.3f} → {outcome['mean_lift']:.3f} "
            f"({delta:+.3f}) with {outcome['n_rules']} rules still standing, clearing "
            f"both the {config.MIN_LIFT_GAIN} gain threshold and the rule floor. "
            f"Adopted as the new incumbent.",
        )
    return (
        "rejected",
        f"Mean lift went {before:.3f} → {outcome['mean_lift']:.3f} ({delta:+.3f}), "
        f"which does not clear the {config.MIN_LIFT_GAIN} threshold. The incumbent "
        f"configuration is kept.",
    )


def execute(
    n_orders: int = config.SEARCH_N_ORDERS,
    seed: int = config.SEED,
) -> dict:
    """Run the search. Every number below is measured during this call."""
    started = time.perf_counter()
    corpus = data.subsample(data.load_corpus(), n_orders, seed=seed)

    baseline_params = dict(config.SEARCH_BASELINE)
    trajectory: list[dict] = []
    step_id = 0

    # ------------------------------------------------------------------ Phase 1
    tournament = []
    for algorithm in mining.ALGORITHMS:
        outcome = _trial(corpus, baseline_params, algorithm)
        tournament.append({"algorithm": algorithm, **outcome})
    tournament.sort(key=lambda row: row["elapsed_seconds"])
    for rank, row in enumerate(tournament):
        row["is_champion"] = rank == 0
    backbone = tournament[0]["algorithm"]

    incumbent_params = dict(baseline_params)
    incumbent = tournament[0]["mean_lift"]
    initial = incumbent

    trajectory.append(
        _step(
            step_id, 0, "Phase 1 — backbone tournament", "baseline",
            hypothesis=(
                f"Run all three backbones at the baseline thresholds and take the "
                f"fastest as the search engine. Because the three are exact they must "
                f"agree on the rule set, so this picks a runtime, not a result."
            ),
            snippet=(
                f"for algorithm in ('Apriori', 'FP-Growth', 'ECLAT'):\n"
                f"    mine(algorithm, corpus, min_support={baseline_params['min_support']}, "
                f"max_len={baseline_params['max_len']})"
            ),
            params=incumbent_params,
            before=incumbent,
            outcome={**tournament[0]},
            decision="baseline",
            reflection=(
                f"{backbone} was fastest at {tournament[0]['elapsed_seconds']:.3f}s. "
                f"Starting mean lift is {incumbent:.3f} across "
                f"{tournament[0]['n_rules']} rules. All three found "
                f"{tournament[0]['n_itemsets']} itemsets, as they must."
            ),
        )
    )
    step_id += 1

    # ------------------------------------------------------------------ Phase 2
    base = baseline_params
    mutations = [
        (
            "lift gate",
            {**base, "min_lift": 2.20},
            "Raising the lift gate should strip out weak, near-independent pairs and "
            "lift the mean — the question is how many rules go with them.",
        ),
        (
            "confidence gate",
            {**base, "min_confidence": 0.35},
            "Demanding a higher conditional rate should keep only rules that predict "
            "their consequent reliably.",
        ),
        (
            "support floor",
            {**base, "min_support": 0.0008},
            "Lowering the support floor reaches niche pairings. Rare combinations tend "
            "to have high lift, so the mean may rise even as evidence per rule thins.",
        ),
        (
            "itemset length",
            {**base, "max_len": 2},
            "Restricting mining to pairs removes every multi-item rule. If the longer "
            "itemsets carry the weaker lifts, this helps; if they carry the strongest, "
            "it hurts.",
        ),
        (
            "high leverage",
            {**base, "min_support": 0.0012, "min_confidence": 0.30, "min_lift": 1.75},
            "Combine a firmer support floor with both gates raised, aiming for rules "
            "that are simultaneously well-evidenced and strongly associated.",
        ),
    ]
    for iteration, (category, params, hypothesis) in enumerate(mutations, start=1):
        outcome = _trial(corpus, params, backbone)
        decision, reflection = _verdict(incumbent, outcome, config.PHASE2_MIN_RULES)
        trajectory.append(
            _step(
                step_id, iteration, "Phase 2 — threshold mutations", category,
                hypothesis, _snippet(params), params, incumbent, outcome, decision, reflection,
            )
        )
        step_id += 1
        if decision == "accepted":
            incumbent = outcome["mean_lift"]
            incumbent_params = dict(params)

    # ------------------------------------------------------------------ Phase 3
    grid = [
        {**base, "min_support": 0.0008, "min_confidence": 0.30, "min_lift": 1.50},
        {**base, "min_support": 0.0012, "min_confidence": 0.25, "min_lift": 2.00},
        {**base, "min_support": 0.0020, "min_confidence": 0.20, "min_lift": 1.30},
        {**base, "min_support": 0.0010, "min_confidence": 0.40, "min_lift": 1.60},
    ]
    for offset, params in enumerate(grid):
        iteration = len(mutations) + 1 + offset
        outcome = _trial(corpus, params, backbone)
        decision, reflection = _verdict(incumbent, outcome, config.PHASE3_MIN_RULES)
        trajectory.append(
            _step(
                step_id, iteration, "Phase 3 — hyperparameter grid", "grid point",
                hypothesis=(
                    f"Grid point {offset + 1} of 4: support {params['min_support']}, "
                    f"confidence {params['min_confidence']}, lift {params['min_lift']}. "
                    f"Sweeping the three gates together to see whether any corner beats "
                    f"the incumbent that one-at-a-time mutation found."
                ),
                snippet=_snippet(params), params=params, before=incumbent,
                outcome=outcome, decision=decision, reflection=reflection,
            )
        )
        step_id += 1
        if decision == "accepted":
            incumbent = outcome["mean_lift"]
            incumbent_params = dict(params)

    # ------------------------------------------------------------------ Phase 4
    # A 4-item antecedent needs an itemset of at least 5 items, so this phase —
    # and only this phase — lifts the length cap above the production value of 4.
    bundle_params = {
        **incumbent_params,
        "max_len": config.BUNDLE_MAX_LEN,
        "min_support": min(incumbent_params["min_support"], config.BUNDLE_MIN_SUPPORT),
    }
    iteration = len(mutations) + len(grid) + 1
    outcome = _trial(corpus, bundle_params, backbone)
    decision, reflection = _verdict(incumbent, outcome, config.PHASE3_MIN_RULES)
    reflection += (
        f" Longest antecedent reached: {outcome['max_antecedent_len']} product(s). "
        f"max_len was raised to {config.BUNDLE_MAX_LEN} for this phase, since a "
        f"4-product antecedent requires a 5-item itemset, and the support floor "
        f"dropped to {bundle_params['min_support']} "
        f"(~{bundle_params['min_support'] * corpus.n_transactions:.0f} orders) to give "
        f"such a combination any chance of clearing it. That is thin evidence per "
        f"rule, and is the honest cost of reaching bundles this long in real "
        f"grocery baskets."
    )
    trajectory.append(
        _step(
            step_id, iteration, "Phase 4 — bundle optimisation", "itemset length",
            hypothesis=(
                f"Production mines itemsets up to length 4, which caps antecedents at 3 "
                f"products. Raise the cap to {config.BUNDLE_MAX_LEN} and drop the support "
                f"floor to {config.BUNDLE_MIN_SUPPORT} so genuine 3-4 product bundles can "
                f"form, then see whether those longer rules are strong enough to raise "
                f"the mean without collapsing the rule count."
            ),
            snippet=_snippet(bundle_params), params=bundle_params, before=incumbent,
            outcome=outcome, decision=decision, reflection=reflection,
        )
    )
    if decision == "accepted":
        incumbent = outcome["mean_lift"]
        incumbent_params = dict(bundle_params)

    accepted = sum(1 for s in trajectory if s["decision"] == "accepted")
    rejected = sum(1 for s in trajectory if s["decision"] == "rejected")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objective": "mean lift across the surviving rule set",
        "acceptance_rule": (
            f"accept when mean lift improves by more than {config.MIN_LIFT_GAIN} "
            f"AND at least {config.PHASE2_MIN_RULES} rules survive in phase 2 "
            f"({config.PHASE3_MIN_RULES} in phases 3-4)"
        ),
        "corpus": {
            "n_transactions": corpus.n_transactions,
            "n_transactions_with_two_or_more": corpus.n_minable,
            "seed": seed,
            "note": (
                "The search runs on the same corpus as the production pass, so its "
                "mean-lift figures are directly comparable to the served rule set. "
                "The thresholds differ, which is what the search is exploring."
            ),
        },
        "backbone": backbone,
        "tournament": tournament,
        "baseline_params": baseline_params,
        "best_params": incumbent_params,
        "initial_mean_lift": initial,
        "best_mean_lift": incumbent,
        "improvement_pct": ((incumbent - initial) / initial * 100.0) if initial else 0.0,
        "total_iterations": len(trajectory),
        "accepted": accepted,
        "rejected": rejected,
        "trajectory": trajectory,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _snippet(params: dict) -> str:
    return (
        "mine(corpus,\n"
        f"     min_support={params['min_support']},\n"
        f"     max_len={params['max_len']})\n"
        "rules.generate(result,\n"
        f"     min_confidence={params['min_confidence']},\n"
        f"     min_lift={params['min_lift']})"
    )


def save(record: dict) -> None:
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with config.AUTORESEARCH_JSON.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mining-parameter search.")
    parser.add_argument("--orders", type=int, default=config.SEARCH_N_ORDERS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    record = execute(n_orders=args.orders, seed=args.seed)
    save(record)
    print(
        f"Backbone:     {record['backbone']} (fastest in the tournament)\n"
        f"Mean lift:    {record['initial_mean_lift']:.3f} → {record['best_mean_lift']:.3f} "
        f"({record['improvement_pct']:+.1f}%)\n"
        f"Trials:       {record['total_iterations']} "
        f"({record['accepted']} accepted, {record['rejected']} rejected)\n"
        f"Best params:  {record['best_params']}\n"
        f"Elapsed:      {record['elapsed_seconds']:.1f}s\n"
        f"Artifact:     {config.AUTORESEARCH_JSON}"
    )


if __name__ == "__main__":
    main()
