"""The mining pipeline: benchmark pass, production pass, artifacts.

Two passes over the same corpus:

* the **benchmark** pass compares the three backbones (plus the external
  reference) at one shared set of thresholds;
* the **production** pass mines the rule set the application actually serves, at
  a lower support with stricter rule gates.

Everything the serving layer reads is written here. The application never mines
at request time; it answers from these artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from . import benchmark, config, data, graph as graph_module, mining, rules as rules_module


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)


def catalog_records(catalog_frame) -> list[dict]:
    return catalog_frame.to_dict(orient="records")


def derive_presets(
    production_rules: list[rules_module.Rule],
    names: dict[int, str],
    limit: int = 4,
) -> list[dict]:
    """Quick-start baskets, taken from the data rather than authored.

    Each preset is the antecedent of one of the highest-lift rules in this corpus,
    so clicking it is guaranteed to demonstrate a real recommendation. Multi-item
    antecedents are preferred because they make a more interesting basket; single
    product antecedents fill any remainder.
    """
    presets: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for wanted_multi in (True, False):
        for rule in production_rules:
            if len(presets) >= limit:
                break
            if (len(rule.antecedent) > 1) != wanted_multi:
                continue
            if rule.antecedent in seen:
                continue
            seen.add(rule.antecedent)
            presets.append(
                {
                    "items": list(rule.antecedent),
                    "label": " + ".join(names.get(i, str(i)) for i in rule.antecedent),
                    "top_lift": rule.lift,
                }
            )
    return presets


def basket_coverage(corpus, production_rules, sample: int = 5000) -> float:
    """Share of real baskets in which at least one rule fires.

    Reported honestly: a rule set that covers few baskets is a weak rule set, and
    hiding that behind a rule count would be misleading.
    """
    antecedents = [set(rule.antecedent) for rule in production_rules]
    transactions = corpus.transactions[:sample]
    if not transactions:
        return 0.0
    hits = sum(1 for t in transactions if any(a <= t for a in antecedents))
    return hits / len(transactions)


def run(
    min_support: float | None = None,
    min_confidence: float | None = None,
    min_lift: float | None = None,
    n_orders: int | None = None,
    max_len: int = config.MAX_LEN,
    include_reference: bool = True,
) -> dict:
    """Mine and persist everything. Every argument genuinely changes the output."""
    started = time.perf_counter()

    production = dict(config.PRODUCTION)
    if min_support is not None:
        production["min_support"] = float(min_support)
    if min_confidence is not None:
        production["min_confidence"] = float(min_confidence)
    if min_lift is not None:
        production["min_lift"] = float(min_lift)
    production["max_len"] = max_len

    catalog_frame = data.load_catalog()
    catalog = {int(r["product_id"]): r for r in catalog_records(catalog_frame)}

    corpus = data.load_corpus()
    if n_orders is not None and int(n_orders) < corpus.n_transactions:
        corpus = data.subsample(corpus, int(n_orders), seed=config.SEED)

    names = {pid: meta["product_name"] for pid, meta in catalog.items()}
    item_counts = corpus.item_counts()
    item_frequency = {
        pid: count / corpus.n_transactions for pid, count in item_counts.items()
    }

    # --- benchmark pass: same corpus, same thresholds, three backbones + reference
    benchmark_params = dict(config.BENCHMARK)
    benchmark_params["max_len"] = max_len
    benchmark_summary = benchmark.run(
        corpus,
        min_support=benchmark_params["min_support"],
        max_len=benchmark_params["max_len"],
        min_confidence=benchmark_params["min_confidence"],
        min_lift=benchmark_params["min_lift"],
        include_reference=include_reference,
    )

    # --- production pass: the rule set the app serves
    champion = benchmark_summary["champion"] or "ECLAT"
    production_result = mining.mine(
        champion, corpus, production["min_support"], production["max_len"]
    )
    production_rules = rules_module.generate(
        production_result, production["min_confidence"], production["min_lift"]
    )

    affinity_graph = graph_module.build(production_rules, catalog, item_frequency)
    presets = derive_presets(production_rules, names)

    coverage = basket_coverage(corpus, production_rules)
    headline = {
        "n_transactions": corpus.n_transactions,
        "basket_coverage": coverage,
        "n_transactions_with_two_or_more": corpus.n_minable,
        "catalog_size": len(catalog),
        "n_frequent_itemsets": production_result.n_itemsets,
        "n_active_rules": len(production_rules),
        "top_lift": rules_module.top_lift(production_rules),
        "mean_lift": rules_module.mean_lift(production_rules),
        "mean_confidence": rules_module.mean_confidence(production_rules),
        "n_pairwise_rules": affinity_graph["n_pairwise_rules_available"],
        "production_algorithm": champion,
    }
    benchmark_summary["production"] = headline
    benchmark_summary["production_params"] = production

    _write(config.RULES_JSON, {
        "params": production,
        "algorithm": champion,
        "n_transactions": corpus.n_transactions,
        "presets": presets,
        "preset_note": (
            "Quick-start baskets derived from this corpus: each one is the antecedent "
            "of a high-lift rule found in this run, not an authored scenario."
        ),
        "rules": rules_module.render_all(production_rules, names),
    })
    _write(config.GRAPH_JSON, affinity_graph)
    _write(config.BENCHMARKS_JSON, benchmark_summary)
    _write(config.CATALOG_JSON, {
        "price_note": (
            "price_illustrative is synthetic. Instacart ships no prices; these are "
            "generated from a fixed seed so the basket-economics panel has something "
            "to add up. They are not Instacart data and never influence mining, rule "
            "metrics, the affinity graph or recommendation ranking."
        ),
        "products": [
            {**record, "marginal_frequency_in_corpus": item_frequency.get(int(record["product_id"]), 0.0)}
            for record in catalog_records(catalog_frame)
        ],
    })

    corpus_facts = {}
    if config.CORPUS_META_JSON.exists():
        with config.CORPUS_META_JSON.open(encoding="utf-8") as handle:
            corpus_facts = json.load(handle)

    meta = {
        "corpus": corpus_facts,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": config.DATASET_NAME,
        "dataset_page": config.DATASET_PAGE,
        "seed": config.SEED,
        "benchmark_params": benchmark_params,
        "production_params": production,
        "elapsed_seconds": time.perf_counter() - started,
        "headline": headline,
    }
    _write(config.RUN_META_JSON, meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine the production rule set.")
    parser.add_argument("--min-support", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-lift", type=float, default=None)
    parser.add_argument("--orders", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=config.MAX_LEN)
    args = parser.parse_args()

    meta = run(
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        n_orders=args.orders,
        max_len=args.max_len,
    )
    headline = meta["headline"]
    print(
        f"Corpus:            {headline['n_transactions']:,} orders "
        f"({headline['catalog_size']} catalog products)\n"
        f"Champion backbone: {headline['production_algorithm']} (fastest measured)\n"
        f"Frequent itemsets: {headline['n_frequent_itemsets']:,}\n"
        f"Active rules:      {headline['n_active_rules']:,} "
        f"({headline['n_pairwise_rules']} of them single-product ➔ single-product)\n"
        f"Top lift:          {headline['top_lift']:.2f}x   "
        f"mean confidence {headline['mean_confidence']:.1%}\n"
        f"Basket coverage:   {headline['basket_coverage']:.1%} of baskets fire at least one rule\n"
        f"Elapsed:           {meta['elapsed_seconds']:.1f}s\n"
        f"Artifacts:         {config.ARTIFACT_DIR}"
    )


if __name__ == "__main__":
    main()
