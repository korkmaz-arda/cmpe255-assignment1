"""Acquire the Instacart data and build the mining corpus and product catalog.

Data identity
-------------
The canonical item is the Instacart ``product_id``. Product names, aisles and
departments are display metadata joined from the catalog at the presentation
edge; names are never used as keys.

Determinism
-----------
The catalog is every product appearing in at least ``CATALOG_MIN_FREQUENCY`` of
orders in the **full** train split, so it is a property of the dataset rather
than of the sample. The mining corpus is the whole train split by default, or a
seeded, content-blind random sample of it. Same seed, same corpus.

Support denominator
-------------------
A sampled order whose basket retains fewer than two catalog products is kept. It
cannot contribute a multi-item candidate, but it is a real order that did not
contain the combinations being scored, and dropping it would shrink the
denominator of every support figure and distort every metric derived from it.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from . import config

CHUNK = 1 << 20


class DataUnavailable(RuntimeError):
    """Raised when the dataset cannot be fetched, with manual instructions."""


def _manual_instructions() -> str:
    return (
        f"Could not download the {config.DATASET_NAME} archive.\n"
        f"Download it manually from {config.DATASET_PAGE} and save it as:\n"
        f"  {config.RAW_ARCHIVE}\n"
        f"then re-run `python -m basket.data` (without --download)."
    )


def download(force: bool = False) -> Path:
    """Fetch the dataset archive into ``data/raw/``. Fails loudly, never silently."""
    target = config.RAW_ARCHIVE
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    request = Request(config.DATASET_URL, headers={"User-Agent": "cmpe255-basket/1.0"})
    try:
        with urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
    except (URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DataUnavailable(f"{_manual_instructions()}\n\nUnderlying error: {exc}") from exc

    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise DataUnavailable(
            f"{_manual_instructions()}\n\nThe download did not return a zip archive."
        )
    tmp.replace(target)
    return target


def _read_member(archive: Path, member: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        if member not in names:
            raise DataUnavailable(
                f"Archive {archive} does not contain {member}. "
                f"Found: {sorted(names)}"
            )
        with zf.open(names[member]) as handle:
            return pd.read_csv(io.BytesIO(handle.read()), **kwargs)


def _require_archive() -> Path:
    if not config.RAW_ARCHIVE.exists():
        raise DataUnavailable(
            f"{config.RAW_ARCHIVE} not found. Run `python -m basket.data --download` first."
        )
    return config.RAW_ARCHIVE


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #

def illustrative_prices(product_ids: np.ndarray) -> np.ndarray:
    """Deterministic, clearly-synthetic unit prices.

    Instacart ships no prices. These exist only so the basket-economics panel has
    something to add up; they are generated from a fixed seed keyed on the sorted
    product ids and are never read by mining, scoring, graph construction or
    recommendation ranking.
    """
    rng = np.random.default_rng(config.PRICE_SEED)
    order = np.argsort(product_ids)
    draws = rng.uniform(config.PRICE_MIN, config.PRICE_MAX, size=len(product_ids))
    prices = np.empty(len(product_ids))
    prices[order] = np.round(draws, 2)
    return prices


def build_catalog(archive: Path | None = None) -> pd.DataFrame:
    """Products clearing the frequency floor, over the full train split."""
    archive = archive or _require_archive()
    order_products = _read_member(
        archive, "order_products__train.csv", usecols=["order_id", "product_id"]
    )
    products = _read_member(
        archive, "products.csv", usecols=["product_id", "product_name", "aisle_id", "department_id"]
    )
    aisles = _read_member(archive, "aisles.csv")
    departments = _read_member(archive, "departments.csv")

    total_orders = int(order_products["order_id"].nunique())
    frequency = (
        order_products.drop_duplicates(["order_id", "product_id"])
        .groupby("product_id")
        .size()
        .rename("order_count")
        .reset_index()
    )
    frequency["marginal_frequency"] = frequency["order_count"] / total_orders

    catalog = (
        frequency[frequency["marginal_frequency"] >= config.CATALOG_MIN_FREQUENCY]
        .sort_values(["order_count", "product_id"], ascending=[False, True])
        .merge(products, on="product_id", how="left")
        .merge(aisles, on="aisle_id", how="left")
        .merge(departments, on="department_id", how="left")
        .reset_index(drop=True)
    )
    catalog["product_name"] = catalog["product_name"].fillna("Unknown product")
    catalog["aisle"] = catalog["aisle"].fillna("unknown")
    catalog["department"] = catalog["department"].fillna("unknown")
    catalog["department_id"] = catalog["department_id"].fillna(-1).astype(int)
    catalog["color"] = catalog["department_id"].map(config.department_color)
    catalog["price_illustrative"] = illustrative_prices(catalog["product_id"].to_numpy())
    catalog["rank"] = np.arange(1, len(catalog) + 1)
    return catalog[
        [
            "rank", "product_id", "product_name", "aisle", "aisle_id",
            "department", "department_id", "color", "order_count",
            "marginal_frequency", "price_illustrative",
        ]
    ]


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #

def build_corpus(
    catalog: pd.DataFrame,
    n_orders: int = config.DEFAULT_N_ORDERS,
    seed: int = config.SEED,
    archive: Path | None = None,
) -> pd.DataFrame:
    """Seeded random sample of orders, projected onto the catalog.

    Every sampled order is returned, including those left with fewer than two
    catalog products: they belong in the support denominator.
    """
    archive = archive or _require_archive()
    order_products = _read_member(
        archive, "order_products__train.csv", usecols=["order_id", "product_id"]
    )
    all_orders = np.sort(order_products["order_id"].unique())
    n_orders = int(min(n_orders, len(all_orders)))

    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(all_orders, size=n_orders, replace=False))

    keep = order_products[order_products["order_id"].isin(sampled)]
    keep = keep[keep["product_id"].isin(set(catalog["product_id"]))]
    keep = keep.drop_duplicates(["order_id", "product_id"])

    grouped = (
        keep.sort_values(["order_id", "product_id"])
        .groupby("order_id")["product_id"]
        .apply(list)
    )
    # Reindex onto the full sample so empty baskets survive as empty lists.
    items = grouped.reindex(sampled)
    corpus = pd.DataFrame({"order_id": sampled})
    corpus["items"] = [list(v) if isinstance(v, list) else [] for v in items.to_numpy()]
    return corpus


@dataclass(frozen=True)
class Corpus:
    """The mined transaction database.

    ``transactions`` holds one frozenset of product_ids per sampled order,
    including the sparse ones. ``n_transactions`` is the support denominator.
    """

    transactions: list[frozenset[int]]
    order_ids: list[int]

    @property
    def n_transactions(self) -> int:
        return len(self.transactions)

    @property
    def n_minable(self) -> int:
        """Orders that can contribute a multi-item candidate (reporting only)."""
        return sum(1 for t in self.transactions if len(t) >= 2)

    def item_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for transaction in self.transactions:
            for item in transaction:
                counts[item] = counts.get(item, 0) + 1
        return counts


def to_corpus(frame: pd.DataFrame) -> Corpus:
    return Corpus(
        transactions=[frozenset(int(i) for i in row) for row in frame["items"]],
        order_ids=[int(o) for o in frame["order_id"]],
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def prepare(
    n_orders: int = config.DEFAULT_N_ORDERS,
    seed: int = config.SEED,
    archive: Path | None = None,
) -> dict:
    """Build and persist the catalog and corpus; return a summary."""
    archive = archive or _require_archive()
    catalog = build_catalog(archive)
    corpus_frame = build_corpus(catalog, n_orders=n_orders, seed=seed, archive=archive)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(config.CATALOG_PARQUET, index=False)
    corpus_frame.to_parquet(config.BASKETS_PARQUET, index=False)

    corpus = to_corpus(corpus_frame)
    summary = {
        "n_orders_sampled": corpus.n_transactions,
        "n_orders_with_two_or_more": corpus.n_minable,
        "catalog_size": len(catalog),
        "seed": seed,
        "mean_basket_size": float(np.mean([len(t) for t in corpus.transactions])),
        **split_statistics(archive),
    }
    with config.CORPUS_META_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    return summary


def split_statistics(archive: Path) -> dict:
    """Facts about the whole train split, for the report — measured, not written in."""
    order_products = _read_member(
        archive, "order_products__train.csv", usecols=["order_id", "product_id"]
    ).drop_duplicates(["order_id", "product_id"])
    products = _read_member(archive, "products.csv", usecols=["product_id", "product_name"])
    n_orders = int(order_products["order_id"].nunique())
    frequency = order_products.groupby("product_id").size().sort_values(ascending=False)
    top_id = int(frequency.index[0])
    names = dict(zip(products["product_id"], products["product_name"]))
    return {
        "split_orders": n_orders,
        "split_distinct_products": int(len(frequency)),
        "split_most_common_product": str(names.get(top_id, top_id)),
        "split_most_common_share": float(frequency.iloc[0] / n_orders),
        "split_median_product_share": float(frequency.median() / n_orders),
    }


def load_catalog() -> pd.DataFrame:
    if not config.CATALOG_PARQUET.exists():
        raise DataUnavailable(
            f"{config.CATALOG_PARQUET} not found. Run `python -m basket.data --download`."
        )
    return pd.read_parquet(config.CATALOG_PARQUET)


def load_corpus() -> Corpus:
    if not config.BASKETS_PARQUET.exists():
        raise DataUnavailable(
            f"{config.BASKETS_PARQUET} not found. Run `python -m basket.data --download`."
        )
    return to_corpus(pd.read_parquet(config.BASKETS_PARQUET))


def subsample(corpus: Corpus, n_orders: int, seed: int = config.SEED) -> Corpus:
    """Deterministic sub-sample of an in-memory corpus (used by the search)."""
    if n_orders >= corpus.n_transactions:
        return corpus
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(corpus.n_transactions, size=int(n_orders), replace=False))
    return Corpus(
        transactions=[corpus.transactions[i] for i in idx],
        order_ids=[corpus.order_ids[i] for i in idx],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Instacart basket corpus.")
    parser.add_argument("--download", action="store_true", help="fetch the archive first")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--orders", type=int, default=config.DEFAULT_N_ORDERS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    if args.download or args.force:
        print(f"Downloading {config.DATASET_NAME} ...")
        path = download(force=args.force)
        print(f"  archive: {path} ({path.stat().st_size / 1e6:.0f} MB)")

    summary = prepare(n_orders=args.orders, seed=args.seed)
    print(
        f"Catalog: {summary['catalog_size']} products "
        f"(order frequency >= {config.CATALOG_MIN_FREQUENCY:.1%}, full train split)\n"
        f"Corpus:  {summary['n_orders_sampled']:,} sampled orders "
        f"({summary['n_orders_with_two_or_more']:,} with 2+ catalog products; "
        f"all {summary['n_orders_sampled']:,} count toward support)\n"
        f"Mean basket size: {summary['mean_basket_size']:.2f} catalog products\n"
        f"Written: {config.CATALOG_PARQUET}, {config.BASKETS_PARQUET}"
    )


if __name__ == "__main__":
    main()
