"""
Convert raw downloaded CSVs into clean Parquet files.

Why Parquet?
------------
Parquet is a columnar file format — data is stored column-by-column rather
than row-by-row (like CSV). This makes it dramatically faster when you only
need a few columns from a large table. Reading 3.4M orders from a Parquet
file takes ~0.1s; from a CSV it takes ~3s.

It also stores data types explicitly (integers stay integers, dates stay dates)
so you don't have to parse them every time.

Usage
-----
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --dataset instacart
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    INSTACART_DIR, M5_DIR, OFF_DIR,
    ORDERS_PATH, PRODUCTS_PATH, ORDER_ITEMS_PATH,
    M5_SALES_PATH, OFF_PRODUCTS_PATH,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a cleaned frame to Parquet with zstd compression and log rows + size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    size_mb = path.stat().st_size / 1e6
    logger.info(f"  \u2713 wrote {path.name}: {len(df):,} rows, {size_mb:,.1f} MB")


# ── Instacart ─────────────────────────────────────────────────────────────────

def preprocess_instacart() -> None:
    """
    Clean and merge the Instacart CSV files into three Parquet files:
      - orders.parquet      — one row per order
      - products.parquet    — one row per product (with department + aisle)
      - order_items.parquet — one row per product line in an order
    """
    logger.info("Processing Instacart data…")

    # ── Orders ────────────────────────────────────────────────────────────────
    # The raw file has one row per order.
    # 'eval_set' tells us whether it was in train/test/prior splits for the
    # Kaggle competition — we keep all of them for our purposes.
    orders = pd.read_csv(INSTACART_DIR / "orders.csv")
    orders["order_dow"]         = orders["order_dow"].astype("int8")
    orders["order_hour_of_day"] = orders["order_hour_of_day"].astype("int8")
    # days_since_prior_order is NaN for a user's first-ever order — that's fine
    logger.info(f"  Orders: {len(orders):,} rows")
    write_parquet(orders, ORDERS_PATH)

    # ── Products ──────────────────────────────────────────────────────────────
    # Join products with their aisle and department names
    products    = pd.read_csv(INSTACART_DIR / "products.csv")
    aisles      = pd.read_csv(INSTACART_DIR / "aisles.csv")
    departments = pd.read_csv(INSTACART_DIR / "departments.csv")

    products = (
        products
        .merge(aisles,      on="aisle_id")
        .merge(departments, on="department_id")
    )
    logger.info(f"  Products: {len(products):,} rows")
    write_parquet(products, PRODUCTS_PATH)

    # ── Order items ───────────────────────────────────────────────────────────
    # The Instacart dataset splits order items across three CSV files
    # (prior orders, training orders, test orders).  We concatenate them all.
    frames = []
    for fname in ["order_products__prior.csv", "order_products__train.csv"]:
        fpath = INSTACART_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            frames.append(df)
            logger.info(f"  Loaded {fname}: {len(df):,} rows")

    order_items = pd.concat(frames, ignore_index=True)
    # reordered = 1 means the customer has bought this product before
    order_items["reordered"]      = order_items["reordered"].astype("int8")
    order_items["add_to_cart_order"] = order_items["add_to_cart_order"].astype("int16")
    logger.info(f"  Order items total: {len(order_items):,} rows")
    write_parquet(order_items, ORDER_ITEMS_PATH)

    logger.info("  ✓ Instacart → Parquet complete")


# ── M5 Forecasting ────────────────────────────────────────────────────────────

def preprocess_m5() -> None:
    """
    Reshape M5 sales data from wide format (one column per day) to long
    format (one row per product-day), which is what ML models expect.

    The raw file looks like:
        product_id | store_id | d_1 | d_2 | d_3 | ... | d_1913
    We reshape to:
        product_id | store_id | day | sales
    """
    logger.info("Processing M5 data…")

    sales_file = M5_DIR / "sales_train_evaluation.csv"
    if not sales_file.exists():
        logger.warning(f"  {sales_file} not found — skipping M5")
        return

    df = pd.read_csv(sales_file)
    logger.info(f"  Loaded M5: {df.shape[0]:,} products × {df.shape[1]} columns")

    # Identify day columns (they're named d_1, d_2, ..., d_1941)
    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]

    # pd.melt converts wide → long:
    # before: row per product, column per day
    # after:  row per (product, day) pair
    df_long = df.melt(id_vars=id_cols, value_vars=day_cols,
                      var_name="day", value_name="sales")

    # Convert day string "d_1" → integer 1
    df_long["day_num"] = df_long["day"].str.replace("d_", "").astype("int16")

    # Load calendar to get real dates
    cal = pd.read_csv(M5_DIR / "calendar.csv")
    cal = cal[["d", "date", "wday", "month", "year", "event_name_1", "snap_CA"]]
    cal = cal.rename(columns={"d": "day"})

    df_long = df_long.merge(cal, on="day", how="left")
    df_long["date"] = pd.to_datetime(df_long["date"])

    # Keep only rows with non-zero sales to keep the file manageable
    # (most product-days have zero sales — we can reconstruct zeros where needed)
    df_long = df_long[df_long["sales"] > 0].reset_index(drop=True)

    logger.info(f"  M5 long format: {len(df_long):,} non-zero rows")
    write_parquet(df_long, M5_SALES_PATH)
    logger.info("  ✓ M5 → Parquet complete")


# ── Open Food Facts ───────────────────────────────────────────────────────────

def preprocess_open_food_facts() -> None:
    """
    Extract the useful columns from the Open Food Facts export for the RAG
    knowledge base.

    The export is ~9 GB uncompressed across 211 columns, which is too large to
    load into pandas on a laptop (that read swaps for hours). We stream it with
    DuckDB, which reads the gzipped TSV, keeps only the columns and rows we need,
    and writes Parquet directly with bounded memory. Usually a few minutes.
    """
    logger.info("Processing Open Food Facts...")

    gz_file = OFF_DIR / "products.csv.gz"
    if not gz_file.exists():
        logger.warning(f"  {gz_file} not found, skipping Open Food Facts")
        return

    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")

    read = (
        f"read_csv_auto('{gz_file.as_posix()}', all_varchar=true, "
        f"ignore_errors=true, quote='')"
    )
    out_path = OFF_PRODUCTS_PATH.as_posix()

    logger.info("  Streaming Open Food Facts with DuckDB (a few minutes)...")
    con.execute(f"""
        COPY (
            SELECT
                code,
                product_name,
                brands,
                categories_en,
                countries_en,
                ingredients_text,
                TRY_CAST("energy-kcal_100g" AS DOUBLE) AS "energy-kcal_100g",
                TRY_CAST(fat_100g AS DOUBLE)           AS fat_100g,
                TRY_CAST(carbohydrates_100g AS DOUBLE) AS carbohydrates_100g,
                TRY_CAST(proteins_100g AS DOUBLE)      AS proteins_100g,
                TRY_CAST(fiber_100g AS DOUBLE)         AS fiber_100g,
                TRY_CAST(sugars_100g AS DOUBLE)        AS sugars_100g,
                TRY_CAST(salt_100g AS DOUBLE)          AS salt_100g,
                TRY_CAST(nova_group AS INTEGER)        AS nova_group,
                nutriscore_grade,
                main_category_en
            FROM {read}
            WHERE product_name  IS NOT NULL
              AND categories_en IS NOT NULL
              AND (countries_en IS NULL OR lower(countries_en) LIKE '%australia%')
        ) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)
    """)

    kept = con.execute(f"SELECT count(*) FROM '{out_path}'").fetchone()[0]
    size_mb = OFF_PRODUCTS_PATH.stat().st_size / 1e6
    con.close()
    logger.info(f"  wrote {OFF_PRODUCTS_PATH.name}: {kept:,} rows, {size_mb:,.1f} MB")
    logger.info("  Open Food Facts -> Parquet complete")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess GroceryIQ datasets")
    parser.add_argument("--dataset", choices=["instacart", "m5", "off", "all"],
                        default="all")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("instacart", "all"):
        preprocess_instacart()
    if args.dataset in ("m5", "all"):
        preprocess_m5()
    if args.dataset in ("off", "all"):
        preprocess_open_food_facts()

    logger.info("\nPreprocessing complete.")
    logger.info("Next step: jupyter notebook notebooks/01_eda.ipynb")


if __name__ == "__main__":
    main()
