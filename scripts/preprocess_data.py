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
    orders.to_parquet(ORDERS_PATH, index=False)

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
    products.to_parquet(PRODUCTS_PATH, index=False)

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
    order_items.to_parquet(ORDER_ITEMS_PATH, index=False)

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
    df_long.to_parquet(M5_SALES_PATH, index=False)
    logger.info("  ✓ M5 → Parquet complete")


# ── Open Food Facts ───────────────────────────────────────────────────────────

def preprocess_open_food_facts() -> None:
    """
    Extract the useful columns from the Open Food Facts CSV and clean them
    for use as a RAG knowledge base.

    Open Food Facts has ~185 columns per product — most are empty.
    We keep the columns that matter for product search and recommendation.
    """
    logger.info("Processing Open Food Facts…")

    gz_file = OFF_DIR / "products.csv.gz"
    if not gz_file.exists():
        logger.warning(f"  {gz_file} not found — skipping Open Food Facts")
        return

    # Only load the columns we actually need — much faster than loading all 185
    cols = [
        "code",                   # barcode
        "product_name",
        "brands",
        "categories_en",
        "countries_en",
        "ingredients_text",
        "nutriments",             # JSON-like string of nutritional values
        "energy-kcal_100g",
        "fat_100g",
        "carbohydrates_100g",
        "proteins_100g",
        "fiber_100g",
        "sugars_100g",
        "salt_100g",
        "nova_group",             # food processing level (1=unprocessed, 4=ultra-processed)
        "nutriscore_grade",       # A–E health rating
        "main_category_en",
    ]

    logger.info("  Reading Open Food Facts (this may take 1–2 minutes)…")
    df = pd.read_csv(
        gz_file,
        sep="\t",                 # OFF uses tab-separated values, not comma
        usecols=[c for c in cols if c != "nutriments"],
        low_memory=False,
        on_bad_lines="skip",      # some rows have encoding issues
    )

    logger.info(f"  Raw: {len(df):,} products")

    # Filter to products with at least a name and category
    df = df[df["product_name"].notna() & df["categories_en"].notna()]

    # Restrict to products available in Australia or with no country listed
    # (WiQ is Woolworths Australia — keep it relevant)
    mask_au = (
        df["countries_en"].isna()
        | df["countries_en"].str.contains("Australia", na=False, case=False)
    )
    df = df[mask_au].reset_index(drop=True)

    logger.info(f"  After filtering: {len(df):,} products")
    df.to_parquet(OFF_PRODUCTS_PATH, index=False)
    logger.info("  ✓ Open Food Facts → Parquet complete")


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
