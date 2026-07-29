"""
Centralised data loader for GroceryIQ.

Every other module imports from here rather than reading files directly.
This means if we ever change a file path or format, we only update one place.

Usage
-----
    from src.data.loader import load_orders, load_products, load_order_items

    orders = load_orders()         # returns a pandas DataFrame
    products = load_products()
"""

from __future__ import annotations

import logging
from functools import lru_cache

import duckdb
import pandas as pd

from src.config import (
    ORDERS_PATH, PRODUCTS_PATH, ORDER_ITEMS_PATH,
    M5_SALES_PATH, OFF_PRODUCTS_PATH,
)

logger = logging.getLogger(__name__)


def _check_exists(path) -> None:
    """Raise a helpful error if a data file hasn't been generated yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run: python scripts/download_data.py && python scripts/preprocess_data.py"
        )


# ── Instacart loaders ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_orders() -> pd.DataFrame:
    """
    Load the orders table.
    lru_cache means the DataFrame is only read from disk once per session —
    subsequent calls return the cached copy instantly.
    """
    _check_exists(ORDERS_PATH)
    df = pd.read_parquet(ORDERS_PATH)
    logger.info(f"Loaded orders: {len(df):,} rows")
    return df


@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    """Load products joined with aisle and department names."""
    _check_exists(PRODUCTS_PATH)
    df = pd.read_parquet(PRODUCTS_PATH)
    logger.info(f"Loaded products: {len(df):,} rows")
    return df


@lru_cache(maxsize=1)
def load_order_items() -> pd.DataFrame:
    """Load individual product lines within each order."""
    _check_exists(ORDER_ITEMS_PATH)
    df = pd.read_parquet(ORDER_ITEMS_PATH)
    logger.info(f"Loaded order items: {len(df):,} rows")
    return df


# ── M5 loader ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_m5_sales(
    category: str | None = None,
    store_id: str | None = None,
) -> pd.DataFrame:
    """
    Load M5 sales data, optionally filtered by product category or store.

    Because M5 has millions of rows, filtering here avoids loading
    data you don't need into memory.
    """
    _check_exists(M5_SALES_PATH)

    # DuckDB lets us write SQL directly against a Parquet file on disk.
    # It's much faster than reading the whole file and filtering in pandas.
    query = f"SELECT * FROM '{M5_SALES_PATH}' WHERE 1=1"
    if category:
        query += f" AND cat_id = '{category}'"
    if store_id:
        query += f" AND store_id = '{store_id}'"

    df = duckdb.query(query).df()
    logger.info(f"Loaded M5 sales: {len(df):,} rows")
    return df


# ── Open Food Facts loader ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_off_products(min_name_length: int = 3) -> pd.DataFrame:
    """Load Open Food Facts products for the RAG knowledge base."""
    _check_exists(OFF_PRODUCTS_PATH)
    df = pd.read_parquet(OFF_PRODUCTS_PATH)
    # Drop rows where the product name is too short to be useful
    df = df[df["product_name"].str.len() >= min_name_length]
    logger.info(f"Loaded OFF products: {len(df):,} rows")
    return df


# ── SQL analytics layer (DuckDB) ──────────────────────────────────────────────

def query(sql: str) -> pd.DataFrame:
    """
    Run arbitrary SQL against the processed Parquet files.

    DuckDB can query Parquet files directly using their file paths.
    Just reference them by their config variable names in the FROM clause.

    Example
    -------
        from src.data.loader import query
        from src.config import ORDERS_PATH, ORDER_ITEMS_PATH, PRODUCTS_PATH

        result = query(f'''
            SELECT p.department, COUNT(*) as n_items
            FROM '{ORDER_ITEMS_PATH}' oi
            JOIN '{PRODUCTS_PATH}' p USING (product_id)
            GROUP BY 1 ORDER BY 2 DESC
        ''')
    """
    return duckdb.query(sql).df()
