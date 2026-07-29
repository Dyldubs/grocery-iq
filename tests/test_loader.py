"""
Tests for src/data/loader.py

These tests verify that the loader raises helpful errors when data hasn't
been downloaded yet, and that the SQL query helper works correctly.

We use pytest's tmp_path fixture to create throwaway Parquet files so we
can test the happy path without needing the real 200 MB Instacart download.
"""

import pandas as pd
import pytest


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_parquet(path, df: pd.DataFrame) -> None:
    """Write a small DataFrame to a Parquet file at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ── Error handling ────────────────────────────────────────────────────────────

def test_load_orders_raises_when_file_missing(tmp_path, monkeypatch):
    """
    If the data hasn't been downloaded yet, load_orders() should raise a
    FileNotFoundError with a helpful message telling the user what to run.

    We use monkeypatch to temporarily redirect ORDERS_PATH to a path inside
    tmp_path (a throwaway directory pytest creates per test) so this test
    doesn't depend on whether you've actually run the download script.
    """
    import src.config as cfg
    monkeypatch.setattr(cfg, "ORDERS_PATH", tmp_path / "nonexistent.parquet")

    # Clear lru_cache so the patched path is used
    from src.data.loader import load_orders
    load_orders.cache_clear()

    import src.data.loader as loader
    monkeypatch.setattr(loader, "ORDERS_PATH", tmp_path / "nonexistent.parquet")

    with pytest.raises(FileNotFoundError, match="preprocess_data"):
        loader.load_orders()

    load_orders.cache_clear()   # clean up cache after test


def test_load_products_raises_when_file_missing(tmp_path, monkeypatch):
    """Same check for the products table."""
    import src.data.loader as loader
    monkeypatch.setattr(loader, "PRODUCTS_PATH", tmp_path / "nonexistent.parquet")

    from src.data.loader import load_products
    load_products.cache_clear()

    with pytest.raises(FileNotFoundError):
        loader.load_products()

    load_products.cache_clear()


# ── Happy path (with synthetic Parquet files) ─────────────────────────────────

def test_load_orders_returns_dataframe(tmp_path, monkeypatch):
    """
    When the Parquet file exists, load_orders() should return a DataFrame
    with the expected columns.

    We create a tiny synthetic Parquet file (5 rows) to prove the loader
    works correctly without needing the full 3.4M row download.
    """
    import src.data.loader as loader
    from src.data.loader import load_orders

    fake_orders = pd.DataFrame({
        "order_id":           [1, 2, 3, 4, 5],
        "user_id":            [10, 10, 20, 30, 30],
        "order_dow":          [0, 1, 2, 3, 4],
        "order_hour_of_day":  [8, 9, 10, 11, 12],
        "days_since_prior_order": [None, 7, 14, 21, 7],
        "eval_set":           ["prior"] * 5,
    })

    parquet_path = tmp_path / "orders.parquet"
    _make_parquet(parquet_path, fake_orders)

    load_orders.cache_clear()
    monkeypatch.setattr(loader, "ORDERS_PATH", parquet_path)

    df = loader.load_orders()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "order_id" in df.columns
    assert "user_id" in df.columns

    load_orders.cache_clear()


def test_query_runs_sql_against_parquet(tmp_path):
    """
    The query() helper should be able to run SQL directly against a Parquet
    file using DuckDB — this is the SQL analytics layer of the project.

    This tests one of the most important concepts in the project:
    DuckDB can treat a Parquet file like a database table in a SQL query,
    with no database server needed.
    """
    from src.data.loader import query

    # Create a small parquet file to query
    df = pd.DataFrame({
        "department": ["produce", "produce", "dairy", "bakery", "bakery", "bakery"],
        "n_sales":    [100, 200, 150, 50, 60, 70],
    })
    parquet_path = tmp_path / "test_data.parquet"
    _make_parquet(parquet_path, df)

    # Run a SQL aggregation directly against the parquet file
    result = query(f"""
        SELECT department, SUM(n_sales) as total_sales
        FROM '{parquet_path}'
        GROUP BY department
        ORDER BY total_sales DESC
    """)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3                        # 3 unique departments
    assert result.iloc[0]["department"] == "produce"   # highest sales first
    assert result.iloc[0]["total_sales"] == 300
