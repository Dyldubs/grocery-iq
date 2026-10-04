"""
Feature engineering for the M5 demand forecast.

Turns the modelling grid (one row per series-day, from scripts/build_m5_grid.py)
into a tabular feature matrix for XGBoost. Every sales-derived feature looks only
backwards by at least the 28-day forecast horizon, so a single model can predict
the whole horizon with no leakage and no recursion.

Usage (as a library)
---------------------
    import pandas as pd
    from src.config import M5_GRID_PATH
    from src.models.features import build_features

    grid = pd.read_parquet(M5_GRID_PATH)
    feats, feature_cols = build_features(grid)

Usage (CLI, writes a features Parquet)
--------------------------------------
    python -m src.models.features --store CA_1 --out data/processed/m5_features_CA1.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config knobs ──────────────────────────────────────────────────────────────
HORIZON = 28                                  # forecast 28 days ahead
LAGS = [28, 29, 30, 35, 42, 49, 56]           # all >= HORIZON (no recursion needed)
ROLL_WINDOWS = [7, 28, 90]
CATEGORICALS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
TARGET = "sales"

logger = logging.getLogger(__name__)


# ── Sales lags ────────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame, lags: list[int] = LAGS) -> list[str]:
    """Per-series sales lags. Requires df sorted by ['id', 'day_num']."""
    g = df.groupby("id", observed=True)[TARGET]
    cols = []
    for lag in lags:
        name = f"sales_lag_{lag}"
        df[name] = g.shift(lag).astype("float32")
        cols.append(name)
    return cols


# ── Rolling sales stats ───────────────────────────────────────────────────────
def add_rolling_features(df: pd.DataFrame, windows: list[int] = ROLL_WINDOWS,
                         shift: int = HORIZON) -> list[str]:
    """
    Rolling mean/std of sales over the series shifted back by `shift` days
    (default 28), so nothing inside the forecast horizon can leak in.
    """
    base = df.groupby("id", observed=True)[TARGET].shift(shift)
    by_id = base.groupby(df["id"], observed=True)
    cols = []
    for w in windows:
        mean_name, std_name = f"sales_rmean_{w}", f"sales_rstd_{w}"
        mp = max(2, w // 4)
        df[mean_name] = (by_id.rolling(w, min_periods=mp).mean()
                              .reset_index(level=0, drop=True).astype("float32"))
        df[std_name] = (by_id.rolling(w, min_periods=mp).std()
                             .reset_index(level=0, drop=True).astype("float32"))
        cols += [mean_name, std_name]
    return cols


# ── Calendar, events, SNAP ────────────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> list[str]:
    """Calendar/event/SNAP features. All are known in advance, so no shifting."""
    d = pd.to_datetime(df["date"])
    df["dow"] = d.dt.dayofweek.astype("int8")
    df["dom"] = d.dt.day.astype("int8")
    df["week"] = d.dt.isocalendar().week.astype("int16")
    df["month_f"] = d.dt.month.astype("int8")
    df["year_f"] = d.dt.year.astype("int16")

    df["is_event"] = df["event_name_1"].notna().astype("int8")

    # SNAP flag for the series' own state (CA/TX/WI)
    df["snap"] = np.select(
        [df["state_id"] == "CA", df["state_id"] == "TX", df["state_id"] == "WI"],
        [df["snap_CA"], df["snap_TX"], df["snap_WI"]],
        default=0,
    ).astype("int8")

    # Event name/type as integer codes (-1 = none)
    event_cols = []
    for c in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        code = f"{c}_code"
        df[code] = df[c].astype("category").cat.codes.astype("int16")
        event_cols.append(code)

    return ["dow", "dom", "week", "month_f", "year_f", "is_event", "snap"] + event_cols


# ── Price ─────────────────────────────────────────────────────────────────────
def add_price_features(df: pd.DataFrame) -> list[str]:
    """Price level, week-on-week change, and price vs. the item's own history."""
    g = df.groupby("id", observed=True)["sell_price"]

    # Weekly change: price is constant within a week, so a 7-day shift approximates it.
    prev_wk = g.shift(7)
    df["price_chg_wk"] = (df["sell_price"] / prev_wk - 1).astype("float32")

    # Price relative to the item's expanding mean up to the previous day (no leakage).
    # NOTE: this lambda transform is the slow step on the full grid. For all stores,
    # consider a cheaper cumulative mean; it is fine per-store.
    exp_mean = g.transform(lambda s: s.shift(1).expanding().mean())
    df["price_rel_item"] = (df["sell_price"] / exp_mean).astype("float32")

    df["sell_price"] = df["sell_price"].astype("float32")
    return ["sell_price", "price_chg_wk", "price_rel_item"]


# ── Categoricals ──────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> list[str]:
    """Cast id columns to pandas 'category' for XGBoost native categorical support
    (train with enable_categorical=True)."""
    for c in CATEGORICALS:
        df[c] = df[c].astype("category")
    return list(CATEGORICALS)


# ── Orchestrator ──────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, lags: list[int] = LAGS,
                   windows: list[int] = ROLL_WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    """
    Build all features and return (dataframe, feature_column_names).

    Keeps id/day_num/date for the time-based train/validation split and `sales`
    as the target. Rows early in each series will have NaN lags/rollings; XGBoost
    handles NaN natively, so they are left in by default.
    """
    df = df.sort_values(["id", "day_num"]).reset_index(drop=True)
    feature_cols: list[str] = []
    feature_cols += add_lag_features(df, lags)
    feature_cols += add_rolling_features(df, windows)
    feature_cols += add_calendar_features(df)
    feature_cols += add_price_features(df)
    feature_cols += encode_categoricals(df)
    # TODO (extend): weeks-since-last-price-change, rolling max, lag of same weekday,
    # item/store mean-encoding. Add here and append the new names to feature_cols.
    return df, feature_cols


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import M5_GRID_PATH

    ap = argparse.ArgumentParser(description="Build M5 features from the modelling grid")
    ap.add_argument("--grid", default=str(M5_GRID_PATH), help="Input grid Parquet")
    ap.add_argument("--store", default=None, help="Limit to one store_id (e.g. CA_1)")
    ap.add_argument("--out", default=None, help="Optional output Parquet path")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    df = pd.read_parquet(args.grid)
    if args.store:
        df = df[df["store_id"] == args.store].copy()

    feats, cols = build_features(df)
    logger.info(f"features: {len(feats):,} rows x {len(cols)} feature columns")
    logger.info(f"feature columns: {cols}")

    if args.out:
        feats.to_parquet(args.out, index=False, compression="zstd")
        logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
