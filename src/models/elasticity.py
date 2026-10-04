"""
Price elasticity of demand for M5, estimated per category with log-log OLS.

Own-price elasticity
--------------------
On an item-store-week panel (units vs weekly price), we fit, per category, a
within-item fixed-effects log-log regression: regress demeaned ln(units) on
demeaned ln(price). Demeaning by item-store removes every fixed cross-sectional
difference, so the slope is the *within* price response, i.e. the own-price
elasticity. Standard errors are clustered by item-store. A well-behaved estimate
is negative (demand falls as price rises).

Cross-price elasticities
------------------------
At the category grain, we build a weekly panel of each category's total units
and average price, then regress ln(units) for each category on the ln(price) of
every category. The own term is the diagonal, the cross terms are off-diagonal.

statsmodels / mlflow are imported lazily so the panel builder can be tested
without them.

CLI
---
    python -m src.models.elasticity --store CA_1   # dev subset
    python -m src.models.elasticity                 # all stores
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Panel (DuckDB) ────────────────────────────────────────────────────────────
def build_panel(grid_path: str, store: str | None = None) -> pd.DataFrame:
    """Item-store-week panel of units and price from the M5 grid."""
    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    where = "WHERE sell_price IS NOT NULL" + (f" AND store_id = '{store}'" if store else "")
    sql = f"""
        SELECT item_id, store_id, cat_id, wm_yr_wk,
               sum(sales)      AS units,
               avg(sell_price) AS price
        FROM '{grid_path}'
        {where}
        GROUP BY item_id, store_id, cat_id, wm_yr_wk
        HAVING sum(sales) > 0
    """
    df = con.execute(sql).df()
    con.close()
    df["entity"] = df["item_id"].astype(str) + "@" + df["store_id"].astype(str)
    df["ln_units"] = np.log(df["units"])
    df["ln_price"] = np.log(df["price"])
    return df


# ── Own-price elasticity (within-item FE) ─────────────────────────────────────
def own_price_elasticity(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-category own-price elasticity via a within-item fixed-effects log-log OLS."""
    import statsmodels.api as sm
    rows = []
    for cat, sub in panel.groupby("cat_id", observed=True):
        sub = sub.copy()
        # within (fixed-effects) transform: subtract item-store means
        sub["lu"] = sub["ln_units"] - sub.groupby("entity")["ln_units"].transform("mean")
        sub["lp"] = sub["ln_price"] - sub.groupby("entity")["ln_price"].transform("mean")
        # keep only entities whose price actually moves (others carry no elasticity info)
        price_var = sub.groupby("entity")["ln_price"].transform("std").fillna(0)
        sub = sub[price_var > 0]
        if len(sub) < 100:
            continue
        model = sm.OLS(sub["lu"], sm.add_constant(sub["lp"])).fit(
            cov_type="cluster", cov_kwds={"groups": sub["entity"]})
        rows.append({
            "cat_id": cat,
            "own_elasticity": round(float(model.params["lp"]), 4),
            "std_err": round(float(model.bse["lp"]), 4),
            "t_stat": round(float(model.tvalues["lp"]), 2),
            "n_obs": int(len(sub)),
        })
    return pd.DataFrame(rows)


# ── Cross-price elasticities (category-level system) ──────────────────────────
def cross_price_elasticities(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Category x category elasticity matrix: ln(units) of each category regressed
    on ln(price) of every category, on a weekly panel. Diagonal = own, off = cross.
    """
    import statsmodels.api as sm
    cw = (panel.groupby(["cat_id", "wm_yr_wk"], observed=True)
                .agg(units=("units", "sum"), price=("price", "mean")).reset_index())
    ln_u = np.log(cw.pivot(index="wm_yr_wk", columns="cat_id", values="units"))
    ln_p = np.log(cw.pivot(index="wm_yr_wk", columns="cat_id", values="price"))
    cats = list(ln_p.columns)

    matrix = {}
    for demand_cat in cats:
        data = pd.concat([ln_u[demand_cat].rename("y"), ln_p], axis=1).dropna()
        model = sm.OLS(data["y"], sm.add_constant(data[cats])).fit()
        matrix[demand_cat] = {f"price_{c}": round(float(model.params[c]), 4) for c in cats}
    # rows = demand category, columns = price category
    return pd.DataFrame(matrix).T


# ── Orchestration (MLflow) ────────────────────────────────────────────────────
def run(grid_path: str | None = None, store: str | None = None,
        run_name: str = "elasticity-ols-v1"):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import mlflow
    from src.config import M5_GRID_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODELS_DIR

    grid_path = grid_path or str(M5_GRID_PATH)
    panel = build_panel(grid_path, store)
    logger.info(f"panel rows={len(panel):,}  entities={panel['entity'].nunique():,}  "
                f"categories={panel['cat_id'].nunique()}")

    own = own_price_elasticity(panel)
    cross = cross_price_elasticities(panel)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"store": store or "all", "model": "loglog_ols_within_fe",
                           "panel_rows": len(panel)})
        for _, r in own.iterrows():
            mlflow.log_metric(f"own_elasticity_{r['cat_id']}", r["own_elasticity"])

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        own_path = MODELS_DIR / "elasticity_own.csv"
        cross_path = MODELS_DIR / "elasticity_cross.csv"
        own.to_csv(own_path, index=False)
        cross.to_csv(cross_path)
        mlflow.log_artifact(str(own_path))
        mlflow.log_artifact(str(cross_path))

        logger.info("own-price elasticity:\n" + own.to_string(index=False))
        logger.info("cross-price matrix (row=demand, col=price):\n" + cross.to_string())
    return own, cross


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate M5 price elasticities (log-log OLS)")
    ap.add_argument("--grid", default=None, help="Grid Parquet (defaults to M5_GRID_PATH)")
    ap.add_argument("--store", default=None, help="Limit to one store_id (dev)")
    ap.add_argument("--run-name", default="elasticity-ols-v1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    run(grid_path=args.grid, store=args.store, run_name=args.run_name)


if __name__ == "__main__":
    main()
