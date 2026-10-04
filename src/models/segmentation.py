"""
Customer segmentation for Instacart (RFM-style KMeans).

Instacart has no prices, so this is an RFM-style segmentation adapted to the
data: Recency and Frequency come straight from order cadence, and the Monetary
axis is proxied by basket size (items per order), with reorder ratio and product
breadth added as behavioural signals. Users are clustered with KMeans on
standard-scaled features; quality is reported with the silhouette score.

Features are aggregated in DuckDB (low memory). sklearn / mlflow / joblib are
lazy-imported so the feature builder can be tested without them.

CLI
---
    python -m src.models.segmentation --sample-users 20000 --k 4
    python -m src.models.segmentation                          # all users
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "u_orders",          # Frequency: number of prior orders
    "recency_days",      # Recency: days since prior order on the latest order
    "avg_days_between",  # cadence
    "avg_basket",        # Monetary proxy: items per order
    "reorder_ratio",     # loyalty
    "distinct_products", # breadth
]

logger = logging.getLogger(__name__)


# ── User features (DuckDB) ────────────────────────────────────────────────────
def build_user_features(orders_path: str, order_items_path: str,
                        sample_users: int | None = None) -> pd.DataFrame:
    """One row per user with RFM-style behavioural features from prior orders."""
    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")

    sample = f"ORDER BY hash(user_id) LIMIT {int(sample_users)}" if sample_users else ""

    sql = f"""
    WITH orders AS (SELECT * FROM '{orders_path}'),
         items  AS (SELECT * FROM '{order_items_path}'),
         users  AS (SELECT user_id FROM (SELECT DISTINCT user_id FROM orders) {sample}),
         prior_orders AS (
            SELECT * FROM orders
            WHERE eval_set = 'prior' AND user_id IN (SELECT user_id FROM users)
         ),
         o AS (
            SELECT user_id,
                   max(order_number)                          AS u_orders,
                   avg(days_since_prior_order)                AS avg_days_between,
                   arg_max(days_since_prior_order, order_number) AS recency_days
            FROM prior_orders GROUP BY user_id
         ),
         oi AS (
            SELECT po.user_id, i.product_id, i.reordered
            FROM items i JOIN prior_orders po USING (order_id)
         ),
         it AS (
            SELECT user_id,
                   count(*)                 AS total_items,
                   avg(reordered)           AS reorder_ratio,
                   count(DISTINCT product_id) AS distinct_products
            FROM oi GROUP BY user_id
         )
    SELECT o.user_id, o.u_orders, o.recency_days, o.avg_days_between,
           it.total_items::DOUBLE / o.u_orders AS avg_basket,
           it.reorder_ratio, it.distinct_products
    FROM o JOIN it USING (user_id)
    """
    df = con.execute(sql).df()
    con.close()
    return df


# ── Clustering ────────────────────────────────────────────────────────────────
def fit_segments(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS,
                 k: int = 4, random_state: int = 42):
    """Standard-scale, KMeans, and score with silhouette (sampled for speed)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    work = df.dropna(subset=feature_cols).copy()
    if len(work) < len(df):
        logger.info(f"dropped {len(df) - len(work):,} users with missing features")

    scaler = StandardScaler().fit(work[feature_cols])
    X = scaler.transform(work[feature_cols])

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    work["segment"] = km.fit_predict(X)

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(work), size=min(len(work), 10000), replace=False)
    sil = float(silhouette_score(X[idx], work["segment"].to_numpy()[idx]))

    metrics = {"k": k, "silhouette": sil, "inertia": float(km.inertia_)}
    return scaler, km, work, metrics


def profile_segments(work: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """Mean feature values and size per segment, for interpretation/labelling."""
    prof = work.groupby("segment")[feature_cols].mean().round(2)
    prof["n_users"] = work.groupby("segment").size()
    prof["pct"] = (prof["n_users"] / len(work) * 100).round(1)
    return prof.sort_values("u_orders", ascending=False)


# ── Orchestration (MLflow) ────────────────────────────────────────────────────
def run(sample_users: int | None = None, k: int = 4,
        run_name: str = "segmentation-kmeans-v1"):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import mlflow
    import joblib
    from src.config import (ORDERS_PATH, ORDER_ITEMS_PATH,
                            MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODELS_DIR)

    df = build_user_features(str(ORDERS_PATH), str(ORDER_ITEMS_PATH), sample_users)
    logger.info(f"users={len(df):,}")
    scaler, km, work, metrics = fit_segments(df, k=k)
    prof = profile_segments(work)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"k": k, "sample_users": sample_users or "all",
                           "n_users": len(work), "features": ",".join(FEATURE_COLS)})
        mlflow.log_metrics({"silhouette": metrics["silhouette"], "inertia": metrics["inertia"]})

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "segmentation_kmeans.joblib"
        joblib.dump({"scaler": scaler, "kmeans": km, "features": FEATURE_COLS}, model_path)
        prof_path = MODELS_DIR / "segment_profiles.csv"
        prof.to_csv(prof_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(prof_path))

        logger.info(f"k={k}  silhouette={metrics['silhouette']:.3f}  inertia={metrics['inertia']:.0f}")
        logger.info("segment profile:\n" + prof.to_string())
    return km, work, prof


def main() -> None:
    ap = argparse.ArgumentParser(description="Instacart customer segmentation (RFM KMeans)")
    ap.add_argument("--sample-users", type=int, default=None,
                    help="Cluster a deterministic subsample of N users (dev)")
    ap.add_argument("--k", type=int, default=4, help="Number of segments")
    ap.add_argument("--run-name", default="segmentation-kmeans-v1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    run(sample_users=args.sample_users, k=args.k, run_name=args.run_name)


if __name__ == "__main__":
    main()
