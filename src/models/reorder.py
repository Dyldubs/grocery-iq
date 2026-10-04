"""
Reorder classifier for Instacart.

Predicts whether a user will reorder a product they have bought before, in their
next order. Candidate rows are every (user, product) the user purchased in their
PRIOR orders; the label is 1 if that product appears in the user's TRAIN order.

Features are aggregated in DuckDB (streaming, low memory) at three grains:
user, product, and user-product. The classifier is XGBoost, evaluated with ROC
AUC and PR AUC (average precision), and tracked in MLflow.

xgboost / scikit-learn / mlflow / matplotlib are imported lazily, so the feature
builder can be used and tested without them installed.

CLI
---
    python -m src.models.reorder --sample-users 20000   # dev subsample (by user)
    python -m src.models.reorder                          # all train users
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "reordered_next"

FEATURE_COLS = [
    # user-product grain
    "up_orders", "up_reorders", "up_avg_cart_pos", "up_orders_since_last",
    "up_order_rate", "up_reorder_rate",
    # user grain
    "u_orders", "u_items", "u_reorder_ratio", "u_distinct_products",
    "u_avg_days", "u_avg_basket",
    # product grain
    "p_orders", "p_reorder_rate", "p_users", "p_avg_cart_pos",
]

logger = logging.getLogger(__name__)


# ── Feature/label table (DuckDB) ──────────────────────────────────────────────
def build_training_table(orders_path: str, order_items_path: str,
                         sample_users: int | None = None) -> pd.DataFrame:
    """
    One row per (user, product) candidate from the user's prior orders, with
    user/product/user-product features and the next-order reorder label.
    Only users who have a 'train' order are included (test users are excluded).
    """
    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")

    sample = f"ORDER BY hash(user_id) LIMIT {int(sample_users)}" if sample_users else ""

    sql = f"""
    WITH orders AS (SELECT * FROM '{orders_path}'),
         items  AS (SELECT * FROM '{order_items_path}'),
         oi AS (
            SELECT i.order_id, i.product_id, i.add_to_cart_order, i.reordered,
                   o.user_id, o.order_number, o.eval_set, o.days_since_prior_order
            FROM items i JOIN orders o USING (order_id)
         ),
         train_users AS (
            SELECT user_id FROM (
                SELECT DISTINCT user_id FROM orders WHERE eval_set = 'train'
            ) {sample}
         ),
         prior AS (
            SELECT * FROM oi
            WHERE eval_set = 'prior' AND user_id IN (SELECT user_id FROM train_users)
         ),
         uf AS (
            SELECT user_id,
                   max(order_number)          AS u_orders,
                   count(*)                   AS u_items,
                   avg(reordered)             AS u_reorder_ratio,
                   count(DISTINCT product_id) AS u_distinct_products,
                   avg(days_since_prior_order) AS u_avg_days
            FROM prior GROUP BY user_id
         ),
         pf AS (
            SELECT product_id,
                   count(*)                 AS p_orders,
                   avg(reordered)           AS p_reorder_rate,
                   count(DISTINCT user_id)  AS p_users,
                   avg(add_to_cart_order)   AS p_avg_cart_pos
            FROM prior GROUP BY product_id
         ),
         upf AS (
            SELECT user_id, product_id,
                   count(*)               AS up_orders,
                   sum(reordered)         AS up_reorders,
                   avg(add_to_cart_order) AS up_avg_cart_pos,
                   max(order_number)      AS up_last_order
            FROM prior GROUP BY user_id, product_id
         ),
         train_items AS (
            SELECT DISTINCT user_id, product_id FROM oi WHERE eval_set = 'train'
         )
    SELECT
        c.user_id, c.product_id,
        c.up_orders, c.up_reorders, c.up_avg_cart_pos,
        (u.u_orders - c.up_last_order)              AS up_orders_since_last,
        c.up_orders::DOUBLE / u.u_orders            AS up_order_rate,
        c.up_reorders::DOUBLE / c.up_orders         AS up_reorder_rate,
        u.u_orders, u.u_items, u.u_reorder_ratio, u.u_distinct_products, u.u_avg_days,
        u.u_items::DOUBLE / u.u_orders              AS u_avg_basket,
        p.p_orders, p.p_reorder_rate, p.p_users, p.p_avg_cart_pos,
        CASE WHEN t.product_id IS NOT NULL THEN 1 ELSE 0 END AS {TARGET}
    FROM upf c
    JOIN uf u USING (user_id)
    JOIN pf p USING (product_id)
    LEFT JOIN train_items t ON t.user_id = c.user_id AND t.product_id = c.product_id
    """
    df = con.execute(sql).df()
    con.close()
    return df


# ── Model ─────────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS,
                params: dict | None = None, early_stopping_rounds: int = 50):
    """Train XGBoost with a user-grouped holdout so no user spans train and valid."""
    import xgboost as xgb
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(gss.split(df, groups=df["user_id"]))
    train, valid = df.iloc[tr_idx], df.iloc[va_idx]

    pos = max(train[TARGET].mean(), 1e-6)
    defaults = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(1 - pos) / pos,     # counter class imbalance
        random_state=42,
    )
    model = xgb.XGBClassifier(early_stopping_rounds=early_stopping_rounds,
                              **{**defaults, **(params or {})})
    model.fit(train[feature_cols], train[TARGET],
              eval_set=[(valid[feature_cols], valid[TARGET])], verbose=False)
    return model, valid


def evaluate(model, valid: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS,
             pr_curve_path: str | None = None) -> dict:
    """ROC AUC, PR AUC (average precision), and optionally a saved PR curve."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve)
    proba = model.predict_proba(valid[feature_cols])[:, 1]
    scores = {
        "roc_auc": float(roc_auc_score(valid[TARGET], proba)),
        "pr_auc": float(average_precision_score(valid[TARGET], proba)),
        "base_rate": float(valid[TARGET].mean()),
    }
    if pr_curve_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        prec, rec, _ = precision_recall_curve(valid[TARGET], proba)
        plt.figure(figsize=(5, 4))
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Reorder PR curve (AP={scores['pr_auc']:.3f})")
        plt.tight_layout(); plt.savefig(pr_curve_path, dpi=120); plt.close()
    return scores


# ── Orchestration (MLflow) ────────────────────────────────────────────────────
def run(sample_users: int | None = None, params: dict | None = None,
        run_name: str = "reorder-xgb-v1"):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import mlflow, mlflow.xgboost
    from src.config import (ORDERS_PATH, ORDER_ITEMS_PATH,
                            MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODELS_DIR)

    df = build_training_table(str(ORDERS_PATH), str(ORDER_ITEMS_PATH), sample_users)
    logger.info(f"candidates={len(df):,}  base reorder rate={df[TARGET].mean():.3f}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        model, valid = train_model(df, params=params)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pr_path = MODELS_DIR / "reorder_pr_curve.png"
        scores = evaluate(model, valid, pr_curve_path=str(pr_path))

        mlflow.log_params({"sample_users": sample_users or "all",
                           "n_features": len(FEATURE_COLS), "n_candidates": len(df)})
        mlflow.log_metrics(scores)
        model_path = MODELS_DIR / "reorder_xgb.json"
        model.save_model(model_path)
        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(pr_path))

        logger.info(f"ROC AUC={scores['roc_auc']:.4f}  PR AUC={scores['pr_auc']:.4f}  "
                    f"(base rate {scores['base_rate']:.3f})")
    return model, scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Instacart reorder classifier")
    ap.add_argument("--sample-users", type=int, default=None,
                    help="Train on a deterministic subsample of N users (dev)")
    ap.add_argument("--run-name", default="reorder-xgb-v1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    run(sample_users=args.sample_users, run_name=args.run_name)


if __name__ == "__main__":
    main()
