"""
XGBoost demand forecast for M5, with MLflow tracking and a WRMSSE evaluator.

Pipeline
--------
1. build_features (src.models.features) on the modelling grid
2. time-based split: train on day_num < 1914, validate on 1914..1941 (28 days)
3. XGBoost (tweedie) with early stopping on the validation window
4. predict the 28-day window, clip negatives to zero
5. score with WRMSSE across the 12 M5 aggregation levels
6. log params, metrics and the model to MLflow

xgboost and mlflow are imported lazily inside the functions that need them, so the
WRMSSE evaluator and the split can be used and tested without them installed.

CLI
---
    python -m src.models.forecast --store CA_1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

VALID_START = 1914          # first day of the 28-day validation horizon
VALID_END = 1941            # last day with known actuals (sales_train_evaluation)

# The 12 M5 aggregation levels (keys to group the bottom series by). [] == grand total.
LEVELS: list[list[str]] = [
    [], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"],
    ["state_id", "cat_id"], ["state_id", "dept_id"],
    ["store_id", "cat_id"], ["store_id", "dept_id"],
    ["item_id"], ["item_id", "state_id"], ["item_id", "store_id"],
]

DEFAULT_PARAMS = dict(
    objective="reg:tweedie",
    tweedie_variance_power=1.1,     # tuned for intermittent retail demand
    tree_method="hist",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    random_state=42,
)

logger = logging.getLogger(__name__)


# ── Split ─────────────────────────────────────────────────────────────────────
def train_valid_split(feats: pd.DataFrame, valid_start: int = VALID_START,
                      valid_end: int = VALID_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by day_num into training (before the horizon) and the 28-day validation window."""
    train = feats[feats["day_num"] < valid_start].copy()
    valid = feats[(feats["day_num"] >= valid_start) & (feats["day_num"] <= valid_end)].copy()
    return train, valid


# ── Model ─────────────────────────────────────────────────────────────────────
def train_model(train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str],
                params: dict | None = None, early_stopping_rounds: int = 50):
    """Fit an XGBoost regressor with early stopping on the validation window."""
    import xgboost as xgb
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = xgb.XGBRegressor(enable_categorical=True, eval_metric="rmse",
                             early_stopping_rounds=early_stopping_rounds, **p)
    model.fit(train[feature_cols], train["sales"],
              eval_set=[(valid[feature_cols], valid["sales"])], verbose=False)
    return model


def predict(model, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Predict and clip negative demand to zero."""
    return np.clip(model.predict(frame[feature_cols]), 0, None)


# ── WRMSSE ────────────────────────────────────────────────────────────────────
def _level_score(train: pd.DataFrame, valid_av: pd.DataFrame, keys: list[str],
                 w_window: pd.DataFrame, total_dollars: float) -> float:
    """Weighted RMSSE for one aggregation level."""
    gk = keys if keys else ["_all_"]

    # Scale: mean squared day-to-day change of the aggregated TRAINING series.
    tr = (train.groupby(gk + ["day_num"], observed=True)["sales"].sum()
                .reset_index().sort_values(gk + ["day_num"]))
    tr["diff2"] = tr.groupby(gk, observed=True)["sales"].diff() ** 2
    scale = tr.groupby(gk, observed=True)["diff2"].mean()      # mean ignores the leading NaN

    # Numerator: mean squared error of the aggregated forecast over the 28-day horizon.
    a = valid_av.groupby(gk + ["day_num"], observed=True)["sales"].sum().rename("a")
    p = valid_av.groupby(gk + ["day_num"], observed=True)["pred"].sum().rename("p")
    ap = pd.concat([a, p], axis=1).fillna(0.0)
    ap["e2"] = (ap["a"] - ap["p"]) ** 2
    num = ap.groupby(level=list(range(len(gk))), observed=True)["e2"].mean()

    # Weight: share of dollar sales over the last 28 training days.
    weight = w_window.groupby(gk, observed=True)["dollars"].sum() / total_dollars

    rmsse = np.sqrt(num / scale)
    out = pd.concat([rmsse.rename("rmsse"), weight.rename("w")], axis=1)
    out = out[np.isfinite(out["rmsse"]) & (out["w"] > 0)]
    if out.empty or out["w"].sum() == 0:
        return float("nan")
    w_norm = out["w"] / out["w"].sum()                         # weights sum to 1 within the level
    return float((w_norm * out["rmsse"]).sum())


def wrmsse(train: pd.DataFrame, valid: pd.DataFrame, pred) -> dict:
    """
    WRMSSE over the 12 M5 aggregation levels.

    train / valid are grid rows (need id columns, day_num, sales, sell_price);
    pred is aligned to valid's rows. Each level contributes equally (the 12-level
    mean); within a level, series are weighted by their last-28-day dollar sales.
    """
    train = train.copy()
    valid = valid.copy()
    valid["pred"] = np.asarray(pred)
    train["_all_"] = 0
    valid["_all_"] = 0

    w_window = train[train["day_num"] > train["day_num"].max() - 28].copy()
    w_window["dollars"] = w_window["sales"] * w_window["sell_price"].fillna(0.0)
    w_window["_all_"] = 0
    total_dollars = float(w_window["dollars"].sum())

    level_scores = {}
    for keys in LEVELS:
        name = "total" if not keys else "_".join(keys)
        level_scores[name] = _level_score(train, valid, keys, w_window, total_dollars)

    vals = [v for v in level_scores.values() if np.isfinite(v)]
    return {"wrmsse": float(np.mean(vals)) if vals else float("nan"),
            "levels": level_scores}


# ── Orchestration (MLflow) ────────────────────────────────────────────────────
def run(grid_path: str | Path, store: str | None = None, params: dict | None = None,
        run_name: str = "xgb-tweedie-v1"):
    """Full pipeline wrapped in an MLflow run. Returns (model, scores)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import mlflow
    import mlflow.xgboost
    from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODELS_DIR
    from src.models.features import build_features

    df = pd.read_parquet(grid_path)
    if store:
        df = df[df["store_id"] == store].copy()

    feats, feature_cols = build_features(df)
    train, valid = train_valid_split(feats)
    logger.info(f"train={len(train):,} rows  valid={len(valid):,} rows  features={len(feature_cols)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        model = train_model(train, valid, feature_cols, params)
        preds = predict(model, valid, feature_cols)

        scores = wrmsse(train, valid, preds)
        rmse = float(np.sqrt(np.mean((valid["sales"].to_numpy() - preds) ** 2)))

        mlflow.log_params({**DEFAULT_PARAMS, **(params or {}),
                           "store": store or "all", "n_features": len(feature_cols)})
        mlflow.log_metric("wrmsse", scores["wrmsse"])
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, "model")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"xgb_forecast_{store or 'all'}.json"
        model.save_model(model_path)
        mlflow.log_artifact(str(model_path))

        logger.info(f"WRMSSE={scores['wrmsse']:.4f}  RMSE={rmse:.4f}  -> model saved to {model_path}")
    return model, scores


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import M5_GRID_PATH

    ap = argparse.ArgumentParser(description="Train the M5 XGBoost demand forecast")
    ap.add_argument("--grid", default=str(M5_GRID_PATH))
    ap.add_argument("--store", default=None, help="Limit to one store_id (e.g. CA_1)")
    ap.add_argument("--run-name", default="xgb-tweedie-v1")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    run(args.grid, store=args.store, run_name=args.run_name)


if __name__ == "__main__":
    main()
