# GroceryIQ model card

Versioned record of the Milestone 1 models. Each is saved to `models/`, tracked
in MLflow (experiment `grocery-iq`), and reproducible from the command shown.
Large model binaries (`*.json`, `*.joblib`) are gitignored and rebuilt on demand;
the small result files (`*.csv`, the PR curve `*.png`) and this card are tracked.

Last updated: 2026-10-04

## Demand forecast (XGBoost, tweedie)
- Artefact: `models/xgb_forecast_CA_1.json` (gitignored, ~77 MB)
- Reproduce: `python -m src.models.forecast --store CA_1`
- Scope: M5, store CA_1, 3,049 series, 28-day validation (days 1914 to 1941)
- Metrics: WRMSSE 0.5441, RMSE 2.2443 (naive baseline about 0.83)
- Features: 32 (sales lags >= 28, rolling stats, calendar/event/SNAP, price)

## Reorder classifier (XGBoost)
- Artefacts: `models/reorder_xgb.json` (gitignored), `models/reorder_pr_curve.png`
- Reproduce: `python -m src.models.reorder`
- Scope: Instacart, all train users, 8.47M candidate (user, product) rows
- Metrics: ROC AUC 0.8302, PR AUC 0.4060 (base reorder rate 9.7%)
- Features: 16 at user, product, and user-product grains

## Customer segmentation (KMeans, RFM-style)
- Artefacts: `models/segmentation_kmeans.joblib` (scaler + kmeans + feature list), `models/segment_profiles.csv`
- Reproduce: `python -m src.models.segmentation --k 4`
- Scope: Instacart, 206,209 users, k=4, silhouette 0.261
- Segments: loyal champions (14%), big-basket stock-up (17%), mainstream regulars (36%), at-risk/lapsing (33%)

## Price elasticity (log-log OLS)
- Artefacts: `models/elasticity_own.csv`, `models/elasticity_cross.csv`
- Reproduce: `python -m src.models.elasticity`
- Scope: M5, all stores, 5.11M item-store-week panel rows
- Own-price elasticity (within-item FE): FOODS -0.72, HOBBIES -0.62, HOUSEHOLD -0.46 (all significant)
- Cross-price matrix: exploratory only; see the limitation note in `src/models/elasticity.py`
