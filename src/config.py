"""
Central configuration for GroceryIQ.

All paths and constants live here so nothing is hard-coded elsewhere.
Import this module anywhere you need a path or setting:

    from src.config import DATA_DIR, QDRANT_HOST
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file so environment variables are available via os.getenv()
load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
# Path(__file__) is this file (src/config.py)
# .parent gives src/
# .parent.parent gives the project root grocery-iq/
ROOT_DIR = Path(__file__).parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Raw dataset subdirectories (created by download script)
INSTACART_DIR  = RAW_DIR / "instacart"
M5_DIR         = RAW_DIR / "m5"
OFF_DIR        = RAW_DIR / "open_food_facts"   # OFF = Open Food Facts

# ── Processed file paths ──────────────────────────────────────────────────────
# We convert raw CSVs to Parquet for fast columnar reads
ORDERS_PATH       = PROCESSED_DIR / "orders.parquet"
PRODUCTS_PATH     = PROCESSED_DIR / "products.parquet"
ORDER_ITEMS_PATH  = PROCESSED_DIR / "order_items.parquet"
M5_SALES_PATH     = PROCESSED_DIR / "m5_sales.parquet"
OFF_PRODUCTS_PATH = PROCESSED_DIR / "off_products.parquet"

# ── Models directory ──────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR}/mlflow.db")
MLFLOW_EXPERIMENT_NAME  = os.getenv("MLFLOW_EXPERIMENT_NAME", "grocery-iq")

# ── Qdrant vector database ────────────────────────────────────────────────────
QDRANT_HOST             = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT             = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION       = "grocery_products"

# ── Google Gemini ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL            = "gemini-1.5-flash"        # free-tier model
EMBEDDING_MODEL         = "all-MiniLM-L6-v2"        # local sentence-transformer

# ── Model hyperparameters (defaults; overridden by Optuna / MLflow) ───────────
XGBOOST_DEFAULTS = {
    "n_estimators":  400,
    "max_depth":     6,
    "learning_rate": 0.05,
    "subsample":     0.8,
    "random_state":  42,
}

# ── Ensure directories exist on import ───────────────────────────────────────
for _dir in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR,
             INSTACART_DIR, M5_DIR, OFF_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
