"""
Tests for src/config.py

These tests verify that the config module:
- Resolves all paths relative to the project root correctly
- Creates required directories on import
- Exposes the expected constants

No downloaded data is needed — these are pure unit tests.
"""

def test_project_root_is_a_real_directory():
    """The ROOT_DIR should point to the grocery-iq/ folder."""
    from src.config import ROOT_DIR
    assert ROOT_DIR.is_dir(), f"ROOT_DIR {ROOT_DIR} does not exist"


def test_data_directories_are_created():
    """
    Config should auto-create data/raw, data/processed, data/embeddings
    on import so that downstream scripts don't need to mkdir themselves.
    """
    from src.config import RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR
    for d in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, MODELS_DIR]:
        assert d.exists(), f"Expected {d} to be created by config import"


def test_instacart_dir_is_under_raw():
    """Dataset directories should be nested inside data/raw/."""
    from src.config import RAW_DIR, INSTACART_DIR, M5_DIR, OFF_DIR
    for d in [INSTACART_DIR, M5_DIR, OFF_DIR]:
        assert str(d).startswith(str(RAW_DIR)), f"{d} is not under RAW_DIR"


def test_processed_paths_are_parquet():
    """All processed file paths should end in .parquet."""
    from src.config import (
        ORDERS_PATH, PRODUCTS_PATH, ORDER_ITEMS_PATH,
        M5_SALES_PATH, OFF_PRODUCTS_PATH,
    )
    for p in [ORDERS_PATH, PRODUCTS_PATH, ORDER_ITEMS_PATH, M5_SALES_PATH, OFF_PRODUCTS_PATH]:
        assert p.suffix == ".parquet", f"Expected .parquet, got {p.suffix} for {p.name}"


def test_xgboost_defaults_are_present():
    """The XGBoost default dict should have the keys we rely on."""
    from src.config import XGBOOST_DEFAULTS
    required = {"n_estimators", "max_depth", "learning_rate", "subsample", "random_state"}
    assert required.issubset(XGBOOST_DEFAULTS), (
        f"Missing keys: {required - set(XGBOOST_DEFAULTS)}"
    )


def test_gemini_model_name_is_set():
    """GEMINI_MODEL should be a non-empty string."""
    from src.config import GEMINI_MODEL
    assert isinstance(GEMINI_MODEL, str) and len(GEMINI_MODEL) > 0


def test_google_api_key_reads_from_env(monkeypatch):
    """
    GOOGLE_API_KEY should come from the environment.

    monkeypatch is a pytest built-in that temporarily overrides environment
    variables for the duration of a single test, then restores them.
    This means the test is isolated — it doesn't affect other tests or your
    real .env file.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-abc123")
    # Re-import to pick up the patched env (dotenv runs at import time,
    # but os.getenv reads the live env, so this works)
    import importlib
    import src.config as cfg
    importlib.reload(cfg)
    assert cfg.GOOGLE_API_KEY == "test-key-abc123"
