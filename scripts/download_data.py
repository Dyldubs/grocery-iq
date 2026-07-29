"""
Download all datasets needed by GroceryIQ.

Datasets
--------
1. Instacart Market Basket Analysis (Kaggle)
   ~200 MB compressed | 3.4M orders, 206K users, 50K products

2. M5 Forecasting — Accuracy (Kaggle)
   ~130 MB compressed | hierarchical daily sales data (Walmart)

3. Open Food Facts (direct download, no Kaggle needed)
   ~1 GB compressed | 800K+ product records for the RAG knowledge base

Usage
-----
    # First time: authenticate with Kaggle
    # 1. Go to kaggle.com → your profile → Settings → API → Create New Token
    # 2. This downloads a kaggle.json file to your ~/Downloads
    # 3. Move it: mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    # 4. Set permissions: chmod 600 ~/.kaggle/kaggle.json

    python scripts/download_data.py
    python scripts/download_data.py --skip-off      # skip Open Food Facts
    python scripts/download_data.py --dataset instacart  # one dataset only
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import urllib.request
from pathlib import Path
import sys

# Allow running from project root: python scripts/download_data.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import INSTACART_DIR, M5_DIR, OFF_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _kaggle_download(dataset: str, dest: Path) -> None:
    """
    Use the Kaggle CLI to download and unzip a competition dataset.

    The Kaggle CLI reads credentials from ~/.kaggle/kaggle.json.
    'dataset' is the competition slug shown in the Kaggle URL.
    """
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {dataset} → {dest}")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", dataset, "-p", str(dest), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Kaggle download failed for {dataset}:\n{result.stderr}")
    logger.info(f"  ✓ {dataset} downloaded")


def download_open_food_facts(dest: Path) -> None:
    """
    Download the Open Food Facts CSV export directly (no Kaggle needed).

    This is a ~1 GB gzipped CSV of 800K+ products with nutritional info,
    categories, ingredients, brands — everything we need for the RAG knowledge base.
    """
    url  = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    dest.mkdir(parents=True, exist_ok=True)
    out  = dest / "products.csv.gz"

    if out.exists():
        logger.info(f"  Open Food Facts already downloaded — skipping")
        return

    logger.info(f"Downloading Open Food Facts (~1 GB)…")
    logger.info(f"  URL: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded / total_size * 100)
            print(f"\r  {pct:5.1f}%  {downloaded/1e6:.0f} MB / {total_size/1e6:.0f} MB",
                  end="", flush=True)

    urllib.request.urlretrieve(url, out, reporthook=_progress)
    print()  # newline after progress bar
    logger.info(f"  ✓ Open Food Facts downloaded → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GroceryIQ datasets")
    parser.add_argument("--dataset", choices=["instacart", "m5", "off", "all"],
                        default="all", help="Which dataset to download (default: all)")
    parser.add_argument("--skip-off", action="store_true",
                        help="Skip Open Food Facts (large file, ~1 GB)")
    args = parser.parse_args()

    do_instacart = args.dataset in ("instacart", "all")
    do_m5        = args.dataset in ("m5", "all")
    do_off       = args.dataset in ("off", "all") and not args.skip_off

    if do_instacart:
        _kaggle_download("instacart-market-basket-analysis", INSTACART_DIR)

    if do_m5:
        _kaggle_download("m5-forecasting-accuracy", M5_DIR)

    if do_off:
        download_open_food_facts(OFF_DIR)

    logger.info("\nAll downloads complete.")
    logger.info("Next step: python scripts/preprocess_data.py")


if __name__ == "__main__":
    main()
