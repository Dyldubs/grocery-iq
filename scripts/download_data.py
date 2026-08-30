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
import zipfile
from pathlib import Path
import sys

# Allow running from project root: python scripts/download_data.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import INSTACART_DIR, M5_DIR, OFF_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _kaggle_competition_download(slug: str, dest: Path) -> None:
    """Download a Kaggle competition dataset by competition slug."""
    _kaggle_run(["kaggle", "competitions", "download", "-c", slug, "-p", str(dest)], dest, slug)


def _kaggle_dataset_download(handle: str, dest: Path) -> None:
    """Download a Kaggle dataset by owner/dataset-name handle."""
    _kaggle_run(["kaggle", "datasets", "download", "-d", handle, "-p", str(dest)], dest, handle)


def _kaggle_run(cmd: list, dest: Path, name: str) -> None:
    """Run a kaggle CLI command, then unzip any resulting zip files."""
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {name} → {dest}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        logger.error(output)
        raise RuntimeError(f"Kaggle download failed for {name}:\n{output}")

    # Unzip any downloaded zip files (--unzip flag not available on all CLI versions)
    for zip_path in dest.glob("*.zip"):
        logger.info(f"  Unzipping {zip_path.name}…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()
    logger.info(f"  ✓ {name} downloaded")


def download_open_food_facts(dest: Path) -> None:
    """
    Download the Open Food Facts CSV export directly (no Kaggle needed).

    This is a ~1 GB gzipped CSV of 800K+ products with nutritional info,
    categories, ingredients, brands — everything we need for the RAG knowledge base.

    Uses HTTP Range requests to resume interrupted downloads, with up to 3 retries.
    """
    import time
    url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "products.csv.gz"

    # Check remote file size to detect completed vs partial downloads
    import urllib.error
    try:
        with urllib.request.urlopen(url) as r:
            total_size = int(r.headers.get("Content-Length", 0))
    except urllib.error.URLError:
        total_size = 0

    if out.exists() and total_size > 0 and out.stat().st_size >= total_size:
        logger.info("  Open Food Facts already downloaded — skipping")
        return

    existing = out.stat().st_size if out.exists() else 0
    if existing:
        logger.info(f"  Resuming Open Food Facts from {existing/1e6:.0f} MB…")
    else:
        logger.info("Downloading Open Food Facts (~1 GB)…")
    logger.info(f"  URL: {url}")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url)
            current = out.stat().st_size if out.exists() else 0
            if current:
                req.add_header("Range", f"bytes={current}-")

            with urllib.request.urlopen(req) as response, open(out, "ab") as f:
                chunk_size = 1024 * 256  # 256 KB
                downloaded = current
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = min(100, downloaded / total_size * 100)
                        print(f"\r  {pct:5.1f}%  {downloaded/1e6:.0f} MB / {total_size/1e6:.0f} MB",
                              end="", flush=True)
            print()
            logger.info(f"  ✓ Open Food Facts downloaded → {out}")
            return
        except Exception as e:
            print()
            if attempt < max_retries:
                logger.warning(f"  Download interrupted ({e}), retrying in 5 s… (attempt {attempt}/{max_retries})")
                time.sleep(5)
            else:
                raise RuntimeError(f"Open Food Facts download failed after {max_retries} attempts: {e}") from e
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
        # Instacart competition has ended — data is now hosted as a dataset
        _kaggle_dataset_download("psparks/instacart-market-basket-analysis", INSTACART_DIR)

    if do_m5:
        # M5 is still an active competition
        _kaggle_competition_download("m5-forecasting-accuracy", M5_DIR)

    if do_off:
        download_open_food_facts(OFF_DIR)

    logger.info("\nAll downloads complete.")
    logger.info("Next step: python scripts/preprocess_data.py")


if __name__ == "__main__":
    main()
