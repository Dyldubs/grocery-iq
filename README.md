# GroceryIQ — Agentic Retail Analytics System

An agentic AI system that combines a **RAG-powered product knowledge base** with **ML-driven demand forecasting, customer segmentation, and price elasticity** — letting a retail analyst ask business questions in plain English and get data-grounded, source-cited answers.

Built to demonstrate skills directly relevant to retail AI/ML roles (Woolworths/wiq, Quantium, Coles360).

---

## What it does

A buyer or category manager can ask:

> *"Which snack products are at risk of stockout this week?"*  
> *"What are the best substitutes for Kettle Chips 175g if it goes out of stock?"*  
> *"Which customer segments are most price-sensitive to soft drink discounts?"*

The agent decomposes the question, routes it to the right tool (demand forecast, RAG product search, or customer segmentation), and returns a grounded, cited answer.

---

## Architecture

```
User query (natural language)
        │
        ▼
LangChain ReAct Agent  (Gemini 1.5 Flash)
        │
   ┌────┴──────────────────────────────────────┐
   │                                           │
Tool 1: product_search()          Tool 2: forecast_demand()
RAG over Open Food Facts          XGBoost on M5 Walmart data
Qdrant vector DB                  4-week sales forecast
sentence-transformers embeddings  MLflow experiment tracking
   │                                           │
Tool 3: segment_customers()       Tool 4: price_elasticity()
RFM clustering on Instacart       Log-log OLS per category
KMeans customer segments          Own-price & cross-price elasticity
   │                                           │
   └─────────────────┬─────────────────────────┘
                     ▼
         Synthesised answer + citations
                     │
                     ▼
           FastAPI  /ask  endpoint
           Docker container
           Ragas evaluation dashboard
```

---

## Datasets

| Dataset | Size | Purpose |
|---|---|---|
| [Instacart Market Basket](https://www.kaggle.com/c/instacart-market-basket-analysis) | ~200 MB | Customer segmentation, reorder classifier |
| [M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy) | ~130 MB | Demand forecasting |
| [Open Food Facts](https://world.openfoodfacts.org/data) | ~1 GB | RAG product knowledge base |

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash (free tier) via LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector DB | Qdrant |
| RAG pipeline | LlamaIndex |
| ML models | XGBoost, scikit-learn, statsmodels |
| SQL analytics | DuckDB (queries Parquet files directly) |
| Experiment tracking | MLflow |
| RAG evaluation | Ragas |
| API | FastAPI + Uvicorn |
| Containerisation | Docker + docker-compose |
| CI | GitHub Actions |

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/grocery-iq.git
cd grocery-iq
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your Google Gemini API key (free at aistudio.google.com)

# 3. Download datasets (requires Kaggle account)
# Set up Kaggle credentials first: kaggle.com → Settings → API → Create Token
# Move kaggle.json to ~/.kaggle/kaggle.json
python scripts/download_data.py

# 4. Preprocess (CSV → Parquet)
python scripts/preprocess_data.py

# 5. Explore the data
jupyter notebook notebooks/01_eda.ipynb

# 6. Train ML models
python scripts/train_models.py

# 7. Build RAG knowledge base
python scripts/ingest_products.py

# 8. Start the API
uvicorn src.api.main:app --reload

# 9. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which snack products should we promote this week?"}'
```

---

## Project structure

```
grocery-iq/
├── data/
│   ├── raw/          # Downloaded data (gitignored)
│   ├── processed/    # Cleaned Parquet files (gitignored)
│   └── embeddings/   # Vector index (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_ml_models.ipynb     # Model development and evaluation
│   └── 03_rag_evaluation.ipynb # RAG pipeline evaluation with Ragas
├── src/
│   ├── config.py              # All paths and settings
│   ├── data/loader.py         # Data loading layer
│   ├── models/                # ML models (forecast, segmentation, elasticity)
│   ├── rag/                   # RAG pipeline (ingestion, retriever, evaluation)
│   ├── agent/                 # LangChain agent and tools
│   └── api/                   # FastAPI endpoint
├── scripts/
│   ├── download_data.py       # Download all datasets
│   ├── preprocess_data.py     # CSV → Parquet
│   ├── train_models.py        # Train all ML models
│   └── ingest_products.py     # Embed products into Qdrant
├── tests/                     # pytest test suite
└── .github/workflows/ci.yml   # GitHub Actions CI
```

---

## Modules

| Module | Status | What it covers |
|---|---|---|
| 1 — Data layer | ✅ Complete | Download, preprocess, EDA |
| 2 — ML models | 🔄 In progress | Demand forecasting, segmentation, elasticity |
| 3 — RAG knowledge base | ⏳ Pending | Product embeddings, Qdrant, retrieval eval |
| 4 — Agentic layer | ⏳ Pending | LangChain ReAct agent, LangSmith traces |
| 5 — Evaluation + API | ⏳ Pending | Ragas metrics, FastAPI, Docker |
