# 🍷 Wine Sommelier

An AI-powered wine recommender that combines machine learning, semantic
search, and 110,000+ professional sommelier reviews from WineEnthusiast.

## What it does

Type a natural-language query like *"fruity bold red under $25 for steak
dinner"* and the app returns ranked wine recommendations with:

- **Predicted vs. actual price** (XGBoost regression with 90% prediction
  intervals). Flags Great Value / Fair / Overpriced.
- **Predicted rating** (XGBoost classifier on points).
- **Character vibes**. Each wine is scored across 14 personality
  dimensions using sentence-transformer embedding similarity.
- **Food pairing match**. 20 food categories scored per wine, with
  percentile ranking against the full corpus.
- **Grounded explanation**. The most query-relevant sentence is pulled
  from the actual professional review. No hallucinated text.

## Architecture

| Layer | Technology |
|-------|-----------|
| Data | WineEnthusiast 130k reviews, cleaned to 110,550 wines |
| Price/rating ML | XGBoost (R² = 0.627, MAE = $11.31) with quantile regression |
| Semantic search | sentence-transformers `all-MiniLM-L6-v2` + FAISS index |
| Keyword search | BM25 (rank-bm25) |
| Intent parsing | Pure-Python regex (no LLM, no API costs) |
| UI | Streamlit |

## Running locally

```bash
git clone https://github.com/YOUR-USERNAME/wine-sommelier-app.git
cd wine-sommelier-app
pip install -r requirements.txt
streamlit run app.py
```

The big artifact files (embeddings, FAISS index, cleaned dataset) are
hosted on Hugging Face Hub and download automatically on first run.
First launch takes about 60 seconds while everything loads. After that,
queries return in under a second.

## Live deployment

Deployed on Streamlit Community Cloud. The live URL is in the repo's
About section.
