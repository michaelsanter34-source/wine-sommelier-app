
import warnings, re, os, joblib
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title='Wine Sommelier', page_icon='🍷', layout='wide',
                   initial_sidebar_state='expanded')

# --- Hugging Face download helper ---
# Big artifacts (>25 MB) are hosted on HF Hub instead of GitHub. On first run
# the app downloads them into the working directory. After that they're
# cached locally and load instantly.
HF_REPO_ID = 'Msanter/wine-sommelier-artifacts'  # <-- EDIT THIS

LARGE_FILES = [
     'wine_clean.parquet',
    'embeddings.npy',
    'faiss_index.bin',
    'retrieval_meta.joblib',
    'vibe_vectors.npy',
    'food_vectors.npy',
    'vibe_scores.npy',
    'food_scores.npy',
    'sommelier_meta.joblib',
    'models.joblib',
]

def ensure_artifacts():
    from huggingface_hub import hf_hub_download
    for fname in LARGE_FILES:
        if not os.path.exists(fname):
            local = hf_hub_download(repo_id=HF_REPO_ID, filename=fname,
                                    repo_type='dataset', local_dir='.')
            # hf_hub_download returns a symlink in newer versions; ensure
            # the file is at the expected path
            if local != os.path.abspath(fname) and not os.path.exists(fname):
                import shutil
                shutil.copy(local, fname)

@st.cache_resource(show_spinner='Loading wine database and AI models (one-time, ~30s)...')
def load_all_artifacts():
    ensure_artifacts()
    df = pd.read_parquet('wine_clean.parquet').reset_index(drop=True)
    df['embed_text'] = df['variety'].astype(str) + '. ' + df['description'].astype(str)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.load('embeddings.npy')
    faiss_index = faiss.read_index('faiss_index.bin')
    retrieval_meta = joblib.load('retrieval_meta.joblib')
    bm25 = BM25Okapi(retrieval_meta['tokenized_corpus'])
    vibe_vectors = np.load('vibe_vectors.npy')
    food_vectors = np.load('food_vectors.npy')
    vibe_scores = np.load('vibe_scores.npy')
    food_scores = np.load('food_scores.npy')
    sommelier_meta = joblib.load('sommelier_meta.joblib')
    artifacts = joblib.load('models.joblib')
    return {
        'df': df, 'model': model, 'embeddings': embeddings,
        'faiss_index': faiss_index, 'bm25': bm25,
        'vibe_vectors': vibe_vectors, 'food_vectors': food_vectors,
        'vibe_scores': vibe_scores, 'food_scores': food_scores,
        'vibe_names': sommelier_meta['vibe_names'],
        'food_names': sommelier_meta['food_names'],
        'flavor_keys': sommelier_meta['FLAVOR_KEYS'],
        'food_keywords': sommelier_meta['FOOD_KEYWORDS'],
        'vibe_keywords': sommelier_meta['VIBE_KEYWORDS'],
        'xgb_price_median': artifacts['xgb_price_median'],
        'xgb_price_lower': artifacts['xgb_price_lower'],
        'xgb_price_upper': artifacts['xgb_price_upper'],
        'xgb_points': artifacts['xgb_points'],
        'encoders': artifacts['encoders'],
        'all_features': artifacts['feature_names'],
        'points_features': artifacts['points_features'],
        'numeric_features': artifacts['numeric_features'],
        'flavor_features': artifacts['flavor_features'],
        'categorical_features': artifacts['categorical_features'],
    }
A = load_all_artifacts()

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return [w for w in text.split() if len(w) > 2]

def parse_query(query):
    q = (query or '').lower()
    intent = {'max_price': None, 'min_price': None, 'min_points': None,
              'foods': [], 'vibes': [], 'flavors': [], 'varieties': [], 'countries': []}
    m = (re.search(r'under\s*\$?(\d+)', q) or re.search(r'less than\s*\$?(\d+)', q)
         or re.search(r'<\s*\$?(\d+)', q) or re.search(r'max(?:imum)?\s*\$?(\d+)', q))
    if m: intent['max_price'] = float(m.group(1))
    m = (re.search(r'over\s*\$?(\d+)', q) or re.search(r'at least\s*\$?(\d+)', q)
         or re.search(r'>\s*\$?(\d+)', q))
    if m: intent['min_price'] = float(m.group(1))
    m = re.search(r'\$?(\d+)\s*[-–to]+\s*\$?(\d+)', q)
    if m:
        lo, hi = sorted([float(m.group(1)), float(m.group(2))])
        intent['min_price'], intent['max_price'] = lo, hi
    m = re.search(r'(\d+)\+?\s*(?:points|pts|rating)', q)
    if m: intent['min_points'] = int(m.group(1))
    for kw, food in A['food_keywords'].items():
        if kw in q and food not in intent['foods']: intent['foods'].append(food)
    for kw, vibe in A['vibe_keywords'].items():
        if kw in q and vibe not in intent['vibes']: intent['vibes'].append(vibe)
    for fl in A['flavor_keys']:
        if fl in q: intent['flavors'].append(fl)
    common_varieties = ['pinot noir', 'cabernet sauvignon', 'merlot', 'syrah', 'shiraz',
        'zinfandel', 'malbec', 'tempranillo', 'sangiovese', 'nebbiolo',
        'chardonnay', 'sauvignon blanc', 'riesling', 'pinot grigio', 'pinot gris',
        'gewürztraminer', 'gewurztraminer', 'champagne', 'prosecco', 'rosé', 'rose',
        'red blend', 'white blend', 'bordeaux']
    for v in common_varieties:
        if v in q:
            canonical = {'shiraz': 'syrah', 'pinot gris': 'pinot grigio',
                         'gewurztraminer': 'gewürztraminer', 'rose': 'rosé'}.get(v, v)
            if canonical not in intent['varieties']: intent['varieties'].append(canonical)
    country_map = {'france': 'France', 'italy': 'Italy', 'spain': 'Spain', 'germany': 'Germany',
                   'austria': 'Austria', 'portugal': 'Portugal', 'argentina': 'Argentina',
                   'chile': 'Chile', 'australia': 'Australia', 'new zealand': 'New Zealand',
                   'south africa': 'South Africa'}
    for c, name in country_map.items():
        if c in q: intent['countries'].append(name)
    if any(w in q for w in ['california', 'napa', 'sonoma', 'oregon', 'american']):
        if 'US' not in intent['countries']: intent['countries'].append('US')
    return intent

def split_sentences(text):
    if not isinstance(text, str): return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if len(p) > 15]

def hybrid_search(query, q_emb, max_price=None, min_price=None, min_points=None,
                  variety=None, country=None, required_flavors=None,
                  top_k=10, semantic_weight=0.65, bm25_weight=0.35, candidate_pool=300):
    sem_scores, sem_idx = A['faiss_index'].search(q_emb, candidate_pool)
    sem_scores, sem_idx = sem_scores[0], sem_idx[0]
    sem_norm = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-9)
    q_tokens = tokenize(query)
    df = A['df']
    bm25_all = A['bm25'].get_scores(q_tokens) if q_tokens else np.zeros(len(df))
    bm25_pool = bm25_all[sem_idx]
    bm25_norm = bm25_pool / (bm25_pool.max() + 1e-9) if bm25_pool.max() > 0 else np.zeros_like(bm25_pool)
    combined = semantic_weight * sem_norm + bm25_weight * bm25_norm
    candidates = df.iloc[sem_idx].copy()
    candidates['_orig_idx'] = sem_idx
    candidates['semantic_score'] = sem_scores
    candidates['bm25_score'] = bm25_pool
    candidates['combined_score'] = combined
    if max_price is not None:    candidates = candidates[candidates['price'] <= max_price]
    if min_price is not None:    candidates = candidates[candidates['price'] >= min_price]
    if min_points is not None:   candidates = candidates[candidates['points'] >= min_points]
    if variety is not None:
        if isinstance(variety, str): variety = [variety]
        pat = '|'.join(re.escape(v) for v in variety)
        candidates = candidates[candidates['variety'].str.contains(pat, case=False, na=False)]
    if country is not None:
        if isinstance(country, str): country = [country]
        candidates = candidates[candidates['country'].isin(country)]
    if required_flavors:
        for flav in required_flavors:
            col = f'has_{flav}'
            if col in candidates.columns:
                candidates = candidates[candidates[col] == 1]
    return candidates.sort_values('combined_score', ascending=False).head(top_k).reset_index(drop=True)

def build_features_for_inference(wine_rows):
    out = wine_rows.copy()
    for col, (enc, gmean) in A['encoders'].items():
        out[f'{col}_te'] = out[col].map(enc).fillna(gmean)
    for col in A['numeric_features']:
        if col not in out.columns: out[col] = 0
        out[col] = out[col].fillna(0)
    for col in A['flavor_features']:
        if col not in out.columns: out[col] = 0
    X_full = out[A['all_features']].values
    points_idx = [A['all_features'].index(f) for f in A['points_features']]
    return X_full, X_full[:, points_idx]

def value_label(actual, predicted, lower, upper):
    if actual < lower:
        return 'GREAT VALUE', f'predicted ${predicted:.0f} vs. actual ${actual:.0f} ({predicted/actual:.1f}× value)'
    if actual > upper:
        return 'OVERPRICED', f'predicted ${predicted:.0f} vs. actual ${actual:.0f} ({actual/predicted:.1f}× over)'
    return 'FAIR PRICED', f'predicted ${predicted:.0f}, within expected range'

def batch_relevant_sentences(descriptions, q_emb):
    sentence_lists = [split_sentences(d) for d in descriptions]
    flat_sentences = []
    offsets = [0]
    for sents in sentence_lists:
        flat_sentences.extend(sents)
        offsets.append(len(flat_sentences))
    if not flat_sentences:
        return [(d or '')[:200] for d in descriptions]
    sent_embs = A['model'].encode(flat_sentences, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    sims = (sent_embs @ q_emb.T).flatten()
    out = []
    for i in range(len(sentence_lists)):
        start, end = offsets[i], offsets[i+1]
        if end - start == 0:
            out.append((descriptions[i] or '')[:200])
            continue
        out.append(sentence_lists[i][int(np.argmax(sims[start:end]))])
    return out

def recommend(query=None, *, max_price=None, min_price=None, min_points=None,
              variety=None, country=None, top_k=5):
    query = query or ''
    intent = parse_query(query)
    eff_max_price = max_price if max_price is not None else intent['max_price']
    eff_min_price = min_price if min_price is not None else intent['min_price']
    eff_min_points = min_points if min_points is not None else intent['min_points']
    eff_variety = variety if variety is not None else (intent['varieties'] or None)
    eff_country = country if country is not None else (intent['countries'] or None)
    eff_flavors = intent['flavors'] or None
    search_query = query if query.strip() else \
        ' '.join(filter(None, [' '.join(intent['vibes']), ' '.join(intent['flavors']), 'wine']))
    if not search_query.strip(): search_query = 'wine'
    q_emb = A['model'].encode([search_query], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
    candidates = hybrid_search(search_query, q_emb,
        max_price=eff_max_price, min_price=eff_min_price, min_points=eff_min_points,
        variety=eff_variety, country=eff_country, required_flavors=eff_flavors, top_k=top_k)
    if len(candidates) == 0 and eff_flavors:
        candidates = hybrid_search(search_query, q_emb,
            max_price=eff_max_price, min_price=eff_min_price, min_points=eff_min_points,
            variety=eff_variety, country=eff_country, top_k=top_k)
    if len(candidates) == 0:
        return {'recommendations': [], 'intent': intent,
                'effective_filters': {'max_price': eff_max_price, 'min_price': eff_min_price,
                                       'min_points': eff_min_points, 'variety': eff_variety,
                                       'country': eff_country, 'flavors': eff_flavors},
                'note': 'No wines matched these filters. Try relaxing them.'}
    df = A['df']
    title_to_idx = {df.iloc[i]['title']: i for i in range(len(df))}
    orig_indices = [title_to_idx[t] for t in candidates['title']]
    X_full, X_pts = build_features_for_inference(candidates)
    log_pred = A['xgb_price_median'].predict(X_full)
    log_lo = A['xgb_price_lower'].predict(X_full)
    log_hi = A['xgb_price_upper'].predict(X_full)
    pts_pred = A['xgb_points'].predict(X_pts)
    log_lo = np.minimum(log_lo, log_pred); log_hi = np.maximum(log_hi, log_pred)
    pred_price = 10 ** log_pred; pred_low = 10 ** log_lo; pred_high = 10 ** log_hi
    snippets = batch_relevant_sentences(candidates['description'].tolist(), q_emb)
    recs = []
    vibe_names = A['vibe_names']; food_names = A['food_names']
    for i, (_, row) in enumerate(candidates.iterrows()):
        orig_idx = orig_indices[i]
        wine_vibe_scores = A['vibe_scores'][orig_idx]
        wine_food_scores = A['food_scores'][orig_idx]
        top_vibes = [vibe_names[j] for j in np.argsort(wine_vibe_scores)[::-1][:2]]
        if intent['foods'] and intent['foods'][0] in food_names:
            user_food = intent['foods'][0]; f_idx = food_names.index(user_food)
            user_food_score = wine_food_scores[f_idx]
            pct = (A['food_scores'][:, f_idx] < user_food_score).mean() * 100
            food_info = {'matched_user_food': user_food, 'match_percentile': round(float(pct), 0),
                         'top_default_pairings': [food_names[j] for j in np.argsort(wine_food_scores)[::-1][:2]]}
        else:
            food_info = {'matched_user_food': None, 'match_percentile': None,
                         'top_default_pairings': [food_names[j] for j in np.argsort(wine_food_scores)[::-1][:2]]}
        val_lab, val_explain = value_label(row['price'], pred_price[i], pred_low[i], pred_high[i])
        recs.append({
            'rank': i + 1, 'title': row['title'], 'variety': row['variety'],
            'country': row['country'], 'province': row.get('province', ''), 'winery': row['winery'],
            'vintage': int(row['vintage']) if pd.notna(row.get('vintage')) else None,
            'actual_price': float(row['price']), 'actual_rating': int(row['points']),
            'predicted_price': round(float(pred_price[i]), 2),
            'predicted_price_low': round(float(pred_low[i]), 2),
            'predicted_price_high': round(float(pred_high[i]), 2),
            'predicted_rating': round(float(pts_pred[i]), 1),
            'value_label': val_lab, 'value_explanation': val_explain,
            'top_vibes': top_vibes, 'food_pairing': food_info,
            'grounded_explanation': snippets[i], 'full_description': row['description'],
            'retrieval_score': round(float(row['combined_score']), 3),
        })
    return {'recommendations': recs, 'intent': intent,
            'effective_filters': {'max_price': eff_max_price, 'min_price': eff_min_price,
                                   'min_points': eff_min_points, 'variety': eff_variety,
                                   'country': eff_country, 'flavors': eff_flavors}, 'note': None}

# --- UI ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #1a0a14 0%, #2c1820 100%); color: #f5f0e8; }
    h1, h2, h3 { color: #f5f0e8 !important; }
    .wine-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(212,175,55,0.3);
                 border-radius: 12px; padding: 24px; margin-bottom: 16px; }
    .wine-title { font-size: 1.3rem; font-weight: 600; color: #d4af37; margin-bottom: 4px; }
    .wine-meta { font-size: 0.95rem; color: #b8a99a; margin-bottom: 12px; }
    .price-row { font-size: 1.05rem; margin-bottom: 8px; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
             font-size: 0.85rem; font-weight: 600; margin-right: 8px; }
    .badge-value { background: #2d6a4f; color: #fff; }
    .badge-fair { background: #5a5a5a; color: #fff; }
    .badge-overpriced { background: #7a3030; color: #fff; }
    .vibe-pill { display: inline-block; padding: 3px 10px; margin: 2px;
                 border-radius: 12px; background: rgba(212,175,55,0.15);
                 color: #d4af37; font-size: 0.85rem; }
    .quote { font-style: italic; color: #d4cab4; padding-left: 16px;
             border-left: 2px solid #d4af37; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('# 🍷 Wine Sommelier')
st.markdown('*An AI sommelier that combines machine learning, semantic search, and 110,000+ professional wine reviews to recommend the perfect bottle.*')

with st.sidebar:
    st.markdown('## Optional Filters')
    st.caption('Leave blank to let the AI infer from your query.')
    price_range = st.slider('Price range ($)', 4, 500, (4, 500), step=5)
    min_rating = st.slider('Minimum rating', 80, 100, 80, step=1)
    df = A['df']
    top_varieties = ['Any'] + df['variety'].value_counts().head(40).index.tolist()
    variety_choice = st.selectbox('Variety', top_varieties)
    top_countries = ['Any'] + df['country'].value_counts().head(20).index.tolist()
    country_choice = st.selectbox('Country', top_countries)
    top_k = st.slider('Number of recommendations', 3, 10, 5)
    st.markdown('---')
    st.caption(f'Database: {len(df):,} wines from WineEnthusiast')

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input('', placeholder="e.g. 'fruity bold red under $25 for steak dinner'",
                          label_visibility='collapsed')
with col2:
    search_clicked = st.button('🔍 Find My Wine', type='primary', use_container_width=True)

with st.expander('💡 Example queries'):
    st.markdown("""
    - *"fruity bold red wine under $25 for steak dinner"*
    - *"elegant pinot noir from france, 92+ points"*
    - *"crisp refreshing white from new zealand for sushi"*
    - *"oaky california chardonnay around $30"*
    - *"value pick for a tuesday night pasta"*
    - *"big bold cabernet sauvignon for celebration"*
    """)

if search_clicked:
    if not query.strip() and variety_choice == 'Any' and country_choice == 'Any' \
            and price_range == (4, 500) and min_rating == 80:
        st.warning('Please enter a query or select at least one filter.')
    else:
        with st.spinner('Analyzing your request and searching 110k wines...'):
            ui_max = price_range[1] if price_range[1] < 500 else None
            ui_min = price_range[0] if price_range[0] > 4 else None
            ui_min_pts = min_rating if min_rating > 80 else None
            ui_variety = variety_choice if variety_choice != 'Any' else None
            ui_country = [country_choice] if country_choice != 'Any' else None
            result = recommend(query, max_price=ui_max, min_price=ui_min,
                               min_points=ui_min_pts, variety=ui_variety,
                               country=ui_country, top_k=top_k)
        if result['note']:
            st.error(f"⚠️ {result['note']}")
            st.json(result['effective_filters'])
        else:
            intent = result['intent']
            picked_up = []
            if intent['foods']:    picked_up.append(f"food: {', '.join(intent['foods'])}")
            if intent['vibes']:    picked_up.append(f"vibes: {', '.join(intent['vibes'])}")
            if intent['flavors']:  picked_up.append(f"flavors: {', '.join(intent['flavors'])}")
            if intent['varieties']:picked_up.append(f"variety: {', '.join(intent['varieties'])}")
            if intent['countries']:picked_up.append(f"country: {', '.join(intent['countries'])}")
            if intent['max_price']:picked_up.append(f"max ${intent['max_price']:.0f}")
            if intent['min_points']:picked_up.append(f"{intent['min_points']}+ pts")
            if picked_up:
                st.info('🤖 **AI parsed from your query:** ' + ' • '.join(picked_up))
            st.markdown(f"## Top {len(result['recommendations'])} Recommendations")
            for rec in result['recommendations']:
                badge_class = {'GREAT VALUE': 'badge-value', 'FAIR PRICED': 'badge-fair',
                               'OVERPRICED': 'badge-overpriced'}[rec['value_label']]
                vibes_html = ' '.join(f'<span class="vibe-pill">{v}</span>' for v in rec['top_vibes'])
                fp = rec['food_pairing']
                if fp['matched_user_food']:
                    food_line = f"<strong>Pairs with {fp['matched_user_food']}:</strong> {fp['match_percentile']:.0f}th percentile match"
                else:
                    food_line = f"<strong>Best with:</strong> {', '.join(fp['top_default_pairings'])}"
                meta = f"{rec['variety']} • {rec['country']}"
                if rec['province'] and rec['province'] != 'Unknown':
                    meta += f" / {rec['province']}"
                if rec['vintage']: meta += f" • {rec['vintage']}"
                card_html = f"""
                <div class="wine-card">
                    <div class="wine-title">#{rec['rank']}  {rec['title']}</div>
                    <div class="wine-meta">{meta}</div>
                    <div class="price-row">
                        <strong>${rec['actual_price']:.0f}</strong> &nbsp;•&nbsp; {rec['actual_rating']} pts &nbsp;
                        <span class="badge {badge_class}">{rec['value_label']}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #b8a99a; margin-bottom: 12px;">
                        ML predicted: ${rec['predicted_price']:.0f}
                        (90% interval: ${rec['predicted_price_low']:.0f}–${rec['predicted_price_high']:.0f})
                        &nbsp;•&nbsp; predicted rating: {rec['predicted_rating']:.1f} pts
                    </div>
                    <div style="margin-bottom: 8px;"><strong>Character:</strong> {vibes_html}</div>
                    <div style="margin-bottom: 8px;">{food_line}</div>
                    <div class="quote">"{rec['grounded_explanation']}"</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
else:
    st.markdown('---')
    st.markdown('### How it works')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**🤖 ML Price Prediction**')
        st.caption('XGBoost model trained on 110k wines predicts fair price + 90% interval. R² = 0.63 on held-out test set.')
    with c2:
        st.markdown('**🔎 Hybrid Semantic Search**')
        st.caption('Combines sentence-transformer embeddings, BM25 keyword search, and structured filters with FAISS for fast retrieval.')
    with c3:
        st.markdown('**🍷 Sommelier Intelligence**')
        st.caption('Each wine is profiled across 14 vibes and 20 food pairings. Explanations are extracted from real expert reviews.')
