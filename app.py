# =============================================================================
# app.py
# -----------------------------------------------------------------------------
# Streamlit web application for the Fake News Detection System.
#
# Run:
#   streamlit run app.py
#
# Make sure you have run  main.py  first so that the model artefacts exist
# inside the  models/  directory.
# =============================================================================

import os
import sys
import re
import string
import warnings
warnings.filterwarnings("ignore")

import joblib
import streamlit as st
import nltk

# ---------------------------------------------------------------------------
# NLTK resource check
# ---------------------------------------------------------------------------
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "Fake News Detector",
    page_icon  = "🔍",
    layout     = "centered",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- background ---- */
    .stApp { background: #0f1117; }

    /* ---- hero header ---- */
    .hero {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e74c3c, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero p {
        color: #aaa;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* ---- result boxes ---- */
    .result-fake {
        background: linear-gradient(135deg, #e74c3c22, #c0392b33);
        border: 1.5px solid #e74c3c;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-real {
        background: linear-gradient(135deg, #2ecc7122, #27ae6033);
        border: 1.5px solid #2ecc71;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    .result-sub {
        color: #ccc;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* ---- model badge ---- */
    .badge {
        display: inline-block;
        background: #1e2130;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        color: #aaa;
        margin: 0.3rem 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load artefacts (cached so they load only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models …")
def load_models():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    lr_model   = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    nb_model   = joblib.load(os.path.join(MODELS_DIR, "naive_bayes.pkl"))
    return vectorizer, lr_model, nb_model


# ---------------------------------------------------------------------------
# Text cleaning (mirrors src/preprocessing.py  without pandas dependency)
# ---------------------------------------------------------------------------
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"https?://\S+|www\.\S+", "", text)
    text   = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🔍 Fake News Detector</h1>
    <p>Paste any news article below — our NLP models will judge its credibility.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Model selector
col1, col2 = st.columns([2, 1])
with col1:
    news_input = st.text_area(
        "📰 Paste your news article here:",
        height     = 220,
        placeholder = "Type or paste a full news article …",
    )
with col2:
    model_choice = st.radio(
        "Choose model",
        options = ["Logistic Regression", "Naïve Bayes", "Both (majority vote)"],
        index   = 0,
    )
    st.markdown("&nbsp;")
    predict_btn = st.button("🔍 Analyse Article", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_btn:
    if not news_input.strip():
        st.warning("⚠️  Please paste some text before clicking Analyse.")
    else:
        try:
            vectorizer, lr_model, nb_model = load_models()
        except FileNotFoundError:
            st.error(
                "🚨  Model files not found in `models/`.  "
                "Please run `python main.py` first to train and save the models."
            )
            st.stop()

        # --- preprocess ---
        cleaned = clean_text(news_input)
        X_vec   = vectorizer.transform([cleaned])

        # --- predict ---
        lr_pred  = lr_model.predict(X_vec)[0]
        lr_prob  = lr_model.predict_proba(X_vec)[0]
        nb_pred  = nb_model.predict(X_vec)[0]
        nb_prob  = nb_model.predict_proba(X_vec)[0]

        if model_choice == "Logistic Regression":
            final_pred = lr_pred
            confidence = lr_prob[final_pred]
            used       = "Logistic Regression"
        elif model_choice == "Naïve Bayes":
            final_pred = nb_pred
            confidence = nb_prob[final_pred]
            used       = "Naïve Bayes"
        else:   # majority vote
            vote = lr_pred + nb_pred   # 0,1, or 2
            final_pred = 1 if vote >= 1 else 0
            confidence = (lr_prob[final_pred] + nb_prob[final_pred]) / 2
            used       = "Ensemble (majority vote)"

        # --- display result ---
        label_text  = "❌  FAKE NEWS"  if final_pred == 0 else "✅  REAL NEWS"
        box_class   = "result-fake"    if final_pred == 0 else "result-real"
        label_color = "#e74c3c"        if final_pred == 0 else "#2ecc71"

        st.markdown(f"""
        <div class="{box_class}">
            <div class="result-label" style="color:{label_color}">{label_text}</div>
            <div class="result-sub">
                Confidence : <strong>{confidence * 100:.1f}%</strong> &nbsp;|&nbsp;
                Model used : <strong>{used}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- per-model breakdown (when "Both" selected) ---
        if model_choice == "Both (majority vote)":
            st.markdown("&nbsp;")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Logistic Regression",
                    "FAKE" if lr_pred == 0 else "REAL",
                    f"Confidence: {max(lr_prob)*100:.1f}%",
                )
            with c2:
                st.metric(
                    "Naïve Bayes",
                    "FAKE" if nb_pred == 0 else "REAL",
                    f"Confidence: {max(nb_prob)*100:.1f}%",
                )

        # --- word count info ---
        word_count = len(news_input.split())
        st.caption(
            f"📊  Input: {word_count} words  ·  "
            f"After cleaning: {len(cleaned.split())} words  ·  "
            f"TF-IDF features: {X_vec.shape[1]:,}"
        )

st.divider()

# ---------------------------------------------------------------------------
# About section
# ---------------------------------------------------------------------------
with st.expander("ℹ️  About this app"):
    st.markdown("""
**Fake News Detection System**  
Built with Python · scikit-learn · NLTK · Streamlit

### How it works
1. Your text is cleaned: lowercased, punctuation & stopwords removed, lemmatized.
2. The cleaned text is converted to a **TF-IDF** numerical vector (up to 50 000 features, unigrams + bigrams).
3. The vector is passed to the selected ML model(s) for classification.

### Models
| Model | Typical Accuracy |
|---|---|
| Logistic Regression | ~98–99 % |
| Naïve Bayes | ~94–96 % |

### Dataset
Trained on the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
(≈ 44 000 articles, balanced between Fake and Real).
    """)
