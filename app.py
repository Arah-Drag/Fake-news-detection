# =============================================================================
# app.py
# Streamlit web application for the Fake News Detection System.
# Run: streamlit run app.py
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
# NLTK resource check — download silently if missing
# ---------------------------------------------------------------------------
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
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
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Auto-train models if not already saved
# This runs automatically on Streamlit Cloud first deploy
# ---------------------------------------------------------------------------
def auto_train():
    """Train and save models if pkl files are missing."""
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    if not os.path.exists(lr_path):
        st.info("⏳ First launch detected — training models now. This takes 3-5 minutes...")
        progress = st.progress(0, text="Loading data...")

        # --- imports ---
        import pandas as pd
        from src.preprocessing import load_data, preprocess_dataframe
        from src.model import (
            build_tfidf, split_data,
            train_logistic_regression, train_naive_bayes,
            save_artefacts,
        )

        # --- load data ---
        fake_path = os.path.join(DATA_DIR, "Fake.csv")
        true_path = os.path.join(DATA_DIR, "True.csv")

        if not os.path.exists(fake_path) or not os.path.exists(true_path):
            st.error(
                "Dataset files not found! "
                "Please make sure Fake.csv and True.csv are inside the data/ folder."
            )
            st.stop()

        progress.progress(10, text="Loading datasets...")
        df = load_data(fake_path, true_path)

        progress.progress(30, text="Preprocessing text (this takes a while)...")
        df_clean = preprocess_dataframe(df)

        progress.progress(60, text="Building TF-IDF features...")
        X_train, X_test, y_train, y_test = split_data(df_clean)
        vectorizer     = build_tfidf(X_train, max_features=50000)
        X_train_tfidf  = vectorizer.transform(X_train)

        progress.progress(75, text="Training Logistic Regression...")
        lr_model = train_logistic_regression(X_train_tfidf, y_train)

        progress.progress(88, text="Training Naive Bayes...")
        nb_model = train_naive_bayes(X_train_tfidf, y_train)

        progress.progress(95, text="Saving models...")
        save_artefacts(vectorizer, lr_model, nb_model, save_dir=MODELS_DIR)

        progress.progress(100, text="Done!")
        st.success("✅ Models trained and saved successfully! Refreshing...")
        st.rerun()


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
    .stApp { background: #0f1117; }

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

    .stProgress > div > div {
        background-color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Run auto-train check before anything else
# ---------------------------------------------------------------------------
auto_train()


# ---------------------------------------------------------------------------
# Load artefacts (cached so they load only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    lr_model   = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    nb_model   = joblib.load(os.path.join(MODELS_DIR, "naive_bayes.pkl"))
    return vectorizer, lr_model, nb_model


# ---------------------------------------------------------------------------
# Text cleaning
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
# UI — Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🔍 Fake News Detector</h1>
    <p>Paste any news article below — our NLP models will judge its credibility.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# UI — Input + Model Selection
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    news_input = st.text_area(
        "📰 Paste your news article here:",
        height      = 220,
        placeholder = "Type or paste a full news article...",
    )

with col2:
    model_choice = st.radio(
        "Choose model",
        options = ["Logistic Regression", "Naive Bayes", "Both (majority vote)"],
        index   = 0,
    )
    st.markdown("&nbsp;")
    predict_btn = st.button(
        "🔍 Analyse Article",
        use_container_width = True,
        type                = "primary",
    )

# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------
if predict_btn:
    if not news_input.strip():
        st.warning("Please paste some text before clicking Analyse.")
    else:
        try:
            vectorizer, lr_model, nb_model = load_models()
        except FileNotFoundError:
            st.error(
                "Model files not found. "
                "Please wait for auto-training to complete or run python main.py locally."
            )
            st.stop()

        # Step 1 — clean text
        cleaned = clean_text(news_input)

        if not cleaned.strip():
            st.warning("Text became empty after cleaning. Please enter a longer article.")
            st.stop()

        # Step 2 — vectorize
        X_vec = vectorizer.transform([cleaned])

        # Step 3 — predict
        lr_pred = lr_model.predict(X_vec)[0]
        lr_prob = lr_model.predict_proba(X_vec)[0]
        nb_pred = nb_model.predict(X_vec)[0]
        nb_prob = nb_model.predict_proba(X_vec)[0]

        # Step 4 — decide final prediction
        if model_choice == "Logistic Regression":
            final_pred = lr_pred
            confidence = lr_prob[final_pred]
            used       = "Logistic Regression"

        elif model_choice == "Naive Bayes":
            final_pred = nb_pred
            confidence = nb_prob[final_pred]
            used       = "Naive Bayes"

        else:
            # Both — majority vote with confidence-based tie breaking
            if lr_pred == nb_pred:
                # both models agree
                final_pred = lr_pred
                confidence = (lr_prob[final_pred] + nb_prob[final_pred]) / 2
            else:
                # tie — pick the model with higher confidence
                lr_confidence = max(lr_prob)
                nb_confidence = max(nb_prob)
                if lr_confidence >= nb_confidence:
                    final_pred = lr_pred
                    confidence = lr_confidence
                else:
                    final_pred = nb_pred
                    confidence = nb_confidence
            used = "Ensemble (majority vote)"

        # Step 5 — display result
        label_text  = "FAKE NEWS" if final_pred == 0 else "REAL NEWS"
        emoji       = "❌" if final_pred == 0 else "✅"
        box_class   = "result-fake" if final_pred == 0 else "result-real"
        label_color = "#e74c3c" if final_pred == 0 else "#2ecc71"

        st.markdown(f"""
        <div class="{box_class}">
            <div class="result-label" style="color:{label_color}">{emoji} &nbsp; {label_text}</div>
            <div class="result-sub">
                Confidence : <strong>{confidence * 100:.1f}%</strong> &nbsp;|&nbsp;
                Model used : <strong>{used}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 6 — per model breakdown (only for Both)
        if model_choice == "Both (majority vote)":
            st.markdown("&nbsp;")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Logistic Regression",
                    "FAKE" if lr_pred == 0 else "REAL",
                    f"Confidence: {max(lr_prob) * 100:.1f}%",
                )
            with c2:
                st.metric(
                    "Naive Bayes",
                    "FAKE" if nb_pred == 0 else "REAL",
                    f"Confidence: {max(nb_prob) * 100:.1f}%",
                )

        # Step 7 — stats footer
        word_count = len(news_input.split())
        st.caption(
            f"Input: {word_count} words  |  "
            f"After cleaning: {len(cleaned.split())} words  |  "
            f"TF-IDF features: {X_vec.shape[1]:,}"
        )

st.divider()

# ---------------------------------------------------------------------------
# About section
# ---------------------------------------------------------------------------
with st.expander("ℹ️ About this app"):
    st.markdown("""
**Fake News Detection System**
Built with Python, scikit-learn, NLTK, and Streamlit.

**How it works:**
1. Your text is cleaned — lowercased, punctuation and stopwords removed, lemmatized.
2. The cleaned text is converted to a TF-IDF numerical vector (50,000 features, unigrams + bigrams).
3. The vector is passed to the selected ML model for classification.

**Majority Vote Tie-Breaking:**
When both models disagree, the model with the higher confidence score wins.

**Models:**
| Model | Typical Accuracy |
|---|---|
| Logistic Regression | 98-99% |
| Naive Bayes | 94-96% |

**Dataset:**
Trained on the Kaggle Fake and Real News Dataset (~44,000 articles).
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    """)