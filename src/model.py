# =============================================================================
# model.py
# -----------------------------------------------------------------------------
# Feature engineering (TF-IDF) + model training for the Fake News Detector.
# =============================================================================

import os
import joblib
import numpy  as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.linear_model            import LogisticRegression
from sklearn.naive_bayes             import MultinomialNB


# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING  –  TF-IDF
# ---------------------------------------------------------------------------

def build_tfidf(train_texts, max_features: int = 50_000):
    """
    Fit a TF-IDF vectoriser on the training corpus.

    Parameters
    ----------
    train_texts   : iterable of str   Training documents (clean text)
    max_features  : int               Vocabulary size cap

    Returns
    -------
    TfidfVectorizer   Fitted vectoriser
    """
    print(f"[INFO] Building TF-IDF vectoriser  (max_features={max_features:,}) ...")
    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = (1, 2),    # unigrams + bigrams
        sublinear_tf = True,      # apply log(1+tf) scaling
    )
    vectorizer.fit(train_texts)
    print("  Vocabulary size :", len(vectorizer.vocabulary_), "\n")
    return vectorizer


# ---------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the cleaned DataFrame into training and test sets.

    Parameters
    ----------
    df           : pd.DataFrame   Must contain 'clean_text' and 'label'
    test_size    : float          Fraction for testing (default 0.20)
    random_state : int

    Returns
    -------
    X_train, X_test : pd.Series   Raw text splits
    y_train, y_test : pd.Series   Label splits
    """
    print(f"[INFO] Splitting data  (test_size={test_size}) ...")
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y,           # keep class balance in both splits
    )
    print(f"  Train : {len(X_train):,}   Test : {len(X_test):,}\n")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train_tfidf, y_train):
    """
    Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train_tfidf : sparse matrix   TF-IDF features for the training set
    y_train       : array-like      Training labels

    Returns
    -------
    LogisticRegression   Fitted model
    """
    print("[INFO] Training Logistic Regression ...")
    model = LogisticRegression(
        max_iter    = 1000,
        solver      = "lbfgs",
        C           = 1.0,
        random_state = 42,
    )
    model.fit(X_train_tfidf, y_train)
    print("  Done.\n")
    return model


def train_naive_bayes(X_train_tfidf, y_train):
    """
    Train a Multinomial Naïve Bayes classifier.

    Parameters
    ----------
    X_train_tfidf : sparse matrix   TF-IDF features (must be non-negative)
    y_train       : array-like      Training labels

    Returns
    -------
    MultinomialNB   Fitted model
    """
    print("[INFO] Training Multinomial Naïve Bayes ...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    print("  Done.\n")
    return model


# ---------------------------------------------------------------------------
# 4. SAVE / LOAD ARTEFACTS
# ---------------------------------------------------------------------------

def save_artefacts(vectorizer, lr_model, nb_model, save_dir: str = "models"):
    """
    Persist the vectoriser and both trained models to disk using joblib.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
    lr_model   : LogisticRegression
    nb_model   : MultinomialNB
    save_dir   : str   Directory to save artefacts (created if absent)
    """
    os.makedirs(save_dir, exist_ok=True)

    vectorizer_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
    lr_path         = os.path.join(save_dir, "logistic_regression.pkl")
    nb_path         = os.path.join(save_dir, "naive_bayes.pkl")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(lr_model,   lr_path)
    joblib.dump(nb_model,   nb_path)

    print(f"[INFO] Artefacts saved to '{save_dir}/'")
    print(f"  {vectorizer_path}")
    print(f"  {lr_path}")
    print(f"  {nb_path}\n")


def load_artefacts(save_dir: str = "models"):
    """
    Load vectoriser and models from disk.

    Returns
    -------
    (TfidfVectorizer, LogisticRegression, MultinomialNB)
    """
    vectorizer = joblib.load(os.path.join(save_dir, "tfidf_vectorizer.pkl"))
    lr_model   = joblib.load(os.path.join(save_dir, "logistic_regression.pkl"))
    nb_model   = joblib.load(os.path.join(save_dir, "naive_bayes.pkl"))
    return vectorizer, lr_model, nb_model
