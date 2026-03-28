# =============================================================================
# preprocessing.py
# -----------------------------------------------------------------------------
# Handles all data loading and text cleaning steps for the Fake News Detector.
# =============================================================================

import os
import re
import string
import pandas as pd
import nltk

# Download required NLTK resources (runs only the first time)
nltk.download("stopwords",   quiet=True)
nltk.download("punkt",       quiet=True)
nltk.download("wordnet",     quiet=True)
nltk.download("omw-1.4",     quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

# Shared objects
_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_data(fake_path: str, true_path: str) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv, attach labels, and merge into one DataFrame.

    Labels:
        0  →  Fake news
        1  →  Real news

    Parameters
    ----------
    fake_path : str   Path to Fake.csv
    true_path : str   Path to True.csv

    Returns
    -------
    pd.DataFrame  Combined, shuffled DataFrame with a 'label' column.
    """
    print("[INFO] Loading datasets ...")

    # --- load ---
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    print(f"  Fake news rows : {len(fake_df):,}")
    print(f"  Real news rows : {len(true_df):,}")

    # --- attach labels ---
    fake_df["label"] = 0   # 0 = Fake
    true_df["label"] = 1   # 1 = Real

    # --- combine & shuffle ---
    combined = pd.concat([fake_df, true_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Total rows     : {len(combined):,}\n")
    return combined


# ---------------------------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Apply a full NLP cleaning pipeline to a single string:

      1. Lowercase
      2. Remove URLs
      3. Remove punctuation & digits
      4. Tokenise
      5. Remove stopwords
      6. Lemmatize

    Parameters
    ----------
    text : str   Raw news text

    Returns
    -------
    str   Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (http / https / www)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 4. Tokenise by whitespace
    tokens = text.split()

    # 5. Remove stopwords (also drop very short tokens)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

    # 6. Lemmatize
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the combined DataFrame for modelling:

      - Keep only 'text' and 'label' columns
      - Drop rows with missing text
      - Apply clean_text() to every article
      - Drop rows that become empty after cleaning

    Parameters
    ----------
    df : pd.DataFrame   Raw combined DataFrame

    Returns
    -------
    pd.DataFrame   Cleaned DataFrame with columns ['text', 'label', 'clean_text']
    """
    print("[INFO] Preprocessing text ...")

    # Keep essential columns
    df = df[["text", "label"]].copy()

    # Drop nulls
    before = len(df)
    df.dropna(subset=["text"], inplace=True)
    print(f"  Dropped {before - len(df)} rows with missing text.")

    # Clean text (this may take ~30 s on large datasets)
    df["clean_text"] = df["text"].apply(clean_text)

    # Drop rows that became empty strings after cleaning
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    print(f"  Rows after cleaning : {len(df):,}\n")
    return df
