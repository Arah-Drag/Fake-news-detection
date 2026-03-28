# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# End-to-end pipeline for the Fake News Detection System.
#
# Run:
#   python main.py
#
# Prerequisites:
#   pip install -r requirements.txt
#   Place Fake.csv and True.csv inside the  data/  folder.
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all OSes
import matplotlib.pyplot as plt
import seaborn           as sns
from wordcloud           import WordCloud

# ---------------------------------------------------------------------------
# Add project root to path so sub-module imports work from any cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_data, preprocess_dataframe
from src.model         import (
    build_tfidf,
    split_data,
    train_logistic_regression,
    train_naive_bayes,
    save_artefacts,
)
from src.evaluation    import evaluate_model, plot_confusion_matrix, compare_models


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
FAKE_CSV   = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV   = os.path.join(DATA_DIR, "True.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR  = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ===========================================================================
# STEP 1 – DATA LOADING
# ===========================================================================
print("\n" + "█" * 60)
print("  STEP 1 : DATA LOADING")
print("█" * 60)

df = load_data(FAKE_CSV, TRUE_CSV)
print(df.head(3))
print("\nDataset shape :", df.shape)
print("Columns       :", list(df.columns))


# ===========================================================================
# STEP 2 – DATA PREPROCESSING
# ===========================================================================
print("\n" + "█" * 60)
print("  STEP 2 : DATA PREPROCESSING")
print("█" * 60)

df_clean = preprocess_dataframe(df)
print(df_clean[["text", "clean_text", "label"]].head(3).to_string(max_colwidth=80))


# ===========================================================================
# STEP 3 – EXPLORATORY DATA ANALYSIS (EDA)
# ===========================================================================
print("\n" + "█" * 60)
print("  STEP 3 : EXPLORATORY DATA ANALYSIS")
print("█" * 60)

# --- 3a. Class distribution (count) ---
label_counts = df_clean["label"].value_counts()
print("\nClass distribution:\n", label_counts.rename({0: "Fake", 1: "Real"}))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Bar chart
axes[0].bar(["Fake (0)", "Real (1)"],
            [label_counts[0], label_counts[1]],
            color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=0.8)
axes[0].set_title("Class Distribution", fontsize=13)
axes[0].set_ylabel("Number of Articles", fontsize=11)
for bar, val in zip(axes[0].patches, [label_counts[0], label_counts[1]]):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 150, f"{val:,}",
                 ha="center", fontsize=10)

# Pie chart
axes[1].pie([label_counts[0], label_counts[1]],
            labels=["Fake", "Real"],
            colors=["#e74c3c", "#2ecc71"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Class Proportion", fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[INFO] Class distribution plot saved.")

# --- 3b. Article length distribution ---
df_clean["text_length"] = df_clean["clean_text"].str.split().str.len()

fig, ax = plt.subplots(figsize=(9, 4))
for label, color, name in [(0, "#e74c3c", "Fake"), (1, "#2ecc71", "Real")]:
    subset = df_clean[df_clean["label"] == label]["text_length"]
    ax.hist(subset, bins=50, alpha=0.6, color=color, label=f"{name} (median={int(subset.median())})")
ax.set_title("Article Length Distribution (word count after cleaning)", fontsize=12)
ax.set_xlabel("Word Count", fontsize=11)
ax.set_ylabel("Frequency",  fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "length_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[INFO] Length distribution plot saved.")

# --- 3c. Word clouds ---
def make_wordcloud(text_series, title, color_func, save_path):
    combined_text = " ".join(text_series.dropna().tolist())
    wc = WordCloud(
        width            = 900,
        height           = 450,
        background_color = "white",
        colormap         = color_func,
        max_words        = 200,
        collocations     = False,
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Word cloud saved → {save_path}")

make_wordcloud(
    df_clean[df_clean["label"] == 0]["clean_text"],
    "Most Frequent Words in FAKE News",
    "Reds",
    os.path.join(PLOTS_DIR, "wordcloud_fake.png"),
)
make_wordcloud(
    df_clean[df_clean["label"] == 1]["clean_text"],
    "Most Frequent Words in REAL News",
    "Greens",
    os.path.join(PLOTS_DIR, "wordcloud_real.png"),
)


# ===========================================================================
# STEP 4 – FEATURE ENGINEERING (TF-IDF)
# ===========================================================================
print("\n" + "█" * 60)
print("  STEP 4 : FEATURE ENGINEERING")
print("█" * 60)

X_train, X_test, y_train, y_test = split_data(df_clean)
vectorizer = build_tfidf(X_train, max_features=50_000)

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"  TF-IDF matrix shape — Train : {X_train_tfidf.shape}")
print(f"  TF-IDF matrix shape — Test  : {X_test_tfidf.shape}\n")


# ===========================================================================
# STEP 5 – MODEL TRAINING
# ===========================================================================
print("\n" + "█" * 60)
print("  STEP 5 : MODEL TRAINING")
print("█" * 60)

lr_model = train_logistic_regression(X_train_tfidf, y_train)
nb_model = train_naive_bayes(X_train_tfidf, y_train)


# ===========================================================================
# STEP 6 – MODEL EVALUATION
# ===========================================================================
print("\n" + "█" * 60)
print("  STEP 6 : MODEL EVALUATION")
print("█" * 60)

lr_metrics = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")
nb_metrics = evaluate_model(nb_model, X_test_tfidf, y_test, "Naïve Bayes")

plot_confusion_matrix(
    lr_model, X_test_tfidf, y_test,
    model_name = "Logistic Regression",
    save_path  = os.path.join(PLOTS_DIR, "cm_logistic_regression.png"),
)
plot_confusion_matrix(
    nb_model, X_test_tfidf, y_test,
    model_name = "Naïve Bayes",
    save_path  = os.path.join(PLOTS_DIR, "cm_naive_bayes.png"),
)


# ===========================================================================
# STEP 7 – MODEL COMPARISON
# ===========================================================================
print("\n" + "█" * 60)
print("  STEP 7 : MODEL COMPARISON")
print("█" * 60)

results = {
    "Logistic Regression" : lr_metrics,
    "Naïve Bayes"         : nb_metrics,
}
compare_models(
    results,
    save_path = os.path.join(PLOTS_DIR, "model_comparison.png"),
)


# ===========================================================================
# STEP 8 – SAVE ARTEFACTS
# ===========================================================================
print("\n" + "█" * 60)
print("  STEP 8 : SAVING ARTEFACTS")
print("█" * 60)

save_artefacts(vectorizer, lr_model, nb_model, save_dir=MODELS_DIR)

print("\n✅  Pipeline complete!  All plots are in the 'plots/' folder.")
print("    To launch the web app run:  streamlit run app.py\n")
