# =============================================================================
# evaluation.py
# -----------------------------------------------------------------------------
# Model evaluation utilities: metrics, confusion matrix, comparison table.
# =============================================================================

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn           as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# 1. METRICS
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test_tfidf, y_test, model_name: str = "Model"):
    """
    Print and return evaluation metrics for a single trained model.

    Parameters
    ----------
    model         : fitted sklearn estimator
    X_test_tfidf  : sparse matrix   TF-IDF features for the test set
    y_test        : array-like      True labels
    model_name    : str             Label used in printouts

    Returns
    -------
    dict  Keys: accuracy, precision, recall, f1
    """
    y_pred = model.predict(X_test_tfidf)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=["Fake", "Real"])

    # Pretty-print
    separator = "=" * 55
    print(f"\n{separator}")
    print(f"  {model_name}  –  Evaluation Results")
    print(separator)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print()
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    return {
        "accuracy"  : round(acc * 100, 2),
        "precision" : round(report["weighted avg"]["precision"] * 100, 2),
        "recall"    : round(report["weighted avg"]["recall"]    * 100, 2),
        "f1"        : round(report["weighted avg"]["f1-score"]  * 100, 2),
    }


# ---------------------------------------------------------------------------
# 2. CONFUSION MATRIX PLOT
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model,
    X_test_tfidf,
    y_test,
    model_name : str = "Model",
    save_path  : str = None,
):
    """
    Plot a colour-coded confusion matrix for a single model.

    Parameters
    ----------
    model         : fitted sklearn estimator
    X_test_tfidf  : sparse matrix
    y_test        : array-like
    model_name    : str
    save_path     : str | None   If provided, save the figure to this path.
    """
    y_pred = model.predict(X_test_tfidf)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot        = True,
        fmt          = "d",
        cmap         = "Blues",
        xticklabels  = ["Fake", "Real"],
        yticklabels  = ["Fake", "Real"],
        linewidths   = 0.5,
        linecolor    = "white",
        ax           = ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(f"Confusion Matrix  –  {model_name}", fontsize=13, pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved → {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 3. MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------

def compare_models(results: dict, save_path: str = None):
    """
    Print and optionally save a side-by-side comparison of model metrics.

    Parameters
    ----------
    results   : dict   { model_name: {accuracy, precision, recall, f1} }
    save_path : str | None
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model"          : name,
            "Accuracy (%)"   : metrics["accuracy"],
            "Precision (%)"  : metrics["precision"],
            "Recall (%)"     : metrics["recall"],
            "F1-Score (%)"   : metrics["f1"],
        })

    comparison_df = pd.DataFrame(rows).set_index("Model")

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string())
    print("=" * 60 + "\n")

    # --- bar chart ---
    fig, ax = plt.subplots(figsize=(8, 4))
    comparison_df.plot(
        kind   = "bar",
        ax     = ax,
        color  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
        edgecolor = "white",
        width  = 0.55,
    )
    ax.set_title("Model Comparison – Key Metrics", fontsize=13, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(85, 102)
    ax.legend(loc="lower right", fontsize=9)
    ax.tick_params(axis="x", rotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Comparison chart saved → {save_path}")
    plt.show()
    plt.close()

    return comparison_df
