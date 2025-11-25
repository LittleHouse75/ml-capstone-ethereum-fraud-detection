"""
Model evaluation utilities for the Ethereum scam detection project.

This module provides:
  - A helper to consistently get class-1 probabilities from any sklearn-style model
  - Plotting helpers: ROC, PR, threshold curves, confusion matrix, calibration
  - A one-call "plot_all_evals" convenience wrapper
  - A sweep_thresholds utility to evaluate many decision thresholds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from typing import Any, Dict, Tuple

from src.seed_util import SEED


# ============================================================
# Core helper: always get class-1 probabilities
# ============================================================
def get_probas(model, X):
    """
    Return probability for the positive class (1) for a wide range of models.

    Priority:
      1) predict_proba(...)[1]
      2) decision_function(...) rescaled to [0, 1]
      3) predict(...) cast to float
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        s_min, s_max = scores.min(), scores.max()
        # Min-max scale decision scores into [0, 1] as a pseudo-probability
        return (scores - s_min) / (s_max - s_min + 1e-9)

    # Fallback: use hard predictions as "probabilities"
    return model.predict(X).astype(float)


# ============================================================
# ROC Curve
# ============================================================
def plot_roc_curve(model, X, y, title):
    """
    Plot ROC curve and report ROC AUC for a given model + dataset.
    """
    prob = get_probas(model, X)
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Precision–Recall Curve
# ============================================================
def plot_pr_curve(model, X, y, title):
    """
    Plot Precision–Recall curve and report Average Precision (AP).
    """
    prob = get_probas(model, X)
    precision, recall, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Threshold Sweeps (Precision / Recall / F1 vs Threshold)
# ============================================================
def plot_threshold_curves(model, X, y, title):
    """
    Plot how Precision, Recall, and F1 change as the decision threshold moves.
    """
    prob = get_probas(model, X)
    thresholds = np.linspace(0.0, 1.0, 200)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        pred = (prob >= t).astype(int)
        precisions.append(precision_score(y, pred, zero_division=0))
        recalls.append(recall_score(y, pred, zero_division=0))
        f1s.append(f1_score(y, pred, zero_division=0))

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions, label="Precision", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall", linewidth=2)
    plt.plot(thresholds, f1s, label="F1", linewidth=2)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Confusion Matrix Heatmap
# ============================================================
def plot_confusion_heatmap(model, X, y, threshold, title):
    """
    Plot a confusion-matrix heatmap at a chosen decision threshold.
    """
    prob = get_probas(model, X)
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred)

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# ============================================================
# Calibration Curve
# ============================================================
def plot_calibration_curve(model, X, y, title):
    """
    Plot a calibration curve: predicted probability vs. empirical fraction of positives.
    """
    prob = get_probas(model, X)
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=12)

    plt.figure(figsize=(5.5, 5))
    # x = mean predicted prob in each bin, y = empirical fraction of positives
    plt.plot(mean_pred, frac_pos, "o-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# One-call master function
# ============================================================
def plot_all_evals(model, X, y, name="Model", threshold=0.5):
    """
    Run all evaluation plots + basic metrics for a given model and dataset.

    Prints:
      - Accuracy / Precision / Recall / F1 at the chosen threshold
      - ROC AUC and Average Precision on the underlying score distribution

    Produces:
      - ROC curve
      - Precision–Recall curve
      - Threshold vs (Precision, Recall, F1) curves
      - Confusion-matrix heatmap
      - Calibration curve
    """
    print(f"\n=== Evaluation for: {name} ===")
    print(f"threshold = {threshold}")

    prob = get_probas(model, X)
    pred = (prob >= threshold).astype(int)

    # Core thresholded metrics
    print(f"Accuracy:  {accuracy_score(y, pred):.4f}")
    print(f"Precision: {precision_score(y, pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y, pred, zero_division=0):.4f}")

    # Ranking metrics (threshold-independent)
    try:
        print(f"ROC AUC:   {roc_auc_score(y, prob):.4f}")
    except ValueError:
        print("ROC AUC:   n/a (labels may be constant)")

    try:
        print(f"Avg Precision (AP): {average_precision_score(y, prob):.4f}")
    except ValueError:
        print("Avg Precision: n/a (labels may be constant)")

    # Plots
    plot_roc_curve(model, X, y, f"ROC — {name}")
    plot_pr_curve(model, X, y, f"Precision–Recall — {name}")
    plot_threshold_curves(model, X, y, f"Threshold Curves — {name}")
    plot_confusion_heatmap(model, X, y, threshold, f"Confusion Matrix — {name}")
    plot_calibration_curve(model, X, y, f"Calibration — {name}")


# ============================================================
# Threshold sweep table (for business-rule selection)
# ============================================================
def sweep_thresholds(y_true, y_prob, thresholds=None):
    """
    Sweep over a grid of thresholds and compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    thresholds : iterable of float, optional
        Thresholds to evaluate. If None, use np.linspace(0.01, 0.99, 99).

    Returns
    -------
    pandas.DataFrame
        One row per threshold with:
        accuracy, precision, recall, f1, roc_auc, avg_precision, base_rate.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    base_rate = y_true.mean()

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc = np.nan

        try:
            ap = average_precision_score(y_true, y_prob)
        except ValueError:
            ap = np.nan

        rows.append(
            {
                "threshold": thr,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc,
                "avg_precision": ap,
                "base_rate": base_rate,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Comprehensive threshold evaluation
# ============================================================
def eval_at_threshold(y_true, y_prob, threshold: float):
    """Compute classification + ranking metrics at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    m = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # Ranking metrics on probabilities
    try:
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        m["roc_auc"] = np.nan

    try:
        m["avg_precision"] = average_precision_score(y_true, y_prob)
    except ValueError:
        m["avg_precision"] = np.nan

    m["base_rate"] = float(y_true.mean())
    m["cm"] = confusion_matrix(y_true, y_pred)
    return m


def precision_first_threshold_tuning(
    model,
    X_trainval,
    y_trainval,
    X_test,
    y_test,
    prec_min: float = 0.75,
    seed: int = SEED,
    name: str | None = None,
):
    """
    Split Train/Val from Train+Val, sweep thresholds on Val,
    pick the highest-recall threshold with precision >= prec_min,
    then evaluate on Test.
    """
    # 1) Train/Val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=seed,
    )

    print("Val positives:", int(y_val.sum()), "Val negatives:", int((y_val == 0).sum()))
    print("Test positives:", int(y_test.sum()), "Test negatives:", int((y_test == 0).sum()))

    # 2) Sweep thresholds on validation
    prob_val = get_probas(model, X_val)
    val_sweep_df = sweep_thresholds(
        y_true=y_val,
        y_prob=prob_val,
        thresholds=np.linspace(0.01, 0.99, 99),
    )

    # 3) Apply business rule
    candidates = val_sweep_df[val_sweep_df["precision"] >= prec_min].copy()
    if len(candidates) == 0:
        print(f"No thresholds reached precision >= {prec_min:.2f}; falling back to best F1.")
        best_row = val_sweep_df.sort_values("f1", ascending=False).iloc[0]
    else:
        best_row = (
            candidates
            .sort_values(["recall", "f1"], ascending=[False, False])
            .iloc[0]
        )

    best_threshold = float(best_row["threshold"])
    print("\nChosen threshold (from validation):", best_threshold)
    print(best_row[["threshold", "precision", "recall", "f1", "accuracy"]])

    # 4) Evaluate on test
    prob_test = get_probas(model, X_test)
    test_metrics = eval_at_threshold(y_test, prob_test, best_threshold)

    print("\n=== Final Test Metrics at Chosen Threshold ===")
    for k, v in test_metrics.items():
        if k == "cm":
            continue
        print(f"{k}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"{k}: {v}")

    print("\nConfusion matrix (Test):")
    print(test_metrics["cm"])

    # 5) Plots
    if name is None:
        name = f"Final Tuned Model (threshold={best_threshold:.3f})"

    plot_all_evals(
        model,
        X_test,
        y_test,
        name=name,
        threshold=best_threshold,
    )

    return best_threshold, val_sweep_df, test_metrics

# ============================================================
# Split evaluation
# ============================================================
def evaluate_split(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute classification metrics for a given probability vector and threshold.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold used to convert probabilities into labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1,
        roc_auc, avg_precision, and the confusion matrix (cm).
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, Any] = dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan

    try:
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["avg_precision"] = np.nan

    metrics["cm"] = confusion_matrix(y_true, y_pred)
    return metrics


def print_metrics_block(label, m):
    """
    Pretty-print a block of classification metrics.

    Parameters
    ----------
    label : str
        Title for the metrics block (e.g. "SGAN — Train").
    m : Mapping[str, Any]
        Dictionary of metric names to values. Scalar metrics are
        printed as floats with 4 decimal places. If the key "cm"
        is present, its value is treated as a confusion matrix and
        printed on its own lines without float formatting.
    """
    print(f"\n{label}")
    print("-" * len(label))
    for k, v in m.items():
        if k == "cm":
            print(f"{k:>14}:")
            print(v)
        else:
            print(f"{k:>14}: {v:0.4f}")

# ============================================================
# SHAP-based explainability for tree models (e.g., XGBoost)
# ============================================================

def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_background: int = 200,
    max_samples: int = 500,
    random_state: int = 42,
):
    """
    Compute SHAP values using KernelExplainer (model-agnostic, slower but robust).

    NOTE ON WHY WE USE KernelExplainer INSTEAD OF TreeExplainer
    ----------------------------------------------------------------
    We *originally* used shap.TreeExplainer for XGBoost models because it's much
    faster and works directly on the tree ensemble. However, with the versions
    of XGBoost + SHAP in this environment, TreeExplainer crashes when it tries
    to read XGBoost's internal `base_score`:

        ValueError: could not convert string to float: '[1.3987352E-1]'

    Newer XGBoost stores `base_score` as a JSON-style array string (e.g. 
    "[1.3987352E-1]"), while SHAP expects a simple float ("0.13987352") and
    calls `float(...)` on it. That mismatch makes TreeExplainer unusable here.

    To avoid version-specific hacks against XGBoost internals, we use
    shap.KernelExplainer instead. It is slower (model-agnostic approximation),
    but stable and compatible with our current library versions.

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X : pd.DataFrame
        Feature matrix to explain.
    max_background : int
        Max rows to use as background distribution.
    max_samples : int
        Max rows to actually explain.
    random_state : int
        Seed for reproducible sampling.

    Returns
    -------
    shap_values_pos : np.ndarray
        SHAP values for the positive (scam) class.
    X_sample : pd.DataFrame
        The (subsampled) feature matrix used for SHAP.
    """
    # Ensure DataFrame for convenient sampling
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X

    n_rows = len(X_df)
    if n_rows == 0:
        raise ValueError("X has zero rows; cannot compute SHAP values.")

    # Background sample (for KernelExplainer)
    bg_size = min(max_background, n_rows)
    background = X_df.sample(bg_size, random_state=random_state)

    # Sample points to explain
    sample_size = min(max_samples, n_rows)
    X_sample = X_df.sample(sample_size, random_state=random_state + 1)

    # Model function that returns positive-class probability
    def model_f(batch):
        batch_arr = np.array(batch)
        proba = model.predict_proba(batch_arr)
        # positive class is column 1
        return proba[:, 1]

    explainer = shap.KernelExplainer(model_f, background.values)

    # nsamples="auto" is slower but usually OK with our small sample_size
    shap_values = explainer.shap_values(X_sample.values, nsamples="auto")

    # KernelExplainer returns an array (n_samples, n_features) for binary setup
    # If it's a list, take the positive class.
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    return shap_values_pos, X_sample


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    feature_names=None,
    max_samples: int = 5000,
    random_state: int = 42,
    show_bar: bool = True,
    show_beeswarm: bool = True,
):
    """
    Summary plots showing which features drive the positive (scam) class.

    Produces:
      - Bar plot of mean |SHAP| per feature
      - Beeswarm plot of SHAP values vs feature values
    """
    shap_values, X_sample = compute_shap_values(
        model=model,
        X=X,
        max_samples=max_samples,
        random_state=random_state,
    )

    if isinstance(X_sample, pd.DataFrame):
        data_for_plot = X_sample
        feats = X_sample.columns if feature_names is None else feature_names
    else:
        data_for_plot = X_sample
        feats = feature_names

    if show_bar:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values,
            data_for_plot,
            feature_names=feats,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.show()

    if show_beeswarm:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values,
            data_for_plot,
            feature_names=feats,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.show()
