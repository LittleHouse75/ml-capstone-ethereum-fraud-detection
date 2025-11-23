"""
Baseline classification models for the Ethereum scam detection project.

This module trains and evaluates a small suite of baseline classifiers on
pre-built address-level feature tables:

- Logistic Regression (on scaled features)
- Random Forest (on raw features)
- ExtraTrees (on raw features)
- XGBoost (on raw features, with scale_pos_weight)
- MLPClassifier (on scaled features)

It handles:
  - Train / validation split within the Train+Val feature table
  - Scaling for models that need it (linear / neural nets)
  - Per-model metrics on Train / Val / Test splits
  - Optional full diagnostic plots via src.model_eval.plot_all_evals
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import utilities as util
from src.model_eval import get_probas, plot_all_evals
from src.seed_util import SEED

# Make the project root importable when running notebooks from subfolders
sys.path.append(os.path.abspath(".."))


def run_baselines(
    features_trainval: pd.DataFrame,
    features_test: pd.DataFrame,
    target_col: str = "Scam",
    threshold: float = 0.5,
    random_state: int = SEED,
    verbose: bool = True,
    do_full_eval: bool = False,
    eval_split: str = "test",
):
    """
    Fit baseline models and compute metrics on Train / Val / Test splits.

    This function expects pre-engineered address-level feature tables
    for a Train+Val set and a held-out Test set. It then:

      1) Splits Train+Val into Train / Validation (stratified by target)
      2) Fits a StandardScaler on Train and applies it to models that
         require scaling (LogisticRegression, MLP)
      3) Trains the following baselines:
           - LogisticRegression (scaled)
           - RandomForestClassifier (raw)
           - ExtraTreesClassifier (raw)
           - XGBClassifier (raw, with scale_pos_weight)
           - MLPClassifier (scaled)
      4) Computes metrics on Train / Val / Test for each model:
           - accuracy, precision, recall, f1
           - roc_auc (on probabilities)
           - avg_precision (Average Precision)
           - confusion matrix
      5) Optionally runs full evaluation plots for a chosen split
         (ROC, PR, threshold curves, confusion, calibration).

    Parameters
    ----------
    features_trainval : pd.DataFrame
        Feature table for Train+Val addresses. Must include `target_col`.
    features_test : pd.DataFrame
        Feature table for Test addresses. Must include the same feature
        columns as `features_trainval` and `target_col`.
    target_col : str, default "Scam"
        Name of the binary target column (0/1).
    threshold : float, default 0.5
        Classification threshold used when converting probabilities to labels.
    random_state : int, default SEED
        Random seed for the train/val split and applicable models.
    verbose : bool, default True
        If True, prints headings and per-model validation metrics.
    do_full_eval : bool, default False
        If True, generates full diagnostic plots for each model on the
        split specified by `eval_split`.
    eval_split : {"train", "val", "test"}, default "test"
        Which split to use for the full evaluation plots.

    Returns
    -------
    results : list of dict
        One dict per model with keys:
          - "model_name" : str
          - "model"      : fitted estimator
          - "space"      : "scaled" or "raw"
          - "train"      : metrics dict
          - "val"        : metrics dict
          - "test"       : metrics dict
    train_table : pd.DataFrame
        Summary metrics per model on the train split.
    val_table : pd.DataFrame
        Summary metrics per model on the validation split.
    test_table : pd.DataFrame
        Summary metrics per model on the test split.
    scaler : StandardScaler
        Fitted StandardScaler instance (for reuse with linear/NN models).
    """

    # --------------------------------------------------------
    # 0. Separate features from target
    # --------------------------------------------------------
    feature_cols = [c for c in features_trainval.columns if c != target_col]

    X_trainval = features_trainval[feature_cols].copy()
    y_trainval = features_trainval[target_col].astype(int).values

    # Strict: raises if any feature is missing in test
    X_test = features_test[feature_cols].copy()
    y_test = features_test[target_col].astype(int).values

    # --------------------------------------------------------
    # 1. Train / Validation split (within Train+Val addresses)
    # --------------------------------------------------------
    if verbose:
        util.print_heading("Train / Validation Split (Within Train+Val Addresses)")

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=random_state,
    )

    if verbose:
        print("Train+Val shape:", X_trainval.shape)
        print("Test shape:     ", X_test.shape)
        print("Train+Val positives:", int(y_trainval.sum()))
        print("Train+Val negatives:", int((y_trainval == 0).sum()))
        print("Test positives:      ", int(y_test.sum()))
        print("Test negatives:      ", int((y_test == 0).sum()))
        print()
        print("Train size:", X_train.shape[0])
        print("Val size:  ", X_val.shape[0])
        print("Test size: ", X_test.shape[0])

    # --------------------------------------------------------
    # 2. Scaling (for linear / NN models)
    # --------------------------------------------------------
    if verbose:
        util.print_heading("Scaling Numeric Features (for Linear/NN Models)")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # 3. Baseline model definitions
    # --------------------------------------------------------
    if verbose:
        util.print_heading("Defining Baseline Models")

    baseline_models = []

    # Logistic Regression (scaled)
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )
    baseline_models.append(("LogisticRegression", log_reg, "scaled"))

    # Random Forest (raw)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )
    baseline_models.append(("RandomForest", rf, "raw"))

    # ExtraTrees (raw)
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )
    baseline_models.append(("ExtraTrees", et, "raw"))

    # XGBoost (raw, with scale_pos_weight for imbalance)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    baseline_models.append(("XGBoost", xgb, "raw"))

    if verbose:
        print("XGBoost baseline defined.")

    # MLP (scaled)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=50,
        early_stopping=True,
        random_state=random_state,
    )
    baseline_models.append(("MLP", mlp, "scaled"))

    # --------------------------------------------------------
    # 4. Split-level evaluation helper
    # --------------------------------------------------------
    def evaluate_split(y_true, y_prob, threshold: float = threshold):
        """
        Convert probabilities to labels at the given threshold and
        compute standard classification metrics.
        """
        y_pred = (y_prob >= threshold).astype(int)

        metrics = dict(
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

    # --------------------------------------------------------
    # 5. Run all models
    # --------------------------------------------------------
    if verbose:
        util.print_heading("Running Baseline Models")

    results = []

    for name, model, space in baseline_models:
        if space == "scaled":
            Xtr, Xv, Xte = X_train_scaled, X_val_scaled, X_test_scaled
        else:
            Xtr, Xv, Xte = X_train, X_val, X_test

        if verbose:
            util.print_sub_heading(f"Fitting {name}")

        model.fit(Xtr, y_train)

        if verbose:
            print("Scoring on Train / Val / Test...")

        prob_train = get_probas(model, Xtr)
        prob_val = get_probas(model, Xv)
        prob_test = get_probas(model, Xte)

        train_m = evaluate_split(y_train, prob_train)
        val_m = evaluate_split(y_val, prob_val)
        test_m = evaluate_split(y_test, prob_test)

        if verbose:
            print(f"{name} — Validation metrics:")
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]:
                print(f"  val_{k}: {val_m[k]:.4f}")
            print("Confusion matrix (val):")
            print(val_m["cm"])
            print()

        # Optional: full evaluation plots (ROC / PR / thresholds / confusion / calibration)
        if do_full_eval:
            if eval_split == "test":
                X_eval, y_eval = Xte, y_test
                split_label = "Test"
            elif eval_split == "val":
                X_eval, y_eval = Xv, y_val
                split_label = "Validation"
            else:  # "train"
                X_eval, y_eval = Xtr, y_train
                split_label = "Train"

            util.print_heading(f"Full Evaluation — {name} ({split_label} Split)")
            plot_all_evals(
                model,
                X_eval,
                y_eval,
                name=f"{name} — {split_label}",
                threshold=threshold,
            )

        results.append(
            {
                "model_name": name,
                "model": model,
                "space": space,
                "train": train_m,
                "val": val_m,
                "test": test_m,
            }
        )

    # --------------------------------------------------------
    # 6. Comparison tables (Train / Val / Test)
    # --------------------------------------------------------
    def _split_table(split: str) -> pd.DataFrame:
        """
        Build a summary metric table for a given split.

        Parameters
        ----------
        split : {"train", "val", "test"}
            Which metrics to pull from the `results` list.

        Returns
        -------
        pd.DataFrame
            One row per model with columns:
            ["model", "accuracy", "precision", "recall",
             "f1", "roc_auc", "avg_precision"],
            sorted by avg_precision (descending).
        """
        rows = []
        for r in results:
            m = r[split]
            rows.append(
                {
                    "model": r["model_name"],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                    "avg_precision": m["avg_precision"],
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("avg_precision", ascending=False)
            .reset_index(drop=True)
        )

    train_table = _split_table("train")
    val_table = _split_table("val")
    test_table = _split_table("test")

    return results, train_table, val_table, test_table, scaler