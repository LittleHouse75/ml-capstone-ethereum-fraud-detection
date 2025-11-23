"""
Hyperparameter tuning utilities for the Ethereum scam detection project.

This module provides:
  - A multi-model RandomizedSearchCV pipeline (`run_tuning`) for tree-based models
  - A narrower, XGBoost-only search (`run_xgb_narrow`)
  - Shared helpers for computing metrics and building comparison tables

Both entrypoints assume:
  - Pre-engineered Train+Val and Test feature tables
  - A binary target column (default: "Scam")
"""

import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import utilities as util
from src.model_eval import get_probas, plot_all_evals
from src.seed_util import SEED

# Make the project root importable when running notebooks from subfolders
sys.path.append(os.path.abspath(".."))


# ============================================================
# Small metric helpers (shared by both tuning entrypoints)
# ============================================================
def _evaluate_split(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
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


_METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]


def _row_from_metrics(model_name: str, m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a single summary row from a metric dict.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "XGBoost").
    m : dict
        Metrics dictionary as returned by `_evaluate_split`.

    Returns
    -------
    dict
        Row with columns ["model"] + _METRIC_KEYS.
    """
    return {"model": model_name, **{k: m[k] for k in _METRIC_KEYS}}


# ============================================================
# Search space definitions
# ============================================================
def _build_search_spaces(scale_pos_weight: float, random_state: int = SEED) -> Dict[str, Dict[str, Any]]:
    """
    Construct default RandomizedSearchCV spaces for the main tree models.

    Parameters
    ----------
    scale_pos_weight : float
        Class-imbalance weight used for XGBoost's `scale_pos_weight`.
    random_state : int, default SEED
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary keyed by model name ("XGBoost", "RandomForest", "ExtraTrees"),
        each containing:
          - "model": base estimator
          - "params": parameter distributions
          - "scaled": whether to use scaled inputs
          - "n_iter": number of RandomizedSearchCV iterations
    """
    return {
        "XGBoost": {
            "model": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": randint(300, 900),
                "max_depth": randint(3, 12),
                "learning_rate": uniform(0.005, 0.05),
                "min_child_weight": randint(1, 8),
                "subsample": uniform(0.6, 0.4),        # 0.6–1.0
                "colsample_bytree": uniform(0.6, 0.4),  # 0.6–1.0
                "gamma": uniform(0.0, 0.5),
                "scale_pos_weight": [
                    1.0,
                    np.sqrt(scale_pos_weight),
                    scale_pos_weight,
                    scale_pos_weight * 2,
                ],
            },
            "scaled": False,
            "n_iter": 45,
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": randint(300, 800),
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": randint(2, 12),
                "min_samples_leaf": randint(1, 6),
                "max_features": ["sqrt", 0.5],
                "bootstrap": [True, False],
            },
            "scaled": False,
            "n_iter": 25,
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": randint(300, 800),
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": randint(2, 12),
                "min_samples_leaf": randint(1, 6),
                "max_features": ["sqrt", 0.5],
            },
            "scaled": False,
            "n_iter": 25,
        },
    }


# ============================================================
# Public entrypoint: multi-model tuning
# ============================================================
def run_tuning(
    features_trainval: pd.DataFrame,
    features_test: pd.DataFrame,
    target_col: str = "Scam",
    threshold: float = 0.5,
    random_state: int = SEED,
    verbose: bool = True,
    do_full_eval: bool = True,
    eval_split: str = "test",   # "train" | "val" | "test"
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], StandardScaler]:
    """
    Run RandomizedSearchCV for a set of tree models on address-level features.

    Parameters
    ----------
    features_trainval : pandas.DataFrame
        Feature table for the Train+Val pool (contains `target_col`).
    features_test : pandas.DataFrame
        Feature table for the held-out Test split (contains `target_col`).
    target_col : str, default "Scam"
        Name of the binary target column.
    threshold : float, default 0.5
        Decision threshold used when computing metrics.
    random_state : int, default SEED
        Random seed for reproducibility.
    verbose : bool, default True
        If True, print progress and summary information.
    do_full_eval : bool, default True
        If True, produce evaluation plots (`plot_all_evals`) for tuned models.
    eval_split : {"train", "val", "test"}, default "test"
        Which split to use for full evaluation plots.

    Returns
    -------
    best_models : dict
        Mapping from model name to the tuned estimator.
    train_df : pandas.DataFrame
        Per-model metrics on the training split.
    val_df : pandas.DataFrame
        Per-model metrics on the validation split.
    test_df : pandas.DataFrame
        Per-model metrics on the test split.
    search_spaces : dict
        The search space configuration used for each model.
    scaler : StandardScaler
        Fitted scaler (used only if a scaled model is later added).
    """
    # ----------------------------------------
    # 1. Basic split + scaling
    # ----------------------------------------
    feature_cols = [c for c in features_trainval.columns if c != target_col]

    X_trainval = features_trainval[feature_cols].copy()
    y_trainval = features_trainval[target_col].astype(int).values

    # Reindex test to be strict about feature alignment
    X_test = features_test.reindex(columns=feature_cols).copy()
    y_test = features_test[target_col].astype(int).values

    if verbose:
        util.print_heading("Train / Validation Split for Tuning")

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

    if verbose:
        util.print_heading("Scaling (for any scaled models, if added later)")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------------
    # 2. Search spaces
    # ----------------------------------------
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / max(pos, 1)

    if verbose:
        util.print_heading("Defining Random Search Spaces")

    search_spaces = _build_search_spaces(spw, random_state=random_state)

    # ----------------------------------------
    # 3. RandomizedSearchCV loops
    # ----------------------------------------
    if verbose:
        util.print_heading("Running RandomizedSearchCV")

    best_models: Dict[str, Any] = {}
    tuned_rows = []

    for name, cfg in search_spaces.items():
        if verbose:
            util.print_sub_heading(f"Tuning {name}")

        model = cfg["model"]
        param_dist = cfg["params"]
        n_iter = cfg["n_iter"]

        Xtr = X_train_scaled if cfg["scaled"] else X_train
        Xv = X_val_scaled if cfg["scaled"] else X_val
        Xte = X_test_scaled if cfg["scaled"] else X_test

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="average_precision",
            n_jobs=-1,
            cv=3,
            verbose=2 if verbose else 0,
            random_state=random_state,
            refit=True,
        )

        search.fit(Xtr, y_train)

        if verbose:
            print(f"\nBest AP Score (CV): {search.best_score_:.4f}")
            print("Best Params:")
            for k, v in search.best_params_.items():
                print(f"  {k}: {v}")

        best_model = search.best_estimator_
        best_models[name] = best_model

        # Eval on train / val / test
        prob_train = get_probas(best_model, Xtr)
        prob_val = get_probas(best_model, Xv)
        prob_test = get_probas(best_model, Xte)

        train_m = _evaluate_split(y_train, prob_train, threshold=threshold)
        val_m = _evaluate_split(y_val, prob_val, threshold=threshold)
        test_m = _evaluate_split(y_test, prob_test, threshold=threshold)

        tuned_rows.append(
            {
                "model": name,
                **{f"train_{k}": train_m[k] for k in _METRIC_KEYS},
                **{f"val_{k}": val_m[k] for k in _METRIC_KEYS},
                **{f"test_{k}": test_m[k] for k in _METRIC_KEYS},
            }
        )

    tuned_df = pd.DataFrame(tuned_rows)

    # --- build 3 separate metric tables ---
    def _make_split_df(split: str) -> pd.DataFrame:
        """
        Extract metrics for a single split ("train" / "val" / "test")
        from the combined tuned_df table.
        """
        cols = ["model"] + [f"{split}_{k}" for k in _METRIC_KEYS]
        rename_map = {f"{split}_{k}": k for k in _METRIC_KEYS}
        df_split = tuned_df[cols].rename(columns=rename_map)
        df_split = df_split.sort_values("avg_precision", ascending=False).reset_index(drop=True)
        return df_split

    train_df = _make_split_df("train")
    val_df = _make_split_df("val")
    test_df = _make_split_df("test")

    # ----------------------------------------
    # 4. Optional full evaluation plots
    # ----------------------------------------
    if do_full_eval:
        util.print_heading("Detailed Evaluation Plots — Tuned Models")

        split_map_X = {
            "train": (X_train, X_train_scaled),
            "val": (X_val, X_val_scaled),
            "test": (X_test, X_test_scaled),
        }
        split_map_y = {"train": y_train, "val": y_val, "test": y_test}

        X_raw, X_scaled = split_map_X[eval_split]
        y_eval = split_map_y[eval_split]
        split_label = eval_split.capitalize()

        for name, model in best_models.items():
            cfg = search_spaces[name]
            X_eval = X_scaled if cfg["scaled"] else X_raw

            util.print_sub_heading(f"Plots for Tuned {name} ({split_label})")
            plot_all_evals(
                model,
                X_eval,
                y_eval,
                name=f"Tuned {name} — {split_label}",
                threshold=threshold,
            )

    return best_models, train_df, val_df, test_df, search_spaces, scaler


# ============================================================
# Narrow XGBoost tuning
# ============================================================
def run_xgb_narrow(
    features_trainval: pd.DataFrame,
    features_test: pd.DataFrame,
    target_col: str = "Scam",
    threshold: float = 0.5,
    random_state: int = SEED,
    verbose: bool = True,
    do_full_eval: bool = True,
    eval_split: str = "test",   # "train" | "val" | "test"
) -> Tuple[XGBClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame, RandomizedSearchCV]:
    """
    Narrow RandomizedSearchCV focused only on XGBoost, using a tighter search space.

    Parameters
    ----------
    features_trainval : pandas.DataFrame
        Feature table for the Train+Val pool (contains `target_col`).
    features_test : pandas.DataFrame
        Feature table for the held-out Test split (contains `target_col`).
    target_col : str, default "Scam"
        Name of the binary target column.
    threshold : float, default 0.5
        Decision threshold used when computing metrics.
    random_state : int, default SEED
        Random seed for reproducibility.
    verbose : bool, default True
        If True, print progress and summary information.
    do_full_eval : bool, default True
        If True, produce evaluation plots (`plot_all_evals`) for the tuned model.
    eval_split : {"train", "val", "test"}, default "test"
        Which split to use for full evaluation plots.

    Returns
    -------
    best_xgb : XGBClassifier
        Tuned XGBoost model from the narrow search.
    train_df : pandas.DataFrame
        Single-row metrics for the training split.
    val_df : pandas.DataFrame
        Single-row metrics for the validation split.
    test_df : pandas.DataFrame
        Single-row metrics for the test split.
    xgb_search : RandomizedSearchCV
        The fitted RandomizedSearchCV object (for inspection).
    """
    if verbose:
        util.print_heading("XGBoost Narrow Hyperparameter Search")

    # ----------------------------------------
    # 1. Train/Val/Test split (raw features only)
    # ----------------------------------------
    feature_cols = [c for c in features_trainval.columns if c != target_col]

    X_trainval = features_trainval[feature_cols].copy()
    y_trainval = features_trainval[target_col].astype(int).values

    X_test = features_test.reindex(columns=feature_cols).copy()
    y_test = features_test[target_col].astype(int).values

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

    # ----------------------------------------
    # 2. Narrow XGBoost search space
    # ----------------------------------------
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / max(pos, 1)

    param_dist_xgb = {
        "max_depth": randint(6, 11),           # 6–10
        "learning_rate": uniform(0.03, 0.12),  # 0.03–0.15
        "n_estimators": randint(300, 601),     # 300–600
        "subsample": uniform(0.7, 0.3),        # 0.7–1.0
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0.0, 1.0),
        "reg_alpha": uniform(0.0, 0.5),
        "reg_lambda": uniform(0.5, 1.5),
        "min_child_weight": randint(1, 11),    # 1–10
    }

    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=spw,
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist_xgb,
        n_iter=40,
        scoring="average_precision",
        n_jobs=-1,
        cv=3,
        verbose=2 if verbose else 0,
        random_state=random_state,
        refit=True,
    )

    if verbose:
        print("Running narrowed XGBoost random search...")

    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    if verbose:
        util.print_sub_heading("Best Parameters (Narrow Search)")
        print(xgb_search.best_params_)

    # ----------------------------------------
    # 3. Metrics on train/val/test
    # ----------------------------------------
    prob_train = get_probas(best_xgb, X_train)
    prob_val = get_probas(best_xgb, X_val)
    prob_test = get_probas(best_xgb, X_test)

    train_m = _evaluate_split(y_train, prob_train, threshold=threshold)
    val_m = _evaluate_split(y_val, prob_val, threshold=threshold)
    test_m = _evaluate_split(y_test, prob_test, threshold=threshold)

    train_df = pd.DataFrame([_row_from_metrics("XGBoost (Narrow)", train_m)])
    val_df = pd.DataFrame([_row_from_metrics("XGBoost (Narrow)", val_m)])
    test_df = pd.DataFrame([_row_from_metrics("XGBoost (Narrow)", test_m)])

    if verbose:
        util.print_sub_heading("Performance of Tuned XGBoost (Narrow Search)")
        print("Train:")
        print(train_df)
        print("\nValidation:")
        print(val_df)
        print("\nTest:")
        print(test_df)

    # ----------------------------------------
    # 4. Optional full eval plots
    # ----------------------------------------
    if do_full_eval:
        util.print_heading("Detailed Evaluation Plots — Tuned Narrow-Search XGBoost")

        if eval_split == "train":
            X_eval, y_eval, label = X_train, y_train, "Train"
        elif eval_split == "val":
            X_eval, y_eval, label = X_val, y_val, "Validation"
        else:
            X_eval, y_eval, label = X_test, y_test, "Test"

        plot_all_evals(
            best_xgb,
            X_eval,
            y_eval,
            name=f"Tuned XGBoost (Narrow Search) — {label}",
            threshold=threshold,
        )

    return best_xgb, train_df, val_df, test_df, xgb_search