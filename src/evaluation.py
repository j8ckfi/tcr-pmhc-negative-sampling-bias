"""
evaluation.py
=============
Core evaluation engine for the negative-sampling bias benchmark.

Main entry point
----------------
    run_benchmark(df_positives, sampling_strategy_name, feature_type, n_repeats)

Returns a results dict containing:
  - auc_standard     : mean AUC under standard random-swap CV (optimistic)
  - auc_lopo         : mean AUC under leave-one-peptide-out CV (honest)
  - auc_inflation    : auc_standard - auc_lopo
  - per_peptide_auc  : {peptide: AUC} for the LOPO setting
  - feature_type     : str
  - sampling_strategy: str
  - model_lr_*       : sub-results for LogisticRegression
  - model_rf_*       : sub-results for RandomForestClassifier

Design notes
------------
- Standard CV uses stratified k-fold (k=5) on a dataset built with the
  chosen sampling strategy, repeated n_repeats times.
- LOPO CV iterates over peptides; for each fold the test set is the held-out
  peptide's positives paired with equal random-swap negatives drawn from
  training peptides.
- Both LR and RF are evaluated; the reported aggregates average across models.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features, FeatureType
from negative_sampling import (
    STRATEGY_REGISTRY,
    leave_one_peptide_out,
    random_swap,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_lr(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=500,
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=1,
        )),
    ])


def _build_rf(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        )),
    ])


_MODELS = {
    "lr": _build_lr,
    "rf": _build_rf,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return ROC-AUC; return NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    model_builder,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[float]:
    """
    Run stratified k-fold CV and return per-fold AUC scores.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        model = model_builder(random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(_safe_auc(y[test_idx], proba))
    return aucs


# ---------------------------------------------------------------------------
# Standard CV benchmark (random-swap and other fixed strategies)
# ---------------------------------------------------------------------------

def _standard_cv_benchmark(
    df_positives: pd.DataFrame,
    sampling_fn,
    feature_type: FeatureType,
    n_repeats: int = 5,
    n_splits:  int = 5,
) -> dict:
    """
    Build a balanced dataset with *sampling_fn*, extract features,
    and evaluate with stratified k-fold CV over *n_repeats* seeds.

    Returns dict with per-model mean/std AUC.
    """
    all_aucs: dict[str, list[float]] = {m: [] for m in _MODELS}

    for seed in range(n_repeats):
        df_balanced = sampling_fn(df_positives, random_state=seed)
        X = extract_features(df_balanced, feature_type=feature_type)
        y = df_balanced["label"].values

        for model_name, builder in _MODELS.items():
            fold_aucs = _cv_auc(X, y, builder, n_splits=n_splits, random_state=seed)
            valid = [a for a in fold_aucs if not np.isnan(a)]
            all_aucs[model_name].extend(valid)

    results: dict = {}
    combined_aucs: list[float] = []
    for model_name, aucs in all_aucs.items():
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        std_auc  = float(np.std(aucs))  if aucs else float("nan")
        results[f"model_{model_name}_mean_auc"] = mean_auc
        results[f"model_{model_name}_std_auc"]  = std_auc
        combined_aucs.extend(aucs)

    results["auc_standard_mean"] = float(np.mean(combined_aucs)) if combined_aucs else float("nan")
    results["auc_standard_std"]  = float(np.std(combined_aucs))  if combined_aucs else float("nan")
    return results


# ---------------------------------------------------------------------------
# LOPO benchmark
# ---------------------------------------------------------------------------

def _lopo_benchmark(
    df_positives: pd.DataFrame,
    feature_type: FeatureType,
) -> dict:
    """
    Leave-one-peptide-out evaluation.

    For each fold:
      - train on balanced data (random-swap) from all other peptides
      - test on held-out peptide positives + equal random-swap negatives
        drawn from other-peptide pool
    """
    per_peptide: dict[str, dict[str, float]] = {}

    for pep, train_df, test_pos in leave_one_peptide_out(df_positives):
        # Build test negatives from other peptides only
        other_positives = df_positives[df_positives["peptide"] != pep]
        if len(other_positives) == 0:
            logger.warning(f"LOPO: no other positives for test negatives (pep={pep}); skipping.")
            continue

        # Sample len(test_pos) negatives from other peptides for the test set
        n_test_neg = len(test_pos)
        test_neg_pool = random_swap(other_positives, random_state=0)
        test_neg_pool = test_neg_pool[test_neg_pool["label"] == 0]
        if len(test_neg_pool) == 0:
            logger.warning(f"LOPO: could not build test negatives for pep={pep}; skipping.")
            continue
        test_neg = test_neg_pool.sample(
            n=min(n_test_neg, len(test_neg_pool)),
            replace=len(test_neg_pool) < n_test_neg,
            random_state=0,
        )
        test_df = pd.concat([test_pos, test_neg], ignore_index=True)

        if len(test_df["label"].unique()) < 2:
            logger.warning(f"LOPO test set for pep={pep} has only one class; skipping.")
            continue

        X_train = extract_features(train_df,  feature_type=feature_type)
        X_test  = extract_features(test_df,   feature_type=feature_type)
        y_train = train_df["label"].values
        y_test  = test_df["label"].values

        pep_aucs: dict[str, float] = {}
        for model_name, builder in _MODELS.items():
            model = builder(random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pep_aucs[model_name] = _safe_auc(y_test, proba)

        per_peptide[pep] = pep_aucs
        logger.info(
            f"  LOPO pep={pep}: "
            + "  ".join(f"{k}={v:.3f}" for k, v in pep_aucs.items() if not np.isnan(v))
        )

    if not per_peptide:
        logger.warning("LOPO: no folds completed successfully.")
        return {
            "auc_lopo_mean": float("nan"),
            "auc_lopo_std":  float("nan"),
            "per_peptide_auc": {},
        }

    # Aggregate
    all_lopo_aucs: list[float] = []
    per_pep_combined: dict[str, float] = {}
    model_lopo: dict[str, list[float]] = {m: [] for m in _MODELS}

    for pep, aucs in per_peptide.items():
        valid = [v for v in aucs.values() if not np.isnan(v)]
        if valid:
            per_pep_combined[pep] = float(np.mean(valid))
            all_lopo_aucs.extend(valid)
        for m, v in aucs.items():
            if not np.isnan(v):
                model_lopo[m].append(v)

    results: dict = {
        "auc_lopo_mean":    float(np.mean(all_lopo_aucs)) if all_lopo_aucs else float("nan"),
        "auc_lopo_std":     float(np.std(all_lopo_aucs))  if all_lopo_aucs else float("nan"),
        "per_peptide_auc":  per_pep_combined,
    }
    for model_name, aucs in model_lopo.items():
        results[f"model_{model_name}_lopo_mean"] = float(np.mean(aucs)) if aucs else float("nan")
        results[f"model_{model_name}_lopo_std"]  = float(np.std(aucs))  if aucs else float("nan")

    return results


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def run_benchmark(
    df_positives: pd.DataFrame,
    sampling_strategy_name: str,
    feature_type: FeatureType,
    n_repeats: int = 5,
) -> dict:
    """
    Run the full benchmark for one (sampling strategy, feature type) condition.

    Parameters
    ----------
    df_positives : pd.DataFrame
        DataFrame containing only positive TCR-pMHC pairs (label == 1).
        Must have columns: CDR3b, peptide, label (and optionally CDR3a, mhc).
    sampling_strategy_name : str
        One of: 'random_swap', 'epitope_balanced', 'within_cluster',
                'shuffled_cdr3', 'leave_one_peptide_out'.
    feature_type : str
        One of: 'sequence', 'biophysical'.
    n_repeats : int
        Number of random seeds for the standard CV component (default 5).

    Returns
    -------
    dict with keys:
        auc_standard      – mean AUC under standard CV with this strategy
        auc_lopo          – mean AUC under LOPO CV (random-swap negatives)
        auc_inflation     – auc_standard - auc_lopo  (bias measure)
        per_peptide_auc   – {peptide: AUC} from LOPO
        feature_type      – echoed back
        sampling_strategy – echoed back
        model_lr_*        – LogisticRegression sub-results
        model_rf_*        – RandomForestClassifier sub-results
    """
    logger.info(
        f"run_benchmark: strategy={sampling_strategy_name}  "
        f"features={feature_type}  n_repeats={n_repeats}"
    )

    # --- Standard CV component ---
    if sampling_strategy_name == "leave_one_peptide_out":
        # For LOPO strategy, use random_swap for the standard CV comparison
        # (LOPO itself is not a dataset-building strategy but a CV scheme)
        sampling_fn = random_swap
    elif sampling_strategy_name in STRATEGY_REGISTRY:
        sampling_fn = STRATEGY_REGISTRY[sampling_strategy_name]
    else:
        raise ValueError(
            f"Unknown sampling strategy: '{sampling_strategy_name}'. "
            f"Choose from: {list(STRATEGY_REGISTRY.keys()) + ['leave_one_peptide_out']}"
        )

    std_results  = _standard_cv_benchmark(
        df_positives, sampling_fn, feature_type, n_repeats=n_repeats
    )
    lopo_results = _lopo_benchmark(df_positives, feature_type)

    auc_standard = std_results["auc_standard_mean"]
    auc_lopo     = lopo_results["auc_lopo_mean"]
    auc_inflation = (
        auc_standard - auc_lopo
        if not (np.isnan(auc_standard) or np.isnan(auc_lopo))
        else float("nan")
    )

    return {
        # Top-level summary
        "sampling_strategy": sampling_strategy_name,
        "feature_type":      feature_type,
        "auc_standard":      auc_standard,
        "auc_standard_std":  std_results["auc_standard_std"],
        "auc_lopo":          auc_lopo,
        "auc_lopo_std":      lopo_results["auc_lopo_std"],
        "auc_inflation":     auc_inflation,
        "per_peptide_auc":   lopo_results["per_peptide_auc"],
        # Per-model standard CV
        "model_lr_standard_mean": std_results.get("model_lr_mean_auc", float("nan")),
        "model_lr_standard_std":  std_results.get("model_lr_std_auc",  float("nan")),
        "model_rf_standard_mean": std_results.get("model_rf_mean_auc", float("nan")),
        "model_rf_standard_std":  std_results.get("model_rf_std_auc",  float("nan")),
        # Per-model LOPO
        "model_lr_lopo_mean": lopo_results.get("model_lr_lopo_mean", float("nan")),
        "model_lr_lopo_std":  lopo_results.get("model_lr_lopo_std",  float("nan")),
        "model_rf_lopo_mean": lopo_results.get("model_rf_lopo_mean", float("nan")),
        "model_rf_lopo_std":  lopo_results.get("model_rf_lopo_std",  float("nan")),
    }
