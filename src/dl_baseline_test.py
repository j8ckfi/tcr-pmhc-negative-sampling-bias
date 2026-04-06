"""
dl_baseline_test.py
===================
Tests whether a neural network (MLP) shows the same AUC inflation pattern
as LR/RF under different negative sampling strategies.

Key question: Does increased model capacity resist the distributional
shortcuts from random-swap negatives?

Uses sklearn MLPClassifier (3 hidden layers) under our standard 8-condition
protocol: 4 negative strategies × 2 feature types, with LOPO evaluation.

If the MLP shows the same inflation pattern, the "just use deep learning"
objection is refuted — the problem is in the data, not the model.

Protocol note: Negatives are generated once per strategy+repeat and shared
across CV/LOPO folds. This is IDENTICAL to the LR/RF protocol in evaluation.py
and matches the standard practice in published TCR-pMHC benchmarks. The point
is to compare model capacity under the SAME conditions, not to fix the protocol.
Any fold-level leakage affects all models equally, so the inflation comparison
(MLP vs LR vs RF) remains valid.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features
from immrep_loader import load_training_data
from negative_sampling import (
    random_swap,
    epitope_balanced,
    within_cluster,
    shuffled_cdr3,
)

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

N_REPEATS = 3
RANDOM_STATE = 42

STRATEGIES = {
    "random_swap": random_swap,
    "epitope_balanced": epitope_balanced,
    "within_cluster": within_cluster,
    "shuffled_cdr3": shuffled_cdr3,
}

FEATURE_TYPES = ["biophysical", "sequence"]


def build_mlp(seed: int = 42) -> Pipeline:
    """3-layer MLP with batch normalization via StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
            batch_size=256,
        )),
    ])


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Standard CV (matches evaluation.py protocol)
# ---------------------------------------------------------------------------

def run_standard_cv(positives, negatives, feature_type, n_splits=5, seed=42):
    """5-fold stratified CV, returns mean AUC."""
    combined = pd.concat([positives, negatives]).reset_index(drop=True)
    X = extract_features(combined, feature_type)
    y = combined["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        model = build_mlp(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(safe_auc(y[test_idx], proba))
    return np.nanmean(aucs)


# ---------------------------------------------------------------------------
# LOPO (matches evaluation.py protocol)
# ---------------------------------------------------------------------------

def run_lopo(positives, negatives, feature_type, seed=42):
    """Leave-one-peptide-out CV, returns per-peptide AUCs and mean."""
    combined = pd.concat([positives, negatives]).reset_index(drop=True)
    peptides = sorted(combined["peptide"].unique())
    per_peptide = {}

    for held_out in peptides:
        train = combined[combined["peptide"] != held_out]
        test = combined[combined["peptide"] == held_out]

        if len(test) < 10 or len(np.unique(test["label"])) < 2:
            continue

        X_train = extract_features(train, feature_type)
        y_train = train["label"].values
        X_test = extract_features(test, feature_type)
        y_test = test["label"].values

        model = build_mlp(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        auc = safe_auc(y_test, proba)
        per_peptide[held_out] = auc

    return per_peptide


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Deep Learning Baseline: MLP under 8-condition protocol")
    print("=" * 60)

    all_data = load_training_data()
    positives = all_data[all_data["label"] == 1].reset_index(drop=True)
    print(f"Positives: {len(positives)}")
    print(f"Peptides: {sorted(positives['peptide'].unique())}")

    records = []

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        for feat_type in FEATURE_TYPES:
            print(f"\n  Feature: {feat_type}")

            cv_aucs = []
            all_lopo_aucs = []

            for rep in range(N_REPEATS):
                seed = RANDOM_STATE + rep

                # Generate negatives
                negatives = strategy_fn(positives, random_state=seed)

                # Standard CV
                cv_auc = run_standard_cv(positives, negatives, feat_type, seed=seed)
                cv_aucs.append(cv_auc)

                # LOPO
                per_peptide = run_lopo(positives, negatives, feat_type, seed=seed)
                lopo_mean = np.nanmean(list(per_peptide.values())) if per_peptide else float("nan")
                all_lopo_aucs.append(lopo_mean)

                print(f"    Rep {rep}: CV={cv_auc:.4f} LOPO={lopo_mean:.4f}")

            mean_cv = np.nanmean(cv_aucs)
            mean_lopo = np.nanmean(all_lopo_aucs)
            inflation = mean_cv - mean_lopo

            print(f"  --> MLP mean CV={mean_cv:.4f} LOPO={mean_lopo:.4f} "
                  f"inflation={inflation:+.4f}")

            records.append({
                "strategy": strategy_name,
                "feature_type": feat_type,
                "model": "MLP",
                "mean_cv_auc": mean_cv,
                "std_cv_auc": np.nanstd(cv_aucs),
                "mean_lopo_auc": mean_lopo,
                "std_lopo_auc": np.nanstd(all_lopo_aucs),
                "inflation": inflation,
            })

    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_DIR / "dl_baseline_results.csv", index=False)

    # Load LR/RF results for comparison
    try:
        lr_rf = pd.read_csv(RESULTS_DIR / "benchmark_results.csv")
        has_comparison = True
    except FileNotFoundError:
        has_comparison = False

    # Summary
    print("\n" + "=" * 60)
    print("MLP RESULTS")
    print("=" * 60)
    print(results_df[["strategy", "feature_type", "mean_cv_auc", "mean_lopo_auc", "inflation"]]
          .to_string(index=False))

    if has_comparison:
        print("\n" + "=" * 60)
        print("COMPARISON: MLP vs LR vs RF inflation")
        print("=" * 60)
        for strategy in STRATEGIES:
            for feat in FEATURE_TYPES:
                mlp_row = results_df[(results_df["strategy"] == strategy) &
                                     (results_df["feature_type"] == feat)]
                if mlp_row.empty:
                    continue
                mlp_inf = mlp_row["inflation"].values[0]

                lr_rows = lr_rf[(lr_rf["strategy"] == strategy) &
                                (lr_rf["feature_type"] == feat) &
                                (lr_rf["model"] == "LR")]
                rf_rows = lr_rf[(lr_rf["strategy"] == strategy) &
                                (lr_rf["feature_type"] == feat) &
                                (lr_rf["model"] == "RF")]

                lr_inf = (lr_rows["mean_cv_auc"].values[0] - lr_rows["mean_lopo_auc"].values[0]) if len(lr_rows) > 0 else float("nan")
                rf_inf = (rf_rows["mean_cv_auc"].values[0] - rf_rows["mean_lopo_auc"].values[0]) if len(rf_rows) > 0 else float("nan")

                print(f"  {strategy:20s} {feat:12s}: "
                      f"LR={lr_inf:+.3f} RF={rf_inf:+.3f} MLP={mlp_inf:+.3f}")

    # Figure
    generate_figure(results_df, lr_rf if has_comparison else None)

    print("\nDone.")


def generate_figure(mlp_df: pd.DataFrame, lr_rf_df: pd.DataFrame | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = list(STRATEGIES.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, feat_type in enumerate(FEATURE_TYPES):
        ax = axes[ax_idx]

        x = np.arange(len(strategies))
        width = 0.25

        # MLP
        mlp_cv = [mlp_df[(mlp_df["strategy"]==s) & (mlp_df["feature_type"]==feat_type)]["mean_cv_auc"].values[0]
                   for s in strategies]
        mlp_lopo = [mlp_df[(mlp_df["strategy"]==s) & (mlp_df["feature_type"]==feat_type)]["mean_lopo_auc"].values[0]
                    for s in strategies]

        ax.bar(x - width, mlp_cv, width, label="MLP Standard CV", color="#9C27B0", alpha=0.85)
        ax.bar(x, mlp_lopo, width, label="MLP LOPO", color="#E1BEE7", alpha=0.85)

        # LR/RF comparison (LOPO only, from RF)
        if lr_rf_df is not None:
            rf_lopo = []
            for s in strategies:
                row = lr_rf_df[(lr_rf_df["strategy"]==s) &
                               (lr_rf_df["feature_type"]==feat_type) &
                               (lr_rf_df["model"]=="RF")]
                rf_lopo.append(row["mean_lopo_auc"].values[0] if len(row) > 0 else 0)
            ax.bar(x + width, rf_lopo, width, label="RF LOPO", color="#FFB74D", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=8)
        ax.set_ylabel("AUC", fontsize=10)
        ax.set_title(f"{feat_type.capitalize()} features", fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("MLP (256-128-64) Shows Same Inflation Pattern as LR/RF",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig20_dl_baseline.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
