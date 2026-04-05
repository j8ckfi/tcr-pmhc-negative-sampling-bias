"""
tchard_comparison.py
====================
Tests whether the directional AUC inflation finding from IMMREP_2022 replicates
on the TChard dataset.

Hypothesis: random-swap negatives inflate biophysical AUC and deflate sequence AUC
relative to experimental (assay) negatives.

Expected directional pattern:
  biophysical AUC(sampled_negs) > biophysical AUC(neg_assays)  [positive diff]
  sequence    AUC(sampled_negs) < sequence    AUC(neg_assays)  [negative diff]
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features

TCHARD_DIR = REPO_ROOT / "data" / "tcr_h" / "tc-hard" / "tc-hard"
SPLITS_DIR = TCHARD_DIR / "ds.hard-splits" / "pep+cdr3b"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FOLDS = 5
NEG_TYPES = ["only-neg-assays", "only-sampled-negs"]
FEATURE_TYPES = ["biophysical", "sequence"]
MAX_TRAIN_BIOPHYS = 20_000   # biophysical features are fast
MAX_TRAIN_SEQUENCE = 10_000  # sequence features (8900-d) are slow

MODELS = {
    "LR":  LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs"),
    "RF":  RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(neg_type: str, split: str, fold: int) -> pd.DataFrame:
    """Load one CSV split and rename columns for feature extraction."""
    path = SPLITS_DIR / split / neg_type / f"{split}-{fold}.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"cdr3.beta": "CDR3b", "antigen.epitope": "peptide"})
    # Drop rows missing CDR3b or peptide
    df = df.dropna(subset=["CDR3b", "peptide"])
    df["label"] = df["label"].astype(float)
    return df


def stratified_subsample(df: pd.DataFrame, max_n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Stratified subsample up to max_n rows, preserving label ratio."""
    if len(df) <= max_n:
        return df
    pos = df[df["label"] == 1.0]
    neg = df[df["label"] == 0.0]
    ratio = len(pos) / len(df)
    n_pos = int(round(max_n * ratio))
    n_neg = max_n - n_pos
    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))
    sampled_pos = pos.sample(n=n_pos, random_state=int(rng.integers(0, 2**31)))
    sampled_neg = neg.sample(n=n_neg, random_state=int(rng.integers(0, 2**31)))
    return pd.concat([sampled_pos, sampled_neg]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run():
    rng = np.random.default_rng(42)
    records = []

    for neg_type in NEG_TYPES:
        print(f"\n{'='*60}")
        print(f"Negative type: {neg_type}")
        print(f"{'='*60}")

        for feature_type in FEATURE_TYPES:
            max_train = MAX_TRAIN_SEQUENCE if feature_type == "sequence" else MAX_TRAIN_BIOPHYS
            print(f"\n  Feature type: {feature_type} (max train={max_train})")

            fold_aucs: dict[str, list[float]] = {m: [] for m in MODELS}

            for fold in range(N_FOLDS):
                print(f"    Fold {fold}...", end=" ", flush=True)

                # Load data
                train_df = load_split(neg_type, "train", fold)
                test_df  = load_split(neg_type, "test",  fold)

                # Subsample train
                train_df = stratified_subsample(train_df, max_train, rng)

                print(f"train={len(train_df)} test={len(test_df)}", end=" ", flush=True)

                # Extract features
                X_train = extract_features(train_df, feature_type)
                X_test  = extract_features(test_df,  feature_type)
                y_train = train_df["label"].values
                y_test  = test_df["label"].values

                # Check we have both classes
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    print("SKIP (single class)")
                    continue

                # Train and evaluate each model
                model_results = []
                for model_name, model_proto in MODELS.items():
                    # Clone model to avoid state leakage
                    import copy
                    model = copy.deepcopy(model_proto)
                    model.fit(X_train, y_train)
                    proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, proba)
                    fold_aucs[model_name].append(auc)
                    records.append({
                        "fold": fold,
                        "neg_type": neg_type,
                        "feature_type": feature_type,
                        "model": model_name,
                        "auc": auc,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                    })
                    model_results.append(f"{model_name}={auc:.3f}")
                print(" | ".join(model_results), flush=True)

            # Fold summary
            for model_name in MODELS:
                aucs = fold_aucs[model_name]
                if aucs:
                    print(f"    [{model_name}] mean AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    return records


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def analyse(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "tchard_comparison.csv", index=False)
    print(f"\nResults saved to results/tchard_comparison.csv")

    # Mean AUC per (neg_type × feature_type × model)
    summary = (
        df.groupby(["neg_type", "feature_type", "model"])["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["neg_type", "feature_type", "model", "mean_auc", "std_auc", "n_folds"]

    print("\n" + "="*70)
    print("MEAN AUC BY CONDITION")
    print("="*70)
    print(summary.to_string(index=False))

    # Directional analysis: sampled_negs - neg_assays
    print("\n" + "="*70)
    print("AUC DIFFERENCE (sampled_negs - neg_assays)")
    print("="*70)

    pivot = summary.pivot_table(
        index=["feature_type", "model"],
        columns="neg_type",
        values="mean_auc"
    ).reset_index()

    # Handle possible column name variations
    if "only-sampled-negs" in pivot.columns and "only-neg-assays" in pivot.columns:
        pivot["diff_sampled_minus_assay"] = pivot["only-sampled-negs"] - pivot["only-neg-assays"]
    else:
        print("WARNING: Could not compute diff — missing columns in pivot")
        pivot["diff_sampled_minus_assay"] = np.nan

    print(pivot[["feature_type", "model", "only-neg-assays", "only-sampled-negs", "diff_sampled_minus_assay"]].to_string(index=False))

    return summary, pivot


def check_directional_bias(pivot: pd.DataFrame) -> bool:
    """
    Directional bias confirmed if:
      - biophysical diff > 0 (sampled negs inflate biophysical AUC)
      - sequence diff <= 0 (sampled negs deflate or don't inflate sequence AUC)
    for at least one model.
    """
    confirmed_models = []
    all_models = pivot["model"].unique()

    print("\n" + "="*70)
    print("DIRECTIONAL BIAS CHECK")
    print("="*70)

    for model in all_models:
        biophys_row = pivot[(pivot["feature_type"] == "biophysical") & (pivot["model"] == model)]
        seq_row     = pivot[(pivot["feature_type"] == "sequence")    & (pivot["model"] == model)]

        if biophys_row.empty or seq_row.empty:
            print(f"  [{model}] insufficient data")
            continue

        biophys_diff = biophys_row["diff_sampled_minus_assay"].values[0]
        seq_diff     = seq_row["diff_sampled_minus_assay"].values[0]

        biophys_inflated = biophys_diff > 0
        seq_deflated     = seq_diff < 0

        print(f"  [{model}] biophysical diff={biophys_diff:+.4f} ({'INFLATED' if biophys_inflated else 'not inflated'})  "
              f"sequence diff={seq_diff:+.4f} ({'deflated' if seq_deflated else 'NOT deflated'})")

        if biophys_inflated and seq_deflated:
            confirmed_models.append(model)

    # Also allow: biophysical inflated AND sequence less inflated (partial confirmation)
    # Count overall directional signal
    biophys_diffs = pivot[pivot["feature_type"] == "biophysical"]["diff_sampled_minus_assay"].values
    seq_diffs     = pivot[pivot["feature_type"] == "sequence"]["diff_sampled_minus_assay"].values

    biophys_pos = np.sum(biophys_diffs > 0)
    seq_neg     = np.sum(seq_diffs < 0)

    print(f"\n  Biophysical models with positive diff: {biophys_pos}/{len(biophys_diffs)}")
    print(f"  Sequence models with negative diff:    {seq_neg}/{len(seq_diffs)}")

    confirmed = len(confirmed_models) > 0
    return confirmed, confirmed_models


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(summary: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter to RF model for the main figure
    rf_data = summary[summary["model"] == "RF"].copy()

    # Pivot for grouped bar chart
    pivot = rf_data.pivot_table(
        index="feature_type",
        columns="neg_type",
        values="mean_auc"
    ).reset_index()

    neg_types_present = [c for c in ["only-neg-assays", "only-sampled-negs"] if c in pivot.columns]
    feature_types = pivot["feature_type"].tolist()

    x = np.arange(len(feature_types))
    width = 0.35
    colors = {"only-neg-assays": "#2196F3", "only-sampled-negs": "#FF5722"}
    labels = {"only-neg-assays": "Experimental negatives\n(assay)", "only-sampled-negs": "Swapped negatives\n(sampled)"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, neg_type in enumerate(neg_types_present):
        offset = (i - 0.5) * width
        vals = pivot[neg_type].values
        # Get std from summary for error bars
        stds = []
        for ft in feature_types:
            row = summary[(summary["model"] == "RF") &
                          (summary["neg_type"] == neg_type) &
                          (summary["feature_type"] == ft)]
            stds.append(row["std_auc"].values[0] if not row.empty else 0)
        bars = ax.bar(x + offset, vals, width,
                      label=labels.get(neg_type, neg_type),
                      color=colors.get(neg_type, None),
                      yerr=stds, capsize=4, alpha=0.85)
        # Label bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Feature type", fontsize=12)
    ax.set_ylabel("Mean AUC (5-fold)", fontsize=12)
    ax.set_title("TChard: AUC by Negative Type and Feature Type\n(Random Forest, 5-fold)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([ft.capitalize() for ft in feature_types], fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (AUC=0.5)")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig15_tchard_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to results/figures/fig15_tchard_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("TChard Directional Bias Replication Study")
    print("=" * 60)
    print(f"Data: {SPLITS_DIR}")
    print(f"Folds: {N_FOLDS}  |  Neg types: {NEG_TYPES}")
    print(f"Feature types: {FEATURE_TYPES}  |  Models: {list(MODELS.keys())}")

    records = run()

    if not records:
        print("\nERROR: No results collected. Check data paths.")
        sys.exit(1)

    summary, pivot = analyse(records)
    confirmed, confirmed_models = check_directional_bias(pivot)

    print("\n" + "="*70)
    if confirmed:
        print("CONCLUSION: Directional bias CONFIRMED on TChard")
        print(f"  Pattern seen in models: {', '.join(confirmed_models)}")
        print("  Biophysical AUC is higher with swapped negatives (inflation).")
        print("  Sequence AUC is lower with swapped negatives (deflation).")
        print("  Finding generalizes beyond IMMREP_2022.")
    else:
        # Check partial confirmation
        biophys_diffs = pivot[pivot["feature_type"] == "biophysical"]["diff_sampled_minus_assay"].values
        seq_diffs     = pivot[pivot["feature_type"] == "sequence"]["diff_sampled_minus_assay"].values
        if np.any(biophys_diffs > 0) and not np.any(seq_diffs < 0):
            print("CONCLUSION: Directional bias PARTIALLY CONFIRMED on TChard")
            print("  Biophysical inflation present but sequence deflation not observed.")
        elif not np.any(biophys_diffs > 0):
            print("CONCLUSION: Directional bias NOT CONFIRMED on TChard")
            print("  Biophysical AUC not consistently higher with swapped negatives.")
        else:
            print("CONCLUSION: Directional bias NOT CONFIRMED on TChard")
            print("  Pattern does not match IMMREP_2022 finding.")
    print("="*70)

    generate_figure(summary)
    print("\nDone.")
