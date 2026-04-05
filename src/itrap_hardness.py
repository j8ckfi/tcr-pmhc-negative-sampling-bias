"""
itrap_hardness.py
=================
Characterizes the "hardness" of negative samples using ITRAP's 10x dextramer data.

ITRAP uniquely provides BOTH experimental non-binders (neg_control from 10x Genomics
dextramer assays) AND swapped negatives in one dataset. This lets us directly compare:

1. Biophysical distance distributions: experimental vs swapped negatives to positives
2. Per-negative-type model performance (AUC)
3. Cross-evaluation: train on one negative type, test on the other

Key question: Are swapped negatives systematically "easier" (farther in biophysical
space from positives) than experimental non-binders?
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features

ITRAP_PATH = REPO_ROOT / "data" / "itrap" / "ITRAP_benchmark" / "ITRAP_train.csv"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load and prepare ITRAP data
# ---------------------------------------------------------------------------

def load_itrap() -> pd.DataFrame:
    df = pd.read_csv(ITRAP_PATH)
    # Standardize columns for feature extraction
    df = df.rename(columns={"CDR3b": "CDR3b"})  # already correct
    # Label: binder column (1=positive, 0=negative)
    # Origin: NaN=positive, neg_control=experimental neg, swapped=swapped neg
    print(f"Loaded ITRAP: {len(df)} rows")
    print(f"  Positives: {(df['binder']==1).sum()}")
    print(f"  Neg control (experimental): {(df['origin']=='neg_control').sum()}")
    print(f"  Swapped negatives: {(df['origin']=='swapped').sum()}")
    print(f"  Peptides: {df['peptide'].nunique()} ({', '.join(df['peptide'].unique())})")
    return df


# ---------------------------------------------------------------------------
# Analysis 1: Biophysical distance distributions
# ---------------------------------------------------------------------------

def compute_distance_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute biophysical distance from each negative to nearest positive, per peptide."""
    print("\n--- Distance Distribution Analysis ---")

    records = []
    for peptide in sorted(df["peptide"].unique()):
        pdf = df[df["peptide"] == peptide]
        pos = pdf[pdf["binder"] == 1]
        neg_ctrl = pdf[(pdf["binder"] == 0) & (pdf["origin"] == "neg_control")]
        neg_swap = pdf[(pdf["binder"] == 0) & (pdf["origin"] == "swapped")]

        if len(pos) < 2 or (len(neg_ctrl) < 2 and len(neg_swap) < 2):
            print(f"  {peptide}: skipping (insufficient data)")
            continue

        # Extract biophysical features
        X_pos = extract_features(pos, "biophysical")

        for neg_type, neg_df in [("neg_control", neg_ctrl), ("swapped", neg_swap)]:
            if len(neg_df) < 2:
                continue
            X_neg = extract_features(neg_df, "biophysical")

            # Distance from each negative to nearest positive
            dists = cdist(X_neg, X_pos, metric="euclidean")
            min_dists = dists.min(axis=1)

            for d in min_dists:
                records.append({
                    "peptide": peptide,
                    "neg_type": neg_type,
                    "dist_to_nearest_pos": d,
                })

        print(f"  {peptide}: pos={len(pos)} neg_ctrl={len(neg_ctrl)} swapped={len(neg_swap)}")

    dist_df = pd.DataFrame(records)

    # Summary statistics
    print("\n  Distance summary (mean ± std):")
    for neg_type in ["neg_control", "swapped"]:
        subset = dist_df[dist_df["neg_type"] == neg_type]["dist_to_nearest_pos"]
        if len(subset) > 0:
            print(f"    {neg_type}: {subset.mean():.3f} ± {subset.std():.3f} (median={subset.median():.3f})")

    # Statistical test
    from scipy.stats import mannwhitneyu
    ctrl_dists = dist_df[dist_df["neg_type"] == "neg_control"]["dist_to_nearest_pos"]
    swap_dists = dist_df[dist_df["neg_type"] == "swapped"]["dist_to_nearest_pos"]
    if len(ctrl_dists) > 0 and len(swap_dists) > 0:
        U, p = mannwhitneyu(ctrl_dists, swap_dists, alternative="two-sided")
        print(f"\n  Mann-Whitney U test (ctrl vs swap): U={U:.0f}, p={p:.2e}")
        if swap_dists.mean() > ctrl_dists.mean():
            print("  --> Swapped negatives are FARTHER from positives (easier)")
        else:
            print("  --> Experimental negatives are FARTHER from positives (harder to distinguish)")

    return dist_df


# ---------------------------------------------------------------------------
# Analysis 2: Per-negative-type AUC
# ---------------------------------------------------------------------------

def per_negtype_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Train/test with each negative type separately using 5-fold CV."""
    print("\n--- Per-Negative-Type AUC ---")

    pos = df[df["binder"] == 1].copy()
    neg_ctrl = df[(df["binder"] == 0) & (df["origin"] == "neg_control")].copy()
    neg_swap = df[(df["binder"] == 0) & (df["origin"] == "swapped")].copy()

    records = []

    for neg_name, neg_df in [("neg_control", neg_ctrl), ("swapped", neg_swap)]:
        # Combine positives with this negative type
        combined = pd.concat([pos, neg_df]).reset_index(drop=True)
        y = combined["binder"].values

        for feat_type in ["biophysical", "sequence"]:
            print(f"\n  {neg_name} + {feat_type}:")
            X = extract_features(combined, feat_type)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for model_name, model_cls in [
                ("LR", lambda: LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs")),
                ("RF", lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
            ]:
                fold_aucs = []
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    model = model_cls()
                    model.fit(X[train_idx], y[train_idx])
                    proba = model.predict_proba(X[test_idx])[:, 1]
                    auc = roc_auc_score(y[test_idx], proba)
                    fold_aucs.append(auc)

                mean_auc = np.mean(fold_aucs)
                std_auc = np.std(fold_aucs)
                print(f"    {model_name}: {mean_auc:.4f} ± {std_auc:.4f}")
                records.append({
                    "neg_type": neg_name,
                    "feature_type": feat_type,
                    "model": model_name,
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                    "n_pos": len(pos),
                    "n_neg": len(neg_df),
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analysis 3: Cross-evaluation
# ---------------------------------------------------------------------------

def cross_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Train on one negative type, test on the other."""
    print("\n--- Cross-Evaluation ---")
    print("  Train on neg_type A + positives, test on neg_type B + positives")

    pos = df[df["binder"] == 1].copy()
    neg_ctrl = df[(df["binder"] == 0) & (df["origin"] == "neg_control")].copy()
    neg_swap = df[(df["binder"] == 0) & (df["origin"] == "swapped")].copy()

    records = []

    for feat_type in ["biophysical", "sequence"]:
        print(f"\n  Feature: {feat_type}")

        # Build train/test datasets
        # Split positives 80/20 to avoid test leakage
        rng = np.random.default_rng(42)
        pos_idx = rng.permutation(len(pos))
        n_train_pos = int(0.8 * len(pos))
        pos_train = pos.iloc[pos_idx[:n_train_pos]]
        pos_test = pos.iloc[pos_idx[n_train_pos:]]

        for train_neg_name, train_neg, test_neg_name, test_neg in [
            ("neg_control", neg_ctrl, "swapped", neg_swap),
            ("swapped", neg_swap, "neg_control", neg_ctrl),
        ]:
            # Also split negatives 80/20
            train_neg_sub = train_neg.sample(frac=0.8, random_state=42)
            test_neg_sub = test_neg.sample(frac=0.2, random_state=42)

            train_df = pd.concat([pos_train, train_neg_sub]).reset_index(drop=True)

            # Same-type test (sanity check)
            test_same_neg = train_neg.drop(train_neg_sub.index)
            test_same = pd.concat([pos_test, test_same_neg]).reset_index(drop=True)

            # Cross-type test
            test_cross = pd.concat([pos_test, test_neg_sub]).reset_index(drop=True)

            X_train = extract_features(train_df, feat_type)
            y_train = train_df["binder"].values
            X_test_same = extract_features(test_same, feat_type)
            y_test_same = test_same["binder"].values
            X_test_cross = extract_features(test_cross, feat_type)
            y_test_cross = test_cross["binder"].values

            for model_name, model_cls in [
                ("LR", lambda: LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs")),
                ("RF", lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
            ]:
                model = model_cls()
                model.fit(X_train, y_train)

                auc_same = roc_auc_score(y_test_same, model.predict_proba(X_test_same)[:, 1])
                auc_cross = roc_auc_score(y_test_cross, model.predict_proba(X_test_cross)[:, 1])

                drop = auc_same - auc_cross
                print(f"    Train={train_neg_name} {model_name}: same={auc_same:.3f} cross={auc_cross:.3f} drop={drop:+.3f}")

                records.append({
                    "feature_type": feat_type,
                    "model": model_name,
                    "train_neg": train_neg_name,
                    "test_neg_same": train_neg_name,
                    "test_neg_cross": test_neg_name,
                    "auc_same": auc_same,
                    "auc_cross": auc_cross,
                    "auc_drop": drop,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(dist_df: pd.DataFrame, auc_df: pd.DataFrame, cross_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Distance distributions
    ax = axes[0]
    for neg_type, color, label in [
        ("neg_control", "#2196F3", "Experimental\n(10x dextramer)"),
        ("swapped", "#FF5722", "Swapped\n(random pair)"),
    ]:
        subset = dist_df[dist_df["neg_type"] == neg_type]["dist_to_nearest_pos"]
        if len(subset) > 0:
            ax.hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel("Biophysical distance to nearest positive", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("A. Negative hardness distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Per-negative-type AUC (RF only for clarity)
    ax = axes[1]
    rf_data = auc_df[auc_df["model"] == "RF"]
    x = np.arange(2)
    width = 0.35
    colors = {"neg_control": "#2196F3", "swapped": "#FF5722"}
    labels_map = {"neg_control": "Experimental", "swapped": "Swapped"}

    for i, neg_type in enumerate(["neg_control", "swapped"]):
        subset = rf_data[rf_data["neg_type"] == neg_type]
        vals = [subset[subset["feature_type"] == ft]["mean_auc"].values[0] for ft in ["biophysical", "sequence"]]
        stds = [subset[subset["feature_type"] == ft]["std_auc"].values[0] for ft in ["biophysical", "sequence"]]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=labels_map[neg_type], color=colors[neg_type],
                      yerr=stds, capsize=4, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Biophysical", "Sequence"], fontsize=10)
    ax.set_ylabel("Mean AUC (5-fold CV)", fontsize=10)
    ax.set_title("B. AUC by negative type (RF)", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Cross-evaluation AUC drop
    ax = axes[2]
    rf_cross = cross_df[cross_df["model"] == "RF"]
    x_labels = []
    same_vals = []
    cross_vals = []
    for _, row in rf_cross.iterrows():
        x_labels.append(f"{row['feature_type'][:3].upper()}\ntrain={row['train_neg'][:4]}")
        same_vals.append(row["auc_same"])
        cross_vals.append(row["auc_cross"])

    x = np.arange(len(x_labels))
    width = 0.3
    ax.bar(x - width/2, same_vals, width, label="Same-type test", color="#4CAF50", alpha=0.85)
    ax.bar(x + width/2, cross_vals, width, label="Cross-type test", color="#9C27B0", alpha=0.85)

    for i, (s, c) in enumerate(zip(same_vals, cross_vals)):
        ax.text(i - width/2, s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width/2, c + 0.01, f"{c:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_title("C. Cross-evaluation (RF)", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("ITRAP: Experimental vs Swapped Negative Characterization",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig17_itrap_hardness.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("ITRAP Negative Hardness Characterization")
    print("=" * 60)

    df = load_itrap()

    # 1. Distance distributions
    dist_df = compute_distance_distributions(df)
    dist_df.to_csv(RESULTS_DIR / "itrap_distances.csv", index=False)

    # 2. Per-negative-type AUC
    auc_df = per_negtype_auc(df)
    auc_df.to_csv(RESULTS_DIR / "itrap_negtype_auc.csv", index=False)

    # 3. Cross-evaluation
    cross_df = cross_evaluation(df)
    cross_df.to_csv(RESULTS_DIR / "itrap_cross_eval.csv", index=False)

    # 4. Figures
    generate_figures(dist_df, auc_df, cross_df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ctrl_dist = dist_df[dist_df["neg_type"] == "neg_control"]["dist_to_nearest_pos"]
    swap_dist = dist_df[dist_df["neg_type"] == "swapped"]["dist_to_nearest_pos"]
    print(f"\nDistance to nearest positive:")
    print(f"  Experimental negs: {ctrl_dist.mean():.3f} ± {ctrl_dist.std():.3f}")
    print(f"  Swapped negs:      {swap_dist.mean():.3f} ± {swap_dist.std():.3f}")
    print(f"  Ratio (swap/ctrl): {swap_dist.mean()/ctrl_dist.mean():.2f}x")

    print(f"\nAUC by negative type (RF):")
    rf_auc = auc_df[auc_df["model"] == "RF"]
    for neg_type in ["neg_control", "swapped"]:
        for feat in ["biophysical", "sequence"]:
            row = rf_auc[(rf_auc["neg_type"] == neg_type) & (rf_auc["feature_type"] == feat)]
            if not row.empty:
                print(f"  {neg_type:12s} {feat:12s}: {row['mean_auc'].values[0]:.4f}")

    auc_swap_bio = rf_auc[(rf_auc["neg_type"]=="swapped") & (rf_auc["feature_type"]=="biophysical")]["mean_auc"].values[0]
    auc_ctrl_bio = rf_auc[(rf_auc["neg_type"]=="neg_control") & (rf_auc["feature_type"]=="biophysical")]["mean_auc"].values[0]
    print(f"\n  Biophysical AUC inflation (swapped - experimental): {auc_swap_bio - auc_ctrl_bio:+.4f}")

    print(f"\nCross-evaluation AUC drop (RF):")
    rf_cross = cross_df[cross_df["model"] == "RF"]
    for _, row in rf_cross.iterrows():
        print(f"  Train={row['train_neg']:12s} {row['feature_type']:12s}: "
              f"same={row['auc_same']:.3f} cross={row['auc_cross']:.3f} drop={row['auc_drop']:+.3f}")

    print("\nDone.")
