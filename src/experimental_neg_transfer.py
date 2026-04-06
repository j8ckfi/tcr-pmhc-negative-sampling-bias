"""
experimental_neg_transfer.py
============================
"Show the cure": Does training on experimental non-binders produce models
that generalize better than training on swapped negatives?

Design:
  - Dataset: ITRAP (has both experimental neg_control and swapped negatives)
  - Protocol: Leave-One-Peptide-Out (LOPO) on ITRAP's 4 peptides
  - Compare: train with experimental negs vs train with swapped negs
  - If experimental-neg-trained model has higher LOPO AUC, we've shown the fix

Also tests cross-dataset transfer:
  - Train on ITRAP (experimental or swapped), test on IMMREP22 TRUE SET (held-out)
  - Uses the untouched test partition, not training data, for transfer evaluation

Uncertainty:
  - Multi-seed negative resampling (10 seeds)
  - Paired Wilcoxon signed-rank test per peptide
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features
from immrep_loader import load_training_data

ITRAP_PATH = REPO_ROOT / "data" / "itrap" / "ITRAP_benchmark" / "ITRAP_train.csv"
IMMREP_TRUE_SET_DIR = (REPO_ROOT / "data" / "IMMREP_2022_TCRSpecificity" /
                       "IMMREP_2022_TCRSpecificity-main" / "true_set")

N_SEEDS = 10  # multi-seed for uncertainty
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def build_model(name: str, seed: int = 42):
    if name == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs",
                                       random_state=seed)),
        ])
    else:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=8,
                                            random_state=seed, n_jobs=-1)),
        ])


# ---------------------------------------------------------------------------
# Experiment 1: LOPO within ITRAP
# ---------------------------------------------------------------------------

def itrap_lopo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-One-Peptide-Out on ITRAP, comparing experimental vs swapped negatives.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: ITRAP Leave-One-Peptide-Out")
    print("=" * 60)

    peptides = sorted(df["peptide"].unique())
    pos = df[df["binder"] == 1]
    neg_ctrl = df[(df["binder"] == 0) & (df["origin"] == "neg_control")]
    neg_swap = df[(df["binder"] == 0) & (df["origin"] == "swapped")]

    records = []

    for held_out in peptides:
        print(f"\n  Held-out peptide: {held_out}")

        test_pos = pos[pos["peptide"] == held_out]
        test_neg_exp = neg_ctrl[neg_ctrl["peptide"] == held_out]

        if len(test_pos) < 5:
            print(f"    SKIP (test_pos={len(test_pos)})")
            continue

        for neg_name, neg_source in [("experimental", neg_ctrl), ("swapped", neg_swap)]:
            test_neg = neg_source[neg_source["peptide"] == held_out]

            if len(test_neg) < 5:
                print(f"    {neg_name}: SKIP (test_neg={len(test_neg)})")
                continue

            train_pos = pos[pos["peptide"] != held_out]
            train_neg_full = neg_source[neg_source["peptide"] != held_out]

            if len(train_neg_full) < 10:
                print(f"    {neg_name}: SKIP (train_neg too small)")
                continue

            for feat_type in ["biophysical", "sequence"]:
                # Pre-extract test features (constant across seeds)
                test_same = pd.concat([test_pos, test_neg]).reset_index(drop=True)
                X_test_same = extract_features(test_same, feat_type)
                y_test_same = test_same["binder"].values

                X_test_exp, y_test_exp = None, None
                if len(test_neg_exp) >= 5:
                    test_exp = pd.concat([test_pos, test_neg_exp]).reset_index(drop=True)
                    X_test_exp = extract_features(test_exp, feat_type)
                    y_test_exp = test_exp["binder"].values

                for model_name in ["LR", "RF"]:
                    seed_aucs_same = []
                    seed_aucs_exp = []

                    for seed in range(N_SEEDS):
                        # Resample negatives each seed
                        n_train_neg = min(len(train_neg_full), len(train_pos) * 2)
                        train_neg = train_neg_full.sample(n=n_train_neg, random_state=seed)
                        train_df = pd.concat([train_pos, train_neg]).reset_index(drop=True)

                        X_train = extract_features(train_df, feat_type)
                        y_train = train_df["binder"].values

                        model = build_model(model_name, seed=seed)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(X_train, y_train)

                        if len(np.unique(y_test_same)) >= 2:
                            auc_same = roc_auc_score(y_test_same,
                                                     model.predict_proba(X_test_same)[:, 1])
                        else:
                            auc_same = float("nan")
                        seed_aucs_same.append(auc_same)

                        auc_exp = float("nan")
                        if X_test_exp is not None and len(np.unique(y_test_exp)) >= 2:
                            auc_exp = roc_auc_score(y_test_exp,
                                                    model.predict_proba(X_test_exp)[:, 1])
                        seed_aucs_exp.append(auc_exp)

                    mean_same = np.nanmean(seed_aucs_same)
                    mean_exp = np.nanmean(seed_aucs_exp)
                    std_exp = np.nanstd(seed_aucs_exp)

                    records.append({
                        "experiment": "itrap_lopo",
                        "held_out_peptide": held_out,
                        "train_neg_type": neg_name,
                        "feature_type": feat_type,
                        "model": model_name,
                        "auc_same_neg": mean_same,
                        "auc_vs_experimental": mean_exp,
                        "auc_exp_std": std_exp,
                        "n_seeds": N_SEEDS,
                        "n_train": len(train_pos) + min(len(train_neg_full), len(train_pos) * 2),
                        "n_test": len(test_same),
                    })

                    print(f"    {neg_name:12s} {feat_type:12s} {model_name}: "
                          f"same={mean_same:.3f} exp={mean_exp:.3f}±{std_exp:.3f}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Experiment 2: Cross-dataset transfer (ITRAP → IMMREP22)
# ---------------------------------------------------------------------------

def load_immrep_true_set() -> pd.DataFrame:
    """Load IMMREP22 held-out TRUE SET (not training data) for transfer evaluation."""
    dfs = []
    for f in sorted(IMMREP_TRUE_SET_DIR.glob("*.txt")):
        if f.stem.startswith("testSet"):
            continue
        peptide = f.stem
        df = pd.read_csv(f, sep="\t")
        df["peptide"] = peptide
        df = df.rename(columns={"TRB_CDR3": "CDR3b", "Label": "label"})
        df = df[["CDR3b", "peptide", "label"]].dropna()
        # Add dummy CDR3a and mhc for feature extraction compatibility
        df["CDR3a"] = ""
        df["mhc"] = "HLA-A*02:01"
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded IMMREP22 TRUE SET: {len(combined)} rows, "
          f"{combined['peptide'].nunique()} peptides")
    return combined


def cross_dataset_transfer(itrap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train on ITRAP, test on IMMREP22 TRUE SET (held-out, untouched)
    for overlapping peptides. Multi-seed with paired testing.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Cross-Dataset Transfer (ITRAP -> IMMREP22 TRUE SET)")
    print("=" * 60)

    # Load IMMREP22 TRUE SET (held-out test partition)
    immrep_test = load_immrep_true_set()
    immrep_peptides = set(immrep_test["peptide"].unique())
    itrap_peptides = set(itrap_df["peptide"].unique())
    overlap = sorted(immrep_peptides & itrap_peptides)

    print(f"  ITRAP peptides: {sorted(itrap_peptides)}")
    print(f"  IMMREP22 true_set peptides: {sorted(immrep_peptides)}")
    print(f"  Overlap: {overlap}")

    if not overlap:
        print("  No overlapping peptides — skipping cross-dataset transfer")
        return pd.DataFrame()

    pos_itrap = itrap_df[itrap_df["binder"] == 1]
    neg_ctrl = itrap_df[(itrap_df["binder"] == 0) & (itrap_df["origin"] == "neg_control")]
    neg_swap = itrap_df[(itrap_df["binder"] == 0) & (itrap_df["origin"] == "swapped")]

    records = []

    for test_peptide in overlap:
        print(f"\n  Test peptide: {test_peptide}")

        test_df = immrep_test[immrep_test["peptide"] == test_peptide].reset_index(drop=True)

        if len(test_df) < 10 or len(np.unique(test_df["label"])) < 2:
            print(f"    SKIP (test={len(test_df)}, classes={len(np.unique(test_df['label']))})")
            continue

        print(f"    Test set: {len(test_df)} (pos={int((test_df['label']==1).sum())}, "
              f"neg={int((test_df['label']==0).sum())})")

        for neg_name, neg_source in [("experimental", neg_ctrl), ("swapped", neg_swap)]:
            train_pos = pos_itrap[pos_itrap["peptide"] != test_peptide]
            train_neg_full = neg_source[neg_source["peptide"] != test_peptide]

            if len(train_neg_full) < 10:
                print(f"    {neg_name}: SKIP (train_neg too small)")
                continue

            for feat_type in ["biophysical", "sequence"]:
                X_test = extract_features(test_df, feat_type)
                y_test = test_df["label"].values

                for model_name in ["LR", "RF"]:
                    seed_aucs = []

                    for seed in range(N_SEEDS):
                        n_train_neg = min(len(train_neg_full), len(train_pos) * 2)
                        train_neg = train_neg_full.sample(n=n_train_neg, random_state=seed)

                        train_pos_std = train_pos.copy()
                        train_neg_std = train_neg.copy()
                        train_pos_std["label"] = 1
                        train_neg_std["label"] = 0
                        train_df = pd.concat([train_pos_std, train_neg_std]).reset_index(drop=True)

                        X_train = extract_features(train_df, feat_type)
                        y_train = train_df["label"].values

                        if len(np.unique(y_train)) < 2:
                            continue

                        model = build_model(model_name, seed=seed)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(X_train, y_train)

                        proba = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, proba)
                        seed_aucs.append(auc)

                    mean_auc = np.mean(seed_aucs) if seed_aucs else float("nan")
                    std_auc = np.std(seed_aucs) if seed_aucs else float("nan")

                    records.append({
                        "experiment": "cross_dataset",
                        "test_peptide": test_peptide,
                        "train_neg_type": neg_name,
                        "feature_type": feat_type,
                        "model": model_name,
                        "auc": mean_auc,
                        "auc_std": std_auc,
                        "n_seeds": len(seed_aucs),
                        "n_train": len(train_pos) + min(len(train_neg_full), len(train_pos) * 2),
                        "n_test": len(test_df),
                    })

                    print(f"    {neg_name:12s} {feat_type:12s} {model_name}: "
                          f"AUC={mean_auc:.3f}±{std_auc:.3f}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(lopo_df: pd.DataFrame, transfer_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 1 + (1 if len(transfer_df) > 0 else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Panel A: ITRAP LOPO — experimental vs swapped (tested against experimental negs)
    ax = axes[0]
    rf_lopo = lopo_df[lopo_df["model"] == "RF"]
    for neg_type, color, label in [
        ("experimental", "#2196F3", "Train: experimental"),
        ("swapped", "#FF5722", "Train: swapped"),
    ]:
        subset = rf_lopo[rf_lopo["train_neg_type"] == neg_type]
        for feat_type, marker in [("biophysical", "o"), ("sequence", "s")]:
            feat_sub = subset[subset["feature_type"] == feat_type]
            if feat_sub.empty:
                continue
            peptides = feat_sub["held_out_peptide"].values
            aucs = feat_sub["auc_vs_experimental"].values
            x_pos = np.arange(len(peptides))
            offset = 0.15 if feat_type == "sequence" else -0.15
            neg_offset = 0.02 if neg_type == "swapped" else -0.02
            ax.scatter(x_pos + offset + neg_offset, aucs, marker=marker,
                      color=color, s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
                      label=f"{label} ({feat_type[:3]})" if feat_type == "biophysical" else None)

    # Mean bars
    for neg_type, color in [("experimental", "#2196F3"), ("swapped", "#FF5722")]:
        subset = rf_lopo[(rf_lopo["train_neg_type"] == neg_type) &
                         (rf_lopo["feature_type"] == "biophysical")]
        mean_auc = subset["auc_vs_experimental"].mean()
        ax.axhline(mean_auc, color=color, linestyle="--", alpha=0.5, linewidth=1.5)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 3, mean_auc + 0.01,
                f"mean={mean_auc:.3f}", color=color, fontsize=8, ha="right")

    peptide_labels = rf_lopo["held_out_peptide"].unique()
    ax.set_xticks(range(len(peptide_labels)))
    ax.set_xticklabels(peptide_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("LOPO AUC (tested vs experimental negs)", fontsize=10)
    ax.set_title("A. ITRAP LOPO: Experimental vs Swapped Training", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Cross-dataset transfer
    if len(transfer_df) > 0 and n_panels > 1:
        ax = axes[1]
        rf_transfer = transfer_df[transfer_df["model"] == "RF"]

        peptides = sorted(rf_transfer["test_peptide"].unique())
        x = np.arange(len(peptides))
        width = 0.35
        colors = {"experimental": "#2196F3", "swapped": "#FF5722"}

        for i, neg_type in enumerate(["experimental", "swapped"]):
            vals = []
            for pep in peptides:
                row = rf_transfer[(rf_transfer["test_peptide"] == pep) &
                                  (rf_transfer["train_neg_type"] == neg_type) &
                                  (rf_transfer["feature_type"] == "biophysical")]
                vals.append(row["auc"].values[0] if not row.empty else 0)
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                         label=f"Train: {neg_type}", color=colors[neg_type], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(peptides, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("AUC on IMMREP22 test", fontsize=10)
        ax.set_title("B. Cross-Dataset: ITRAP → IMMREP22", fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Does Training on Experimental Negatives Improve Generalization?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig19_experimental_neg_transfer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Experimental Negative Transfer Study")
    print("=" * 60)

    # Load ITRAP
    itrap = pd.read_csv(ITRAP_PATH)
    print(f"ITRAP: {len(itrap)} rows, {itrap['peptide'].nunique()} peptides")
    print(f"  Positives: {(itrap['binder']==1).sum()}")
    print(f"  Neg control: {(itrap['origin']=='neg_control').sum()}")
    print(f"  Swapped: {(itrap['origin']=='swapped').sum()}")

    # Experiment 1: ITRAP LOPO
    lopo_df = itrap_lopo(itrap)
    lopo_df.to_csv(RESULTS_DIR / "experimental_neg_lopo.csv", index=False)

    # Experiment 2: Cross-dataset transfer
    transfer_df = cross_dataset_transfer(itrap)
    if len(transfer_df) > 0:
        transfer_df.to_csv(RESULTS_DIR / "experimental_neg_transfer.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if len(lopo_df) > 0:
        print("\nITRAP LOPO (RF, tested vs experimental negs):")
        rf_lopo = lopo_df[(lopo_df["model"] == "RF") & (lopo_df["feature_type"] == "biophysical")]
        for neg_type in ["experimental", "swapped"]:
            subset = rf_lopo[rf_lopo["train_neg_type"] == neg_type]
            aucs = subset["auc_vs_experimental"].dropna()
            if len(aucs) > 0:
                print(f"  Train={neg_type:12s}: mean LOPO AUC = {aucs.mean():.4f} ± "
                      f"{aucs.std():.4f} (n={len(aucs)} peptides)")

        # Paired comparison
        exp_aucs = rf_lopo[rf_lopo["train_neg_type"]=="experimental"].sort_values("held_out_peptide")["auc_vs_experimental"].dropna().values
        swap_aucs = rf_lopo[rf_lopo["train_neg_type"]=="swapped"].sort_values("held_out_peptide")["auc_vs_experimental"].dropna().values

        if len(exp_aucs) >= 2 and len(swap_aucs) >= 2 and len(exp_aucs) == len(swap_aucs):
            diff = exp_aucs - swap_aucs
            print(f"\n  Paired per-peptide difference (exp - swap):")
            print(f"    Mean: {diff.mean():+.4f}, individual: {[f'{d:+.3f}' for d in diff]}")

            if len(exp_aucs) >= 4 and not np.all(diff == 0):
                stat, p = wilcoxon(exp_aucs, swap_aucs, alternative="greater")
                print(f"    Wilcoxon signed-rank (one-sided, exp > swap): stat={stat:.1f}, p={p:.4f}")
                if p < 0.05:
                    print("    --> SIGNIFICANT: experimental negs improve LOPO generalization")
                else:
                    print(f"    --> Not significant at p<0.05 (p={p:.4f})")
            else:
                print("    (Too few peptides for Wilcoxon test)")
        else:
            exp_mean = np.nanmean(exp_aucs) if len(exp_aucs) > 0 else float("nan")
            swap_mean = np.nanmean(swap_aucs) if len(swap_aucs) > 0 else float("nan")
            print(f"\n  Improvement from experimental negs: {exp_mean - swap_mean:+.4f}")

    if len(transfer_df) > 0:
        print("\nCross-dataset transfer (RF, biophysical):")
        rf_xfer = transfer_df[(transfer_df["model"] == "RF") &
                              (transfer_df["feature_type"] == "biophysical")]
        for neg_type in ["experimental", "swapped"]:
            subset = rf_xfer[rf_xfer["train_neg_type"] == neg_type]
            if len(subset) > 0:
                print(f"  Train={neg_type:12s}: mean AUC = {subset['auc'].mean():.4f}")

    # Generate figure
    generate_figure(lopo_df, transfer_df)

    print("\nDone.")
