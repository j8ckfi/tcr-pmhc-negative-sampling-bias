"""
tchard_tripeptide_baseline.py
=============================
Runs a trivial 3-mer (tripeptide) baseline on TChard data to demonstrate that:
1. Easy splits (random CV) yield high AUC even with trivial features
2. Hard splits (peptide+CDR3b held out) collapse to chance
3. The gap proves that TCR-H's claimed AUC is driven by data leakage, not binding logic

Compares: random CV (easy) vs peptide-held-out (hard) splits,
          with randomized negatives vs experimental negatives.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features

TCHARD_DIR = REPO_ROOT / "data" / "tcr_h" / "tc-hard" / "tc-hard"
SPLITS_DIR = TCHARD_DIR / "ds.hard-splits" / "pep+cdr3b"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

MAX_TRAIN = 10_000
MAX_TEST = 10_000
N_FOLDS = 5

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs"),
    "RF": lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
}


def load_full_dataset() -> pd.DataFrame:
    """Load the full TChard dataset."""
    df = pd.read_csv(TCHARD_DIR / "ds.csv", low_memory=False)
    df = df.rename(columns={"cdr3.beta": "CDR3b", "antigen.epitope": "peptide"})
    df = df.dropna(subset=["CDR3b", "peptide"])
    df["label"] = df["label"].astype(float)
    return df


def subsample(df: pd.DataFrame, max_n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= max_n:
        return df
    pos = df[df["label"] == 1.0]
    neg = df[df["label"] == 0.0]
    ratio = len(pos) / len(df)
    n_pos = min(int(round(max_n * ratio)), len(pos))
    n_neg = min(max_n - n_pos, len(neg))
    sampled = pd.concat([
        pos.sample(n=n_pos, random_state=int(rng.integers(0, 2**31))),
        neg.sample(n=n_neg, random_state=int(rng.integers(0, 2**31))),
    ])
    return sampled.reset_index(drop=True)


def load_hard_split(neg_type: str, split: str, fold: int) -> pd.DataFrame:
    path = SPLITS_DIR / split / neg_type / f"{split}-{fold}.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"cdr3.beta": "CDR3b", "antigen.epitope": "peptide"})
    df = df.dropna(subset=["CDR3b", "peptide"])
    df["label"] = df["label"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Experiment 1: Easy splits (random CV) on full dataset
# ---------------------------------------------------------------------------

def run_easy_splits(df_full: pd.DataFrame, neg_type_filter: str) -> list[dict]:
    """Random 5-fold CV (easy) — peptides leak across folds."""
    print(f"\n{'='*60}")
    print(f"EASY SPLITS (random CV) — neg_type={neg_type_filter}")
    print(f"{'='*60}")

    # Filter by negative source
    if neg_type_filter == "randomized":
        df = df_full[(df_full["label"] == 1.0) |
                     (df_full["negative.source"] == "randomized")].copy()
    else:  # experimental: nettcr-2.0, iedb, mira
        df = df_full[(df_full["label"] == 1.0) |
                     (df_full["negative.source"].isin(["nettcr-2.0", "iedb", "mira"]))].copy()

    print(f"  Total: {len(df)} (pos={int((df['label']==1).sum())}, neg={int((df['label']==0).sum())})")

    # Subsample for speed
    rng = np.random.default_rng(42)
    df_sub = subsample(df, MAX_TRAIN * 2, rng)
    y = df_sub["label"].values

    records = []
    for feat_type in ["sequence"]:  # 3-mer is within sequence features
        print(f"\n  Feature: {feat_type}")
        X = extract_features(df_sub, feat_type)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for model_name, model_fn in MODELS.items():
            fold_aucs = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                model = model_fn()
                model.fit(X[train_idx], y[train_idx])
                proba = model.predict_proba(X[test_idx])[:, 1]
                auc = roc_auc_score(y[test_idx], proba)
                fold_aucs.append(auc)
                print(f"    Fold {fold}: {model_name}={auc:.3f}", flush=True)

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            print(f"    [{model_name}] mean={mean_auc:.4f} ± {std_auc:.4f}")
            records.append({
                "split_type": "easy_random_cv",
                "neg_type": neg_type_filter,
                "feature_type": feat_type,
                "model": model_name,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
            })

    return records


# ---------------------------------------------------------------------------
# Experiment 2: Hard splits (peptide+CDR3b held out) — from pre-built splits
# ---------------------------------------------------------------------------

def run_hard_splits(neg_type: str) -> list[dict]:
    """Use TChard's pre-built hard splits."""
    print(f"\n{'='*60}")
    print(f"HARD SPLITS (peptide+CDR3b held out) — neg_type={neg_type}")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    records = []

    for feat_type in ["sequence"]:
        print(f"\n  Feature: {feat_type}")
        for model_name, model_fn in MODELS.items():
            fold_aucs = []
            for fold in range(N_FOLDS):
                print(f"    Fold {fold}...", end=" ", flush=True)
                train_df = load_hard_split(neg_type, "train", fold)
                test_df = load_hard_split(neg_type, "test", fold)

                train_df = subsample(train_df, MAX_TRAIN, rng)
                test_df = subsample(test_df, MAX_TEST, rng)

                X_train = extract_features(train_df, feat_type)
                X_test = extract_features(test_df, feat_type)
                y_train = train_df["label"].values
                y_test = test_df["label"].values

                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    print("SKIP", flush=True)
                    continue

                model = model_fn()
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
                fold_aucs.append(auc)
                print(f"{model_name}={auc:.3f}", flush=True)

            if fold_aucs:
                mean_auc = np.mean(fold_aucs)
                std_auc = np.std(fold_aucs)
                print(f"    [{model_name}] mean={mean_auc:.4f} ± {std_auc:.4f}")
                records.append({
                    "split_type": "hard_pep_cdr3b",
                    "neg_type": neg_type,
                    "feature_type": feat_type,
                    "model": model_name,
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                })

    return records


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(results_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, neg_type in enumerate(["only-sampled-negs", "only-neg-assays"]):
        ax = axes[ax_idx]
        neg_label = "Swapped negatives" if "sampled" in neg_type else "Experimental negatives"

        # Map neg_type names
        if "sampled" in neg_type:
            easy_key = "randomized"
            hard_key = neg_type
        else:
            easy_key = "experimental"
            hard_key = neg_type

        subset = results_df[results_df["neg_type"].isin([easy_key, hard_key])]

        x = np.arange(2)  # LR, RF
        width = 0.35
        colors = {"easy_random_cv": "#4CAF50", "hard_pep_cdr3b": "#F44336"}
        labels_map = {"easy_random_cv": "Easy (random CV)", "hard_pep_cdr3b": "Hard (pep+CDR3b)"}

        for i, split_type in enumerate(["easy_random_cv", "hard_pep_cdr3b"]):
            split_data = subset[subset["split_type"] == split_type]
            if split_data.empty:
                continue
            vals = []
            stds = []
            for model in ["LR", "RF"]:
                row = split_data[split_data["model"] == model]
                if not row.empty:
                    vals.append(row["mean_auc"].values[0])
                    stds.append(row["std_auc"].values[0])
                else:
                    vals.append(0)
                    stds.append(0)

            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=labels_map[split_type], color=colors[split_type],
                          yerr=stds, capsize=4, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "RF"], fontsize=11)
        ax.set_ylabel("Mean AUC", fontsize=10)
        ax.set_title(f"{neg_label}", fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("TChard: 3-mer Baseline — Easy vs Hard Splits",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig18_tchard_easy_vs_hard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("TChard Tripeptide Baseline: Easy vs Hard Splits")
    print("=" * 60)

    all_records = []

    # Load full dataset for easy splits
    df_full = load_full_dataset()

    # Easy splits with randomized (swapped) negatives
    all_records.extend(run_easy_splits(df_full, "randomized"))

    # Easy splits with experimental negatives
    all_records.extend(run_easy_splits(df_full, "experimental"))

    # Hard splits with sampled (swapped) negatives
    all_records.extend(run_hard_splits("only-sampled-negs"))

    # Hard splits with experimental negatives
    all_records.extend(run_hard_splits("only-neg-assays"))

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(RESULTS_DIR / "tchard_easy_vs_hard.csv", index=False)
    print(f"\nResults saved to results/tchard_easy_vs_hard.csv")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Key comparison
    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)
    for model in ["LR", "RF"]:
        easy_swap = results_df[(results_df["split_type"] == "easy_random_cv") &
                               (results_df["neg_type"] == "randomized") &
                               (results_df["model"] == model)]
        hard_swap = results_df[(results_df["split_type"] == "hard_pep_cdr3b") &
                               (results_df["neg_type"] == "only-sampled-negs") &
                               (results_df["model"] == model)]
        if not easy_swap.empty and not hard_swap.empty:
            easy_auc = easy_swap["mean_auc"].values[0]
            hard_auc = hard_swap["mean_auc"].values[0]
            print(f"  {model} with swapped negs: Easy={easy_auc:.3f} -> Hard={hard_auc:.3f} (drop={easy_auc-hard_auc:+.3f})")

    generate_figure(results_df)
    print("\nDone.")
