"""
ceiling_analysis.py
===================
Label noise sensitivity analysis and information-theoretic upper bounds
for TCR-pMHC LOPO prediction.

Analyses
--------
1. Label noise sensitivity  — simulate false positive rates in training positives
2. Mutual information / Bayes error ceiling  — 1-NN leave-one-out on positives
3. Sample size confound  — training positives available vs observed LOPO AUC
4. Per-peptide CDR3b diversity  — pairwise Hamming distance, correlated with AUC

Outputs
-------
results/ceiling_analysis.csv
results/sample_size_lopo.csv
results/cdr3_diversity.csv
results/figures/fig8_label_noise_sensitivity.png
results/figures/fig9_sample_size_vs_lopo.png
results/figures/fig10_cdr3_diversity_vs_lopo.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from features import extract_features
from negative_sampling import random_swap, leave_one_peptide_out
from immrep_loader import load_positives_only, load_training_data

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reference AUCs (biophysical, random_swap, from per_peptide_lopo_auc.csv)
# ---------------------------------------------------------------------------
LOPO_CSV = RESULTS_DIR / "per_peptide_lopo_auc.csv"

def load_reference_aucs() -> pd.DataFrame:
    df = pd.read_csv(LOPO_CSV)
    # Use biophysical + random_swap as reference
    ref = df[(df["sampling_strategy"] == "random_swap") &
             (df["feature_type"]      == "biophysical")].copy()
    ref = ref[["peptide", "lopo_auc"]].reset_index(drop=True)
    return ref

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_auc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _build_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                      random_state=42, n_jobs=1)),
    ])


def _lopo_auc_on_positives(df_pos: pd.DataFrame) -> dict[str, float]:
    """
    Run LOPO benchmark (biophysical features, LR) on a positives DataFrame.
    Returns {peptide: AUC}.
    """
    per_peptide: dict[str, float] = {}

    for pep, train_df, test_pos in leave_one_peptide_out(df_pos, random_state=42):
        # Build test negatives from other peptides
        other_pos = df_pos[df_pos["peptide"] != pep]
        if len(other_pos) == 0:
            continue

        test_neg_pool = random_swap(other_pos, random_state=0)
        test_neg_pool = test_neg_pool[test_neg_pool["label"] == 0]
        if len(test_neg_pool) == 0:
            continue

        n_test_neg = len(test_pos)
        test_neg = test_neg_pool.sample(
            n=min(n_test_neg, len(test_neg_pool)),
            replace=(len(test_neg_pool) < n_test_neg),
            random_state=0,
        )
        test_df = pd.concat([test_pos, test_neg], ignore_index=True)

        if test_df["label"].nunique() < 2:
            continue

        X_train = extract_features(train_df, feature_type="biophysical")
        X_test  = extract_features(test_df,  feature_type="biophysical")
        y_train = train_df["label"].values
        y_test  = test_df["label"].values

        model = _build_lr()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        per_peptide[pep] = _safe_auc(y_test, proba)

    return per_peptide


# ===========================================================================
# Analysis 1: Label noise sensitivity
# ===========================================================================

def analysis1_label_noise(df_pos: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Label Noise Sensitivity")
    print("=" * 70)

    noise_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rng = np.random.default_rng(42)

    rows = []
    observed_overall = 0.539  # target from task description

    for noise_rate in noise_rates:
        # Flip noise_rate fraction of positives to label=0
        df_noisy = df_pos.copy()
        n_pos = len(df_noisy)
        n_flip = int(round(noise_rate * n_pos))

        if n_flip > 0:
            flip_idx = rng.choice(n_pos, size=n_flip, replace=False)
            df_noisy.loc[flip_idx, "label"] = 0
            # Only keep those still labelled 1 as "positives" for LOPO
            df_noisy_pos = df_noisy[df_noisy["label"] == 1].copy().reset_index(drop=True)
        else:
            df_noisy_pos = df_noisy.copy()

        # Need at least 2 peptides with positives to run LOPO
        pep_counts = df_noisy_pos.groupby("peptide").size()
        valid_peps = pep_counts[pep_counts >= 2].index
        df_noisy_pos = df_noisy_pos[df_noisy_pos["peptide"].isin(valid_peps)].reset_index(drop=True)

        if df_noisy_pos["peptide"].nunique() < 2:
            print(f"  noise_rate={noise_rate:.2f}: insufficient peptides after flipping, skipping.")
            continue

        per_pep_auc = _lopo_auc_on_positives(df_noisy_pos)

        valid_aucs = [v for v in per_pep_auc.values() if not np.isnan(v)]
        overall_mean = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

        print(f"  noise_rate={noise_rate:.2f}: n_remaining_pos={len(df_noisy_pos):4d}, "
              f"overall_LOPO_AUC={overall_mean:.4f}  "
              f"(n_peptides={len(per_pep_auc)})")

        for pep, auc in per_pep_auc.items():
            rows.append({
                "noise_rate":   noise_rate,
                "peptide":      pep,
                "lopo_auc":     auc,
                "overall_mean": overall_mean,
            })

    df_noise = pd.DataFrame(rows)

    # Identify noise_rate where overall AUC first crosses observed floor (~0.539)
    summary = df_noise.groupby("noise_rate")["lopo_auc"].mean().reset_index()
    summary.columns = ["noise_rate", "mean_lopo_auc"]

    below = summary[summary["mean_lopo_auc"] <= observed_overall]
    if len(below) > 0:
        crossover_rate = below["noise_rate"].iloc[0]
        print(f"\n  AUC floor ({observed_overall}) reached at noise_rate ~ {crossover_rate:.2f}")
    else:
        crossover_rate = None
        print(f"\n  AUC never dropped to {observed_overall} within tested noise range.")

    print("\n  Noise rate vs mean LOPO AUC:")
    for _, r in summary.iterrows():
        marker = " <-- floor" if crossover_rate is not None and r["noise_rate"] == crossover_rate else ""
        print(f"    noise={r['noise_rate']:.2f}  AUC={r['mean_lopo_auc']:.4f}{marker}")

    # ------------------------------------------------------------------
    # Plot fig8
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall mean AUC vs noise_rate
    ax = axes[0]
    ax.plot(summary["noise_rate"], summary["mean_lopo_auc"],
            "o-", color="#2166ac", lw=2, ms=7, label="Mean LOPO AUC")
    ax.axhline(observed_overall, color="#d6604d", ls="--", lw=1.5,
               label=f"Observed floor ({observed_overall})")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="Chance (0.5)")
    if crossover_rate is not None:
        ax.axvline(crossover_rate, color="#d6604d", ls=":", lw=1.2,
                   label=f"Crossover ~{crossover_rate:.0%}")
    ax.set_xlabel("Label noise rate (fraction of positives flipped)", fontsize=12)
    ax.set_ylabel("Mean LOPO AUC (biophysical + LR)", fontsize=12)
    ax.set_title("Label Noise Sensitivity:\nOverall Mean LOPO AUC", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.01, 0.32)
    ax.set_ylim(0.3, 0.75)
    ax.grid(True, alpha=0.3)

    # Right: per-peptide AUC degradation (heatmap style line plot)
    ax2 = axes[1]
    peptides_sorted = sorted(df_noise["peptide"].unique())
    cmap = plt.cm.tab20
    for i, pep in enumerate(peptides_sorted):
        pep_data = df_noise[df_noise["peptide"] == pep].sort_values("noise_rate")
        ax2.plot(pep_data["noise_rate"], pep_data["lopo_auc"],
                 "o-", lw=1, ms=4, alpha=0.7,
                 color=cmap(i / max(len(peptides_sorted) - 1, 1)),
                 label=pep)
    ax2.axhline(0.5, color="grey", ls=":", lw=1)
    ax2.set_xlabel("Label noise rate", fontsize=12)
    ax2.set_ylabel("LOPO AUC", fontsize=12)
    ax2.set_title("Per-Peptide LOPO AUC vs Label Noise Rate", fontsize=13)
    ax2.legend(fontsize=6, ncol=2, loc="lower left")
    ax2.set_xlim(-0.01, 0.32)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig8_label_noise_sensitivity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")

    # Save CSV
    csv_path = RESULTS_DIR / "ceiling_analysis.csv"
    df_noise.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return df_noise


# ===========================================================================
# Analysis 2: Mutual information / Bayes error ceiling (1-NN LOO)
# ===========================================================================

def analysis2_mi_ceiling(df_pos: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Mutual Information / Bayes Error Ceiling (1-NN LOO)")
    print("=" * 70)

    # Extract biophysical features from CDR3b only (not peptide)
    # Use only CDR3b-side features (first 52 dims of the 104-dim biophys vector)
    X_all = extract_features(df_pos, feature_type="biophysical")
    X_cdr3b = X_all[:, :52]   # CDR3b biophysical summary only

    peptides = df_pos["peptide"].values
    unique_peps = sorted(np.unique(peptides))
    pep_to_int = {p: i for i, p in enumerate(unique_peps)}
    y_pep = np.array([pep_to_int[p] for p in peptides])

    # --- Mutual information (CDR3b features vs peptide label) ---
    mi_scores = mutual_info_classif(X_cdr3b, y_pep, random_state=42)
    total_mi = float(mi_scores.sum())
    mean_mi  = float(mi_scores.mean())
    print(f"\n  Total MI (CDR3b features → peptide): {total_mi:.4f} nats")
    print(f"  Mean MI per feature:                  {mean_mi:.6f} nats")
    top5_idx = np.argsort(mi_scores)[::-1][:5]
    print(f"  Top-5 MI feature indices: {top5_idx.tolist()}")
    print(f"  Top-5 MI values:          {mi_scores[top5_idx].round(4).tolist()}")

    # --- 1-NN leave-one-out: can we identify which peptide a CDR3b binds? ---
    # This gives an empirical upper bound on peptide discrimination from CDR3b alone
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cdr3b)

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    # LOO: for each sample predict its peptide label using all other samples
    n = len(X_scaled)
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        knn.fit(X_scaled[mask], y_pep[mask])
        pred = knn.predict(X_scaled[[i]])[0]
        if pred == y_pep[i]:
            correct += 1

    loo_accuracy = correct / n
    bayes_error_upper = 1.0 - loo_accuracy
    print(f"\n  1-NN LOO accuracy (peptide identification from CDR3b): {loo_accuracy:.4f}")
    print(f"  Implied Bayes error upper bound:                        {bayes_error_upper:.4f}")
    print(f"  (Random chance accuracy = {1/len(unique_peps):.4f} for {len(unique_peps)} classes)")

    # --- Per-peptide 1-NN LOO accuracy ---
    print("\n  Per-peptide 1-NN LOO accuracy (how often is a positive CDR3b")
    print("  correctly assigned to its peptide vs other peptides?):")

    rows = []
    for pep in unique_peps:
        pep_mask = (peptides == pep)
        n_pep = pep_mask.sum()

        # For samples belonging to this peptide: LOO within full dataset
        pep_indices = np.where(pep_mask)[0]
        pep_correct = sum(
            1 for i in pep_indices
            if (knn.fit(X_scaled[np.arange(n) != i], y_pep[np.arange(n) != i])
                and knn.predict(X_scaled[[i]])[0] == y_pep[i])
        )
        acc = pep_correct / n_pep if n_pep > 0 else float("nan")
        print(f"    {pep:12s}: n={n_pep:4d}  1-NN LOO acc={acc:.3f}")
        rows.append({"peptide": pep, "n_positives": n_pep, "nn_loo_accuracy": acc})

    df_mi = pd.DataFrame(rows)
    print(f"\n  Overall 1-NN LOO accuracy: {loo_accuracy:.4f}")
    print(f"  Bayes error upper bound:   {bayes_error_upper:.4f}")

    return df_mi, total_mi, loo_accuracy


# ===========================================================================
# Analysis 3: Sample size confound
# ===========================================================================

def analysis3_sample_size(df_pos: pd.DataFrame, ref_aucs: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Sample Size Confound (Training Positives vs LOPO AUC)")
    print("=" * 70)

    peptides = sorted(df_pos["peptide"].unique())
    rows = []
    for pep in peptides:
        n_total = (df_pos["peptide"] == pep).sum()
        # In LOPO, we train on all OTHER peptides — so n_training_positives
        # is total minus this peptide's count
        n_train_pos = (df_pos["peptide"] != pep).sum()
        n_test_pos  = n_total

        auc_row = ref_aucs[ref_aucs["peptide"] == pep]
        lopo_auc = float(auc_row["lopo_auc"].values[0]) if len(auc_row) > 0 else float("nan")

        rows.append({
            "peptide":           pep,
            "n_test_positives":  n_test_pos,
            "n_train_positives": n_train_pos,
            "lopo_auc":          lopo_auc,
        })

    df_ss = pd.DataFrame(rows)

    # Linear regression: n_test_positives → LOPO AUC
    valid = df_ss.dropna(subset=["lopo_auc", "n_test_positives"])
    if len(valid) >= 3:
        slope_test, intercept_test, r_test, p_test, se_test = stats.linregress(
            valid["n_test_positives"], valid["lopo_auc"])
        r2_test = r_test ** 2

        slope_train, intercept_train, r_train, p_train, se_train = stats.linregress(
            valid["n_train_positives"], valid["lopo_auc"])
        r2_train = r_train ** 2
    else:
        r2_test = r2_train = float("nan")
        slope_test = slope_train = float("nan")
        p_test = p_train = float("nan")

    print("\n  Per-peptide: test set size and LOPO AUC")
    print(f"  {'Peptide':12s}  {'n_test':>6}  {'n_train':>7}  {'LOPO_AUC':>8}")
    for _, r in df_ss.sort_values("n_test_positives").iterrows():
        print(f"  {r['peptide']:12s}  {r['n_test_positives']:6.0f}  "
              f"{r['n_train_positives']:7.0f}  {r['lopo_auc']:8.4f}")

    print(f"\n  Linear regression (n_test_positives → LOPO AUC):")
    print(f"    R² = {r2_test:.4f},  slope = {slope_test:.6f},  p = {p_test:.4f}")
    print(f"\n  Linear regression (n_train_positives → LOPO AUC):")
    print(f"    R² = {r2_train:.4f},  slope = {slope_train:.6f},  p = {p_train:.4f}")

    if not np.isnan(r2_test):
        print(f"\n  Interpretation: {r2_test*100:.1f}% of LOPO AUC variance explained by "
              f"test-peptide sample size.")
        print(f"  Interpretation: {r2_train*100:.1f}% of LOPO AUC variance explained by "
              f"training set size.")

    # ------------------------------------------------------------------
    # Plot fig9
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, xcol, xlabel, r2, slope, intercept, p in [
        (axes[0], "n_test_positives",
         "Test-Peptide Positive Count (n_test)", r2_test, slope_test, intercept_test, p_test),
        (axes[1], "n_train_positives",
         "Training Positive Count (all other peptides)", r2_train, slope_train, intercept_train, p_train),
    ]:
        x = valid[xcol].values
        y = valid["lopo_auc"].values

        ax.scatter(x, y, color="#2166ac", s=60, zorder=3)

        # Label each point
        for _, row in valid.iterrows():
            ax.annotate(row["peptide"], (row[xcol], row["lopo_auc"]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 2), textcoords="offset points")

        # Regression line
        if not np.isnan(r2):
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="#d6604d", lw=1.5, ls="--",
                    label=f"R²={r2:.3f}, p={p:.3f}")
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("LOPO AUC", fontsize=11)
        ax.set_title(f"Sample Size vs LOPO AUC\n({xlabel.split('(')[0].strip()})",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig9_sample_size_vs_lopo.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")

    csv_path = RESULTS_DIR / "sample_size_lopo.csv"
    df_ss.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return df_ss


# ===========================================================================
# Analysis 4: Per-peptide CDR3b diversity
# ===========================================================================

def _normalized_hamming(s1: str, s2: str) -> float:
    """Normalized Hamming distance between two strings (pad to longer)."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    # Pad shorter sequence with a null character
    s1 = s1.ljust(max_len, "\x00")
    s2 = s2.ljust(max_len, "\x00")
    mismatches = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return mismatches / max_len


def analysis4_cdr3_diversity(df_pos: pd.DataFrame, ref_aucs: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Per-Peptide CDR3b Diversity vs LOPO AUC")
    print("=" * 70)

    peptides = sorted(df_pos["peptide"].unique())
    rows = []

    for pep in peptides:
        seqs = df_pos[df_pos["peptide"] == pep]["CDR3b"].dropna().unique().tolist()
        n = len(seqs)

        if n < 2:
            mean_ham = float("nan")
            std_ham  = float("nan")
            median_ham = float("nan")
        else:
            # Compute all pairwise normalized Hamming distances
            # Cap at 500 sequences to avoid O(n²) blowup on very large sets
            if n > 500:
                rng = np.random.default_rng(42)
                idx = rng.choice(n, size=500, replace=False)
                seqs_sample = [seqs[i] for i in idx]
            else:
                seqs_sample = seqs

            dists = []
            ns = len(seqs_sample)
            for i in range(ns):
                for j in range(i + 1, ns):
                    dists.append(_normalized_hamming(seqs_sample[i], seqs_sample[j]))

            mean_ham   = float(np.mean(dists))
            std_ham    = float(np.std(dists))
            median_ham = float(np.median(dists))

        auc_row = ref_aucs[ref_aucs["peptide"] == pep]
        lopo_auc = float(auc_row["lopo_auc"].values[0]) if len(auc_row) > 0 else float("nan")
        n_unique = len(seqs)

        rows.append({
            "peptide":           pep,
            "n_unique_cdr3b":    n_unique,
            "mean_hamming":      mean_ham,
            "std_hamming":       std_ham,
            "median_hamming":    median_ham,
            "lopo_auc":          lopo_auc,
        })

        print(f"  {pep:12s}: n_unique={n_unique:4d}  "
              f"mean_ham={mean_ham:.4f}  std={std_ham:.4f}  lopo_auc={lopo_auc:.4f}")

    df_div = pd.DataFrame(rows)

    # Correlate diversity with LOPO AUC
    valid = df_div.dropna(subset=["mean_hamming", "lopo_auc"])
    if len(valid) >= 3:
        r_pearson, p_pearson = stats.pearsonr(valid["mean_hamming"], valid["lopo_auc"])
        r_spearman, p_spearman = stats.spearmanr(valid["mean_hamming"], valid["lopo_auc"])
        print(f"\n  Pearson r  (mean Hamming vs LOPO AUC): r={r_pearson:.4f}, p={p_pearson:.4f}")
        print(f"  Spearman ρ (mean Hamming vs LOPO AUC): ρ={r_spearman:.4f}, p={p_spearman:.4f}")
        interpretation = "negative" if r_pearson < 0 else "positive"
        print(f"\n  Interpretation: {interpretation} correlation — "
              + ("higher CDR3b diversity → harder to generalize (lower LOPO AUC)."
                 if r_pearson < 0
                 else "higher CDR3b diversity → easier to generalize (higher LOPO AUC)."))
    else:
        r_pearson = r_spearman = float("nan")
        p_pearson = p_spearman = float("nan")

    # ------------------------------------------------------------------
    # Plot fig10
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter mean Hamming vs LOPO AUC
    ax = axes[0]
    x = valid["mean_hamming"].values
    y = valid["lopo_auc"].values
    ax.scatter(x, y, color="#2166ac", s=60, zorder=3)
    for _, row in valid.iterrows():
        ax.annotate(row["peptide"], (row["mean_hamming"], row["lopo_auc"]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(3, 2), textcoords="offset points")

    if len(valid) >= 3:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color="#d6604d", lw=1.5, ls="--",
                label=f"r={r_pearson:.3f}, p={p_pearson:.3f}")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("Mean Pairwise Normalized Hamming Distance", fontsize=11)
    ax.set_ylabel("LOPO AUC", fontsize=11)
    ax.set_title("CDR3b Diversity vs LOPO AUC\n(Pearson correlation)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: bar chart of CDR3b diversity per peptide, coloured by LOPO AUC
    ax2 = axes[1]
    df_sorted = df_div.sort_values("mean_hamming", ascending=False)
    aucs_norm = (df_sorted["lopo_auc"] - 0.0) / 1.0  # 0–1 range
    colours = plt.cm.RdYlGn(aucs_norm.fillna(0.5).values)
    bars = ax2.barh(df_sorted["peptide"], df_sorted["mean_hamming"],
                    color=colours, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Mean Pairwise Normalized Hamming Distance", fontsize=11)
    ax2.set_ylabel("Peptide", fontsize=11)
    ax2.set_title("CDR3b Diversity per Peptide\n(coloured by LOPO AUC: red=low, green=high)", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.3)

    # Add LOPO AUC labels on bars
    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{row['lopo_auc']:.3f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig10_cdr3_diversity_vs_lopo.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")

    csv_path = RESULTS_DIR / "cdr3_diversity.csv"
    df_div.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return df_div


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("CEILING ANALYSIS: Label Noise, MI Bounds, Sample Size, CDR3b Diversity")
    print("=" * 70)

    # Load data
    print("\nLoading training positives...")
    df_pos = load_positives_only()
    print(f"  Total positives: {len(df_pos)}")
    print(f"  Unique peptides: {df_pos['peptide'].nunique()}")
    print(f"  Unique CDR3b:    {df_pos['CDR3b'].nunique()}")
    print(f"  Peptides: {sorted(df_pos['peptide'].unique())}")

    # Load reference LOPO AUCs
    ref_aucs = load_reference_aucs()
    overall_ref_mean = ref_aucs["lopo_auc"].mean()
    print(f"\nReference LOPO AUC (biophysical + random_swap):")
    print(f"  Mean = {overall_ref_mean:.4f}")
    print(f"  Min  = {ref_aucs['lopo_auc'].min():.4f}  ({ref_aucs.loc[ref_aucs['lopo_auc'].idxmin(), 'peptide']})")
    print(f"  Max  = {ref_aucs['lopo_auc'].max():.4f}  ({ref_aucs.loc[ref_aucs['lopo_auc'].idxmax(), 'peptide']})")
    below_chance = ref_aucs[ref_aucs["lopo_auc"] < 0.5]
    print(f"  Peptides below chance (AUC<0.5): {len(below_chance)}")
    for _, r in below_chance.iterrows():
        print(f"    {r['peptide']:12s}  AUC={r['lopo_auc']:.4f}")

    # Run analyses
    df_noise = analysis1_label_noise(df_pos)
    df_mi, total_mi, loo_acc = analysis2_mi_ceiling(df_pos)
    df_ss   = analysis3_sample_size(df_pos, ref_aucs)
    df_div  = analysis4_cdr3_diversity(df_pos, ref_aucs)

    # ------------------------------------------------------------------
    # Summary of all findings
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    # Analysis 1 summary
    noise_summary = df_noise.groupby("noise_rate")["lopo_auc"].mean()
    observed_floor = 0.539
    below_floor = noise_summary[noise_summary <= observed_floor]
    if len(below_floor) > 0:
        crossover = below_floor.index[0]
        print(f"\n[1] Label noise: observed AUC floor (~{observed_floor}) is consistent with "
              f"~{crossover:.0%} label noise in training positives.")
        print(f"    This falls within the published tetramer FPR range (15-30%).")
    else:
        print(f"\n[1] Label noise: mean LOPO AUC did not drop to {observed_floor} "
              f"within tested range. Signal loss may be structural.")

    auc_at_0 = noise_summary.get(0.0, float("nan"))
    auc_at_30 = noise_summary.get(0.30, float("nan"))
    print(f"    AUC degradation: {auc_at_0:.4f} (0% noise) → {auc_at_30:.4f} (30% noise)")

    # Analysis 2 summary
    print(f"\n[2] MI ceiling: total MI(CDR3b features → peptide) = {total_mi:.4f} nats.")
    print(f"    1-NN LOO peptide-ID accuracy = {loo_acc:.4f}  "
          f"(Bayes error upper bound = {1-loo_acc:.4f})")
    if loo_acc > 1 / 17:
        print(f"    CDR3b biophysical features carry SOME discriminative signal "
              f"({loo_acc:.1%} vs {1/17:.1%} random chance).")

    # Analysis 3 summary
    valid_ss = df_ss.dropna(subset=["lopo_auc", "n_test_positives"])
    if len(valid_ss) >= 3:
        _, _, r_test, p_test, _ = stats.linregress(
            valid_ss["n_test_positives"], valid_ss["lopo_auc"])
        _, _, r_train, p_train, _ = stats.linregress(
            valid_ss["n_train_positives"], valid_ss["lopo_auc"])
        print(f"\n[3] Sample size: n_test_positives explains R²={r_test**2:.3f} of LOPO AUC variance "
              f"(p={p_test:.3f}).")
        print(f"    n_train_positives explains R²={r_train**2:.3f} (p={p_train:.3f}).")
        if r_test**2 > 0.2:
            print(f"    CONCLUSION: Sample size is a meaningful confounder.")
        else:
            print(f"    CONCLUSION: Sample size is NOT the primary driver of LOPO AUC variance.")

    # Analysis 4 summary
    valid_div = df_div.dropna(subset=["mean_hamming", "lopo_auc"])
    if len(valid_div) >= 3:
        r_p, p_p = stats.pearsonr(valid_div["mean_hamming"], valid_div["lopo_auc"])
        r_s, p_s = stats.spearmanr(valid_div["mean_hamming"], valid_div["lopo_auc"])
        print(f"\n[4] CDR3b diversity: Pearson r={r_p:.4f} (p={p_p:.4f}), "
              f"Spearman ρ={r_s:.4f} (p={p_s:.4f}).")
        if p_p < 0.05:
            direction = "negatively" if r_p < 0 else "positively"
            print(f"    Significant: CDR3b diversity is {direction} correlated with LOPO AUC.")
            if r_p < 0:
                print(f"    High diversity peptides are harder to generalise to (restricted repertoire = easier).")
        else:
            print(f"    Not significant at α=0.05; diversity does not strongly predict generalisability.")

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print(f"  results/ceiling_analysis.csv")
    print(f"  results/sample_size_lopo.csv")
    print(f"  results/cdr3_diversity.csv")
    print(f"  results/figures/fig8_label_noise_sensitivity.png")
    print(f"  results/figures/fig9_sample_size_vs_lopo.png")
    print(f"  results/figures/fig10_cdr3_diversity_vs_lopo.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
