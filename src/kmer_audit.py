"""
kmer_audit.py — k-mer frequency baseline audit for IMMREP_2022 TCR-pMHC specificity benchmark.

Compares pure 3-mer CDR3β frequency features against the 22 published models from the
IMMREP_2022 challenge. Tests whether simple k-mer fingerprints explain a substantial
fraction of published model performance.
"""

import os
import sys
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
TRAIN_DIR  = BASE / "data/IMMREP_2022_TCRSpecificity/IMMREP_2022_TCRSpecificity-main/training_data"
TEST_DIR   = BASE / "data/IMMREP_2022_TCRSpecificity/IMMREP_2022_TCRSpecificity-main/test_set"
TRUE_DIR   = BASE / "data/IMMREP_2022_TCRSpecificity/IMMREP_2022_TCRSpecificity-main/true_set"
EVAL_DIR   = BASE / "data/IMMREP_2022_TCRSpecificity/IMMREP_2022_TCRSpecificity-main/evaluation"
FIG_DIR    = BASE / "results/figures"
RESULTS    = BASE / "results/kmer_audit_results.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — MicroAUC (from evaluate.py: it's just standard per-epitope AUC
#           averaged across epitopes; the "micro" refers to micro-average over
#           epitope files, not the sklearn micro-average formulation).
# ──────────────────────────────────────────────────────────────────────────────

def micro_auc(y_true, y_score):
    """Standard ROC-AUC (same as IMMREP evaluate.py: roc_auc_score(label==1, score))."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true == 1, y_score)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Feature extractors
# ──────────────────────────────────────────────────────────────────────────────

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Pre-generate all 3-mer combos once
ALL_TRIMERS = ["".join(t) for t in itertools.product(AA, repeat=3)]
TRIMER_INDEX = {t: i for i, t in enumerate(ALL_TRIMERS)}  # 8000 features
N_TRIMERS = len(ALL_TRIMERS)   # 8000

def aa_composition(seq: str) -> np.ndarray:
    """20-dim normalized amino-acid composition vector."""
    vec = np.zeros(20, dtype=float)
    seq = seq.upper()
    for aa in seq:
        if aa in AA_INDEX:
            vec[AA_INDEX[aa]] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def trimer_freq(seq: str) -> np.ndarray:
    """8000-dim normalized 3-mer frequency vector."""
    vec = np.zeros(N_TRIMERS, dtype=float)
    seq = seq.upper()
    count = 0
    for i in range(len(seq) - 2):
        tri = seq[i:i+3]
        if tri in TRIMER_INDEX:
            vec[TRIMER_INDEX[tri]] += 1
            count += 1
    if count > 0:
        vec /= count
    return vec


def build_features(sequences, feature_type="trimer"):
    """Build feature matrix for a list of CDR3β sequences."""
    if feature_type == "trimer":
        return np.vstack([trimer_freq(s) for s in sequences])
    elif feature_type == "aa":
        return np.vstack([aa_composition(s) for s in sequences])
    elif feature_type == "random":
        return np.random.rand(len(sequences), 1)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_training(epitope: str) -> pd.DataFrame:
    path = TRAIN_DIR / f"{epitope}.txt"
    df = pd.read_csv(path, sep="\t")
    return df[["Label", "TRB_CDR3"]].dropna()


def load_test_with_labels(epitope: str) -> pd.DataFrame:
    """Load test set and merge with true labels. Returns df with TRB_CDR3 and Label."""
    test_path = TEST_DIR / f"{epitope}.txt"
    true_path = TRUE_DIR / f"{epitope}.txt"

    test_df = pd.read_csv(test_path, sep="\t")
    true_df = pd.read_csv(true_path, sep="\t")

    # true_set has Label column (and Epitope_GroundTruth)
    # test_set has TRA_CDR3, TRB_CDR3 but no Label
    # merge on TRA_CDR3 + TRB_CDR3
    merged = pd.merge(
        test_df[["TRA_CDR3", "TRB_CDR3"]],
        true_df[["TRA_CDR3", "TRB_CDR3", "Label"]],
        on=["TRA_CDR3", "TRB_CDR3"],
        how="left"
    )
    merged = merged.dropna(subset=["Label", "TRB_CDR3"])
    merged["Label"] = merged["Label"].astype(int)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Training + evaluation for one epitope
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_epitope(epitope: str, feature_type: str = "trimer") -> dict:
    """Train on training data, evaluate on test set. Returns dict of AUC results."""
    train_df = load_training(epitope)
    test_df  = load_test_with_labels(epitope)

    X_train = build_features(train_df["TRB_CDR3"].tolist(), feature_type)
    y_train = train_df["Label"].values

    X_test  = build_features(test_df["TRB_CDR3"].tolist(), feature_type)
    y_test  = test_df["Label"].values

    results = {"epitope": epitope, "feature": feature_type}

    if feature_type == "random":
        # Random coin-flip — just use uniform random scores (no fitting)
        np.random.seed(42)
        scores = np.random.rand(len(y_test))
        results["LR_AUC"]  = micro_auc(y_test, scores)
        results["RF_AUC"]  = micro_auc(y_test, scores)
        results["n_train"] = len(y_train)
        results["n_test"]  = len(y_test)
        results["pos_frac_train"] = (y_train == 1).mean()
        results["pos_frac_test"]  = (y_test  == 1).mean()
        return results

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    lr.fit(X_train, y_train)
    lr_scores = lr.predict_proba(X_test)[:, 1]
    results["LR_AUC"] = micro_auc(y_test, lr_scores)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_scores = rf.predict_proba(X_test)[:, 1]
    results["RF_AUC"] = micro_auc(y_test, rf_scores)

    results["n_train"] = len(y_train)
    results["n_test"]  = len(y_test)
    results["pos_frac_train"] = (y_train == 1).mean()
    results["pos_frac_test"]  = (y_test  == 1).mean()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Discover epitopes (exclude README and testSet_Global)
    epitopes = sorted([
        f.stem for f in TRAIN_DIR.iterdir()
        if f.suffix == ".txt" and f.stem != "README"
    ])
    print(f"Found {len(epitopes)} epitopes: {epitopes}\n")

    # ── Run all feature types ─────────────────────────────────────────────────
    feature_types = ["random", "aa", "trimer"]
    all_rows = []

    for feat in feature_types:
        print(f"=== Feature type: {feat} ===")
        feat_rows = []
        for ep in epitopes:
            try:
                row = evaluate_epitope(ep, feature_type=feat)
                feat_rows.append(row)
                print(f"  {ep:20s}  LR={row['LR_AUC']:.4f}  RF={row['RF_AUC']:.4f}"
                      f"  (n_train={row['n_train']}, n_test={row['n_test']}, "
                      f"pos_train={row['pos_frac_train']:.2%}, pos_test={row['pos_frac_test']:.2%})")
            except Exception as e:
                print(f"  {ep}: ERROR — {e}")
        all_rows.extend(feat_rows)

        # Average
        lr_avg = np.nanmean([r["LR_AUC"] for r in feat_rows])
        rf_avg = np.nanmean([r["RF_AUC"] for r in feat_rows])
        print(f"  {'AVERAGE':20s}  LR={lr_avg:.4f}  RF={rf_avg:.4f}\n")

    results_df = pd.DataFrame(all_rows)

    # ── Load published model AUCs ─────────────────────────────────────────────
    print("=== Loading published model MicroAUCs ===")
    pub_df = pd.read_csv(EVAL_DIR / "microaucs.csv", header=0)
    # Columns: unnamed index, model_name, MicroAUC, epitope1..., _Average, ab, cdr, cluster
    # First col is model name (index 0), second is "MicroAUC" label, rest are epitope AUCs
    pub_df.columns = pub_df.columns.str.strip()
    # The CSV uses blank first column as row index
    pub_df = pub_df.rename(columns={pub_df.columns[0]: "model"})
    # Drop the "MicroAUC" label column (col index 1)
    metric_col = pub_df.columns[1]
    pub_df = pub_df.drop(columns=[metric_col])

    # Separate metadata columns
    meta_cols = ["model", "ab", "cdr", "cluster"]
    epi_cols_pub = [c for c in pub_df.columns if c not in meta_cols]

    print(f"Models: {pub_df['model'].tolist()}")
    print(f"Epitope columns in microaucs: {epi_cols_pub}\n")

    # Convert AUC columns to numeric
    for col in epi_cols_pub:
        pub_df[col] = pd.to_numeric(pub_df[col], errors="coerce")

    # ── Build comparison table per epitope ────────────────────────────────────
    # Extract our baselines per epitope and feature
    def get_auc(feat, ep, model_type="LR"):
        sub = results_df[(results_df["feature"] == feat) & (results_df["epitope"] == ep)]
        if len(sub) == 0:
            return float("nan")
        return sub.iloc[0][f"{model_type}_AUC"]

    # Build wide comparison DataFrame
    comp_rows = []
    for ep in epitopes:
        row = {"epitope": ep}
        row["random_LR"]   = get_auc("random", ep, "LR")
        row["aa_LR"]       = get_auc("aa", ep, "LR")
        row["trimer_LR"]   = get_auc("trimer", ep, "LR")
        row["trimer_RF"]   = get_auc("trimer", ep, "RF")

        # Published models for this epitope (column name must match)
        if ep in pub_df.columns:
            pub_vals = pub_df[ep].dropna().values.astype(float)
        else:
            pub_vals = np.array([])

        row["pub_n_models"]  = len(pub_vals)
        row["pub_min"]       = pub_vals.min() if len(pub_vals) > 0 else np.nan
        row["pub_median"]    = np.median(pub_vals) if len(pub_vals) > 0 else np.nan
        row["pub_max"]       = pub_vals.max() if len(pub_vals) > 0 else np.nan
        row["pub_mean"]      = pub_vals.mean() if len(pub_vals) > 0 else np.nan

        # How many published models does 3-mer LR beat?
        if len(pub_vals) > 0:
            row["trimer_LR_beats_n"] = int((row["trimer_LR"] > pub_vals).sum())
            row["trimer_RF_beats_n"] = int((row["trimer_RF"] > pub_vals).sum())
        else:
            row["trimer_LR_beats_n"] = np.nan
            row["trimer_RF_beats_n"] = np.nan

        row["trimer_LR_beats_best"] = row["trimer_LR"] >= row["pub_max"]
        row["trimer_RF_beats_best"] = row["trimer_RF"] >= row["pub_max"]
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)

    # Add average row
    avg_row = {"epitope": "_Average"}
    for col in comp_df.columns[1:]:
        if "beats_best" in col:
            avg_row[col] = comp_df[col].sum()  # count of epitopes beaten
        else:
            avg_row[col] = comp_df[col].mean()
    comp_df = pd.concat([comp_df, pd.DataFrame([avg_row])], ignore_index=True)

    # ── Step 3: Print comparison ──────────────────────────────────────────────
    print("\n" + "="*90)
    print("COMPARISON: 3-mer baseline vs published models")
    print("="*90)
    header = f"{'Epitope':18s} {'rand':6s} {'aa':6s} {'tri-LR':7s} {'tri-RF':7s} | "
    header += f"{'pub_min':7s} {'pub_med':7s} {'pub_max':7s} | {'LR_beats':8s} {'RF_beats':8s}"
    print(header)
    print("-"*90)
    for _, row in comp_df.iterrows():
        beats_lr = f"{int(row['trimer_LR_beats_n'])}/{int(row['pub_n_models'])}" \
            if not pd.isna(row['trimer_LR_beats_n']) else "n/a"
        beats_rf = f"{int(row['trimer_RF_beats_n'])}/{int(row['pub_n_models'])}" \
            if not pd.isna(row['trimer_RF_beats_n']) else "n/a"
        if row['epitope'] == '_Average':
            beats_lr = f"avg {row['trimer_LR_beats_n']:.1f}"
            beats_rf = f"avg {row['trimer_RF_beats_n']:.1f}"
        print(f"{row['epitope']:18s} "
              f"{row['random_LR']:6.4f} {row['aa_LR']:6.4f} "
              f"{row['trimer_LR']:7.4f} {row['trimer_RF']:7.4f} | "
              f"{row['pub_min']:7.4f} {row['pub_median']:7.4f} {row['pub_max']:7.4f} | "
              f"{beats_lr:8s} {beats_rf:8s}")

    # Summary stats
    avg = comp_df[comp_df["epitope"] == "_Average"].iloc[0]
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"Average AUC  — Random baseline LR : {avg['random_LR']:.4f}")
    print(f"Average AUC  — AA-composition LR  : {avg['aa_LR']:.4f}")
    print(f"Average AUC  — 3-mer LR            : {avg['trimer_LR']:.4f}")
    print(f"Average AUC  — 3-mer RF            : {avg['trimer_RF']:.4f}")
    print(f"Average AUC  — Published median    : {avg['pub_median']:.4f}")
    print(f"Average AUC  — Published best      : {avg['pub_max']:.4f}")

    n_ep = len(epitopes)
    lr_beats_best = comp_df[comp_df["epitope"] != "_Average"]["trimer_LR_beats_best"].sum()
    rf_beats_best = comp_df[comp_df["epitope"] != "_Average"]["trimer_RF_beats_best"].sum()
    print(f"\n3-mer LR beats best published model on {int(lr_beats_best)}/{n_ep} epitopes")
    print(f"3-mer RF beats best published model on {int(rf_beats_best)}/{n_ep} epitopes")

    # Per-published-model comparison (average AUC)
    avg_avail = [c for c in ["_Average"] if c in pub_df.columns]
    if "_Average" in pub_df.columns:
        pub_avgs = pub_df[["model", "_Average"]].copy()
        pub_avgs["_Average"] = pd.to_numeric(pub_avgs["_Average"], errors="coerce")
        pub_avgs = pub_avgs.sort_values("_Average")

        trimer_lr_avg = avg["trimer_LR"]
        trimer_rf_avg = avg["trimer_RF"]

        n_lr_beats = (pub_avgs["_Average"] < trimer_lr_avg).sum()
        n_rf_beats = (pub_avgs["_Average"] < trimer_rf_avg).sum()
        print(f"\n3-mer LR (avg {trimer_lr_avg:.4f}) beats {n_lr_beats}/{len(pub_avgs)} published models overall")
        print(f"3-mer RF (avg {trimer_rf_avg:.4f}) beats {n_rf_beats}/{len(pub_avgs)} published models overall")

        print(f"\nPublished model ranking (with 3-mer baselines indicated):")
        print(f"  {'Rank':4s} {'Model':30s} {'Avg AUC':8s}")
        print(f"  {'-'*45}")

        # Merge baselines into ranking
        all_models = list(pub_avgs[["model", "_Average"]].itertuples(index=False, name=None))
        all_models.append(("  [3-mer LR baseline]", trimer_lr_avg))
        all_models.append(("  [3-mer RF baseline]", trimer_rf_avg))
        all_models.append(("  [AA-comp LR baseline]", avg["aa_LR"]))
        all_models.append(("  [Random baseline]", avg["random_LR"]))
        all_models.sort(key=lambda x: x[1])
        for rank, (name, auc) in enumerate(all_models, 1):
            marker = " <--" if "baseline" in name else ""
            print(f"  {rank:4d} {name:30s} {auc:.4f}{marker}")

    # ── Step 4: Interpretation ────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("INTERPRETATION")
    print(f"{'='*90}")
    if avg["trimer_LR"] > 0.7:
        print(f"[!] 3-mer LR achieves average AUC = {avg['trimer_LR']:.4f} > 0.70 threshold.")
        print("    This is a strong signal that published models may be primarily learning")
        print("    CDR3b 3-mer frequency fingerprints, not generalizable binding logic.")
    elif avg["trimer_LR"] > 0.6:
        print(f"[~] 3-mer LR achieves average AUC = {avg['trimer_LR']:.4f} (0.60-0.70 range).")
        print("    Moderate baseline. Published models still learn something beyond k-mer stats,")
        print("    but the gap is smaller than expected for 'genuine binding' models.")
    else:
        print(f"[ok] 3-mer LR achieves average AUC = {avg['trimer_LR']:.4f} < 0.60.")
        print("     Baseline is weak. Published models appear to capture real signal beyond k-mer.")

    gap_lr_to_best = avg["pub_max"] - avg["trimer_LR"]
    gap_lr_to_med  = avg["pub_median"] - avg["trimer_LR"]
    print(f"\n    Gap: 3-mer LR → published median : {gap_lr_to_med:+.4f}")
    print(f"    Gap: 3-mer LR → published best   : {gap_lr_to_best:+.4f}")

    # ── Save numeric results ──────────────────────────────────────────────────
    comp_df.to_csv(RESULTS, index=False)
    print(f"\nNumeric results saved to: {RESULTS}")

    # ── Step 5: Visualizations ────────────────────────────────────────────────
    epi_plot = [ep for ep in epitopes if ep in comp_df["epitope"].values]
    plot_df = comp_df[comp_df["epitope"].isin(epi_plot)].copy()

    # ── Fig 6: Bar chart per epitope ──────────────────────────────────────────
    x = np.arange(len(epi_plot))
    width = 0.20

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 1.5*width, plot_df["trimer_LR"].values,   width, label="3-mer LR",          color="#2196F3", alpha=0.9)
    ax.bar(x - 0.5*width, plot_df["trimer_RF"].values,   width, label="3-mer RF",          color="#03A9F4", alpha=0.9)
    ax.bar(x + 0.5*width, plot_df["pub_median"].values,  width, label="Published median",  color="#FF9800", alpha=0.9)
    ax.bar(x + 1.5*width, plot_df["pub_max"].values,     width, label="Published best",    color="#F44336", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(epi_plot, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AUC (MicroAUC per epitope)")
    ax.set_title("k-mer Baseline Audit — 3-mer CDR3β Frequency vs. Published IMMREP_2022 Models\n"
                 "(Higher = better; bars show LR/RF baselines and published model distribution)")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out6 = FIG_DIR / "fig6_kmer_audit.png"
    fig.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out6}")

    # ── Fig 7: Scatter 3-mer LR AUC vs best published model AUC per epitope ──
    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        plot_df["trimer_LR"].values,
        plot_df["pub_max"].values,
        s=80, zorder=3, alpha=0.85,
        c=plot_df["pub_median"].values, cmap="RdYlGn", vmin=0.4, vmax=1.0
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Published median AUC", fontsize=10)

    # Annotate each point with epitope name
    for _, row in plot_df.iterrows():
        ax.annotate(row["epitope"], (row["trimer_LR"], row["pub_max"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points", alpha=0.8)

    # Diagonal reference line
    lims = [0.35, 1.01]
    ax.plot(lims, lims, "--", color="gray", linewidth=1.0, label="y=x (parity)")
    ax.set_xlabel("3-mer LR AUC (CDR3β k-mer baseline)", fontsize=11)
    ax.set_ylabel("Best Published Model AUC", fontsize=11)
    ax.set_title("Per-epitope: 3-mer LR vs. Best Published Model\n"
                 "(Points above diagonal = published model outperforms baseline;\n"
                 " Color = published median AUC)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(alpha=0.3)

    # Pearson correlation
    corr = np.corrcoef(plot_df["trimer_LR"].values, plot_df["pub_max"].values)[0, 1]
    ax.text(0.05, 0.97, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.tight_layout()
    out7 = FIG_DIR / "fig7_kmer_vs_models_scatter.png"
    fig.savefig(out7, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out7}")

    print("\nDone.")


if __name__ == "__main__":
    main()
