"""
V-gene ablation analysis for TCR-pMHC specificity prediction.

Tests whether germline-encoded N/C-terminal 3-mer features drive model performance,
or whether the hypervariable CDR3 loop itself carries predictive signal.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(
    _PROJECT_ROOT,
    "data/IMMREP_2022_TCRSpecificity/IMMREP_2022_TCRSpecificity-main/training_data"
)
TEST_DIR = os.path.join(
    _PROJECT_ROOT,
    "data/IMMREP_2022_TCRSpecificity/IMMREP_2022_TCRSpecificity-main/true_set"
)
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
ALL_3MERS = ["".join(t) for t in product(AMINO_ACIDS, repeat=3)]
KMER_INDEX = {k: i for i, k in enumerate(ALL_3MERS)}

# Top SHAP features to mask (from prior SHAP analysis)
MASK_KMERS = {"CAS", "ASS", "QYF", "YEQ", "EQY", "TQY", "GYE", "SSE"}

CONDITIONS = ["full", "loop_only", "n_term", "c_term", "masked"]

RF_PARAMS = dict(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)


# ── Feature extraction helpers ───────────────────────────────────────────────

def trim_sequence(seq: str, condition: str) -> str:
    """Return the subsequence corresponding to the ablation condition."""
    if condition == "full":
        return seq
    elif condition == "loop_only":
        # Remove first 3 (V-gene anchor: CAS) and last 2 (J-gene anchor: YF/TF)
        trimmed = seq[3:-2] if len(seq) > 5 else ""
        return trimmed
    elif condition == "n_term":
        # First 5 AAs — V-gene germline region
        return seq[:5]
    elif condition == "c_term":
        # Last 5 AAs — J-gene germline region
        return seq[-5:] if len(seq) >= 5 else seq
    elif condition == "masked":
        # Return full sequence; masking applied at feature vector level
        return seq
    else:
        raise ValueError(f"Unknown condition: {condition}")


def seq_to_3mer_freq(seq: str) -> np.ndarray:
    """Convert a CDR3 sequence to a 3-mer frequency vector (len=8000)."""
    vec = np.zeros(len(ALL_3MERS), dtype=np.float32)
    if len(seq) < 3:
        return vec
    n_kmers = len(seq) - 2
    for i in range(n_kmers):
        kmer = seq[i:i+3]
        if kmer in KMER_INDEX:
            vec[KMER_INDEX[kmer]] += 1
    if n_kmers > 0:
        vec /= n_kmers
    return vec


def build_feature_matrix(seqs: list, condition: str) -> np.ndarray:
    """Build feature matrix for a list of sequences under a given condition."""
    vecs = []
    for seq in seqs:
        trimmed = trim_sequence(seq, condition)
        vec = seq_to_3mer_freq(trimmed)
        if condition == "masked":
            # Zero out masked k-mer frequencies
            for kmer in MASK_KMERS:
                if kmer in KMER_INDEX:
                    vec[KMER_INDEX[kmer]] = 0.0
        vecs.append(vec)
    return np.array(vecs, dtype=np.float32)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_epitope_file(path: str):
    """Load a tab-separated epitope file; returns (sequences, labels) or None."""
    try:
        df = pd.read_csv(path, sep="\t")
        if "TRB_CDR3" not in df.columns or "Label" not in df.columns:
            print(f"  Skipping {path}: missing required columns", flush=True)
            return None
        df = df[["TRB_CDR3", "Label"]].dropna()
        # Map -1 → 0
        df["Label"] = df["Label"].map(lambda x: 0 if int(x) == -1 else int(x))
        seqs = df["TRB_CDR3"].tolist()
        labels = df["Label"].tolist()
        return seqs, labels
    except Exception as e:
        print(f"  Failed to load {path}: {e}", flush=True)
        return None


def get_epitope_names():
    """Return sorted list of epitope names (excluding README)."""
    names = []
    for fname in sorted(os.listdir(TRAIN_DIR)):
        if not fname.endswith(".txt"):
            continue
        if fname.lower().startswith("readme"):
            continue
        epitope = fname.replace(".txt", "")
        # Check test file also exists
        test_path = os.path.join(TEST_DIR, fname)
        if not os.path.exists(test_path):
            print(f"  No test file for {epitope}, skipping", flush=True)
            continue
        names.append(epitope)
    return names


# ── Main experiment ───────────────────────────────────────────────────────────

def run_ablation():
    print("=" * 60, flush=True)
    print("V-GENE ABLATION ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    epitopes = get_epitope_names()
    print(f"\nFound {len(epitopes)} epitopes: {epitopes}\n", flush=True)

    records = []

    for epitope in epitopes:
        train_path = os.path.join(TRAIN_DIR, f"{epitope}.txt")
        test_path = os.path.join(TEST_DIR, f"{epitope}.txt")

        train_data = load_epitope_file(train_path)
        test_data = load_epitope_file(test_path)

        if train_data is None or test_data is None:
            continue

        train_seqs, train_labels = train_data
        test_seqs, test_labels = test_data

        # Check class diversity
        if len(set(train_labels)) < 2:
            print(f"  {epitope}: <2 classes in train, skipping", flush=True)
            continue
        if len(set(test_labels)) < 2:
            print(f"  {epitope}: <2 classes in test, skipping", flush=True)
            continue

        n_train = len(train_labels)
        n_test = len(test_labels)
        print(f"\n--- Epitope: {epitope} (train={n_train}, test={n_test}) ---", flush=True)

        for condition in CONDITIONS:
            # Build features
            X_train = build_feature_matrix(train_seqs, condition)
            X_test = build_feature_matrix(test_seqs, condition)
            y_train = np.array(train_labels)
            y_test = np.array(test_labels)

            # Skip degenerate feature matrices (all zeros)
            if X_train.sum() == 0 or X_test.sum() == 0:
                print(f"  Condition {condition}: all-zero features, skipping", flush=True)
                continue

            # Train RF
            rf = RandomForestClassifier(**RF_PARAMS)
            rf.fit(X_train, y_train)

            # Predict probabilities
            proba = rf.predict_proba(X_test)
            # Find column for class 1
            classes = list(rf.classes_)
            if 1 in classes:
                pos_idx = classes.index(1)
            else:
                print(f"  Condition {condition}: no positive class in RF, skipping", flush=True)
                continue

            scores = proba[:, pos_idx]

            try:
                auc = roc_auc_score(y_test, scores)
            except Exception as e:
                print(f"  Condition {condition}: AUC failed ({e}), skipping", flush=True)
                continue

            print(f"  Condition {condition}, epitope {epitope}: AUC = {auc:.4f}", flush=True)
            records.append({
                "epitope": epitope,
                "condition": condition,
                "n_train": n_train,
                "n_test": n_test,
                "test_auc": auc,
            })

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "vgene_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}", flush=True)

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("MEAN ± SD PER CONDITION", flush=True)
    print("=" * 60, flush=True)

    summary = (
        df.groupby("condition")["test_auc"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Mean AUC", "std": "SD", "count": "N epitopes"})
    )
    # Reorder rows to CONDITIONS order
    summary = summary.reindex([c for c in CONDITIONS if c in summary.index])
    print(summary.to_string(), flush=True)

    # ── Statistical test: full vs loop_only ──────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("STATISTICAL TEST: full vs loop_only (Wilcoxon signed-rank)", flush=True)
    print("=" * 60, flush=True)

    full_aucs = df[df["condition"] == "full"].set_index("epitope")["test_auc"]
    loop_aucs = df[df["condition"] == "loop_only"].set_index("epitope")["test_auc"]

    # Align on common epitopes
    common = full_aucs.index.intersection(loop_aucs.index)
    if len(common) >= 2:
        full_vals = full_aucs[common].values
        loop_vals = loop_aucs[common].values
        diff = full_vals - loop_vals
        print(f"  N epitopes compared: {len(common)}", flush=True)
        print(f"  Mean AUC full:      {full_vals.mean():.4f}", flush=True)
        print(f"  Mean AUC loop_only: {loop_vals.mean():.4f}", flush=True)
        print(f"  Mean drop (full - loop_only): {diff.mean():.4f}", flush=True)

        try:
            stat, pval = wilcoxon(full_vals, loop_vals)
            print(f"  Wilcoxon statistic: {stat:.4f}, p-value: {pval:.4f}", flush=True)
        except Exception as e:
            print(f"  Wilcoxon test failed: {e}", flush=True)
            pval = 1.0

        # Conclusion
        print("\n" + "=" * 60, flush=True)
        mean_full = full_vals.mean()
        mean_loop = loop_vals.mean()
        drop_frac = (mean_full - mean_loop) / max(mean_full - 0.5, 1e-6)

        if mean_loop < 0.6 and drop_frac > 0.5:
            conclusion = "V-gene features account for the majority of model signal"
        elif mean_loop >= 0.65:
            conclusion = "V-gene features do not account for the majority of model signal"
        else:
            conclusion = (
                "V-gene features account for a substantial portion of model signal "
                "(loop_only performance is reduced but not at chance)"
            )
        print(f"CONCLUSION: {conclusion}", flush=True)
        print("=" * 60, flush=True)
    else:
        print("  Not enough common epitopes for statistical test.", flush=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    make_figure(df, summary)

    return df


def make_figure(df: pd.DataFrame, summary: pd.DataFrame):
    """Grouped bar chart with per-epitope jitter."""
    fig, ax = plt.subplots(figsize=(10, 6))

    condition_order = [c for c in CONDITIONS if c in df["condition"].unique()]
    palette = sns.color_palette("Set2", n_colors=len(condition_order))
    color_map = dict(zip(condition_order, palette))

    x_positions = np.arange(len(condition_order))
    bar_width = 0.6

    # Draw bars (mean AUC)
    for i, cond in enumerate(condition_order):
        sub = df[df["condition"] == cond]["test_auc"]
        mean_auc = sub.mean()
        sd_auc = sub.std()
        ax.bar(
            x_positions[i],
            mean_auc,
            width=bar_width,
            color=color_map[cond],
            alpha=0.8,
            label=cond,
            yerr=sd_auc,
            capsize=5,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
        )

    # Overlay per-epitope jitter
    rng = np.random.default_rng(42)
    for i, cond in enumerate(condition_order):
        sub = df[df["condition"] == cond]["test_auc"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(
            x_positions[i] + jitter,
            sub,
            color="black",
            s=20,
            alpha=0.5,
            zorder=5,
        )

    # Reference line at chance
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance (AUC=0.5)")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in condition_order],
        fontsize=11,
    )
    ax.set_ylabel("Test AUC (IMMREP true set)", fontsize=12)
    ax.set_xlabel("Feature Condition", fontsize=12)
    ax.set_title(
        "V-gene Ablation: Mean AUC per CDR3 Feature Condition\n"
        "(bars = mean ± SD, dots = per-epitope)",
        fontsize=13,
    )
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    out_path = os.path.join(FIGURES_DIR, "fig13_vgene_ablation.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {out_path}", flush=True)


if __name__ == "__main__":
    run_ablation()
