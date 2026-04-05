# -*- coding: utf-8 -*-
"""
manifold_analysis.py
====================
Geometric investigation of the directional reversal in Finding 2:

  Biophysical + random-swap  -> AUC INFLATED  (+0.225 vs LOPO)
  Sequence   + random-swap  -> AUC DEFLATED  (-0.086 vs LOPO)

Hypothesis: random-swap negatives are *geometrically closer* to the positive
manifold in SEQUENCE space than in BIOPHYSICAL space — making them harder
(not easier) negatives in that space — which inverts the decision boundary.

Sections
--------
1. Load IMMREP positives, generate random-swap and within-cluster negatives.
2. Extract biophysical (104-d) and sequence (8900-d -> PCA-50) features.
3. Pairwise distance distributions: pos->nearest-neg in each feature space.
4. Geometric hypothesis test: compare mean distances across spaces/strategies.
5. UMAP visualisations (fig5_manifold_biophysical.png / fig5_manifold_sequence.png).
6. Per-peptide correlation: distance-to-nearest-neg vs LOPO AUC.
7. Polarity-inversion test: do true positives get low predicted probability
   after training on random-swap negatives with sequence features?
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import umap

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from immrep_loader import load_positives_only
from features import extract_features
from negative_sampling import random_swap, within_cluster

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LOPO_CSV = os.path.join(RESULTS_DIR, "per_peptide_lopo_auc.csv")

# ---------------------------------------------------------------------------
# 1. Load data and generate both negative sets
# ---------------------------------------------------------------------------
print("=" * 70)
print("MANIFOLD ANALYSIS — Directional Reversal Investigation")
print("=" * 70)

print("\n[1] Loading IMMREP positives ...")
positives = load_positives_only()
print(f"    {len(positives)} positive pairs across {positives['peptide'].nunique()} peptides")

print("\n[1] Generating random-swap negatives ...")
df_rs = random_swap(positives, random_state=42)
rs_neg = df_rs[df_rs["label"] == 0].copy().reset_index(drop=True)
pos_df = df_rs[df_rs["label"] == 1].copy().reset_index(drop=True)
print(f"    {len(pos_df)} positives, {len(rs_neg)} random-swap negatives")

print("\n[1] Generating within-cluster negatives ...")
df_wc = within_cluster(positives, random_state=42)
wc_neg = df_wc[df_wc["label"] == 0].copy().reset_index(drop=True)
print(f"    {len(wc_neg)} within-cluster negatives")

# ---------------------------------------------------------------------------
# 2. Extract features
# ---------------------------------------------------------------------------
print("\n[2] Extracting biophysical features (104-d) ...")
X_pos_bio  = extract_features(pos_df,  feature_type="biophysical")
X_rs_bio   = extract_features(rs_neg,  feature_type="biophysical")
X_wc_bio   = extract_features(wc_neg,  feature_type="biophysical")
print(f"    Shapes — pos: {X_pos_bio.shape}, rs_neg: {X_rs_bio.shape}, wc_neg: {X_wc_bio.shape}")

print("\n[2] Extracting sequence features (full dim) ...")
X_pos_seq_raw = extract_features(pos_df,  feature_type="sequence")
X_rs_seq_raw  = extract_features(rs_neg,  feature_type="sequence")
X_wc_seq_raw  = extract_features(wc_neg,  feature_type="sequence")
print(f"    Raw sequence shape: {X_pos_seq_raw.shape}")

print("\n[2] Reducing sequence features to 50 PCs for distance/UMAP ...")
scaler_seq = StandardScaler()
X_all_seq_raw = np.vstack([X_pos_seq_raw, X_rs_seq_raw, X_wc_seq_raw])
X_all_seq_scaled = scaler_seq.fit_transform(X_all_seq_raw)

pca = PCA(n_components=50, random_state=42)
X_all_seq_pca = pca.fit_transform(X_all_seq_scaled)
print(f"    Explained variance (50 PCs): {pca.explained_variance_ratio_.sum():.3f}")

n_pos = len(pos_df)
n_rs  = len(rs_neg)
n_wc  = len(wc_neg)

X_pos_seq = X_all_seq_pca[:n_pos]
X_rs_seq  = X_all_seq_pca[n_pos : n_pos + n_rs]
X_wc_seq  = X_all_seq_pca[n_pos + n_rs :]

# Scale biophysical features too (for fair distance comparison)
scaler_bio = StandardScaler()
X_all_bio_raw = np.vstack([X_pos_bio, X_rs_bio, X_wc_bio])
X_all_bio_scaled = scaler_bio.fit_transform(X_all_bio_raw)

X_pos_bio_sc = X_all_bio_scaled[:n_pos]
X_rs_bio_sc  = X_all_bio_scaled[n_pos : n_pos + n_rs]
X_wc_bio_sc  = X_all_bio_scaled[n_pos + n_rs :]


# ---------------------------------------------------------------------------
# Helper: vectorised min-distance from each point in A to nearest point in B
# ---------------------------------------------------------------------------
def min_distances_to_set(A, B, batch=256):
    """For each row in A, return the Euclidean distance to the nearest row in B."""
    dists = np.empty(len(A), dtype=np.float32)
    for i in range(0, len(A), batch):
        Ai = A[i : i + batch]
        # (batch, 1, d) - (1, n_B, d)  -> (batch, n_B)
        diff = Ai[:, None, :] - B[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        dists[i : i + batch] = d.min(axis=1)
    return dists


# ---------------------------------------------------------------------------
# 3. Pairwise distance distributions
# ---------------------------------------------------------------------------
print("\n[3] Computing pairwise distance distributions ...")

print("    bio: pos -> nearest random-swap neg ...")
d_pos_rs_bio  = min_distances_to_set(X_pos_bio_sc, X_rs_bio_sc)
print("    bio: pos -> nearest within-cluster neg ...")
d_pos_wc_bio  = min_distances_to_set(X_pos_bio_sc, X_wc_bio_sc)
print("    seq: pos -> nearest random-swap neg ...")
d_pos_rs_seq  = min_distances_to_set(X_pos_seq, X_rs_seq)
print("    seq: pos -> nearest within-cluster neg ...")
d_pos_wc_seq  = min_distances_to_set(X_pos_seq, X_wc_seq)

print("\n    Summary statistics (mean ± std of min-distance-to-nearest-neg):")
print(f"    Biophysical  | random-swap neg : {d_pos_rs_bio.mean():.4f} ± {d_pos_rs_bio.std():.4f}")
print(f"    Biophysical  | within-cluster  : {d_pos_wc_bio.mean():.4f} ± {d_pos_wc_bio.std():.4f}")
print(f"    Sequence PCA | random-swap neg : {d_pos_rs_seq.mean():.4f} ± {d_pos_rs_seq.std():.4f}")
print(f"    Sequence PCA | within-cluster  : {d_pos_wc_seq.mean():.4f} ± {d_pos_wc_seq.std():.4f}")

# Save distance distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Min-Distance from Positive to Nearest Negative\n(standardised feature spaces)", fontsize=13)

titles = [
    ("Biophysical — Random-swap neg", d_pos_rs_bio, "#e74c3c"),
    ("Biophysical — Within-cluster neg", d_pos_wc_bio, "#3498db"),
    ("Sequence PCA — Random-swap neg", d_pos_rs_seq, "#e67e22"),
    ("Sequence PCA — Within-cluster neg", d_pos_wc_seq, "#27ae60"),
]
for ax, (title, data, color) in zip(axes.flat, titles):
    ax.hist(data, bins=50, color=color, alpha=0.75, edgecolor="white")
    ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={data.mean():.3f}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Distance to nearest negative", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "fig5_distance_distributions.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n    Saved: {out_path}")


# ---------------------------------------------------------------------------
# 4. Geometric hypothesis test
# ---------------------------------------------------------------------------
print("\n[4] Geometric hypothesis test ...")
print()

# Key ratios: are random-swap negatives closer than within-cluster in each space?
ratio_bio = d_pos_rs_bio.mean() / d_pos_wc_bio.mean()
ratio_seq = d_pos_rs_seq.mean() / d_pos_wc_seq.mean()

print(f"    Ratio (RS / WC) in BIOPHYSICAL space : {ratio_bio:.4f}")
print(f"    Ratio (RS / WC) in SEQUENCE PCA space: {ratio_seq:.4f}")
print()
print("    Interpretation:")
print("      Ratio < 1 -> random-swap negs are CLOSER (harder) than within-cluster")
print("      Ratio > 1 -> random-swap negs are FARTHER (easier) than within-cluster")
print()

# Wilcoxon signed-rank: is the distance distribution significantly different?
stat_bio, p_bio = stats.wilcoxon(d_pos_rs_bio, d_pos_wc_bio, alternative="two-sided")
stat_seq, p_seq = stats.wilcoxon(d_pos_rs_seq, d_pos_wc_seq, alternative="two-sided")

print(f"    Wilcoxon RS vs WC  — Biophysical: W={stat_bio:.1f}, p={p_bio:.2e}")
print(f"    Wilcoxon RS vs WC  — Sequence:    W={stat_seq:.1f}, p={p_seq:.2e}")
print()

# The critical test: are random-swap negs CLOSER in sequence vs biophysical?
# Normalise distances within each space to make them comparable
d_rs_bio_norm = d_pos_rs_bio / d_pos_rs_bio.mean()
d_rs_seq_norm = d_pos_rs_seq / d_pos_rs_seq.mean()
d_wc_bio_norm = d_pos_wc_bio / d_pos_wc_bio.mean()
d_wc_seq_norm = d_pos_wc_seq / d_pos_wc_seq.mean()

# Relative proximity: how close are RS negs compared to WC negs (normalised)?
rs_relative_bio = d_pos_rs_bio / (d_pos_wc_bio + 1e-9)
rs_relative_seq = d_pos_rs_seq / (d_pos_wc_seq + 1e-9)

print(f"    Mean RS/WC relative proximity (per-sample):")
print(f"      Biophysical : {rs_relative_bio.mean():.4f}  (< 1 = RS closer than WC)")
print(f"      Sequence    : {rs_relative_seq.mean():.4f}  (< 1 = RS closer than WC)")

stat_cross, p_cross = stats.wilcoxon(rs_relative_bio, rs_relative_seq, alternative="two-sided")
print(f"\n    Wilcoxon RS-proximity bio vs seq: W={stat_cross:.1f}, p={p_cross:.2e}")
print()

is_rs_closer_seq = rs_relative_seq.mean() < rs_relative_bio.mean()
print("    VERDICT:")
if is_rs_closer_seq:
    print("    *** HYPOTHESIS CONFIRMED: random-swap negatives are relatively CLOSER")
    print("        to positives in sequence space than in biophysical space. ***")
    print("        This makes them harder negatives in sequence space, inflating")
    print("        decision boundary difficulty and causing AUC deflation.")
else:
    print("    *** HYPOTHESIS NOT CONFIRMED in this direction. ***")
    print("        However, other geometric factors may still explain the reversal.")


# Cross-space proximity summary table
print()
summary_df = pd.DataFrame({
    "Feature space":    ["Biophysical", "Sequence PCA"],
    "Mean d(pos->RS)":   [d_pos_rs_bio.mean(), d_pos_rs_seq.mean()],
    "Mean d(pos->WC)":   [d_pos_wc_bio.mean(), d_pos_wc_seq.mean()],
    "RS/WC ratio":      [ratio_bio, ratio_seq],
    "RS harder than WC?": [ratio_bio < 1.0, ratio_seq < 1.0],
})
print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 5. UMAP visualisations
# ---------------------------------------------------------------------------
print("\n[5] Running UMAP on biophysical features ...")

labels_all = (["positive"] * n_pos +
              ["random-swap neg"] * n_rs +
              ["within-cluster neg"] * n_wc)
colors_map = {"positive": "#2ecc71",
              "random-swap neg": "#e74c3c",
              "within-cluster neg": "#3498db"}

X_bio_all = np.vstack([X_pos_bio_sc, X_rs_bio_sc, X_wc_bio_sc])
X_seq_all = X_all_seq_pca  # already computed

reducer_bio = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
emb_bio = reducer_bio.fit_transform(X_bio_all)

print("    Running UMAP on sequence PCA features ...")
reducer_seq = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
emb_seq = reducer_seq.fit_transform(X_seq_all)


def _plot_umap(emb, labels, title, outfile):
    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, color in colors_map.items():
        mask = np.array(labels) == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, s=6, alpha=0.5, label=lbl, rasterized=True)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    patches = [mpatches.Patch(color=v, label=k) for k, v in colors_map.items()]
    ax.legend(handles=patches, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {outfile}")


_plot_umap(emb_bio, labels_all,
           "UMAP — Biophysical Features (standardised)\nPositives + random-swap + within-cluster negatives",
           os.path.join(FIGURES_DIR, "fig5_manifold_biophysical.png"))

_plot_umap(emb_seq, labels_all,
           "UMAP — Sequence Features (PCA-50, standardised)\nPositives + random-swap + within-cluster negatives",
           os.path.join(FIGURES_DIR, "fig5_manifold_sequence.png"))

# Also save a combined figure
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, (emb, space_label) in zip(axes, [(emb_bio, "Biophysical"), (emb_seq, "Sequence PCA-50")]):
    for lbl, color in colors_map.items():
        mask = np.array(labels_all) == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, s=5, alpha=0.45, label=lbl, rasterized=True)
    ax.set_title(f"UMAP — {space_label}", fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    patches = [mpatches.Patch(color=v, label=k) for k, v in colors_map.items()]
    ax.legend(handles=patches, fontsize=8)
fig.suptitle("Manifold Structure: Positives vs Negative Strategies", fontsize=13)
plt.tight_layout()
combined_path = os.path.join(FIGURES_DIR, "fig5_manifold_combined.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {combined_path}")


# ---------------------------------------------------------------------------
# 6. Per-peptide analysis: distance to nearest neg vs LOPO AUC
# ---------------------------------------------------------------------------
print("\n[6] Per-peptide analysis ...")

lopo_df = pd.read_csv(LOPO_CSV)
# Use random_swap LOPO AUC (the honest benchmark)
lopo_bio = lopo_df[(lopo_df["sampling_strategy"] == "random_swap") &
                   (lopo_df["feature_type"] == "biophysical")][["peptide", "lopo_auc"]].copy()
lopo_seq = lopo_df[(lopo_df["sampling_strategy"] == "random_swap") &
                   (lopo_df["feature_type"] == "sequence")][["peptide", "lopo_auc"]].copy()

peptides = sorted(pos_df["peptide"].unique())

# For each peptide, compute mean distance from its positives to nearest RS neg
# using both feature spaces
pep_results = []
for pep in peptides:
    pep_mask = pos_df["peptide"].values == pep

    if pep_mask.sum() == 0:
        continue

    pos_bio_pep = X_pos_bio_sc[pep_mask]
    pos_seq_pep = X_pos_seq[pep_mask]

    d_rs_bio_pep = min_distances_to_set(pos_bio_pep, X_rs_bio_sc)
    d_rs_seq_pep = min_distances_to_set(pos_seq_pep, X_rs_seq)
    d_wc_bio_pep = min_distances_to_set(pos_bio_pep, X_wc_bio_sc)
    d_wc_seq_pep = min_distances_to_set(pos_seq_pep, X_wc_seq)

    pep_results.append({
        "peptide": pep,
        "n_pos": pep_mask.sum(),
        "mean_d_rs_bio": d_rs_bio_pep.mean(),
        "mean_d_rs_seq": d_rs_seq_pep.mean(),
        "mean_d_wc_bio": d_wc_bio_pep.mean(),
        "mean_d_wc_seq": d_wc_seq_pep.mean(),
    })

pep_df = pd.DataFrame(pep_results)
pep_df = pep_df.merge(lopo_bio.rename(columns={"lopo_auc": "lopo_auc_bio"}), on="peptide", how="left")
pep_df = pep_df.merge(lopo_seq.rename(columns={"lopo_auc": "lopo_auc_seq"}), on="peptide", how="left")

print("\n    Per-peptide distance vs LOPO AUC:")
print(pep_df[["peptide", "n_pos", "mean_d_rs_bio", "mean_d_rs_seq",
              "lopo_auc_bio", "lopo_auc_seq"]].to_string(index=False))

# Pearson correlations: mean d(pos->RS neg) vs LOPO AUC
r_bio, p_r_bio = stats.pearsonr(pep_df["mean_d_rs_bio"], pep_df["lopo_auc_bio"])
r_seq, p_r_seq = stats.pearsonr(pep_df["mean_d_rs_seq"], pep_df["lopo_auc_seq"])
r_seq_bio, p_r_seq_bio = stats.pearsonr(pep_df["mean_d_rs_seq"], pep_df["lopo_auc_bio"])
r_bio_seq, p_r_bio_seq = stats.pearsonr(pep_df["mean_d_rs_bio"], pep_df["lopo_auc_seq"])

print(f"\n    Pearson r: d(pos->RS_neg,bio) vs LOPO_AUC_bio : r={r_bio:.3f}, p={p_r_bio:.3f}")
print(f"    Pearson r: d(pos->RS_neg,seq) vs LOPO_AUC_seq : r={r_seq:.3f}, p={p_r_seq:.3f}")
print(f"    Pearson r: d(pos->RS_neg,seq) vs LOPO_AUC_bio : r={r_seq_bio:.3f}, p={p_r_seq_bio:.3f}")
print(f"    Pearson r: d(pos->RS_neg,bio) vs LOPO_AUC_seq : r={r_bio_seq:.3f}, p={p_r_bio_seq:.3f}")

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (xcol, ycol, xlabel, title, color) in zip(axes, [
    ("mean_d_rs_bio", "lopo_auc_bio",
     "Mean d(pos->RS neg) biophysical", "Biophysical: distance vs LOPO AUC", "#8e44ad"),
    ("mean_d_rs_seq", "lopo_auc_seq",
     "Mean d(pos->RS neg) sequence PCA", "Sequence: distance vs LOPO AUC", "#2980b9"),
]):
    ax.scatter(pep_df[xcol], pep_df[ycol], s=60, color=color, alpha=0.8)
    for _, row in pep_df.iterrows():
        ax.annotate(row["peptide"][:6], (row[xcol], row[ycol]),
                    fontsize=6, ha="left", va="bottom", alpha=0.7)
    # Regression line
    m, b, r, p, _ = stats.linregress(pep_df[xcol], pep_df[ycol])
    xrange = np.linspace(pep_df[xcol].min(), pep_df[xcol].max(), 100)
    ax.plot(xrange, m * xrange + b, "k--", linewidth=1.2,
            label=f"r={r:.3f}, p={p:.3f}")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("LOPO AUC (random-swap eval)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)

fig.suptitle("Per-peptide: Distance to Nearest RS Negative vs LOPO AUC", fontsize=12)
plt.tight_layout()
scatter_path = os.path.join(FIGURES_DIR, "fig5_perpeptide_distance_vs_auc.png")
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n    Saved: {scatter_path}")


# ---------------------------------------------------------------------------
# 7. Polarity inversion test
# ---------------------------------------------------------------------------
print("\n[7] Polarity inversion test ...")
print("    Training LR and RF on random-swap negatives with sequence features,")
print("    evaluating on HELD-OUT test peptides (LOPO scheme)")
print()

# Prepare full sequence features (unscaled, will be scaled inside pipeline)
X_pos_seq_full = X_pos_seq_raw  # original high-dim
X_rs_seq_full  = X_rs_seq_raw

# Combine for LOPO-style evaluation
all_pos_df = pos_df.copy()
all_rs_neg_df = rs_neg.copy()

inversion_records = []

def _build_lr():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=500, C=1.0,
                                                solver="lbfgs", random_state=42))])

def _build_rf():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", RandomForestClassifier(n_estimators=100, max_depth=8,
                                                     random_state=42))])

for pep in peptides:
    # Training: all positives/negs EXCEPT this peptide
    train_pos_mask = all_pos_df["peptide"] != pep
    train_neg_mask = all_rs_neg_df["peptide"] != pep

    train_pos_idx = np.where(train_pos_mask.values)[0]
    train_neg_idx = np.where(train_neg_mask.values)[0]

    if len(train_pos_idx) == 0 or len(train_neg_idx) == 0:
        continue

    X_train_seq = np.vstack([X_pos_seq_full[train_pos_idx],
                              X_rs_seq_full[train_neg_idx]])
    y_train = np.concatenate([np.ones(len(train_pos_idx)),
                               np.zeros(len(train_neg_idx))])

    # Test: this peptide's positives paired with RS negs from OTHER peptides
    test_pos_idx = np.where(all_pos_df["peptide"].values == pep)[0]
    test_neg_idx = np.where(all_rs_neg_df["peptide"].values != pep)[0]
    # Sample equal number of negs
    n_test_pos = len(test_pos_idx)
    if len(test_neg_idx) > n_test_pos:
        rng = np.random.default_rng(42)
        test_neg_idx = rng.choice(test_neg_idx, size=n_test_pos, replace=False)

    if len(test_neg_idx) == 0:
        continue

    X_test_seq = np.vstack([X_pos_seq_full[test_pos_idx],
                             X_rs_seq_full[test_neg_idx]])
    y_test = np.concatenate([np.ones(n_test_pos), np.zeros(len(test_neg_idx))])

    for model_name, builder in [("LR", _build_lr), ("RF", _build_rf)]:
        model = builder()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_seq, y_train)

        proba = model.predict_proba(X_test_seq)[:, 1]

        # AUC
        if len(np.unique(y_test)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_test, proba)

        # Mean predicted probability for true positives and true negatives
        mean_proba_true_pos = proba[y_test == 1].mean()
        mean_proba_true_neg = proba[y_test == 0].mean()

        # Fraction of true positives with proba < 0.5 (inverted predictions)
        frac_tp_inverted = (proba[y_test == 1] < 0.5).mean()
        frac_tn_inverted = (proba[y_test == 0] > 0.5).mean()

        inversion_records.append({
            "peptide": pep,
            "model": model_name,
            "auc": auc,
            "mean_proba_true_pos": mean_proba_true_pos,
            "mean_proba_true_neg": mean_proba_true_neg,
            "frac_tp_inverted": frac_tp_inverted,
            "frac_tn_inverted": frac_tn_inverted,
            "polarity_inverted": mean_proba_true_pos < mean_proba_true_neg,
        })

inv_df = pd.DataFrame(inversion_records)

print("    Per-peptide polarity inversion results (sequence features, RS negatives):")
print()
pivot = inv_df.pivot_table(
    index="peptide",
    columns="model",
    values=["auc", "mean_proba_true_pos", "mean_proba_true_neg",
            "frac_tp_inverted", "polarity_inverted"],
    aggfunc="first"
).round(3)
print(pivot.to_string())
print()

# Summary across peptides
for model_name in ["LR", "RF"]:
    sub = inv_df[inv_df["model"] == model_name]
    n_inverted = sub["polarity_inverted"].sum()
    mean_tp_proba = sub["mean_proba_true_pos"].mean()
    mean_tn_proba = sub["mean_proba_true_neg"].mean()
    mean_frac_inv = sub["frac_tp_inverted"].mean()
    mean_auc = sub["auc"].mean()
    print(f"  {model_name}:")
    print(f"    Mean AUC                    : {mean_auc:.4f}")
    print(f"    Mean P(pos | true positive) : {mean_tp_proba:.4f}")
    print(f"    Mean P(pos | true negative) : {mean_tn_proba:.4f}")
    print(f"    Fraction true pos inverted  : {mean_frac_inv:.4f}")
    print(f"    Peptides with inverted polarity: {n_inverted}/{len(sub)}")
    print()

# Overall polarity inversion fraction
total_inverted = inv_df["polarity_inverted"].sum()
total = len(inv_df)
print(f"    OVERALL: {total_inverted}/{total} ({100*total_inverted/total:.1f}%) "
      f"peptide-model pairs show polarity inversion in sequence space")

# Visualise polarity inversion
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model_name in zip(axes, ["LR", "RF"]):
    sub = inv_df[inv_df["model"] == model_name].sort_values("mean_proba_true_pos")
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["mean_proba_true_pos"], 0.35, label="True Positives",
           color="#2ecc71", alpha=0.85)
    ax.bar(x + 0.2, sub["mean_proba_true_neg"], 0.35, label="True Negatives",
           color="#e74c3c", alpha=0.85)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["peptide"].str[:6], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean predicted P(positive)", fontsize=9)
    ax.set_title(f"{model_name}: Sequence Features + RS Negatives\n"
                 f"Mean predicted proba by true class", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    # Add AUC as text
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(x[i], 0.02, f"{row['auc']:.2f}", ha="center", fontsize=6, color="navy")

fig.suptitle("Polarity Inversion: True Positives Getting Low Predicted Probability\n"
             "(Sequence features trained on random-swap negatives, evaluated LOPO)", fontsize=11)
plt.tight_layout()
inversion_path = os.path.join(FIGURES_DIR, "fig5_polarity_inversion.png")
plt.savefig(inversion_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n    Saved: {inversion_path}")

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("FINAL SUMMARY — WHY DOES THE DIRECTIONAL REVERSAL HAPPEN?")
print("=" * 70)
print()
print("Feature spaces and dimensionality:")
print(f"  Biophysical: {X_pos_bio.shape[1]}-d -> standardised {X_pos_bio_sc.shape[1]}-d")
print(f"  Sequence:    {X_pos_seq_raw.shape[1]}-d -> PCA-50 ({pca.explained_variance_ratio_.sum():.1%} var)")
print()
print("Geometric proximity of random-swap negatives to positive manifold:")
print(f"  Bio  mean d(pos->RS)  = {d_pos_rs_bio.mean():.4f}  |  d(pos->WC) = {d_pos_wc_bio.mean():.4f}")
print(f"  Seq  mean d(pos->RS)  = {d_pos_rs_seq.mean():.4f}  |  d(pos->WC) = {d_pos_wc_seq.mean():.4f}")
print(f"  RS/WC ratio  Bio={ratio_bio:.4f}  Seq={ratio_seq:.4f}")
print()
print("Polarity inversion (sequence + RS negatives, LOPO evaluation):")
for mn in ["LR", "RF"]:
    sub = inv_df[inv_df["model"] == mn]
    print(f"  {mn}: mean P(true pos)={sub['mean_proba_true_pos'].mean():.3f}  "
          f"mean P(true neg)={sub['mean_proba_true_neg'].mean():.3f}  "
          f"inverted={sub['polarity_inverted'].sum()}/{len(sub)}")
print()
print("Per-peptide distance vs LOPO AUC correlations:")
print(f"  r(bio dist, bio AUC) = {r_bio:.3f}  p={p_r_bio:.3f}")
print(f"  r(seq dist, seq AUC) = {r_seq:.3f}  p={p_r_seq:.3f}")
print()
print("MECHANISTIC CONCLUSION:")
if rs_relative_seq.mean() < rs_relative_bio.mean():
    print("""  [CONFIRMED] Random-swap negatives are relatively CLOSER to positives
  in sequence space than in biophysical space (RS/WC ratio lower in seq).

  Mechanism: random-swap negatives preserve real CDR3 sequences from
  actual T-cells. In sequence space (BLOSUM62+3mer) they are geometrically
  near the positive manifold — they are hard negatives. This forces the
  decision boundary through the middle of the positive cluster, causing
  AUC DEFLATION on the LOPO test (where the model must generalise to a
  novel peptide).

  In biophysical space (Kidera+charge+hydrophobicity), co-evolved TCR-
  peptide pairs share specific physicochemical signatures that random swaps
  destroy — so random-swap negatives are far from the positive manifold
  (easy negatives). This gives an artificially generous decision boundary
  and AUC INFLATION on standard CV.""")
else:
    print("""  The distance-based hypothesis is not cleanly confirmed by these metrics.
  However, the polarity inversion results may indicate an alternative
  mechanism: the peptide-conditional structure of the sequence embedding
  creates a different confound.""")

print()
print("Figures saved to:", FIGURES_DIR)
print("=" * 70)
