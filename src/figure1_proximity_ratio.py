"""
figure1_proximity_ratio.py
==========================
Publication-quality Figure 1 for the negative-sampling bias paper.

Panel A: Log10 distribution of per-sample RS/WC proximity ratios
           (biophysical vs sequence feature space)
Panel B: Dual-axis bar chart linking the RS/WC ratio to AUC inflation

Output: results/figures/fig0_proximity_ratio_figure1.png  (300 DPI)
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from immrep_loader import load_positives_only
from features import extract_features
from negative_sampling import random_swap, within_cluster

FIGURES_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

OUT_PATH = os.path.join(FIGURES_DIR, "fig0_proximity_ratio_figure1.png")

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLOR_BIO = "#2166ac"   # blue  — biophysical
COLOR_SEQ = "#d73027"   # red   — sequence

# ---------------------------------------------------------------------------
# 1. Load data and generate negatives
# ---------------------------------------------------------------------------
print("[1] Loading IMMREP positives ...")
positives = load_positives_only()
print(f"    {len(positives)} positive pairs, {positives['peptide'].nunique()} peptides")

print("[1] Generating random-swap negatives ...")
df_rs = random_swap(positives, random_state=42)
pos_df  = df_rs[df_rs["label"] == 1].copy().reset_index(drop=True)
rs_neg  = df_rs[df_rs["label"] == 0].copy().reset_index(drop=True)

print("[1] Generating within-cluster negatives ...")
df_wc  = within_cluster(positives, random_state=42)
wc_neg = df_wc[df_wc["label"] == 0].copy().reset_index(drop=True)

n_pos = len(pos_df)
n_rs  = len(rs_neg)
n_wc  = len(wc_neg)
print(f"    pos={n_pos}, rs_neg={n_rs}, wc_neg={n_wc}")

# ---------------------------------------------------------------------------
# 2. Extract and scale features
# ---------------------------------------------------------------------------
print("[2] Extracting biophysical features (104-d) ...")
X_pos_bio_raw = extract_features(pos_df,  feature_type="biophysical")
X_rs_bio_raw  = extract_features(rs_neg,  feature_type="biophysical")
X_wc_bio_raw  = extract_features(wc_neg,  feature_type="biophysical")

scaler_bio = StandardScaler()
X_all_bio = scaler_bio.fit_transform(
    np.vstack([X_pos_bio_raw, X_rs_bio_raw, X_wc_bio_raw])
)
X_pos_bio = X_all_bio[:n_pos]
X_rs_bio  = X_all_bio[n_pos : n_pos + n_rs]
X_wc_bio  = X_all_bio[n_pos + n_rs :]

print("[2] Extracting sequence features → PCA-50 ...")
X_pos_seq_raw = extract_features(pos_df,  feature_type="sequence")
X_rs_seq_raw  = extract_features(rs_neg,  feature_type="sequence")
X_wc_seq_raw  = extract_features(wc_neg,  feature_type="sequence")

scaler_seq = StandardScaler()
X_all_seq_scaled = scaler_seq.fit_transform(
    np.vstack([X_pos_seq_raw, X_rs_seq_raw, X_wc_seq_raw])
)
pca = PCA(n_components=50, random_state=42)
X_all_seq_pca = pca.fit_transform(X_all_seq_scaled)
print(f"    PCA-50 explains {pca.explained_variance_ratio_.sum():.1%} variance")

X_pos_seq = X_all_seq_pca[:n_pos]
X_rs_seq  = X_all_seq_pca[n_pos : n_pos + n_rs]
X_wc_seq  = X_all_seq_pca[n_pos + n_rs :]

# ---------------------------------------------------------------------------
# 3. Per-sample RS/WC proximity ratios
# ---------------------------------------------------------------------------
def nearest_neighbor_distances(A, B, batch=256):
    """For each row in A return Euclidean distance to its nearest row in B."""
    dists = np.empty(len(A), dtype=np.float32)
    for i in range(0, len(A), batch):
        Ai = A[i : i + batch]
        diff = Ai[:, None, :] - B[None, :, :]      # (b, n_B, d)
        dists[i : i + batch] = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
    return dists

print("[3] Computing nearest-neighbour distances (biophysical) ...")
d_rs_bio = nearest_neighbor_distances(X_pos_bio, X_rs_bio)
d_wc_bio = nearest_neighbor_distances(X_pos_bio, X_wc_bio)

print("[3] Computing nearest-neighbour distances (sequence PCA-50) ...")
d_rs_seq = nearest_neighbor_distances(X_pos_seq, X_rs_seq)
d_wc_seq = nearest_neighbor_distances(X_pos_seq, X_wc_seq)

# Per-sample ratio  (add tiny epsilon to avoid /0)
EPS = 1e-9
ratio_bio = d_rs_bio / (d_wc_bio + EPS)
ratio_seq = d_rs_seq / (d_wc_seq + EPS)

mean_ratio_bio = ratio_bio.mean()
mean_ratio_seq = ratio_seq.mean()

log_ratio_bio = np.log10(ratio_bio)
log_ratio_seq = np.log10(ratio_seq)

print(f"\n    Mean RS/WC ratio  — biophysical : {mean_ratio_bio:.1f}")
print(f"    Mean RS/WC ratio  — sequence    : {mean_ratio_seq:.3f}")
print(f"    Ratio-of-ratios   (bio / seq)   : {mean_ratio_bio / mean_ratio_seq:.0f}x")
print(f"\n    Mean log10(ratio) — biophysical : {log_ratio_bio.mean():.3f}")
print(f"    Mean log10(ratio) — sequence    : {log_ratio_seq.mean():.3f}")

# ---------------------------------------------------------------------------
# 4. AUC inflation numbers (from paper)
# ---------------------------------------------------------------------------
AUC_INFLATION = {
    "Biophysical": +0.225,
    "Sequence":    -0.086,
}
MEAN_RATIO = {
    "Biophysical": mean_ratio_bio,
    "Sequence":    mean_ratio_seq,
}

# ---------------------------------------------------------------------------
# 5. Build the figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 5),
    gridspec_kw={"wspace": 0.42}
)

# ── Panel A: overlapping histograms of log10(RS/WC) ─────────────────────────
ax_a = axes[0]

bins = np.linspace(
    min(log_ratio_bio.min(), log_ratio_seq.min()) - 0.2,
    max(log_ratio_bio.max(), log_ratio_seq.max()) + 0.2,
    55,
)

# Biophysical
ax_a.hist(log_ratio_bio, bins=bins,
          density=True, alpha=0.70,
          color=COLOR_BIO, label="Biophysical",
          edgecolor="white", linewidth=0.4)

# Sequence
ax_a.hist(log_ratio_seq, bins=bins,
          density=True, alpha=0.70,
          color=COLOR_SEQ, label="Sequence (PCA-50)",
          edgecolor="white", linewidth=0.4)

# Mean vertical lines
mean_log_bio = log_ratio_bio.mean()
mean_log_seq = log_ratio_seq.mean()

ax_a.axvline(mean_log_bio, color=COLOR_BIO, linewidth=2.0,
             linestyle="--", zorder=5)
ax_a.axvline(mean_log_seq, color=COLOR_SEQ, linewidth=2.0,
             linestyle="--", zorder=5)

# Annotations for mean lines
ymax_a = ax_a.get_ylim()[1]
ax_a.set_ylim(bottom=0)
fig.canvas.draw()           # force layout so get_ylim works
ymax_a = ax_a.get_ylim()[1]

ax_a.text(
    mean_log_bio + 0.07, ymax_a * 0.78,
    f"mean = {mean_ratio_bio:,.0f}×",
    color=COLOR_BIO, fontsize=11, fontweight="bold",
    va="center", ha="left",
)
ax_a.text(
    mean_log_seq - 0.07, ymax_a * 0.62,
    f"mean = {mean_ratio_seq:.3f}×",
    color=COLOR_SEQ, fontsize=11, fontweight="bold",
    va="center", ha="right",
)

# Gap annotation arrow on x-axis level
ax_a.annotate(
    "",
    xy=(mean_log_bio, ymax_a * 0.44),
    xytext=(mean_log_seq, ymax_a * 0.44),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
)
ax_a.text(
    (mean_log_bio + mean_log_seq) / 2, ymax_a * 0.50,
    "1,116× gap",
    ha="center", va="bottom", fontsize=11, color="black", fontweight="bold",
)

ax_a.set_xlabel(r"$\log_{10}$(RS distance / WC distance)  per positive sample",
                fontsize=12)
ax_a.set_ylabel("Density", fontsize=12)
ax_a.set_title("A   Per-sample RS/WC Proximity Ratio",
               fontsize=13, fontweight="bold", loc="left", pad=8)

ax_a.tick_params(axis="both", labelsize=11)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax_a.set_axisbelow(True)

# Zero-line reference (ratio = 1 → log10 = 0)
ax_a.axvline(0, color="grey", linewidth=0.8, linestyle=":", zorder=1,
             label="ratio = 1 (RS ≡ WC)")

legend_a = ax_a.legend(fontsize=10.5, frameon=False, loc="upper left")

# ── Panel B: dual-axis grouped bars ─────────────────────────────────────────
ax_b = axes[1]
ax_b2 = ax_b.twinx()

labels  = ["Biophysical", "Sequence"]
colors  = [COLOR_BIO, COLOR_SEQ]
x       = np.array([0.0, 1.0])
bar_w   = 0.32

# Left axis: mean RS/WC ratio (log scale)
ratio_vals = [MEAN_RATIO["Biophysical"], MEAN_RATIO["Sequence"]]
bars_ratio = ax_b.bar(
    x - bar_w / 2, ratio_vals,
    width=bar_w,
    color=colors,
    alpha=0.85,
    zorder=3,
    label="Mean RS/WC ratio",
)
ax_b.set_yscale("log")
ax_b.set_ylim(0.5, ratio_vals[0] * 6)
ax_b.set_ylabel("Mean RS/WC proximity ratio  (log scale)", fontsize=12,
                color="black")
ax_b.tick_params(axis="y", labelsize=11)
ax_b.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda v, _: f"{v:,.0f}×" if v >= 10 else f"{v:.2f}×")
)

# Value labels on ratio bars
for bar, val in zip(bars_ratio, ratio_vals):
    label_str = f"{val:,.0f}×" if val >= 10 else f"{val:.3f}×"
    ax_b.text(
        bar.get_x() + bar.get_width() / 2,
        val * 1.4,
        label_str,
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=bar.get_facecolor(),
    )

# Right axis: AUC inflation
auc_vals = [AUC_INFLATION["Biophysical"], AUC_INFLATION["Sequence"]]
auc_colors_alpha = [COLOR_BIO, COLOR_SEQ]

bars_auc = ax_b2.bar(
    x + bar_w / 2, auc_vals,
    width=bar_w,
    color=auc_colors_alpha,
    alpha=0.40,
    edgecolor=colors,
    linewidth=1.5,
    zorder=3,
    label="AUC inflation",
    hatch="///",
)
ax_b2.axhline(0, color="black", linewidth=0.8, linestyle="-", zorder=2)
ax_b2.set_ylim(-0.22, 0.45)
ax_b2.set_ylabel("AUC inflation  (standard CV − LOPO)", fontsize=12,
                 color="dimgray")
ax_b2.tick_params(axis="y", labelsize=11, colors="dimgray")
ax_b2.spines["right"].set_edgecolor("dimgray")

# Value labels on AUC bars
for bar, val in zip(bars_auc, auc_vals):
    sign = "+" if val >= 0 else ""
    va   = "bottom" if val >= 0 else "top"
    offset = 0.010 if val >= 0 else -0.010
    ax_b2.text(
        bar.get_x() + bar.get_width() / 2,
        val + offset,
        f"{sign}{val:.3f}",
        ha="center", va=va, fontsize=12, fontweight="bold",
        color=colors[list(auc_vals).index(val)],
    )

# X-axis formatting
ax_b.set_xticks(x)
ax_b.set_xticklabels(labels, fontsize=12)
ax_b.set_xlim(-0.55, 1.55)
ax_b.spines["top"].set_visible(False)
ax_b2.spines["top"].set_visible(False)
ax_b.spines["left"].set_visible(True)
ax_b.set_axisbelow(True)

ax_b.set_title("B   Geometric Separation Predicts AUC Bias",
               fontsize=13, fontweight="bold", loc="left", pad=8)

# Combined legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_BIO, alpha=0.85, label="Biophysical — RS/WC ratio"),
    Patch(facecolor=COLOR_SEQ, alpha=0.85, label="Sequence — RS/WC ratio"),
    Patch(facecolor=COLOR_BIO, alpha=0.40, edgecolor=COLOR_BIO,
          linewidth=1.5, hatch="///", label="Biophysical — AUC inflation"),
    Patch(facecolor=COLOR_SEQ, alpha=0.40, edgecolor=COLOR_SEQ,
          linewidth=1.5, hatch="///", label="Sequence — AUC inflation"),
]
ax_b.legend(handles=legend_elements, fontsize=9.5, frameon=False,
            loc="upper right", bbox_to_anchor=(1.0, 1.0))

# ---------------------------------------------------------------------------
# 6. Global figure styling
# ---------------------------------------------------------------------------
fig.patch.set_facecolor("white")
for ax in [ax_a, ax_b, ax_b2]:
    ax.set_facecolor("white")

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()

print(f"\nSaved: {OUT_PATH}")
print("Done.")
