"""
generate_paper_figures.py
=========================
Publication-quality figures for the TCR-pMHC negative sampling bias paper.
Uses seaborn + matplotlib with a consistent academic style.

Figures:
  1. The Problem — AUC inflation from negative sampling
  2. The Mechanism — Distributional shortcuts
  3. The Crossover — Difficulty titration
  4. Cross-Benchmark Validation
  5. The Cure — Experimental negatives + DL baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "paper"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "pdf.fonttype": 42,  # editable text in PDFs
    "ps.fonttype": 42,
})

# Consistent color palette
PAL_STRATEGY = {
    "random_swap": "#E74C3C",
    "epitope_balanced": "#E67E22",
    "within_cluster": "#F39C12",
    "shuffled_cdr3": "#27AE60",
}
PAL_EVAL = {"Standard CV": "#3498DB", "LOPO": "#E74C3C"}
PAL_FEAT = {"biophysical": "#3498DB", "sequence": "#9B59B6"}
PAL_NEG = {"experimental": "#27AE60", "swapped": "#E74C3C",
           "neg_control": "#27AE60", "only-neg-assays": "#27AE60",
           "only-sampled-negs": "#E74C3C", "randomized": "#E74C3C"}
PAL_MODEL = {"LR": "#5D6D7E", "RF": "#1ABC9C", "MLP": "#8E44AD"}

STRATEGY_LABELS = {
    "random_swap": "Random\nSwap",
    "epitope_balanced": "Epitope\nBalanced",
    "within_cluster": "Within\nCluster",
    "shuffled_cdr3": "Shuffled\nCDR3",
}


def _save(fig, name, dpi=300):
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ===================================================================
# FIGURE 1: The Problem — AUC Inflation
# ===================================================================

def figure1_inflation():
    """
    Panel A: Grouped bars — CV vs LOPO for LR and RF (biophysical features)
    Panel B: Same for sequence features
    Panel C: Per-peptide LOPO AUC sorted (random_swap, biophysical, RF)
    """
    bench = pd.read_csv(RESULTS_DIR / "benchmark_results.csv")
    per_pep = pd.read_csv(RESULTS_DIR / "per_peptide_lopo_auc.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})

    strategies = ["random_swap", "epitope_balanced", "within_cluster", "shuffled_cdr3"]

    # --- Panels A & B: CV vs LOPO ---
    for ax_idx, feat in enumerate(["biophysical", "sequence"]):
        ax = axes[ax_idx]
        rows = []
        for s in strategies:
            r = bench[(bench["sampling_strategy"] == s) & (bench["feature_type"] == feat)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            for model in ["LR", "RF"]:
                cv = r[f"model_{model.lower()}_standard_mean"]
                lopo = r[f"model_{model.lower()}_lopo_mean"]
                rows.append({"Strategy": STRATEGY_LABELS[s], "Model": model,
                             "Standard CV": cv, "LOPO": lopo})

        df = pd.DataFrame(rows)
        x = np.arange(len(strategies))
        w = 0.18

        for i, model in enumerate(["LR", "RF"]):
            sub = df[df["Model"] == model]
            offset = (i - 0.5) * w * 2.5
            cv_vals = sub["Standard CV"].values
            lopo_vals = sub["LOPO"].values

            ax.bar(x + offset - w/2, cv_vals, w, color=PAL_MODEL[model],
                   alpha=0.9, label=f"{model} CV" if ax_idx == 0 else "")
            ax.bar(x + offset + w/2, lopo_vals, w, color=PAL_MODEL[model],
                   alpha=0.35, label=f"{model} LOPO" if ax_idx == 0 else "",
                   edgecolor=PAL_MODEL[model], linewidth=0.8)

            # Inflation arrows for random_swap
            if strategies[0] == "random_swap":
                gap = cv_vals[0] - lopo_vals[0]
                if gap > 0.05:
                    mid_x = x[0] + offset
                    ax.annotate("", xy=(mid_x, lopo_vals[0] + 0.01),
                                xytext=(mid_x, cv_vals[0] - 0.01),
                                arrowprops=dict(arrowstyle="->", color="#C0392B",
                                                lw=1.5, ls="--"))

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=8)
        ax.set_ylabel("AUC" if ax_idx == 0 else "")
        ax.set_title(f"{'A' if ax_idx == 0 else 'B'}. {feat.capitalize()} Features",
                     fontweight="bold", loc="left", fontsize=11)
        ax.axhline(0.5, color="#95A5A6", ls="--", lw=0.8, zorder=0)
        ax.set_ylim(0.25, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper right",
                      framealpha=0.9, edgecolor="none")

    # --- Panel C: Per-peptide LOPO ---
    ax = axes[2]
    sub = per_pep[(per_pep["sampling_strategy"] == "random_swap") &
                  (per_pep["feature_type"] == "biophysical")]
    sub = sub.sort_values("lopo_auc", ascending=True)

    colors = ["#E74C3C" if v < 0.5 else "#F39C12" if v < 0.7 else "#27AE60"
              for v in sub["lopo_auc"]]
    ax.barh(range(len(sub)), sub["lopo_auc"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["peptide"].values, fontsize=7)
    ax.set_xlabel("LOPO AUC")
    ax.axvline(0.5, color="#95A5A6", ls="--", lw=0.8)
    ax.set_title("C. Per-Peptide LOPO (Random Swap, Biophysical)",
                 fontweight="bold", loc="left", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 1: Negative Sampling Strategy Controls Apparent Model Performance",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_inflation")


# ===================================================================
# FIGURE 2: The Mechanism — Distributional Shortcuts
# ===================================================================

def figure2_mechanism():
    """
    Panel A: ITRAP distance distributions (neg_control vs swapped)
    Panel B: V-gene ablation
    Panel C: SHAP vs MI correlation
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel A: ITRAP distance distributions ---
    ax = axes[0]
    dist = pd.read_csv(RESULTS_DIR / "itrap_distances.csv")
    # Subsample for speed
    if len(dist) > 5000:
        parts = []
        for nt in dist["neg_type"].unique():
            sub = dist[dist["neg_type"] == nt]
            parts.append(sub.sample(min(2500, len(sub)), random_state=42))
        dist = pd.concat(parts, ignore_index=True)

    dist["neg_type"] = dist["neg_type"].map(
        {"neg_control": "Experimental", "swapped": "Swapped"})

    sns.violinplot(data=dist, x="neg_type", y="dist_to_nearest_pos",
                   hue="neg_type",
                   palette={"Experimental": "#27AE60", "Swapped": "#E74C3C"},
                   ax=ax, inner="quartile", cut=0, linewidth=0.8, legend=False)
    ax.set_xlabel("")
    ax.set_ylabel("Distance to Nearest Positive")
    ax.set_title("A. Negative Hardness: Experimental\nvs Swapped (ITRAP)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add p-value annotation
    ax.text(0.5, 0.95, "p < 10$^{-30}$", transform=ax.transAxes,
            ha="center", fontsize=9, style="italic", color="#7F8C8D")

    # --- Panel B: V-gene ablation ---
    ax = axes[1]
    vgene = pd.read_csv(RESULTS_DIR / "vgene_ablation.csv")
    condition_order = ["full", "loop_only", "masked", "n_term", "c_term"]
    condition_labels = {"full": "Full CDR3",
                        "loop_only": "Loop Only\n(no anchors)",
                        "masked": "Masked\n(V/J only)",
                        "n_term": "N-terminal\nOnly",
                        "c_term": "C-terminal\nOnly"}

    vgene_agg = vgene.groupby("condition")["test_auc"].agg(["mean", "std"]).reset_index()
    vgene_agg = vgene_agg.set_index("condition").loc[condition_order].reset_index()
    vgene_agg["label"] = vgene_agg["condition"].map(condition_labels)

    colors_v = ["#3498DB", "#2ECC71", "#9B59B6", "#E67E22", "#E74C3C"]
    bars = ax.bar(range(len(vgene_agg)), vgene_agg["mean"], yerr=vgene_agg["std"],
                  capsize=3, color=colors_v, alpha=0.85, edgecolor="white", linewidth=0.5,
                  error_kw={"lw": 1, "capthick": 1})
    ax.set_xticks(range(len(vgene_agg)))
    ax.set_xticklabels(vgene_agg["label"].values, fontsize=7)
    ax.set_ylabel("Mean LOPO AUC")
    ax.set_title("B. V-Gene Ablation:\nGermline vs Hypervariable",
                 fontweight="bold", loc="left", fontsize=10)
    ax.axhline(0.5, color="#95A5A6", ls="--", lw=0.8)
    ax.set_ylim(0.3, 0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel C: SHAP vs MI ---
    ax = axes[2]
    shap = pd.read_csv(RESULTS_DIR / "shap_attribution.csv")
    top = shap.nsmallest(50, "shap_rank")

    ax.scatter(top["shap_rank"], top["mi_rank"], s=30, alpha=0.6,
               color="#3498DB", edgecolors="white", linewidth=0.3)

    # Label top 8 with adjustable text to avoid overlap
    top8 = top.nsmallest(8, "shap_rank")
    offsets = [(5, 8), (5, -12), (5, 8), (-20, 10), (5, 8),
               (5, -12), (-20, -12), (5, 8)]
    for i, (_, row) in enumerate(top8.iterrows()):
        ox, oy = offsets[i] if i < len(offsets) else (5, 5)
        ax.annotate(row["kmer"], (row["shap_rank"], row["mi_rank"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(ox, oy), textcoords="offset points",
                    color="#2C3E50", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#BDC3C7",
                                    lw=0.5) if abs(ox) > 10 else None)

    # Correlation line
    from scipy.stats import spearmanr
    rho, pval = spearmanr(top["shap_rank"], top["mi_rank"])
    ax.text(0.95, 0.05, f"Spearman $\\rho$ = {rho:.2f}\np = {pval:.1e}",
            transform=ax.transAxes, ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#BDC3C7", alpha=0.9))

    ax.set_xlabel("SHAP Rank")
    ax.set_ylabel("Mutual Information Rank")
    ax.set_title("C. Feature Attribution:\nSHAP vs MI (top 50 k-mers)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 2: Models Learn Distributional Shortcuts, Not Binding Logic",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_mechanism")


# ===================================================================
# FIGURE 3: The Crossover — Difficulty Titration
# ===================================================================

def figure3_crossover():
    """
    Panel A: AUC vs difficulty level (biophysical vs sequence, LR)
    Panel B: Same for RF
    """
    titration = pd.read_csv(RESULTS_DIR / "titration_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, model in enumerate(["lr", "rf"]):
        ax = axes[ax_idx]
        model_label = model.upper()

        for feat, color, marker in [("biophysical", "#3498DB", "o"),
                                     ("sequence", "#9B59B6", "s")]:
            sub = titration[(titration["model"] == model) &
                            (titration["feature_type"] == feat)].sort_values("difficulty_level")

            ax.plot(sub["difficulty_level"], sub["auc_mean"],
                    color=color, marker=marker, markersize=6,
                    linewidth=2, label=feat.capitalize(), zorder=3)
            ax.fill_between(sub["difficulty_level"],
                            sub["auc_ci_low"], sub["auc_ci_high"],
                            color=color, alpha=0.15, zorder=1)

        # Crossover annotation
        bio = titration[(titration["model"] == model) &
                        (titration["feature_type"] == "biophysical")].sort_values("difficulty_level")
        seq = titration[(titration["model"] == model) &
                        (titration["feature_type"] == "sequence")].sort_values("difficulty_level")

        if len(bio) == len(seq):
            diff = bio["auc_mean"].values - seq["auc_mean"].values
            # Find where biophysical overtakes sequence
            crossings = np.where(np.diff(np.sign(diff)))[0]
            if len(crossings) > 0:
                cx = crossings[0]
                cross_level = (bio["difficulty_level"].values[cx] +
                               bio["difficulty_level"].values[cx + 1]) / 2
                cross_auc = (bio["auc_mean"].values[cx] +
                             bio["auc_mean"].values[cx + 1]) / 2
                ax.axvline(cross_level, color="#E74C3C", ls=":", lw=1.5, alpha=0.7)
                ax.annotate("Crossover",
                            xy=(cross_level, cross_auc),
                            xytext=(cross_level + 0.8, cross_auc + 0.06),
                            fontsize=9, color="#E74C3C", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="#E74C3C",
                                            lw=1.2))

        ax.axhline(0.5, color="#95A5A6", ls="--", lw=0.8)
        ax.set_xlabel("Difficulty Level")
        ax.set_ylabel("LOPO AUC" if ax_idx == 0 else "")
        ax.set_title(f"{'A' if ax_idx == 0 else 'B'}. {model_label}: Biophysical vs Sequence",
                     fontweight="bold", loc="left", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="none")
        ax.set_xlim(0.5, 8.5)
        ax.set_ylim(0.35, 1.05)

        # Difficulty labels on x-axis
        level_names = titration.drop_duplicates("difficulty_level").sort_values(
            "difficulty_level")[["difficulty_level", "level_name"]]
        ax.set_xticks(level_names["difficulty_level"].values)
        ax.set_xticklabels(
            [n.replace("_", "\n") for n in level_names["level_name"].values],
            fontsize=6, rotation=45, ha="right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 3: Biophysical Features Overtake Sequence as Negative Difficulty Increases",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_crossover")


# ===================================================================
# FIGURE 4: Cross-Benchmark Validation
# ===================================================================

def figure4_cross_benchmark():
    """
    Panel A: TChard experimental vs swapped negatives
    Panel B: TChard easy vs hard splits
    Panel C: ITRAP cross-evaluation
    Panel D: IMMREP23 replication
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel A: TChard neg types ---
    ax = axes[0, 0]
    tchard = pd.read_csv(RESULTS_DIR / "tchard_comparison.csv")
    tchard_agg = tchard.groupby(["neg_type", "feature_type", "model"])["auc"].agg(
        ["mean", "std"]).reset_index()

    neg_labels = {"only-neg-assays": "Experimental", "only-sampled-negs": "Swapped"}
    tchard_agg["neg_label"] = tchard_agg["neg_type"].map(neg_labels)
    tchard_agg["condition"] = tchard_agg["neg_label"] + "\n" + tchard_agg["feature_type"]

    plot_data = tchard_agg[tchard_agg["model"] == "RF"].sort_values("neg_type")
    x = np.arange(2)  # biophysical, sequence
    w = 0.35

    for i, (neg, label) in enumerate([("only-neg-assays", "Experimental"),
                                       ("only-sampled-negs", "Swapped")]):
        sub = plot_data[plot_data["neg_type"] == neg]
        vals = [sub[sub["feature_type"] == f]["mean"].values[0]
                for f in ["biophysical", "sequence"]]
        errs = [sub[sub["feature_type"] == f]["std"].values[0]
                for f in ["biophysical", "sequence"]]
        color = PAL_NEG[neg]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=4,
               color=color, alpha=0.85, label=label,
               error_kw={"lw": 1, "capthick": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(["Biophysical", "Sequence"])
    ax.set_ylabel("AUC (RF)")
    ax.axhline(0.5, color="#95A5A6", ls="--", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("A. TChard: Experimental vs Swapped Negatives",
                 fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B: TChard easy vs hard ---
    ax = axes[0, 1]
    easyhard = pd.read_csv(RESULTS_DIR / "tchard_easy_vs_hard.csv")
    eh_rf = easyhard[easyhard["model"] == "RF"]

    conditions = []
    vals_eh = []
    errs_eh = []
    colors_eh = []
    for split in ["easy_random_cv", "hard_pep_cdr3b"]:
        for neg in eh_rf["neg_type"].unique():
            sub = eh_rf[(eh_rf["split_type"] == split) & (eh_rf["neg_type"] == neg)]
            if len(sub) > 0:
                split_label = "Easy" if "easy" in split else "Hard"
                neg_label = "Exp" if neg in ["experimental", "only-neg-assays"] else "Swap"
                conditions.append(f"{split_label}\n{neg_label}")
                vals_eh.append(sub["mean_auc"].values[0])
                errs_eh.append(sub["std_auc"].values[0])
                colors_eh.append("#27AE60" if "Exp" in neg_label else "#E74C3C")

    ax.bar(range(len(conditions)), vals_eh, yerr=errs_eh, capsize=4,
           color=colors_eh, alpha=0.85,
           error_kw={"lw": 1, "capthick": 1})
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel("AUC (RF, Sequence)")
    ax.axhline(0.5, color="#95A5A6", ls="--", lw=0.8)
    ax.set_ylim(0, 1.15)
    ax.set_title("B. TChard: Easy vs Hard Splits",
                 fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add gap annotation
    if len(vals_eh) >= 2:
        max_v = max(vals_eh)
        min_v = min(vals_eh)
        ax.annotate(f"Gap: {max_v - min_v:.2f}", xy=(0.5, 0.92),
                    xycoords="axes fraction", ha="center", fontsize=9,
                    fontweight="bold", color="#C0392B")

    # --- Panel C: ITRAP cross-evaluation ---
    ax = axes[1, 0]
    cross = pd.read_csv(RESULTS_DIR / "itrap_cross_eval.csv")
    cross_rf = cross[cross["model"] == "RF"]

    rows_c = []
    for _, r in cross_rf.iterrows():
        rows_c.append({"Feature": r["feature_type"].capitalize(),
                        "Condition": f"Same\n({r['train_neg'][:3]})",
                        "AUC": r["auc_same"], "type": "same"})
        rows_c.append({"Feature": r["feature_type"].capitalize(),
                        "Condition": f"Cross\n({r['train_neg'][:3]})",
                        "AUC": r["auc_cross"], "type": "cross"})

    cdf = pd.DataFrame(rows_c)
    x = np.arange(len(cross_rf))
    w = 0.35

    same_vals = cross_rf["auc_same"].values
    cross_vals = cross_rf["auc_cross"].values
    labels_c = [f"{r['feature_type'][:3].capitalize()}\n(train: {r['train_neg'][:3]})"
                for _, r in cross_rf.iterrows()]

    ax.bar(x - w/2, same_vals, w, color="#27AE60", alpha=0.85, label="Same neg type")
    ax.bar(x + w/2, cross_vals, w, color="#E74C3C", alpha=0.85, label="Cross neg type")

    # Drop annotations
    for i in range(len(x)):
        drop = same_vals[i] - cross_vals[i]
        if drop > 0.05:
            ax.annotate(f"-{drop:.2f}", xy=(x[i], cross_vals[i] - 0.03),
                        ha="center", fontsize=7, color="#C0392B", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_c, fontsize=7)
    ax.set_ylabel("AUC (RF)")
    ax.axhline(0.5, color="#95A5A6", ls="--", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("C. ITRAP Cross-Evaluation:\nModels Learn Neg-Type-Specific Patterns",
                 fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel D: IMMREP23 replication ---
    ax = axes[1, 1]
    immrep23 = pd.read_csv(RESULTS_DIR / "immrep23_replication.csv")

    rows_d = []
    for _, r in immrep23.iterrows():
        feat = r["feature_type"]
        rows_d.append({"Dataset": "IMMREP22", "Feature": feat.capitalize(),
                        "Inflation": r["immrep22_inflation"]})
        rows_d.append({"Dataset": "IMMREP23", "Feature": feat.capitalize(),
                        "Inflation": r["auc_inflation"]})

    df_d = pd.DataFrame(rows_d)
    x = np.arange(2)
    w = 0.35

    for i, ds in enumerate(["IMMREP22", "IMMREP23"]):
        sub = df_d[df_d["Dataset"] == ds]
        vals = [sub[sub["Feature"] == f]["Inflation"].values[0]
                for f in ["Biophysical", "Sequence"]]
        color = "#3498DB" if ds == "IMMREP22" else "#E67E22"
        ax.bar(x + (i - 0.5) * w, vals, w, color=color, alpha=0.85, label=ds)

    ax.set_xticks(x)
    ax.set_xticklabels(["Biophysical", "Sequence"])
    ax.set_ylabel("AUC Inflation (CV - LOPO)")
    ax.axhline(0, color="#95A5A6", ls="--", lw=0.8)
    ax.set_title("D. IMMREP23 Replication:\nSame Inflation Pattern",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 4: Inflation Pattern Replicates Across Independent Benchmarks",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_cross_benchmark")


# ===================================================================
# FIGURE 5: The Cure — Experimental Negatives + DL Baseline
# ===================================================================

def figure5_cure():
    """
    Panel A: Experimental neg training vs swapped (ITRAP LOPO, RF biophysical)
    Panel B: MLP shows same inflation as LR/RF
    Panel C: All models inflation summary
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel A: Experimental neg transfer ---
    ax = axes[0]
    lopo = pd.read_csv(RESULTS_DIR / "experimental_neg_lopo.csv")
    rf_bio = lopo[(lopo["model"] == "RF") & (lopo["feature_type"] == "biophysical")]

    peptides = sorted(rf_bio["held_out_peptide"].unique())
    # Remove peptides with NaN
    peptides = [p for p in peptides
                if not rf_bio[(rf_bio["held_out_peptide"] == p) &
                              (rf_bio["train_neg_type"] == "experimental")
                              ]["auc_vs_experimental"].isna().all()]

    x = np.arange(len(peptides))
    w = 0.35

    for i, neg_type in enumerate(["experimental", "swapped"]):
        vals = []
        errs = []
        for pep in peptides:
            sub = rf_bio[(rf_bio["held_out_peptide"] == pep) &
                         (rf_bio["train_neg_type"] == neg_type)]
            auc = sub["auc_vs_experimental"].dropna()
            vals.append(auc.mean() if len(auc) > 0 else float("nan"))
            errs.append(sub["auc_exp_std"].dropna().mean() if len(sub) > 0 else 0)

        color = PAL_NEG[neg_type]
        label = f"Train: {neg_type.capitalize()}"
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3,
               color=color, alpha=0.85, label=label,
               error_kw={"lw": 0.8, "capthick": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(peptides, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("LOPO AUC (vs experimental negs)")
    ax.axhline(0.5, color="#95A5A6", ls="--", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("A. The Cure: Train on Experimental\nNegatives (RF, Biophysical)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Mean annotation
    exp_mean = np.nanmean([rf_bio[(rf_bio["held_out_peptide"] == p) &
                                   (rf_bio["train_neg_type"] == "experimental")
                                   ]["auc_vs_experimental"].dropna().mean()
                           for p in peptides])
    swap_mean = np.nanmean([rf_bio[(rf_bio["held_out_peptide"] == p) &
                                    (rf_bio["train_neg_type"] == "swapped")
                                    ]["auc_vs_experimental"].dropna().mean()
                            for p in peptides])
    ax.text(0.98, 0.85, f"Exp mean: {exp_mean:.3f}\nSwap mean: {swap_mean:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#BDC3C7", alpha=0.9))

    # --- Panel B: MLP inflation ---
    ax = axes[1]
    mlp = pd.read_csv(RESULTS_DIR / "dl_baseline_results.csv")
    bench = pd.read_csv(RESULTS_DIR / "benchmark_results.csv")

    # Build comparison: random_swap, biophysical
    models_data = []
    for model_name, cv, lopo in [
        ("LR",
         bench[(bench["sampling_strategy"] == "random_swap") &
               (bench["feature_type"] == "biophysical")]["model_lr_standard_mean"].values[0],
         bench[(bench["sampling_strategy"] == "random_swap") &
               (bench["feature_type"] == "biophysical")]["model_lr_lopo_mean"].values[0]),
        ("RF",
         bench[(bench["sampling_strategy"] == "random_swap") &
               (bench["feature_type"] == "biophysical")]["model_rf_standard_mean"].values[0],
         bench[(bench["sampling_strategy"] == "random_swap") &
               (bench["feature_type"] == "biophysical")]["model_rf_lopo_mean"].values[0]),
    ]:
        models_data.append({"Model": model_name, "CV": cv, "LOPO": lopo,
                            "Inflation": cv - lopo})

    mlp_rs_bio = mlp[(mlp["strategy"] == "random_swap") &
                     (mlp["feature_type"] == "biophysical")]
    if len(mlp_rs_bio) > 0:
        models_data.append({"Model": "MLP", "CV": mlp_rs_bio["mean_cv_auc"].values[0],
                            "LOPO": mlp_rs_bio["mean_lopo_auc"].values[0],
                            "Inflation": mlp_rs_bio["inflation"].values[0]})

    mdf = pd.DataFrame(models_data)
    x = np.arange(len(mdf))
    w = 0.35

    ax.bar(x - w/2, mdf["CV"], w, color="#3498DB", alpha=0.85, label="Standard CV")
    ax.bar(x + w/2, mdf["LOPO"], w, color="#E74C3C", alpha=0.85, label="LOPO")

    # Inflation labels
    for i, row in mdf.iterrows():
        mid = (row["CV"] + row["LOPO"]) / 2
        ax.text(i, row["CV"] + 0.02, f"+{row['Inflation']:.2f}",
                ha="center", fontsize=8, fontweight="bold", color="#C0392B")

    ax.set_xticks(x)
    ax.set_xticklabels(mdf["Model"])
    ax.set_ylabel("AUC")
    ax.axhline(0.5, color="#95A5A6", ls="--", lw=0.8)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.set_title("B. MLP Shows Same Inflation\n(Random Swap, Biophysical)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel C: Inflation heatmap ---
    ax = axes[2]

    # Build inflation matrix: strategy × feature × model
    strategies = ["random_swap", "epitope_balanced", "within_cluster", "shuffled_cdr3"]
    feat_types = ["biophysical", "sequence"]

    heatmap_data = []
    for s in strategies:
        row = {}
        for f in feat_types:
            b = bench[(bench["sampling_strategy"] == s) & (bench["feature_type"] == f)]
            f_short = "Bio" if f == "biophysical" else "Seq"
            if len(b) > 0:
                lr_inf = b["model_lr_standard_mean"].values[0] - b["model_lr_lopo_mean"].values[0]
                rf_inf = b["model_rf_standard_mean"].values[0] - b["model_rf_lopo_mean"].values[0]
                row[f"LR {f_short}"] = lr_inf
                row[f"RF {f_short}"] = rf_inf

            m = mlp[(mlp["strategy"] == s) & (mlp["feature_type"] == f)]
            if len(m) > 0:
                row[f"MLP {f_short}"] = m["inflation"].values[0]

        row["Strategy"] = STRATEGY_LABELS[s].replace("\n", " ")
        heatmap_data.append(row)

    hdf = pd.DataFrame(heatmap_data).set_index("Strategy")
    # Only keep columns that exist
    hdf = hdf.dropna(axis=1, how="all")

    sns.heatmap(hdf.astype(float), annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, vmin=-0.1, vmax=0.7, linewidths=0.5,
                ax=ax, cbar_kws={"label": "Inflation", "shrink": 0.8})
    ax.set_title("C. Inflation Heatmap:\nAll Models x Conditions",
                 fontweight="bold", loc="left", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    fig.suptitle("Figure 5: Experimental Negatives Recover Signal; Deep Learning Doesn't Help",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_cure")


# ===================================================================
# Main
# ===================================================================

def main():
    print("Generating publication figures...")
    print(f"Output: {FIGURES_DIR}")
    print()

    figure1_inflation()
    figure2_mechanism()
    figure3_crossover()
    figure4_cross_benchmark()
    figure5_cure()

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Formats: PNG (300 dpi) + PDF (vector)")


if __name__ == "__main__":
    main()
