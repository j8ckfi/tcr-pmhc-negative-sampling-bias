"""
visualize_results.py
====================
Generate publication-quality figures from benchmark_results.csv.

Figures produced:
  1. fig1_auc_inflation_heatmap.png  — 4 strategies × 2 features, AUC inflation
  2. fig2_standard_vs_lopo_bar.png   — grouped bar: AUC_standard vs AUC_lopo
  3. fig3_per_peptide_lopo.png       — LOPO AUC per peptide (biophysical vs sequence)
  4. fig4_model_comparison.png       — LR vs RF: standard and LOPO AUC
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR     = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGS_DIR     = RESULTS_DIR / "figures"
FIGS_DIR.mkdir(exist_ok=True)


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError:
        print("matplotlib not available — skipping plots, producing text tables only.")
        return None, None


def load_results() -> pd.DataFrame:
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_per_peptide() -> pd.DataFrame:
    pp_path = RESULTS_DIR / "per_peptide_lopo_auc.csv"
    if not pp_path.exists():
        return pd.DataFrame()
    return pd.read_csv(pp_path)


# ---------------------------------------------------------------------------
# Figure 1: AUC inflation heatmap
# ---------------------------------------------------------------------------

def fig1_inflation_heatmap(df: pd.DataFrame, plt, patches) -> None:
    strategies = ["random_swap", "epitope_balanced", "within_cluster", "shuffled_cdr3"]
    features   = ["biophysical", "sequence"]
    feat_labels = {"biophysical": "Biophysical", "sequence": "Sequence"}
    strat_labels = {
        "random_swap":      "Random swap",
        "epitope_balanced": "Epitope-balanced",
        "within_cluster":   "Within-cluster",
        "shuffled_cdr3":    "Shuffled CDR3",
    }

    mat = np.full((len(strategies), len(features)), np.nan)
    for i, s in enumerate(strategies):
        for j, f in enumerate(features):
            row = df[(df["sampling_strategy"] == s) & (df["feature_type"] == f)]
            if not row.empty:
                mat[i, j] = row["auc_inflation"].values[0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([feat_labels[f] for f in features], fontsize=11)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([strat_labels[s] for s in strategies], fontsize=11)

    for i in range(len(strategies)):
        for j in range(len(features)):
            val = mat[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.15 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="AUC inflation (standard − LOPO)", shrink=0.85)
    ax.set_title("AUC Inflation by Negative Sampling Strategy\nand Feature Type",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    out = FIGS_DIR / "fig1_auc_inflation_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Standard AUC vs LOPO AUC grouped bar
# ---------------------------------------------------------------------------

def fig2_standard_vs_lopo(df: pd.DataFrame, plt, patches) -> None:
    strategies = ["random_swap", "epitope_balanced", "within_cluster", "shuffled_cdr3"]
    features   = ["biophysical", "sequence"]
    strat_labels = {
        "random_swap":      "Random\nswap",
        "epitope_balanced": "Epitope\nbalanced",
        "within_cluster":   "Within\ncluster",
        "shuffled_cdr3":    "Shuffled\nCDR3",
    }
    colors = {
        ("biophysical", "standard"): "#2166ac",
        ("biophysical", "lopo"):     "#74add1",
        ("sequence",    "standard"): "#d73027",
        ("sequence",    "lopo"):     "#f46d43",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(strategies))
    bar_width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    feat_cv_pairs = [
        ("biophysical", "standard"),
        ("biophysical", "lopo"),
        ("sequence",    "standard"),
        ("sequence",    "lopo"),
    ]
    labels = [
        "Biophysical (standard CV)",
        "Biophysical (LOPO)",
        "Sequence (standard CV)",
        "Sequence (LOPO)",
    ]

    for idx, ((feat, cv), offset, label) in enumerate(zip(feat_cv_pairs, offsets, labels)):
        vals = []
        errs = []
        for s in strategies:
            row = df[(df["sampling_strategy"] == s) & (df["feature_type"] == feat)]
            if row.empty:
                vals.append(np.nan)
                errs.append(0)
            elif cv == "standard":
                vals.append(row["auc_standard"].values[0])
                errs.append(row["auc_standard_std"].values[0])
            else:
                vals.append(row["auc_lopo"].values[0])
                errs.append(row["auc_lopo_std"].values[0])
        ax.bar(
            x + offset * bar_width, vals, bar_width,
            color=colors[(feat, cv)],
            yerr=errs, capsize=3, label=label,
            error_kw={"elinewidth": 1},
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([strat_labels[s] for s in strategies], fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Standard CV vs Leave-One-Peptide-Out AUC\nby Negative Sampling Strategy",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS_DIR / "fig2_standard_vs_lopo_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Per-peptide LOPO AUC
# ---------------------------------------------------------------------------

def fig3_per_peptide(pp_df: pd.DataFrame, plt, patches) -> None:
    if pp_df.empty:
        print("  Skipping Fig 3: no per-peptide data.")
        return

    # Average over strategies (since LOPO is strategy-independent)
    pp_mean = (
        pp_df.groupby(["peptide", "feature_type"])["lopo_auc"]
        .mean()
        .reset_index()
    )

    peptides = sorted(pp_mean["peptide"].unique())
    features = ["biophysical", "sequence"]
    colors   = {"biophysical": "#2166ac", "sequence": "#d73027"}

    x = np.arange(len(peptides))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, feat in enumerate(features):
        sub = pp_mean[pp_mean["feature_type"] == feat].set_index("peptide")
        vals = [sub.loc[p, "lopo_auc"] if p in sub.index else np.nan for p in peptides]
        ax.bar(x + (i - 0.5) * bar_width, vals, bar_width,
               color=colors[feat], label=feat.capitalize(), alpha=0.85)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(peptides, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("LOPO AUC (mean over strategies)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Peptide Leave-One-Peptide-Out AUC\n(honest generalization score)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS_DIR / "fig3_per_peptide_lopo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: LR vs RF comparison
# ---------------------------------------------------------------------------

def fig4_model_comparison(df: pd.DataFrame, plt, patches) -> None:
    features = ["biophysical", "sequence"]
    models   = ["lr", "rf"]
    model_labels = {"lr": "Logistic Reg.", "rf": "Random Forest"}
    colors   = {"standard": "#d73027", "lopo": "#74add1"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, feat in zip(axes, features):
        strategies = df[df["feature_type"] == feat]["sampling_strategy"].unique()
        x = np.arange(len(strategies))
        bar_width = 0.2

        offsets = [-1.5, -0.5, 0.5, 1.5]
        combos = [
            ("lr",  "standard", f"LR standard"),
            ("lr",  "lopo",     f"LR LOPO"),
            ("rf",  "standard", f"RF standard"),
            ("rf",  "lopo",     f"RF LOPO"),
        ]
        color_map = {
            ("lr", "standard"): "#2166ac",
            ("lr", "lopo"):     "#74add1",
            ("rf", "standard"): "#d73027",
            ("rf", "lopo"):     "#f46d43",
        }

        for (model, cv, label), offset in zip(combos, offsets):
            vals = []
            for s in strategies:
                row = df[(df["sampling_strategy"] == s) & (df["feature_type"] == feat)]
                if row.empty:
                    vals.append(np.nan)
                    continue
                col_map = {
                    ("lr", "standard"): "model_lr_standard_mean",
                    ("lr", "lopo"):     "model_lr_lopo_mean",
                    ("rf", "standard"): "model_rf_standard_mean",
                    ("rf", "lopo"):     "model_rf_lopo_mean",
                }
                col = col_map[(model, cv)]
                vals.append(row[col].values[0] if col in row.columns else np.nan)

            ax.bar(x + offset * bar_width, vals, bar_width,
                   color=color_map[(model, cv)], label=label, alpha=0.85)

        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("_", "\n") for s in strategies],
            rotation=0, ha="center", fontsize=8
        )
        ax.set_title(f"{feat.capitalize()} features", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("ROC-AUC", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("LR vs RF: Standard CV and LOPO AUC by Strategy",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIGS_DIR / "fig4_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Text report (always produced, even without matplotlib)
# ---------------------------------------------------------------------------

def print_text_report(df: pd.DataFrame, pp_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  NOVEL RESULT: AUC INFLATION FROM NEGATIVE SAMPLING IN TCR-pMHC")
    print("  SPECIFICITY PREDICTION (IMMREP_2022 BENCHMARK, N=2,445 POSITIVES)")
    print("=" * 70)

    print("\n[1] AUC INFLATION BY STRATEGY (averaged over feature types)")
    strat_stats = (
        df.groupby("sampling_strategy")["auc_inflation"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    for strat, row in strat_stats.iterrows():
        bar = "#" * int(abs(row["mean"]) * 40)
        sign = "+" if row["mean"] >= 0 else ""
        print(f"  {strat:<22} {sign}{row['mean']:.4f} +/- {row['std']:.4f}  {bar}")

    print("\n[2] SEQUENCE vs BIOPHYSICAL INFLATION DIFFERENTIAL")
    for feat in ["biophysical", "sequence"]:
        sub = df[df["feature_type"] == feat]
        mean_inf = sub["auc_inflation"].mean()
        print(f"  {feat:<15}: mean inflation = {mean_inf:+.4f}")
    diff = (df[df.feature_type == "sequence"]["auc_inflation"].mean() -
            df[df.feature_type == "biophysical"]["auc_inflation"].mean())
    print(f"  Differential (sequence - biophysical): {diff:+.4f}")
    if diff > 0.02:
        print("  -> Sequence features show MORE inflation")
        print("     (supports distributional shortcut memorization hypothesis)")
    elif diff < -0.02:
        print("  -> Biophysical features show MORE inflation (unexpected)")
    else:
        print("  -> No differential: both feature types equally susceptible")

    print("\n[3] HONEST GENERALIZATION (LOPO AUC)")
    print(f"  Biophysical mean LOPO AUC: "
          f"{df[df.feature_type=='biophysical']['auc_lopo'].mean():.4f}")
    print(f"  Sequence mean LOPO AUC:    "
          f"{df[df.feature_type=='sequence']['auc_lopo'].mean():.4f}")

    if not pp_df.empty:
        print("\n[4] HARDEST PEPTIDES TO GENERALIZE TO (lowest LOPO AUC)")
        pp_mean = pp_df.groupby("peptide")["lopo_auc"].mean().sort_values()
        for pep, auc in pp_mean.head(5).items():
            flag = " ← zero-shot failure" if auc < 0.55 else ""
            print(f"  {pep:<15}  LOPO AUC = {auc:.4f}{flag}")

        print("\n[5] EASIEST PEPTIDES TO GENERALIZE TO (highest LOPO AUC)")
        for pep, auc in pp_mean.tail(5).sort_values(ascending=False).items():
            print(f"  {pep:<15}  LOPO AUC = {auc:.4f}")

    print("\n[6] HEADLINE FINDING")
    max_row = df.loc[df["auc_inflation"].idxmax()]
    print(
        f"  Random-swap negatives inflate AUC by up to "
        f"{df['auc_inflation'].max():.3f} ({df['auc_inflation'].max()*100:.1f} percentage points)."
    )
    print(
        f"  Replacing with within-cluster (harder) negatives reduces inflation "
        f"by {(df[df.sampling_strategy=='random_swap']['auc_inflation'].mean() - df[df.sampling_strategy=='within_cluster']['auc_inflation'].mean()):.3f} AUC units."
    )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results …")
    df    = load_results()
    pp_df = load_per_peptide()

    print_text_report(df, pp_df)

    plt, patches = _try_import_matplotlib()
    if plt is not None:
        print("\nGenerating figures …")
        fig1_inflation_heatmap(df, plt, patches)
        fig2_standard_vs_lopo(df, plt, patches)
        fig3_per_peptide(pp_df, plt, patches)
        fig4_model_comparison(df, plt, patches)
        print(f"\nAll figures saved to: {FIGS_DIR}")
    else:
        print("Install matplotlib for figures: pip install matplotlib")


if __name__ == "__main__":
    main()
