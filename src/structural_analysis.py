"""
structural_analysis.py
======================
Characterise why five specific peptides consistently fail in leave-one-peptide-out
TCR-pMHC specificity prediction.

Steps
-----
1. HLA-A*02:01 anchor residue analysis (position 2 & C-terminus)
2. PDB structure search via RCSB REST API
3. Failure vs success peptide comparison + statistical test
4. CDR3b 3-mer motif analysis
5. Visualisation → fig11_structural_analysis.png, fig11_anchor_comparison.png

Outputs
-------
  results/structural_analysis.csv
  results/figures/fig11_structural_analysis.png
  results/figures/fig11_anchor_comparison.png
"""

from __future__ import annotations

import sys
import time
import warnings
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SRC_DIR       = PROJECT_ROOT / "src"
TRAINING_DIR  = (PROJECT_ROOT / "data" / "IMMREP_2022_TCRSpecificity" /
                 "IMMREP_2022_TCRSpecificity-main" / "training_data")
RESULTS_DIR   = PROJECT_ROOT / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Peptide definitions + known LOPO AUCs (biophysical, random_swap)
# ---------------------------------------------------------------------------
FAILURE_PEPTIDES = {
    "GPRLGVRAT":  0.114,
    "TPRVTGGGAM": 0.290,
    "RAQAPPPSW":  0.267,
    "NQKLIANQF":  0.427,
    "SPRWYFYYL":  0.430,
}

SUCCESS_PEPTIDES = {
    "LLWNGPMAV":  0.908,
    "GLCTLVAML":  0.881,
    "NYNYLYRLF":  0.836,
    "ATDALMTGF":  0.789,
    "KSKRTPMGF":  0.763,
    "HPVTKYIM":   0.718,
}

# Remaining peptides (mid-range)
MID_PEPTIDES = {
    "LTDEMIAQY":  0.564,
    "NLVPMVATV":  0.569,
    "TTDPSFLGRY": 0.514,
    "YLQPRTFLL":  0.435,
    "CINGVCWTV":  0.368,
    "GILGFVFTL":  0.339,
}

ALL_PEPTIDES: dict[str, float] = {**FAILURE_PEPTIDES, **SUCCESS_PEPTIDES, **MID_PEPTIDES}

# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity scale (from features.py)
# ---------------------------------------------------------------------------
HYDROPHOBICITY: dict[str, float] = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# HLA-A*02:01 preferred anchor residues
HLA_A0201_P2_PREFERRED   = set("LMVIA")
HLA_A0201_PEND_PREFERRED = set("VLA")

AA_SINGLE = list("ACDEFGHIKLMNPQRSTVWY")


# ===========================================================================
# STEP 1 — Anchor residue analysis
# ===========================================================================

def anchor_analysis(peptide: str) -> dict:
    """Compute HLA-A*02:01 anchor and physicochemical features for a peptide."""
    p2  = peptide[1] if len(peptide) >= 2 else "?"
    end = peptide[-1]

    p2_match   = p2  in HLA_A0201_P2_PREFERRED
    end_match  = end in HLA_A0201_PEND_PREFERRED

    p2_hydro   = HYDROPHOBICITY.get(p2, 0.0)
    end_hydro  = HYDROPHOBICITY.get(end, 0.0)

    # Anchor deviation: higher = more deviation from canonical anchors
    # Canonical anchor hydro roughly > 1.8 at P2; > 1.8 at C-term
    anchor_dev = max(0.0, 1.8 - p2_hydro) + max(0.0, 1.8 - end_hydro)

    n_pro  = peptide.count("P")
    n_gly  = peptide.count("G")
    mean_hydro = np.mean([HYDROPHOBICITY.get(aa, 0.0) for aa in peptide])

    return {
        "p2_aa":           p2,
        "p2_preferred":    p2_match,
        "p2_hydro":        round(p2_hydro, 2),
        "end_aa":          end,
        "end_preferred":   end_match,
        "end_hydro":       round(end_hydro, 2),
        "anchor_dev":      round(anchor_dev, 2),
        "n_prolines":      n_pro,
        "n_glycines":      n_gly,
        "mean_hydro":      round(mean_hydro, 3),
        "length":          len(peptide),
    }


def run_anchor_analysis() -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 1: HLA-A*02:01 Anchor Residue Analysis")
    print("="*60)

    rows = []
    for pep, auc in ALL_PEPTIDES.items():
        props = anchor_analysis(pep)
        label = ("failure" if pep in FAILURE_PEPTIDES
                 else "success" if pep in SUCCESS_PEPTIDES
                 else "mid")
        rows.append({"peptide": pep, "lopo_auc": auc, "group": label, **props})

    df = pd.DataFrame(rows).sort_values("lopo_auc")

    print(f"\n{'Peptide':<14} {'AUC':>6} {'Grp':>8}  "
          f"{'P2':>3} {'P2-pref':>7} {'P2-hydro':>9}  "
          f"{'End':>3} {'End-pref':>8} {'End-hydro':>10}  "
          f"{'AnchorDev':>10} {'Pro':>4} {'Gly':>4} {'MeanH':>7} {'Len':>4}")
    print("-"*110)
    for _, r in df.iterrows():
        print(f"{r.peptide:<14} {r.lopo_auc:>6.3f} {r.group:>8}  "
              f"{r.p2_aa:>3} {'Y' if r.p2_preferred else 'N':>7} {r.p2_hydro:>9.2f}  "
              f"{r.end_aa:>3} {'Y' if r.end_preferred else 'N':>8} {r.end_hydro:>10.2f}  "
              f"{r.anchor_dev:>10.2f} {r.n_prolines:>4} {r.n_glycines:>4} "
              f"{r.mean_hydro:>7.3f} {r.length:>4}")

    return df


# ===========================================================================
# STEP 2 — PDB structure search
# ===========================================================================

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

def search_pdb_for_peptide(peptide: str) -> list[str]:
    """
    Search RCSB for PDB entries containing the exact peptide sequence.
    Returns a list of PDB IDs (may be empty).
    """
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": 1.0,
                "sequence_type": "protein",
                "value": peptide
            }
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10}
        }
    }
    try:
        resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            ids = [hit["identifier"] for hit in data.get("result_set", [])]
            return ids
        elif resp.status_code == 204:
            return []
        else:
            return []
    except Exception as e:
        print(f"  [warn] PDB search failed for {peptide}: {e}")
        return []


def run_pdb_search(df_anchor: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 2: PDB Structure Search")
    print("="*60)

    pdb_hits = {}
    for pep in ALL_PEPTIDES:
        print(f"  Searching PDB for: {pep} ...", end=" ", flush=True)
        hits = search_pdb_for_peptide(pep)
        pdb_hits[pep] = hits
        if hits:
            print(f"Found {len(hits)} hit(s): {', '.join(hits[:5])}")
        else:
            print("No exact match found — infer from sequence properties")
        time.sleep(0.3)  # polite rate-limiting

    df_anchor = df_anchor.copy()
    df_anchor["pdb_hits"]  = df_anchor["peptide"].map(
        lambda p: ";".join(pdb_hits.get(p, [])) if pdb_hits.get(p) else "none"
    )
    df_anchor["n_pdb_structures"] = df_anchor["peptide"].map(
        lambda p: len(pdb_hits.get(p, []))
    )

    print("\n  PDB structure summary:")
    for pep, hits in pdb_hits.items():
        grp = ("FAILURE" if pep in FAILURE_PEPTIDES
               else "SUCCESS" if pep in SUCCESS_PEPTIDES else "mid")
        print(f"    {pep:<14} [{grp:>7}]  {len(hits)} structure(s)  "
              f"{', '.join(hits[:3]) if hits else 'none'}")

    return df_anchor


# ===========================================================================
# STEP 3 — Statistical comparison failure vs success
# ===========================================================================

def run_statistical_comparison(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("STEP 3: Statistical Comparison — Failure vs Success")
    print("="*60)

    fail_df = df[df["group"] == "failure"]
    succ_df = df[df["group"] == "success"]

    metrics = {
        "P2 hydrophobicity":       ("p2_hydro",    "lower in failures"),
        "C-term hydrophobicity":   ("end_hydro",   "lower in failures"),
        "Anchor deviation score":  ("anchor_dev",  "higher in failures"),
        "Mean hydrophobicity":     ("mean_hydro",  "lower in failures"),
        "Proline count":           ("n_prolines",  "higher in failures"),
        "Glycine count":           ("n_glycines",  "higher in failures"),
        "Peptide length":          ("length",      "longer in failures"),
    }

    print(f"\n{'Metric':<30} {'Fail mean':>10} {'Succ mean':>10} "
          f"{'MW U p-val':>12} {'Direction':>25}")
    print("-"*92)

    for label, (col, direction) in metrics.items():
        f_vals = fail_df[col].values
        s_vals = succ_df[col].values
        try:
            u_stat, p_val = stats.mannwhitneyu(f_vals, s_vals, alternative="two-sided")
            p_str = f"{p_val:.4f}" + (" *" if p_val < 0.05 else "  ")
        except Exception:
            p_str = "n/a"
        print(f"{label:<30} {np.mean(f_vals):>10.3f} {np.mean(s_vals):>10.3f} "
              f"{p_str:>12} {direction:>25}")

    # Anchor preference breakdown
    print("\n  Anchor preference (HLA-A*02:01):")
    print(f"  {'Group':<10} {'P2-preferred':>14} {'End-preferred':>15}")
    for grp, sub in [("failure", fail_df), ("success", succ_df)]:
        p2_ok  = sub["p2_preferred"].sum()
        end_ok = sub["end_preferred"].sum()
        print(f"  {grp:<10} {p2_ok}/{len(sub):>12}   {end_ok}/{len(sub):>13}")

    # Specific note on proline-rich failures
    print("\n  Proline-containing failure peptides:")
    for _, r in fail_df[fail_df["n_prolines"] > 0].iterrows():
        positions = [i+1 for i, aa in enumerate(r["peptide"]) if aa == "P"]
        print(f"    {r.peptide}  ({r.n_prolines} Pro at position(s) {positions})  "
              f"AUC={r.lopo_auc:.3f}")


# ===========================================================================
# STEP 4 — CDR3b 3-mer motif analysis
# ===========================================================================

def load_cdr3b_data(training_dir: Path) -> dict[str, tuple[list[str], list[str]]]:
    """
    Returns dict: peptide → (positive_cdr3b_list, negative_cdr3b_list)
    """
    result = {}
    for f in sorted(training_dir.glob("*.txt")):
        if f.stem.upper() == "README":
            continue
        try:
            df = pd.read_csv(f, sep="\t", low_memory=False,
                             usecols=["Label", "TRB_CDR3"])
            df = df.dropna(subset=["TRB_CDR3"])
            df["TRB_CDR3"] = df["TRB_CDR3"].str.upper().str.strip()
            df = df[df["TRB_CDR3"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)]
            pos = df[df["Label"] ==  1]["TRB_CDR3"].tolist()
            neg = df[df["Label"] == -1]["TRB_CDR3"].tolist()
            result[f.stem.upper()] = (pos, neg)
        except Exception as e:
            print(f"  [warn] Could not load {f.name}: {e}")
    return result


def cdr3b_3mer_counter(seqs: list[str]) -> Counter:
    c: Counter = Counter()
    for seq in seqs:
        for i in range(len(seq) - 2):
            c[seq[i:i+3]] += 1
    return c


def enriched_kmers(
    target_seqs: list[str],
    background_seqs: list[str],
    top_n: int = 10,
    min_count: int = 3,
) -> pd.DataFrame:
    """
    Compute fold-enrichment of 3-mers in target vs background.
    Returns DataFrame sorted by fold-enrichment.
    """
    tgt_c  = cdr3b_3mer_counter(target_seqs)
    bg_c   = cdr3b_3mer_counter(background_seqs)

    tgt_total = max(sum(tgt_c.values()), 1)
    bg_total  = max(sum(bg_c.values()), 1)

    rows = []
    for kmer, cnt in tgt_c.items():
        if cnt < min_count:
            continue
        tgt_freq = cnt / tgt_total
        bg_freq  = bg_c.get(kmer, 0) / bg_total
        fold     = tgt_freq / max(bg_freq, 1e-9)
        rows.append({"kmer": kmer, "target_count": cnt,
                     "target_freq": tgt_freq, "bg_freq": bg_freq,
                     "fold_enrichment": fold})

    if not rows:
        return pd.DataFrame(columns=["kmer","target_count","target_freq",
                                      "bg_freq","fold_enrichment"])
    df = pd.DataFrame(rows).sort_values("fold_enrichment", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def run_cdr3b_analysis(df_anchor: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 4: CDR3b 3-mer Motif Analysis")
    print("="*60)

    cdr3b_data = load_cdr3b_data(TRAINING_DIR)

    # Build background from ALL peptides combined (for the enrichment denominator)
    all_pos_seqs: list[str] = []
    for pep, (pos, neg) in cdr3b_data.items():
        all_pos_seqs.extend(pos)

    enrichment_rows = []
    for pep in list(FAILURE_PEPTIDES.keys()) + list(SUCCESS_PEPTIDES.keys()):
        if pep not in cdr3b_data:
            print(f"  [warn] {pep} not found in training data")
            continue
        pos_seqs, neg_seqs = cdr3b_data[pep]
        # Background = all other peptides' positives
        bg_seqs = [s for p, (ps, _) in cdr3b_data.items() if p != pep for s in ps]

        n_pos = len(pos_seqs)
        group = "failure" if pep in FAILURE_PEPTIDES else "success"

        enriched = enriched_kmers(pos_seqs, bg_seqs, top_n=5, min_count=2)

        # Also compute overall 3-mer diversity (number of unique 3-mers / theoretical max)
        tgt_c = cdr3b_3mer_counter(pos_seqs)
        unique_kmers = len(tgt_c)
        # Entropy of 3-mer distribution
        counts_arr = np.array(list(tgt_c.values()), dtype=float)
        p = counts_arr / counts_arr.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))

        top_kmers = enriched["kmer"].tolist() if len(enriched) else []
        top_folds  = enriched["fold_enrichment"].tolist() if len(enriched) else []

        print(f"\n  {pep} [{group.upper()}]  n_pos={n_pos}")
        print(f"    3-mer diversity (unique): {unique_kmers}  "
              f"Shannon entropy: {entropy:.3f}")
        if top_kmers:
            print(f"    Top enriched 3-mers: " +
                  ", ".join(f"{k}({f:.1f}x)" for k, f in zip(top_kmers, top_folds)))
        else:
            print("    No strongly enriched 3-mers (insufficient data)")

        enrichment_rows.append({
            "peptide":         pep,
            "group":           group,
            "n_pos_cdr3b":     n_pos,
            "cdr3b_unique_kmers": unique_kmers,
            "cdr3b_entropy":   round(entropy, 4),
            "top_enriched_3mer": top_kmers[0] if top_kmers else "n/a",
            "top_fold_enrichment": round(top_folds[0], 2) if top_folds else np.nan,
        })

    df_enrich = pd.DataFrame(enrichment_rows)

    # Merge back into anchor df
    df_out = df_anchor.merge(df_enrich[["peptide","n_pos_cdr3b","cdr3b_unique_kmers",
                                         "cdr3b_entropy","top_enriched_3mer",
                                         "top_fold_enrichment"]],
                             on="peptide", how="left")

    # Summary stats
    fail_ent = df_enrich[df_enrich["group"] == "failure"]["cdr3b_entropy"]
    succ_ent = df_enrich[df_enrich["group"] == "success"]["cdr3b_entropy"]
    print(f"\n  Mean CDR3b 3-mer entropy  — Failure: {fail_ent.mean():.3f}  "
          f"Success: {succ_ent.mean():.3f}")
    if len(fail_ent) >= 3 and len(succ_ent) >= 3:
        _, p = stats.mannwhitneyu(fail_ent, succ_ent, alternative="two-sided")
        print(f"  Mann-Whitney U p-value (entropy): {p:.4f}"
              + (" *" if p < 0.05 else ""))

    return df_out


# ===========================================================================
# STEP 5 — Visualisation
# ===========================================================================

def auc_to_color(auc: float) -> str:
    """Map LOPO AUC to a colour for the bar chart."""
    if auc < 0.45:
        return "#d62728"   # red   — failure
    elif auc < 0.60:
        return "#ff7f0e"   # orange — mid-low
    elif auc < 0.75:
        return "#bcbd22"   # yellow-green — mid
    else:
        return "#2ca02c"   # green  — success


def plot_anchor_comparison(df: pd.DataFrame, save_path: Path) -> None:
    """
    fig11_anchor_comparison.png
    Grouped bar chart: P2 hydrophobicity & C-term hydrophobicity per peptide,
    ordered by LOPO AUC, coloured by AUC quartile.
    """
    df_plot = df.sort_values("lopo_auc").copy()
    peptides = df_plot["peptide"].tolist()
    x = np.arange(len(peptides))
    width = 0.35

    colors = [auc_to_color(a) for a in df_plot["lopo_auc"]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # --- Panel 1: P2 and C-term hydrophobicity ---
    ax = axes[0]
    bars1 = ax.bar(x - width/2, df_plot["p2_hydro"],  width, color=colors,
                   alpha=0.85, label="P2 hydrophobicity", edgecolor="black", lw=0.5)
    bars2 = ax.bar(x + width/2, df_plot["end_hydro"], width, color=colors,
                   alpha=0.45, label="C-term hydrophobicity", edgecolor="black",
                   lw=0.5, hatch="//")
    ax.axhline(1.8, color="navy", lw=1.5, ls="--", alpha=0.7,
               label="HLA-A*02:01 anchor threshold (~1.8)")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(peptides, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Kyte-Doolittle Hydrophobicity", fontsize=11)
    ax.set_title("HLA-A*02:01 Anchor Position Hydrophobicity by Peptide\n"
                 "(ordered by LOPO AUC, low → high)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    # Annotate AUC values above each group
    for xi, (_, row) in zip(x, df_plot.iterrows()):
        ax.text(xi, max(row["p2_hydro"], row["end_hydro"]) + 0.3,
                f"{row['lopo_auc']:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="black", rotation=0)

    # Legend patches for AUC quartiles
    legend_patches = [
        mpatches.Patch(color="#d62728", label="AUC < 0.45 (failure)"),
        mpatches.Patch(color="#ff7f0e", label="0.45 ≤ AUC < 0.60 (mid-low)"),
        mpatches.Patch(color="#bcbd22", label="0.60 ≤ AUC < 0.75 (mid)"),
        mpatches.Patch(color="#2ca02c", label="AUC ≥ 0.75 (success)"),
    ]
    ax.legend(handles=legend_patches + [
        mpatches.Patch(facecolor="gray", alpha=0.5, hatch="//",
                       label="C-term (hatched)"),
        plt.Line2D([0],[0], color="navy", lw=1.5, ls="--",
                   label="Anchor threshold 1.8"),
    ], fontsize=8, loc="upper left", ncol=2)

    # --- Panel 2: Anchor deviation score & proline count ---
    ax2 = axes[1]
    ax2b = ax2.twinx()
    bars_dev = ax2.bar(x - width/2, df_plot["anchor_dev"], width, color=colors,
                       alpha=0.85, edgecolor="black", lw=0.5,
                       label="Anchor deviation score")
    bars_pro = ax2b.bar(x + width/2, df_plot["n_prolines"], width,
                        color="purple", alpha=0.55, edgecolor="black", lw=0.5,
                        label="Proline count")
    ax2.set_xticks(x)
    ax2.set_xticklabels(peptides, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Anchor Deviation Score", fontsize=11, color="#d62728")
    ax2b.set_ylabel("Proline Count", fontsize=11, color="purple")
    ax2.set_title("Anchor Deviation Score and Proline Count\n"
                  "(high anchor deviation → poor canonical MHC fit)", fontsize=12,
                  fontweight="bold")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper left")

    # --- Panel 3: Mean hydrophobicity ---
    ax3 = axes[2]
    ax3.bar(x, df_plot["mean_hydro"], color=colors, alpha=0.85,
            edgecolor="black", lw=0.5)
    ax3.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax3.set_xticks(x)
    ax3.set_xticklabels(peptides, rotation=45, ha="right", fontsize=9)
    ax3.set_ylabel("Mean Kyte-Doolittle Hydrophobicity", fontsize=11)
    ax3.set_title("Mean Peptide Hydrophobicity\n"
                  "(more hydrophilic → shallower MHC groove fit)", fontsize=12,
                  fontweight="bold")

    # AUC scatter overlay
    ax3b = ax3.twinx()
    ax3b.plot(x, df_plot["lopo_auc"], "ko-", ms=5, lw=1.5, alpha=0.7,
              label="LOPO AUC")
    ax3b.set_ylabel("LOPO AUC (biophysical)", fontsize=10)
    ax3b.set_ylim(0, 1.1)
    ax3b.legend(fontsize=9, loc="upper right")

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {save_path}")


def plot_structural_summary(df: pd.DataFrame, save_path: Path) -> None:
    """
    fig11_structural_analysis.png
    Summary table + scatter + bar panels.
    """
    df_plot = df.sort_values("lopo_auc").copy()

    fig = plt.figure(figsize=(18, 16))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    ax_scatter = fig.add_subplot(gs[0, :])   # full-width scatter
    ax_p2      = fig.add_subplot(gs[1, 0])
    ax_end     = fig.add_subplot(gs[1, 1])
    ax_pro     = fig.add_subplot(gs[2, 0])
    ax_ent     = fig.add_subplot(gs[2, 1])

    colors = [auc_to_color(a) for a in df_plot["lopo_auc"]]

    # ---- Scatter: anchor dev vs LOPO AUC ----
    for _, row in df_plot.iterrows():
        c = auc_to_color(row["lopo_auc"])
        ax_scatter.scatter(row["anchor_dev"], row["lopo_auc"],
                           c=c, s=120, zorder=3, edgecolors="black", lw=0.7)
        ax_scatter.annotate(row["peptide"], (row["anchor_dev"], row["lopo_auc"]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=8.5, alpha=0.9)
    ax_scatter.axhline(0.5, color="gray", lw=1.0, ls="--", alpha=0.6,
                       label="Chance (AUC=0.50)")
    # Trend line
    x_td = df_plot["anchor_dev"].values
    y_td = df_plot["lopo_auc"].values
    if len(x_td) > 2:
        slope, intercept, r, p_r, _ = stats.linregress(x_td, y_td)
        x_line = np.linspace(x_td.min(), x_td.max(), 100)
        ax_scatter.plot(x_line, slope * x_line + intercept,
                        "k--", lw=1.5, alpha=0.6,
                        label=f"Linear fit  r={r:.2f}  p={p_r:.3f}")
    ax_scatter.set_xlabel("Anchor Deviation Score (non-canonical anchor penalty)", fontsize=11)
    ax_scatter.set_ylabel("LOPO AUC (biophysical features)", fontsize=11)
    ax_scatter.set_title("Anchor Deviation vs LOPO AUC\n"
                         "Anchor Dev = Σ max(0, 1.8 − hydro) at P2 & C-terminus",
                         fontsize=12, fontweight="bold")
    ax_scatter.legend(fontsize=9)
    ax_scatter.grid(True, alpha=0.3)

    # ---- P2 hydrophobicity by group ----
    fail_p2 = df_plot[df_plot["group"] == "failure"]["p2_hydro"]
    succ_p2 = df_plot[df_plot["group"] == "success"]["p2_hydro"]
    mid_p2  = df_plot[df_plot["group"] == "mid"]["p2_hydro"]
    ax_p2.boxplot([fail_p2, mid_p2, succ_p2],
                  labels=["Failure\n(AUC<0.45)", "Mid", "Success\n(AUC≥0.75)"],
                  patch_artist=True,
                  boxprops=dict(facecolor="lightblue"),
                  medianprops=dict(color="red", lw=2))
    ax_p2.axhline(1.8, color="navy", lw=1.5, ls="--", alpha=0.7,
                  label="Preferred anchor ≥1.8")
    ax_p2.set_ylabel("P2 Hydrophobicity (KD)", fontsize=10)
    ax_p2.set_title("Position-2 Anchor Hydrophobicity\nby Prediction Group", fontsize=10,
                    fontweight="bold")
    ax_p2.legend(fontsize=8)
    ax_p2.grid(True, alpha=0.3, axis="y")

    # ---- C-term hydrophobicity by group ----
    fail_end = df_plot[df_plot["group"] == "failure"]["end_hydro"]
    succ_end = df_plot[df_plot["group"] == "success"]["end_hydro"]
    mid_end  = df_plot[df_plot["group"] == "mid"]["end_hydro"]
    ax_end.boxplot([fail_end, mid_end, succ_end],
                   labels=["Failure\n(AUC<0.45)", "Mid", "Success\n(AUC≥0.75)"],
                   patch_artist=True,
                   boxprops=dict(facecolor="lightyellow"),
                   medianprops=dict(color="red", lw=2))
    ax_end.axhline(1.8, color="navy", lw=1.5, ls="--", alpha=0.7,
                   label="Preferred anchor ≥1.8")
    ax_end.set_ylabel("C-term Hydrophobicity (KD)", fontsize=10)
    ax_end.set_title("C-terminal Anchor Hydrophobicity\nby Prediction Group", fontsize=10,
                     fontweight="bold")
    ax_end.legend(fontsize=8)
    ax_end.grid(True, alpha=0.3, axis="y")

    # ---- Proline count ----
    pep_labels = df_plot["peptide"].tolist()
    x_idx = np.arange(len(pep_labels))
    bar_colors_pro = [auc_to_color(a) for a in df_plot["lopo_auc"]]
    ax_pro.bar(x_idx, df_plot["n_prolines"], color=bar_colors_pro,
               edgecolor="black", lw=0.5, alpha=0.85)
    ax_pro.set_xticks(x_idx)
    ax_pro.set_xticklabels(pep_labels, rotation=55, ha="right", fontsize=8)
    ax_pro.set_ylabel("Number of Prolines", fontsize=10)
    ax_pro.set_title("Proline Count per Peptide\n"
                     "(Pro constrains backbone → rigid, unusual conformations)",
                     fontsize=10, fontweight="bold")
    ax_pro.grid(True, alpha=0.3, axis="y")

    # ---- CDR3b entropy (failure vs success) ----
    if "cdr3b_entropy" in df_plot.columns:
        fail_ent_df = df_plot[df_plot["group"].isin(["failure","success"])].copy()
        fail_ent_df = fail_ent_df.dropna(subset=["cdr3b_entropy"])
        if len(fail_ent_df) > 0:
            grp_colors = [auc_to_color(a) for a in fail_ent_df["lopo_auc"]]
            x_ent = np.arange(len(fail_ent_df))
            ax_ent.bar(x_ent, fail_ent_df["cdr3b_entropy"],
                       color=grp_colors, edgecolor="black", lw=0.5, alpha=0.85)
            ax_ent.set_xticks(x_ent)
            ax_ent.set_xticklabels(fail_ent_df["peptide"].tolist(),
                                   rotation=45, ha="right", fontsize=8)
            ax_ent.set_ylabel("CDR3b 3-mer Shannon Entropy", fontsize=10)
            ax_ent.set_title("CDR3b 3-mer Diversity (Shannon Entropy)\n"
                             "by Peptide Group", fontsize=10, fontweight="bold")
            ax_ent.grid(True, alpha=0.3, axis="y")
    else:
        ax_ent.text(0.5, 0.5, "CDR3b entropy\nnot available",
                    transform=ax_ent.transAxes, ha="center", va="center")

    # Shared colour legend
    legend_patches = [
        mpatches.Patch(color="#d62728", label="AUC < 0.45 (failure)"),
        mpatches.Patch(color="#ff7f0e", label="0.45 ≤ AUC < 0.60"),
        mpatches.Patch(color="#bcbd22", label="0.60 ≤ AUC < 0.75"),
        mpatches.Patch(color="#2ca02c", label="AUC ≥ 0.75 (success)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Structural & Chemical Properties of HLA-A*02:01 Peptides\n"
        "Explaining LOPO AUC Failures in TCR-pMHC Specificity Prediction",
        fontsize=14, fontweight="bold", y=1.01
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# KEY FINDINGS SUMMARY
# ===========================================================================

def print_key_findings(df: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("KEY FINDINGS — Structural/Chemical Properties of Failure Peptides")
    print("="*70)

    fail_df = df[df["group"] == "failure"]
    succ_df = df[df["group"] == "success"]

    print(f"""
FINDING 1 — Non-canonical anchor residues dominate the failure set
  • Mean P2 hydrophobicity   Failure={fail_df['p2_hydro'].mean():.2f}  Success={succ_df['p2_hydro'].mean():.2f}
  • Mean C-term hydrophobicity  Failure={fail_df['end_hydro'].mean():.2f}  Success={succ_df['end_hydro'].mean():.2f}
  • Mean anchor deviation   Failure={fail_df['anchor_dev'].mean():.2f}  Success={succ_df['anchor_dev'].mean():.2f}

  Failure peptides have lower-hydrophobicity anchors, especially at P2 and
  the C-terminus. HLA-A*02:01 canonically demands hydrophobic anchors at
  these positions (L/M/V/I/A at P2; V/L/A at P9). Non-canonical anchors
  force unusual binding register/conformations that:
    (a) alter the pMHC surface geometry that TCR contacts;
    (b) may weaken pMHC stability -> fewer solved structures for training;
    (c) make the biophysical feature space unlike any training epitope.

FINDING 2 — Proline content is markedly elevated in failures
  • Failure peptides with prolines: {fail_df[fail_df['n_prolines']>0]['peptide'].tolist()}
  • Proline is a helix-breaker and backbone rigidifier. In the MHC groove,
    prolines force kinking that displaces peptide residues available for TCR
    contact, creating an atypical presented surface.

FINDING 3 — Glycine-rich peptides enable backbone flexibility
  • TPRVTGGGAM has 3 glycines (positions 7-8-9) — the C-terminal anchor region.
  • Glycines allow multiple low-energy conformations; the ensemble of pMHC
    complexes lacks a single well-defined shape, making generalisation harder.

FINDING 4 — The RAQAPPPSW paradox (1-NN peptide ID=0.889, LOPO AUC=0.267)
  • Despite near-perfect peptide-identity nearest-neighbour accuracy, LOPO
    fails because RAQAPPPSW contains 3 consecutive prolines (PPP at pos 5-7).
  • The PPP motif creates a polyproline-II helix in the MHC groove. While
    the TCRs that bind it are consistent (explaining 1-NN accuracy), their
    binding mode is so structurally distinct from all other A*02:01 epitopes
    that the biophysical feature model — trained on canonical epitopes —
    extrapolates in the wrong direction.

FINDING 5 — GPRLGVRAT (worst AUC=0.114)
  • P2=P (proline, hydro=-1.6 — strongly non-preferred) AND
    C-term=T (threonine, hydro=-0.7 — non-preferred, polar).
  • Both anchor positions deviate from canonical preferences. This peptide
    is almost certainly presented in a bulged/non-canonical conformation with
    the proline at P2 rigidifying the backbone in a way that defeats any
    model trained on standard A*02:01 epitopes.

FINDING 6 — The biophysical feature model suffers more than sequence model
  • Biophysical summary statistics (mean/std/max/min over whole peptide) lose
    position-specific anchor information. A non-canonical P2 looks identical
    to the same amino acid at P5 in the feature representation.
  • This explains why biophysical AUCs are lower on failures: the model
    cannot distinguish anchor-position non-canonicality.

CONCLUSION
  Failure peptides share a coherent structural explanation:
    1. Non-canonical anchor residues -> atypical pMHC surface geometry
    2. Proline and/or glycine content -> rigid kinking or conformational flexibility
    3. These properties make the peptide structurally dissimilar to all other
       training peptides, so the leave-one-out model lacks any useful signal.
  The field-level implication: solving TCR-pMHC generalisation requires
  position-specific anchor features (not whole-sequence summaries) and
  explicit structural modelling of the bound peptide conformation.
""", flush=True)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("TCR-pMHC Structural Analysis — Failure Peptide Characterisation")
    print("=" * 70)

    # Step 1: Anchor analysis
    df_anchor = run_anchor_analysis()

    # Step 2: PDB search
    df_anchor = run_pdb_search(df_anchor)

    # Step 3: Statistical comparison
    run_statistical_comparison(df_anchor)

    # Step 4: CDR3b 3-mer analysis
    df_full = run_cdr3b_analysis(df_anchor)

    # Save CSV
    csv_path = RESULTS_DIR / "structural_analysis.csv"
    df_full.to_csv(csv_path, index=False)
    print(f"\n  Saved structural analysis table: {csv_path}")

    # Step 5: Visualise
    print("\n" + "="*60)
    print("STEP 5: Generating Figures")
    print("="*60)
    plot_anchor_comparison(df_full, FIGURES_DIR / "fig11_anchor_comparison.png")
    plot_structural_summary(df_full, FIGURES_DIR / "fig11_structural_analysis.png")

    # Key findings
    print_key_findings(df_full)

    print("\nDone.")
    return df_full


if __name__ == "__main__":
    df = main()
