"""
immrep23_replication.py
=======================
Cross-benchmark validation (Tier 2, Item 7).

Scientific question
-------------------
On IMMREP_2022 (17 peptides, HLA-A*02:01), random-swap negatives inflate
biophysical AUC by +22.5pp while deflating sequence AUC by -8.6pp.
IMMREP23 has 20 peptides across 6 HLA alleles with Levenshtein-filtered
negatives (supplied test set). We test whether the same directional bias
pattern is a property of the feature spaces, not one benchmark's quirks.

Analysis design
---------------
- Training positives: VDJdb_paired_chain.csv filtered to the 20 IMMREP23 peptides
- Negative strategies tested: random_swap, within_cluster
- Feature types: biophysical, sequence
- Evaluation: 5-fold stratified CV (n_repeats=3) vs LOPO CV
- AUC inflation = auc_standard - auc_lopo
- Comparison against IMMREP22 reference values

Outputs
-------
  results/immrep23_replication.csv
  results/figures/fig16_immrep23_replication.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── live output ────────────────────────────────────────────────────────────────
sys.stdout.reconfigure(line_buffering=True)

# ── path setup ─────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data" / "immrep23" / "IMMREP23" / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from features          import extract_features
from negative_sampling import random_swap, within_cluster, leave_one_peptide_out, random_swap as _rs
from evaluation        import _standard_cv_benchmark, _lopo_benchmark

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── IMMREP22 reference values (from results/benchmark_results.csv) ─────────────
IMMREP22_REF = {
    ("random_swap",    "biophysical"): {"auc_standard": 0.7637, "auc_lopo": 0.5390, "auc_inflation": +0.2248},
    ("epitope_balanced","biophysical"):{"auc_standard": 0.7539, "auc_lopo": 0.5390, "auc_inflation": +0.2149},
    ("within_cluster", "biophysical"): {"auc_standard": 0.6024, "auc_lopo": 0.5390, "auc_inflation": +0.0635},
    ("shuffled_cdr3",  "biophysical"): {"auc_standard": 0.2811, "auc_lopo": 0.5390, "auc_inflation": -0.2578},
    ("random_swap",    "sequence"):    {"auc_standard": 0.4955, "auc_lopo": 0.5810, "auc_inflation": -0.0855},
    ("epitope_balanced","sequence"):   {"auc_standard": 0.5854, "auc_lopo": 0.5810, "auc_inflation": +0.0044},
    ("within_cluster", "sequence"):    {"auc_standard": 0.5486, "auc_lopo": 0.5810, "auc_inflation": -0.0324},
    ("shuffled_cdr3",  "sequence"):    {"auc_standard": 0.9999, "auc_lopo": 0.5810, "auc_inflation": +0.4189},
}

# Strategies to run in this replication (matching core IMMREP22 finding)
STRATEGIES = ["random_swap", "within_cluster"]
FEATURE_TYPES = ["biophysical", "sequence"]
N_REPEATS = 3


# ── Step 1: Load and explore data ──────────────────────────────────────────────

def load_immrep23_positives() -> pd.DataFrame:
    """
    Load VDJdb training positives, normalise column names to match our pipeline,
    and filter to the 20 IMMREP23 benchmark peptides.
    """
    vdj_path = DATA_DIR / "VDJdb_paired_chain.csv"
    sol_path  = DATA_DIR / "solutions.csv"

    logger.info(f"Loading VDJdb training data from {vdj_path}")
    vdj = pd.read_csv(vdj_path)
    sol = pd.read_csv(sol_path)

    # ── Explore ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  IMMREP23 DATA EXPLORATION")
    print("=" * 70)
    print(f"  VDJdb columns    : {vdj.columns.tolist()}")
    print(f"  VDJdb shape      : {vdj.shape}")
    print(f"  Target values    : {vdj['Target'].value_counts().to_dict()}")
    print(f"  Unique peptides  : {vdj['Peptide'].nunique()}")
    print(f"  Unique HLA       : {vdj['HLA'].nunique()}")
    print()
    print(f"  Solutions columns: {sol.columns.tolist()}")
    print(f"  Solutions shape  : {sol.shape}")
    print(f"  Label values     : {sol['Label'].value_counts().to_dict()}")
    print(f"  Unique peptides  : {sol['Peptide'].nunique()}")
    print(f"  HLA alleles      : {sorted(sol['HLA'].unique())}")
    print()

    # ── Identify IMMREP23 benchmark peptides ───────────────────────────────────
    benchmark_peptides = set(sol["Peptide"].unique())
    print(f"  IMMREP23 benchmark peptides ({len(benchmark_peptides)}):")
    for pep in sorted(benchmark_peptides):
        n_vdj = len(vdj[vdj["Peptide"] == pep])
        n_pos  = len(sol[(sol["Peptide"] == pep) & (sol["Label"] == 1)])
        print(f"    {pep:<15}  VDJdb_n={n_vdj:4d}  test_positives={n_pos:3d}")
    print()

    # ── Build positives: VDJdb rows for IMMREP23 peptides ─────────────────────
    # VDJdb already contains only Target==1 rows (all 11312 are positives)
    pos = vdj[vdj["Peptide"].isin(benchmark_peptides)].copy()

    # Normalise column names to match pipeline expectations
    pos = pos.rename(columns={"Peptide": "peptide", "HLA": "mhc"})
    pos["label"] = 1
    # CDR3b already named correctly

    # Drop rows with missing CDR3b
    before = len(pos)
    pos = pos.dropna(subset=["CDR3b"]).reset_index(drop=True)
    after = len(pos)
    if before != after:
        logger.warning(f"  Dropped {before - after} rows with missing CDR3b")

    print(f"  Positives for analysis : {len(pos)}")
    print(f"  Unique peptides        : {pos['peptide'].nunique()}")
    print(f"  Unique HLA             : {pos['mhc'].nunique()}")
    print()

    # Verify minimum positives per peptide for LOPO
    pep_counts = pos.groupby("peptide").size()
    print("  Positives per peptide:")
    for pep, n in pep_counts.sort_values(ascending=False).items():
        flag = " *** LOW" if n < 5 else ""
        print(f"    {pep:<15}  n={n:4d}{flag}")
    print()

    # Filter out peptides with <2 positives (LOPO requires at least 1 test sample)
    valid_peps = pep_counts[pep_counts >= 2].index
    excluded = pep_counts[pep_counts < 2]
    if len(excluded) > 0:
        print(f"  Excluding peptides with <2 positives: {excluded.to_dict()}")
        pos = pos[pos["peptide"].isin(valid_peps)].reset_index(drop=True)
        print(f"  After filtering: {len(pos)} positives, {pos['peptide'].nunique()} peptides")

    return pos


# ── Step 2: Run analysis ───────────────────────────────────────────────────────

def run_immrep23_analysis(df_positives: pd.DataFrame) -> list[dict]:
    """
    For each (strategy, feature_type) pair: run standard CV + LOPO,
    compute AUC inflation, compare to IMMREP22 reference.
    """
    strategy_fns = {
        "random_swap":   random_swap,
        "within_cluster": within_cluster,
    }

    results = []
    conditions = [(s, f) for s in STRATEGIES for f in FEATURE_TYPES]
    total = len(conditions)

    print("=" * 70)
    print("  RUNNING ANALYSIS")
    print("=" * 70)

    for i, (strategy, feat) in enumerate(conditions, 1):
        print(f"\n[{i}/{total}]  strategy={strategy}  features={feat}")
        sampling_fn = strategy_fns[strategy]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Standard CV
            print(f"  Running standard CV (n_repeats={N_REPEATS}, 5-fold)...")
            std = _standard_cv_benchmark(
                df_positives, sampling_fn, feat, n_repeats=N_REPEATS
            )
            auc_std = std["auc_standard_mean"]
            print(f"  AUC_standard = {auc_std:.4f} ± {std['auc_standard_std']:.4f}")

            # LOPO CV
            print(f"  Running LOPO CV ({df_positives['peptide'].nunique()} folds)...")
            lopo = _lopo_benchmark(df_positives, feat)
            auc_lopo = lopo["auc_lopo_mean"]
            print(f"  AUC_lopo     = {auc_lopo:.4f} ± {lopo['auc_lopo_std']:.4f}")

        auc_inflation = auc_std - auc_lopo if not (np.isnan(auc_std) or np.isnan(auc_lopo)) else float("nan")
        print(f"  Inflation    = {auc_inflation:+.4f}")

        # IMMREP22 reference
        ref = IMMREP22_REF.get((strategy, feat), {})
        ref_inflation = ref.get("auc_inflation", float("nan"))
        directional_match = (
            (auc_inflation > 0 and ref_inflation > 0) or
            (auc_inflation < 0 and ref_inflation < 0)
        ) if not np.isnan(auc_inflation) else None

        print(f"  IMMREP22 ref inflation = {ref_inflation:+.4f}")
        print(f"  Directional match      = {directional_match}")

        # Per-peptide LOPO breakdown
        per_pep = lopo.get("per_peptide_auc", {})
        if per_pep:
            print(f"  Per-peptide LOPO AUC:")
            for pep, auc in sorted(per_pep.items(), key=lambda x: -x[1]):
                flag = " ***" if auc < 0.55 else ""
                print(f"    {pep:<15}  {auc:.4f}{flag}")

        results.append({
            "benchmark":           "IMMREP23",
            "sampling_strategy":   strategy,
            "feature_type":        feat,
            "auc_standard":        auc_std,
            "auc_standard_std":    std["auc_standard_std"],
            "auc_lopo":            auc_lopo,
            "auc_lopo_std":        lopo["auc_lopo_std"],
            "auc_inflation":       auc_inflation,
            "immrep22_inflation":  ref_inflation,
            "directional_match":   directional_match,
            "model_lr_standard":   std.get("model_lr_mean_auc", float("nan")),
            "model_rf_standard":   std.get("model_rf_mean_auc", float("nan")),
            "model_lr_lopo":       lopo.get("model_lr_lopo_mean", float("nan")),
            "model_rf_lopo":       lopo.get("model_rf_lopo_mean", float("nan")),
            "n_peptides":          df_positives["peptide"].nunique(),
            "n_positives":         len(df_positives),
            "per_peptide_auc":     per_pep,
        })

    return results


# ── Step 3: Print comparison table ────────────────────────────────────────────

def print_comparison_table(results: list[dict]) -> None:
    print("\n\n" + "=" * 90)
    print("  CROSS-BENCHMARK COMPARISON: IMMREP22 vs IMMREP23")
    print("=" * 90)
    print(
        f"  {'Strategy':<22} {'Features':<14} "
        f"{'IMMREP22_inf':>13} {'IMMREP23_std':>13} {'IMMREP23_lopo':>14} "
        f"{'IMMREP23_inf':>13} {'DirMatch':>9}"
    )
    print("-" * 90)

    for r in results:
        dm = "YES" if r["directional_match"] else ("NO" if r["directional_match"] is False else "N/A")
        print(
            f"  {r['sampling_strategy']:<22} {r['feature_type']:<14} "
            f"{r['immrep22_inflation']:>+13.4f} "
            f"{r['auc_standard']:>13.4f} "
            f"{r['auc_lopo']:>14.4f} "
            f"{r['auc_inflation']:>+13.4f} "
            f"{dm:>9}"
        )
    print("=" * 90)

    # Directional conclusion
    n_match = sum(1 for r in results if r["directional_match"] is True)
    n_total = sum(1 for r in results if r["directional_match"] is not None)
    print(f"\n  Directional matches: {n_match}/{n_total}")

    # Core claim: biophysical > sequence inflation under random_swap
    bio_rs  = next((r for r in results if r["sampling_strategy"]=="random_swap" and r["feature_type"]=="biophysical"), None)
    seq_rs  = next((r for r in results if r["sampling_strategy"]=="random_swap" and r["feature_type"]=="sequence"), None)
    bio_wc  = next((r for r in results if r["sampling_strategy"]=="within_cluster" and r["feature_type"]=="biophysical"), None)
    seq_wc  = next((r for r in results if r["sampling_strategy"]=="within_cluster" and r["feature_type"]=="sequence"), None)

    print("\n  KEY FINDINGS:")

    if bio_rs and seq_rs:
        diff23 = bio_rs["auc_inflation"] - seq_rs["auc_inflation"]
        diff22 = IMMREP22_REF[("random_swap","biophysical")]["auc_inflation"] - \
                 IMMREP22_REF[("random_swap","sequence")]["auc_inflation"]
        same_dir = (diff23 > 0) == (diff22 > 0)
        print(f"  random_swap: bio_inflation - seq_inflation")
        print(f"    IMMREP22: {diff22:+.4f}  (bio={IMMREP22_REF[('random_swap','biophysical')]['auc_inflation']:+.4f}, seq={IMMREP22_REF[('random_swap','sequence')]['auc_inflation']:+.4f})")
        print(f"    IMMREP23: {diff23:+.4f}  (bio={bio_rs['auc_inflation']:+.4f}, seq={seq_rs['auc_inflation']:+.4f})")
        print(f"    Directional replication: {'YES' if same_dir else 'NO'}")

    if bio_wc and seq_wc:
        diff23_wc = bio_wc["auc_inflation"] - seq_wc["auc_inflation"]
        diff22_wc = IMMREP22_REF[("within_cluster","biophysical")]["auc_inflation"] - \
                    IMMREP22_REF[("within_cluster","sequence")]["auc_inflation"]
        same_dir_wc = (diff23_wc > 0) == (diff22_wc > 0)
        print(f"  within_cluster: bio_inflation - seq_inflation")
        print(f"    IMMREP22: {diff22_wc:+.4f}  (bio={IMMREP22_REF[('within_cluster','biophysical')]['auc_inflation']:+.4f}, seq={IMMREP22_REF[('within_cluster','sequence')]['auc_inflation']:+.4f})")
        print(f"    IMMREP23: {diff23_wc:+.4f}  (bio={bio_wc['auc_inflation']:+.4f}, seq={seq_wc['auc_inflation']:+.4f})")
        print(f"    Directional replication: {'YES' if same_dir_wc else 'NO'}")

    # Within-cluster reduces inflation vs random-swap (for biophysical)?
    if bio_rs and bio_wc:
        less_inf = bio_wc["auc_inflation"] < bio_rs["auc_inflation"]
        print(f"\n  within_cluster < random_swap inflation (biophysical):")
        print(f"    IMMREP22: {'YES' if IMMREP22_REF[('within_cluster','biophysical')]['auc_inflation'] < IMMREP22_REF[('random_swap','biophysical')]['auc_inflation'] else 'NO'}")
        print(f"    IMMREP23: {'YES' if less_inf else 'NO'}")

    print()


# ── Step 4: Generate figure ────────────────────────────────────────────────────

def generate_figure(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Build inflation matrices for both benchmarks
    # Rows: strategies, Cols: feature types
    strategies_plot = ["random_swap", "within_cluster"]
    feats_plot      = ["biophysical", "sequence"]

    # IMMREP22 matrix
    mat22 = np.full((len(strategies_plot), len(feats_plot)), np.nan)
    for i, s in enumerate(strategies_plot):
        for j, f in enumerate(feats_plot):
            val = IMMREP22_REF.get((s, f), {}).get("auc_inflation", np.nan)
            mat22[i, j] = val

    # IMMREP23 matrix
    mat23 = np.full((len(strategies_plot), len(feats_plot)), np.nan)
    for r in results:
        if r["sampling_strategy"] in strategies_plot and r["feature_type"] in feats_plot:
            i = strategies_plot.index(r["sampling_strategy"])
            j = feats_plot.index(r["feature_type"])
            mat23[i, j] = r["auc_inflation"]

    # Colour range symmetric around 0
    vmax = max(
        np.nanmax(np.abs(mat22)),
        np.nanmax(np.abs(mat23)),
    )
    vmax = max(vmax, 0.05)  # at least ±0.05 range

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cmap = "RdYlGn_r"  # red = high inflation, green = low/negative

    for ax, mat, title, bench in zip(
        axes,
        [mat22, mat23],
        ["IMMREP_2022\n(17 peptides, HLA-A*02:01)", "IMMREP23\n(20 peptides, 6 HLA alleles)"],
        ["IMMREP22", "IMMREP23"],
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                            fontsize=13, fontweight="bold", color=color)

        ax.set_xticks(range(len(feats_plot)))
        ax.set_xticklabels([f.capitalize() for f in feats_plot], fontsize=11)
        ax.set_yticks(range(len(strategies_plot)))
        ax.set_yticklabels([s.replace("_", "\n") for s in strategies_plot], fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Feature Type", fontsize=10)
        ax.set_ylabel("Negative Sampling Strategy", fontsize=10)

        plt.colorbar(im, ax=ax, label="AUC Inflation (standard − LOPO)", shrink=0.8)

    # Determine overall conclusion
    bio_rs23 = next((r for r in results if r["sampling_strategy"]=="random_swap"    and r["feature_type"]=="biophysical"), None)
    seq_rs23 = next((r for r in results if r["sampling_strategy"]=="random_swap"    and r["feature_type"]=="sequence"),    None)
    if bio_rs23 and seq_rs23:
        bio_inf23 = bio_rs23["auc_inflation"]
        seq_inf23 = seq_rs23["auc_inflation"]
        bio_inf22 = IMMREP22_REF[("random_swap","biophysical")]["auc_inflation"]
        seq_inf22 = IMMREP22_REF[("random_swap","sequence")]["auc_inflation"]
        same_dir = (bio_inf23 > seq_inf23) == (bio_inf22 > seq_inf22)
        conclusion = (
            "REPLICATES: Biophysical > Sequence inflation under random_swap on both benchmarks"
            if same_dir else
            "DOES NOT REPLICATE: Directional pattern reversed on IMMREP23"
        )
        fig.suptitle(
            f"AUC Inflation: Negative Sampling Bias Cross-Benchmark Validation\n"
            f"Conclusion: {conclusion}",
            fontsize=11, y=1.04
        )
    else:
        fig.suptitle("AUC Inflation: Negative Sampling Bias Cross-Benchmark Validation", fontsize=12)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig16_immrep23_replication.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved to {out_path}")


# ── Step 5: Save results ───────────────────────────────────────────────────────

def save_results(results: list[dict]) -> None:
    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "per_peptide_auc"}
        flat.append(row)
    df = pd.DataFrame(flat)
    out_path = RESULTS_DIR / "immrep23_replication.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")
    print(f"  Results saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("  IMMREP23 CROSS-BENCHMARK REPLICATION STUDY")
    print("  Tier 2 Item 7: Directional AUC inflation replication")
    print("=" * 70 + "\n")

    # 1. Load data
    df_positives = load_immrep23_positives()

    # 2. Run analysis
    results = run_immrep23_analysis(df_positives)

    # 3. Print comparison
    print_comparison_table(results)

    # 4. Save results
    save_results(results)

    # 5. Generate figure
    print("\n  Generating figure...")
    generate_figure(results)

    # 6. Final conclusion
    bio_rs = next((r for r in results if r["sampling_strategy"]=="random_swap" and r["feature_type"]=="biophysical"), None)
    seq_rs = next((r for r in results if r["sampling_strategy"]=="random_swap" and r["feature_type"]=="sequence"), None)

    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    if bio_rs and seq_rs:
        bio_inf = bio_rs["auc_inflation"]
        seq_inf = seq_rs["auc_inflation"]
        bio22   = IMMREP22_REF[("random_swap","biophysical")]["auc_inflation"]
        seq22   = IMMREP22_REF[("random_swap","sequence")]["auc_inflation"]
        replicated = (bio_inf > seq_inf) == (bio22 > seq22)
        print(f"  IMMREP22: bio_inflation={bio22:+.4f}, seq_inflation={seq22:+.4f}")
        print(f"  IMMREP23: bio_inflation={bio_inf:+.4f}, seq_inflation={seq_inf:+.4f}")
        print()
        if replicated:
            print("  RESULT: DIRECTIONAL PATTERN REPLICATES ON IMMREP23.")
            print("  Biophysical features show more (or less-negative) AUC inflation")
            print("  under random_swap negatives than sequence features on both benchmarks.")
            print("  This supports the hypothesis that the directional bias is a property")
            print("  of the feature spaces, not an artefact of the IMMREP_2022 dataset.")
        else:
            print("  RESULT: DIRECTIONAL PATTERN DOES NOT REPLICATE ON IMMREP23.")
            print("  The relative ordering of biophysical vs sequence inflation under")
            print("  random_swap is reversed between benchmarks. The bias may be")
            print("  dataset-specific or modulated by HLA diversity / peptide composition.")
    print("=" * 70)


if __name__ == "__main__":
    main()
