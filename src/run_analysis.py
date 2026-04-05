"""
run_analysis.py
===============
Orchestrates the negative-sampling-bias benchmark on the IMMREP_2022 dataset.

Novel scientific question
-------------------------
By how many AUC points does each negative sampling strategy inflate reported
TCR-pMHC specificity prediction performance relative to the honest
leave-one-peptide-out (LOPO) score?  Does this inflation differ between
sequence-based and biophysical feature sets?

Experimental design
-------------------
  Negative sampling strategies (5):
    1. random_swap         — standard field practice
    2. epitope_balanced    — equal neg contribution per epitope
    3. within_cluster      — harder negatives from same V-gene cluster
    4. shuffled_cdr3       — synthetic negatives (shuffled CDR3b)
    5. leave_one_peptide_out — used as CV scheme (see note below)

  Feature sets (2):
    - sequence    : BLOSUM62 position encoding (faster, skip 3-mers)
    - biophysical : 104 Kidera/charge/hydrophobicity/volume features

  Classifiers: LogisticRegression + RandomForestClassifier
  CV: n_repeats=3 (standard), leave-one-peptide-out (honest)
  Metric: ROC-AUC

  Key output: auc_inflation = auc_standard(strategy) - auc_lopo(random_swap)
              This quantifies the benchmark contamination per strategy.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── imports ──────────────────────────────────────────────────────────────────
from immrep_loader import load_training_data, summarize
from evaluation    import run_benchmark

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
SAMPLING_STRATEGIES = [
    "random_swap",
    "epitope_balanced",
    "within_cluster",
    "shuffled_cdr3",
]

FEATURE_TYPES = ["biophysical", "sequence"]

N_REPEATS = 3  # per condition (× 5 CV folds = 15 AUC estimates per condition)


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_results_table(rows: list[dict]) -> None:
    """Pretty-print results as an ASCII table."""
    print("\n")
    print("=" * 90)
    print(
        f"{'Strategy':<22} {'Features':<14} "
        f"{'AUC_standard':>13} {'AUC_lopo':>10} {'INFLATION':>11} "
        f"{'LR_std':>8} {'RF_std':>8}"
    )
    print("-" * 90)
    for r in sorted(rows, key=lambda x: -x.get("auc_inflation", 0)):
        s  = r.get("sampling_strategy", "")
        f  = r.get("feature_type", "")
        as_= r.get("auc_standard", float("nan"))
        al = r.get("auc_lopo",     float("nan"))
        ai = r.get("auc_inflation",float("nan"))
        lr = r.get("model_lr_standard_mean", float("nan"))
        rf = r.get("model_rf_standard_mean", float("nan"))
        print(
            f"{s:<22} {f:<14} "
            f"{as_:>13.4f} {al:>10.4f} {ai:>+11.4f} "
            f"{lr:>8.4f} {rf:>8.4f}"
        )
    print("=" * 90)


def _print_per_peptide_table(rows: list[dict]) -> None:
    """Show LOPO AUC per peptide for each feature type (averaged over strategies)."""
    # Collect per-peptide AUCs per feature type
    pep_aucs: dict[str, dict[str, list[float]]] = {}  # feat → pep → [aucs]
    for r in rows:
        feat = r["feature_type"]
        for pep, auc in r.get("per_peptide_auc", {}).items():
            pep_aucs.setdefault(feat, {}).setdefault(pep, []).append(auc)

    for feat, pep_dict in pep_aucs.items():
        print(f"\n  LOPO per-peptide AUC  [{feat} features]")
        print(f"  {'Peptide':<15} {'Mean AUC':>9} {'SD':>7} {'N':>4}")
        print(f"  {'-'*15} {'-'*9} {'-'*7} {'-'*4}")
        for pep, aucs in sorted(pep_dict.items(), key=lambda x: -np.nanmean(x[1])):
            valid = [a for a in aucs if not np.isnan(a)]
            mean_a = np.mean(valid) if valid else float("nan")
            std_a  = np.std(valid)  if valid else float("nan")
            flag = "  *** poor" if mean_a < 0.55 else ""
            print(f"  {pep:<15} {mean_a:>9.4f} {std_a:>7.4f} {len(valid):>4}{flag}")


def _print_inflation_by_strategy(rows: list[dict]) -> None:
    """Summarize mean inflation per strategy (averaged over feature types)."""
    strat_inf: dict[str, list[float]] = {}
    for r in rows:
        s = r["sampling_strategy"]
        ai = r.get("auc_inflation", float("nan"))
        if not np.isnan(ai):
            strat_inf.setdefault(s, []).append(ai)

    print("\n  AUC Inflation by strategy (mean ± SD across feature types):")
    print(f"  {'Strategy':<22} {'Inflation':>10} {'SD':>7}")
    print(f"  {'-'*22} {'-'*10} {'-'*7}")
    for s, vals in sorted(strat_inf.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {s:<22} {np.mean(vals):>+10.4f} {np.std(vals):>7.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading IMMREP_2022 training data …")
    df_all = load_training_data()
    summarize(df_all)

    df_positives = df_all[df_all["label"] == 1].reset_index(drop=True)
    logger.info(
        f"Positive examples: {len(df_positives)} "
        f"across {df_positives['peptide'].nunique()} peptides"
    )

    # Verify ≥2 positives per peptide for LOPO to work
    pep_counts = df_positives.groupby("peptide").size()
    small_peps = pep_counts[pep_counts < 5]
    if not small_peps.empty:
        logger.warning(f"Peptides with <5 positives (may give noisy LOPO): {small_peps.to_dict()}")

    # ── Run all conditions ────────────────────────────────────────────────────
    conditions = [
        (strategy, feat)
        for feat in FEATURE_TYPES
        for strategy in SAMPLING_STRATEGIES
    ]
    total = len(conditions)
    results: list[dict] = []

    logger.info(f"Running {total} conditions ({N_REPEATS} repeats each) …")
    for i, (strategy, feat) in enumerate(conditions, 1):
        logger.info(f"[{i}/{total}]  strategy={strategy}  features={feat}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = run_benchmark(
                    df_positives=df_positives,
                    sampling_strategy_name=strategy,
                    feature_type=feat,
                    n_repeats=N_REPEATS,
                )
            results.append(res)
            logger.info(
                f"  → AUC_std={res['auc_standard']:.4f}  "
                f"AUC_lopo={res['auc_lopo']:.4f}  "
                f"inflation={res['auc_inflation']:+.4f}"
            )
        except Exception as exc:
            logger.error(f"  FAILED: {exc}", exc_info=True)

    if not results:
        logger.error("No results produced. Exiting.")
        sys.exit(1)

    # ── Print results ─────────────────────────────────────────────────────────
    _print_results_table(results)
    _print_inflation_by_strategy(results)
    _print_per_peptide_table(results)

    # ── Save results ──────────────────────────────────────────────────────────
    # Flat CSV (drop per_peptide_auc dict)
    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "per_peptide_auc"}
        flat.append(row)
    df_results = pd.DataFrame(flat)
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved flat results → {csv_path}")

    # Full JSON (includes per_peptide_auc)
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=lambda x: None if np.isnan(x) else x)
    logger.info(f"Saved full results → {json_path}")

    # Per-peptide LOPO AUC table
    pep_rows = []
    for r in results:
        for pep, auc in r.get("per_peptide_auc", {}).items():
            pep_rows.append({
                "sampling_strategy": r["sampling_strategy"],
                "feature_type": r["feature_type"],
                "peptide": pep,
                "lopo_auc": auc,
            })
    if pep_rows:
        pep_df = pd.DataFrame(pep_rows)
        pep_path = RESULTS_DIR / "per_peptide_lopo_auc.csv"
        pep_df.to_csv(pep_path, index=False)
        logger.info(f"Saved per-peptide LOPO AUC → {pep_path}")

    # ── Compute headline numbers ───────────────────────────────────────────────
    df_r = pd.DataFrame(flat)
    print("\n\n" + "=" * 60)
    print("  HEADLINE FINDINGS")
    print("=" * 60)

    mean_inflation = df_r["auc_inflation"].mean()
    max_inflation  = df_r.loc[df_r["auc_inflation"].idxmax()]
    min_inflation  = df_r.loc[df_r["auc_inflation"].idxmin()]

    print(f"  Mean AUC inflation (all strategies/features): {mean_inflation:+.4f}")
    print(
        f"  Largest inflation : {max_inflation['sampling_strategy']:20s} "
        f"[{max_inflation['feature_type']}]  {max_inflation['auc_inflation']:+.4f}"
    )
    print(
        f"  Smallest inflation: {min_inflation['sampling_strategy']:20s} "
        f"[{min_inflation['feature_type']}]  {min_inflation['auc_inflation']:+.4f}"
    )

    # Does inflation differ between sequence vs biophysical?
    seq_inf  = df_r[df_r["feature_type"] == "sequence"]["auc_inflation"].mean()
    bio_inf  = df_r[df_r["feature_type"] == "biophysical"]["auc_inflation"].mean()
    diff     = seq_inf - bio_inf
    print(f"\n  Mean inflation — sequence features  : {seq_inf:+.4f}")
    print(f"  Mean inflation — biophysical features: {bio_inf:+.4f}")
    print(f"  Differential (seq - bio)             : {diff:+.4f}")
    if diff > 0.02:
        print("  -> Sequence features show MORE inflation (supports distributional")
        print("     shortcut hypothesis: seq models benefit more from negative leakage)")
    elif diff < -0.02:
        print("  -> Biophysical features show MORE inflation (unexpected, check data)")
    else:
        print("  -> No substantial differential inflation between feature types")

    print("=" * 60)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
