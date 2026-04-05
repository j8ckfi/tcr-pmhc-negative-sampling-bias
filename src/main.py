"""
main.py
=======
Orchestration script for the TCR-pMHC negative-sampling bias benchmark.

Usage
-----
    python src/main.py
    python src/main.py --data data/my_dataset.csv --repeats 10

What it does
------------
1. Loads positive TCR-pMHC pairs from ../data/ (tries multiple formats)
2. Runs all 5 sampling strategies × 2 feature types = 10 conditions
3. Saves a flat results CSV to ../results/benchmark_results.csv
4. Prints a formatted summary table showing AUC inflation per condition

Directory layout expected
-------------------------
    project_root/
        data/           ← put your .csv / .tsv / .txt files here
        results/        ← output written here
        src/
            main.py     ← this file
            data_loader.py
            negative_sampling.py
            features.py
            evaluation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent
_DATA_DIR    = _PROJECT_DIR / "data"
_RESULTS_DIR = _PROJECT_DIR / "results"

# Put src/ on the path so sibling imports work when invoked as a script
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_loader import load_data, load_directory, summarize
from negative_sampling import STRATEGY_NAMES
from evaluation import run_benchmark

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic demo data (used when no real data files are found)
# ---------------------------------------------------------------------------

def _make_demo_data(n_peptides: int = 6, n_per_peptide: int = 40) -> pd.DataFrame:
    """
    Generate a small synthetic TCR-pMHC dataset for demonstration / testing.

    Sequences are random but respect typical length distributions:
      CDR3b : 12–18 aa
      peptide: 8–11 aa
    """
    rng = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")

    peptides = [
        "".join(rng.choice(aa, size=int(rng.integers(8, 12))))
        for _ in range(n_peptides)
    ]

    rows = []
    for pep in peptides:
        for _ in range(n_per_peptide):
            cdr3b_len = int(rng.integers(12, 19))
            cdr3b = "".join(rng.choice(aa, size=cdr3b_len))
            rows.append({
                "CDR3b":   cdr3b,
                "CDR3a":   float("nan"),
                "peptide": pep,
                "mhc":     "HLA-A*02:01",
                "label":   1,
            })

    df = pd.DataFrame(rows)
    logger.info(
        f"Generated synthetic demo dataset: "
        f"{len(df)} positives, {n_peptides} peptides."
    )
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_positives(data_path: Path | None) -> pd.DataFrame:
    """
    Attempt to load positive TCR-pMHC pairs from *data_path*.

    Falls back to the synthetic demo dataset if no real data is found.
    Returns DataFrame with label == 1 only.
    """
    if data_path is not None:
        data_path = Path(data_path)
        if data_path.is_file():
            df = load_data(data_path)
        elif data_path.is_dir():
            df = load_directory(data_path)
        else:
            raise FileNotFoundError(f"--data path not found: {data_path}")
    else:
        # Auto-discover files in ../data/
        if _DATA_DIR.exists():
            try:
                df = load_directory(_DATA_DIR)
                logger.info(f"Auto-loaded data from {_DATA_DIR}")
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(f"No data files found in {_DATA_DIR}: {exc}")
                logger.warning("Using synthetic demo dataset.")
                df = _make_demo_data()
        else:
            logger.warning(f"Data directory not found: {_DATA_DIR}")
            logger.warning("Using synthetic demo dataset.")
            df = _make_demo_data()

    # Keep only positives as the starting pool
    positives = df[df["label"] == 1].reset_index(drop=True)
    if len(positives) == 0:
        raise ValueError("Dataset contains no positive examples (label == 1).")

    n_peptides = positives["peptide"].nunique()
    if n_peptides < 2:
        raise ValueError(
            f"Dataset has only {n_peptides} unique peptide(s). "
            "LOPO evaluation requires at least 2."
        )

    logger.info(
        f"Loaded {len(positives)} positives across {n_peptides} peptides."
    )
    summarize(positives)
    return positives


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    "sampling_strategy",
    "feature_type",
    "auc_standard",
    "auc_lopo",
    "auc_inflation",
    "model_lr_standard_mean",
    "model_rf_standard_mean",
    "model_lr_lopo_mean",
    "model_rf_lopo_mean",
]


def _print_summary_table(results_df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 90)
    print("  NEGATIVE SAMPLING BIAS BENCHMARK — SUMMARY")
    print("=" * 90)
    print(
        f"  {'Strategy':<25} {'Features':<15} "
        f"{'AUC_std':>9} {'AUC_lopo':>9} {'Inflation':>10} "
        f"{'LR_std':>8} {'RF_std':>8} {'LR_lopo':>8} {'RF_lopo':>8}"
    )
    print("-" * 90)

    for _, row in results_df.iterrows():
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else "  NaN"

        print(
            f"  {row['sampling_strategy']:<25} {row['feature_type']:<15} "
            f"{fmt(row['auc_standard']):>9} "
            f"{fmt(row['auc_lopo']):>9} "
            f"{fmt(row['auc_inflation']):>10} "
            f"{fmt(row.get('model_lr_standard_mean', float('nan'))):>8} "
            f"{fmt(row.get('model_rf_standard_mean', float('nan'))):>8} "
            f"{fmt(row.get('model_lr_lopo_mean', float('nan'))):>8} "
            f"{fmt(row.get('model_rf_lopo_mean', float('nan'))):>8}"
        )
    print("=" * 90)
    print()

    # Highlight the biggest inflators
    valid = results_df.dropna(subset=["auc_inflation"])
    if len(valid) > 0:
        max_idx = valid["auc_inflation"].idxmax()
        min_idx = valid["auc_inflation"].idxmin()
        r_max = valid.loc[max_idx]
        r_min = valid.loc[min_idx]
        print(
            f"  Largest  inflation: "
            f"{r_max['sampling_strategy']} + {r_max['feature_type']}  "
            f"→ {r_max['auc_inflation']:+.3f}"
        )
        print(
            f"  Smallest inflation: "
            f"{r_min['sampling_strategy']} + {r_min['feature_type']}  "
            f"→ {r_min['auc_inflation']:+.3f}"
        )
        print()


def _save_per_peptide(all_results: list[dict], out_dir: Path) -> None:
    """Save per-peptide LOPO AUC breakdowns to a separate CSV."""
    rows = []
    for res in all_results:
        ppa = res.get("per_peptide_auc", {})
        for pep, auc in ppa.items():
            rows.append({
                "sampling_strategy": res["sampling_strategy"],
                "feature_type":      res["feature_type"],
                "peptide":           pep,
                "auc_lopo":          auc,
            })
    if rows:
        df = pd.DataFrame(rows)
        path = out_dir / "per_peptide_auc.csv"
        df.to_csv(path, index=False)
        logger.info(f"Per-peptide AUCs saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TCR-pMHC negative-sampling bias benchmark"
    )
    p.add_argument(
        "--data", type=Path, default=None,
        help=(
            "Path to a data file (.csv/.tsv) or directory containing data files. "
            "Defaults to ../data/ relative to src/."
        ),
    )
    p.add_argument(
        "--repeats", type=int, default=5,
        help="Number of random seeds for standard CV component (default: 5).",
    )
    p.add_argument(
        "--strategies", nargs="+", default=STRATEGY_NAMES,
        choices=STRATEGY_NAMES,
        help="Subset of sampling strategies to run (default: all 5).",
    )
    p.add_argument(
        "--features", nargs="+", default=["sequence", "biophysical"],
        choices=["sequence", "biophysical"],
        help="Feature types to evaluate (default: both).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Directory to write results CSV (default: ../results/).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    out_dir = args.output if args.output else _RESULTS_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    df_positives = load_positives(args.data)

    # ---- Run all conditions ----
    conditions = [
        (strategy, feat)
        for strategy in args.strategies
        for feat in args.features
    ]

    n_conditions = len(conditions)
    logger.info(f"Running {n_conditions} conditions ({args.repeats} repeats each).")

    all_results: list[dict] = []
    t0_total = time.time()

    for i, (strategy, feat) in enumerate(conditions, 1):
        label = f"[{i}/{n_conditions}] strategy={strategy}  feature={feat}"
        logger.info(label)
        t0 = time.time()

        try:
            result = run_benchmark(
                df_positives,
                sampling_strategy_name=strategy,
                feature_type=feat,
                n_repeats=args.repeats,
            )
        except Exception as exc:
            logger.error(f"  FAILED: {exc}", exc_info=True)
            result = {
                "sampling_strategy": strategy,
                "feature_type":      feat,
                "auc_standard":      float("nan"),
                "auc_lopo":          float("nan"),
                "auc_inflation":     float("nan"),
                "per_peptide_auc":   {},
                "error":             str(exc),
            }

        elapsed = time.time() - t0
        logger.info(
            f"  Done in {elapsed:.1f}s  "
            f"AUC_std={result.get('auc_standard', float('nan')):.3f}  "
            f"AUC_lopo={result.get('auc_lopo', float('nan')):.3f}  "
            f"inflation={result.get('auc_inflation', float('nan')):+.3f}"
        )
        all_results.append(result)

    total_elapsed = time.time() - t0_total
    logger.info(f"All conditions completed in {total_elapsed:.1f}s.")

    # ---- Save flat CSV (drop nested per_peptide_auc dict) ----
    flat_rows = []
    for res in all_results:
        row = {k: v for k, v in res.items() if k != "per_peptide_auc"}
        flat_rows.append(row)

    results_df = pd.DataFrame(flat_rows)
    csv_path = out_dir / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # ---- Save per-peptide breakdown ----
    _save_per_peptide(all_results, out_dir)

    # ---- Save full results as JSON (includes per_peptide_auc) ----
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info(f"Full results saved to {json_path}")

    # ---- Print summary table ----
    _print_summary_table(results_df)


if __name__ == "__main__":
    main()
