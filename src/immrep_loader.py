"""
immrep_loader.py
================
Custom loader for the IMMREP_2022_TCRSpecificity benchmark dataset.

The IMMREP dataset stores one epitope per file (e.g., GILGFVFTL.txt).
Labels are 1 (positive) or -1 (negative — no 0).
The peptide sequence is encoded in the filename stem.

This module returns a standardised DataFrame compatible with the rest
of the pipeline:
    CDR3a | CDR3b | peptide | mhc | label   (label: 0 or 1)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

IMMREP_DATA_ROOT = Path(__file__).parent.parent / "data" / \
    "IMMREP_2022_TCRSpecificity" / "IMMREP_2022_TCRSpecificity-main"

TRAINING_DIR = IMMREP_DATA_ROOT / "training_data"
TEST_DIR     = IMMREP_DATA_ROOT / "test_set"
TRUE_DIR     = IMMREP_DATA_ROOT / "true_set"

# All epitopes are HLA-A*02:01-restricted
DEFAULT_MHC = "HLA-A*02:01"


def _load_epitope_file(path: Path, peptide: Optional[str] = None) -> pd.DataFrame:
    """
    Load a single per-epitope IMMREP file.

    Parameters
    ----------
    path    : path to the .txt file
    peptide : override the peptide name (default: filename stem)

    Returns
    -------
    DataFrame with columns [CDR3a, CDR3b, peptide, mhc, label]
    """
    if peptide is None:
        peptide = path.stem.upper()

    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Map IMMREP column names → standard names
    cdr3a = df.get("TRA_CDR3", df.get("A3", pd.Series(dtype=str)))
    cdr3b = df.get("TRB_CDR3", df.get("B3", pd.Series(dtype=str)))

    if cdr3b is None or cdr3b.isna().all():
        raise ValueError(f"No CDR3b column found in {path}")

    # IMMREP labels: 1 → 1,  -1 → 0
    raw_label = df.get("Label", df.get("label", pd.Series(dtype=float)))
    label = raw_label.map({1: 1, -1: 0}).fillna(raw_label.map({"1": 1, "-1": 0}))

    out = pd.DataFrame({
        "CDR3a"  : cdr3a.astype(str).str.strip().str.upper().replace("NAN", np.nan),
        "CDR3b"  : cdr3b.astype(str).str.strip().str.upper(),
        "peptide": peptide,
        "mhc"    : DEFAULT_MHC,
        "label"  : label.astype(float),
    })

    out = out.dropna(subset=["CDR3b"])
    out = out[out["CDR3b"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)]
    out["label"] = out["label"].astype(int)

    return out.reset_index(drop=True)


def load_training_data(training_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all per-epitope training files.

    Returns
    -------
    Combined DataFrame with all 17 epitopes, positives + negatives.
    """
    if training_dir is None:
        training_dir = TRAINING_DIR

    frames = []
    for f in sorted(training_dir.glob("*.txt")):
        if f.stem.upper() == "README":
            continue
        try:
            df = _load_epitope_file(f)
            df["source_file"] = f.name
            frames.append(df)
            pos = (df["label"] == 1).sum()
            neg = (df["label"] == 0).sum()
            logger.info(f"  {f.stem}: {pos} pos, {neg} neg")
        except Exception as exc:
            logger.warning(f"  Skipping {f.name}: {exc}")

    if not frames:
        raise FileNotFoundError(f"No epitope files loaded from {training_dir}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total training rows: {len(combined)}")
    return combined


def load_test_data(
    test_dir: Optional[Path] = None,
    true_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load per-epitope test files joined with ground-truth labels.

    Returns
    -------
    DataFrame with same schema as training data.
    """
    if test_dir is None:
        test_dir = TEST_DIR
    if true_dir is None:
        true_dir = TRUE_DIR

    # True-set files carry labels; load them directly
    true_files = list(sorted(true_dir.glob("*.txt")))
    frames = []
    for f in true_files:
        if f.stem.upper() in ("README", "TESTSET_GLOBAL", "TESTSET_GLOBAL_NOHEADERS"):
            continue
        try:
            df = _load_epitope_file(f)
            df["split"] = "test"
            frames.append(df)
        except Exception as exc:
            logger.warning(f"  Skipping true-set {f.name}: {exc}")

    if not frames:
        raise FileNotFoundError(f"No true-set files loaded from {true_dir}")

    return pd.concat(frames, ignore_index=True)


def load_positives_only(training_dir: Optional[Path] = None) -> pd.DataFrame:
    """Return only positive examples from training data."""
    df = load_training_data(training_dir)
    return df[df["label"] == 1].reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    """Print dataset summary."""
    print("=" * 60)
    print(f"  Total rows       : {len(df)}")
    print(f"  Positives        : {(df['label'] == 1).sum()}")
    print(f"  Negatives        : {(df['label'] == 0).sum()}")
    print(f"  Unique CDR3b     : {df['CDR3b'].nunique()}")
    print(f"  Unique peptides  : {df['peptide'].nunique()}")
    print(f"  Peptides         : {sorted(df['peptide'].unique())}")
    mean_len = df["CDR3b"].str.len().mean()
    print(f"  Mean CDR3b len   : {mean_len:.1f} aa")
    print("=" * 60)
