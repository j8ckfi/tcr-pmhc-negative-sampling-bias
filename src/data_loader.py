"""
data_loader.py
==============
Load and preprocess TCR-pMHC specificity datasets.

Supported formats:
  - Tab-separated (.tsv, .txt)
  - Comma-separated (.csv)

Expected columns (case-insensitive aliases accepted):
  CDR3a   : alpha-chain CDR3 amino-acid sequence (optional)
  CDR3b   : beta-chain  CDR3 amino-acid sequence (required)
  peptide : presented peptide sequence
  mhc     : MHC allele string (e.g. "HLA-A*02:01")
  label   : binding label — 1 = binder, 0 = non-binder

If CDR3a is absent the column is filled with NaN and downstream code
handles it gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name aliases
# ---------------------------------------------------------------------------
_CDR3A_ALIASES  = {"cdr3a", "cdr3_alpha", "cdr3.alpha", "alpha_cdr3", "tcr_alpha"}
_CDR3B_ALIASES  = {"cdr3b", "cdr3_beta",  "cdr3.beta",  "beta_cdr3",  "tcr_beta", "cdr3"}
_PEP_ALIASES    = {"peptide", "epitope", "antigen", "pep", "sequence", "peptide_sequence"}
_MHC_ALIASES    = {"mhc", "hla", "allele", "mhc_allele", "hla_allele", "mhc allele"}
_LABEL_ALIASES  = {"label", "binding", "binder", "y", "target", "class", "hit"}

STANDARD_COLS = ["CDR3a", "CDR3b", "peptide", "mhc", "label"]


def _match_column(columns: list[str], aliases: set[str]) -> Optional[str]:
    """Return the first column whose lower-cased name appears in *aliases*."""
    for col in columns:
        if col.strip().lower() in aliases:
            return col
    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw DataFrame columns to the standard schema.
    Raises ValueError if required columns (CDR3b, peptide, label) are missing.
    """
    cols = list(df.columns)
    rename_map: dict[str, str] = {}

    cdr3a_col = _match_column(cols, _CDR3A_ALIASES)
    cdr3b_col = _match_column(cols, _CDR3B_ALIASES)
    pep_col   = _match_column(cols, _PEP_ALIASES)
    mhc_col   = _match_column(cols, _MHC_ALIASES)
    lbl_col   = _match_column(cols, _LABEL_ALIASES)

    # Required columns
    missing = []
    if cdr3b_col is None:
        missing.append("CDR3b")
    if pep_col is None:
        missing.append("peptide")
    if lbl_col is None:
        missing.append("label")
    if missing:
        raise ValueError(
            f"Could not find required column(s): {missing}. "
            f"Available columns: {cols}"
        )

    if cdr3a_col:
        rename_map[cdr3a_col] = "CDR3a"
    rename_map[cdr3b_col] = "CDR3b"
    rename_map[pep_col]   = "peptide"
    if mhc_col:
        rename_map[mhc_col] = "mhc"
    rename_map[lbl_col]   = "label"

    df = df.rename(columns=rename_map)

    # Add missing optional columns
    if "CDR3a" not in df.columns:
        logger.info("CDR3a column not found — filling with NaN.")
        df["CDR3a"] = np.nan
    if "mhc" not in df.columns:
        logger.info("MHC column not found — filling with 'unknown'.")
        df["mhc"] = "unknown"

    return df[STANDARD_COLS]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - Strip whitespace from string columns
      - Upper-case amino-acid sequences
      - Drop rows where CDR3b or peptide is null/empty
      - Coerce label to integer {0, 1}
    """
    for col in ["CDR3a", "CDR3b", "peptide", "mhc"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace({"NAN": np.nan, "": np.nan, "NONE": np.nan})

    # Drop rows with missing essential sequences
    before = len(df)
    df = df.dropna(subset=["CDR3b", "peptide"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with missing CDR3b or peptide.")

    # Coerce label
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    invalid_labels = df["label"].isna().sum()
    if invalid_labels:
        logger.warning(f"Dropping {invalid_labels} rows with non-numeric labels.")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Only keep 0/1 labels
    n_before = len(df)
    df = df[df["label"].isin([0, 1])]
    if len(df) < n_before:
        logger.warning(
            f"Dropped {n_before - len(df)} rows with labels outside {{0,1}}."
        )

    df = df.reset_index(drop=True)
    return df


def load_data(
    path: Union[str, Path],
    sep: Optional[str] = None,
    positives_only: bool = False,
) -> pd.DataFrame:
    """
    Load a TCR-pMHC dataset from a delimited text file.

    Parameters
    ----------
    path : str or Path
        Path to the data file.
    sep : str, optional
        Column delimiter.  If None, inferred from the file extension
        (.csv → ',', everything else → '\\t').
    positives_only : bool
        If True, return only rows where label == 1.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns:
        ``['CDR3a', 'CDR3b', 'peptide', 'mhc', 'label']``

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns cannot be found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if sep is None:
        sep = "," if path.suffix.lower() == ".csv" else "\t"

    logger.info(f"Loading data from {path} (sep={repr(sep)})")
    df = pd.read_csv(path, sep=sep, low_memory=False)
    logger.info(f"  Raw shape: {df.shape}")

    df = _standardize_columns(df)
    df = _clean(df)

    if positives_only:
        df = df[df["label"] == 1].reset_index(drop=True)

    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    logger.info(f"  Final shape: {df.shape}  (pos={pos}, neg={neg})")
    return df


def load_directory(
    directory: Union[str, Path],
    positives_only: bool = False,
) -> pd.DataFrame:
    """
    Scan *directory* for .csv / .tsv / .txt files and concatenate them
    into a single DataFrame.

    Parameters
    ----------
    directory : str or Path
    positives_only : bool

    Returns
    -------
    pd.DataFrame
        Combined, deduplicated dataset.
    """
    directory = Path(directory)
    extensions = {".csv", ".tsv", ".txt"}
    files = [f for f in sorted(directory.iterdir()) if f.suffix.lower() in extensions]

    if not files:
        raise FileNotFoundError(
            f"No .csv/.tsv/.txt files found in {directory}"
        )

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = load_data(f, positives_only=positives_only)
            df["source_file"] = f.name
            frames.append(df)
            logger.info(f"  Loaded {f.name}: {len(df)} rows")
        except Exception as exc:
            logger.warning(f"  Skipping {f.name}: {exc}")

    if not frames:
        raise ValueError(f"No files could be loaded from {directory}")

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate on biological key
    key_cols = ["CDR3b", "peptide", "label"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=key_cols).reset_index(drop=True)
    if len(combined) < before:
        logger.info(f"  Removed {before - len(combined)} duplicate rows.")

    return combined


def summarize(df: pd.DataFrame) -> None:
    """Print a brief summary of the loaded DataFrame to stdout."""
    print("=" * 55)
    print(f"  Rows          : {len(df)}")
    print(f"  Positives     : {(df['label']==1).sum()}")
    print(f"  Negatives     : {(df['label']==0).sum()}")
    print(f"  Unique CDR3b  : {df['CDR3b'].nunique()}")
    print(f"  Unique peptide: {df['peptide'].nunique()}")
    print(f"  Peptides      : {sorted(df['peptide'].unique())}")
    print(f"  CDR3a present : {df['CDR3a'].notna().sum()} / {len(df)}")
    print("=" * 55)
