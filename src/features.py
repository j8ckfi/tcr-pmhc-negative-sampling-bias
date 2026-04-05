"""
features.py
===========
Feature extraction for TCR-pMHC specificity prediction.

Two feature families are provided:

Sequence-based
--------------
  - BLOSUM62 encoding of CDR3b (position-specific, zero-padded to max_len)
  - 3-mer frequency vector of CDR3b
  - BLOSUM62 encoding of peptide (position-specific, zero-padded to max_len)
  Concatenated → one flat numeric vector per row.

Biophysical
-----------
  Per amino acid:
    - 10 Kidera factors
    - Charge at pH 7
    - Kyte-Doolittle hydrophobicity
    - Molecular volume
  Summarised over each sequence as [mean, std, max, min] of every property
  → flat vector for CDR3b + flat vector for peptide, concatenated.

Public API
----------
  extract_features(df, feature_type, max_cdr3b_len, max_pep_len)
      → np.ndarray of shape (n_samples, n_features)

  get_feature_names(feature_type, max_cdr3b_len, max_pep_len)
      → list of str
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Amino-acid alphabet
# ---------------------------------------------------------------------------
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
N_AA = len(AA)  # 20

# ---------------------------------------------------------------------------
# BLOSUM62 matrix (20×20, standard ordering ACDEFGHIKLMNPQRSTVWY)
# ---------------------------------------------------------------------------
# Source: NCBI / Henikoff & Henikoff 1992
_BLOSUM62_RAW = [
    #  A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [  4, -1, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],  # A
    [ -1,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],  # C
    [ -2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],  # D
    [ -1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],  # E
    [ -2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],  # F
    [  0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],  # G
    [ -2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],  # H
    [ -1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],  # I
    [ -1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],  # K
    [ -1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],  # L
    [ -1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],  # M
    [ -2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],  # N
    [ -1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],  # P
    [ -1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],  # Q
    [ -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],  # R
    [  1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],  # S
    [  0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],  # T
    [  0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],  # V
    [ -3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],  # W
    [ -2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],  # Y
]
BLOSUM62 = np.array(_BLOSUM62_RAW, dtype=np.float32)  # shape (20, 20)


def _blosum_encode(seq: str, max_len: int) -> np.ndarray:
    """
    Encode a sequence using BLOSUM62 row lookup, zero-padded to max_len.

    Returns array of shape (max_len * 20,).
    Unknown amino acids are encoded as zeros.
    """
    mat = np.zeros((max_len, N_AA), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        idx = AA_INDEX.get(aa)
        if idx is not None:
            mat[i] = BLOSUM62[idx]
    return mat.ravel()


def _kmer_freq(seq: str, k: int = 3) -> np.ndarray:
    """
    Compute k-mer frequency vector over the standard amino-acid alphabet.

    Returns array of shape (20**k,) with relative frequencies.
    """
    kmers = ["".join(p) for p in product(AA, repeat=k)]
    kmer_index = {km: i for i, km in enumerate(kmers)}
    vec = np.zeros(len(kmers), dtype=np.float32)
    n = len(seq) - k + 1
    if n <= 0:
        return vec
    for i in range(n):
        km = seq[i: i + k]
        idx = kmer_index.get(km)
        if idx is not None:
            vec[idx] += 1.0
    vec /= n  # relative frequency
    return vec


# ---------------------------------------------------------------------------
# Kidera factors (10 orthogonal physicochemical factors per amino acid)
# ---------------------------------------------------------------------------
# Source: Kidera et al. (1985) J Protein Chem 4:23-55
# Ordering: ACDEFGHIKLMNPQRSTVWY
_KIDERA_RAW: dict[str, list[float]] = {
    "A": [ 0.52, -1.78,  0.77, -0.10,  0.15, -0.89, -0.37, -0.98, -0.47, -0.50],
    "C": [ 0.28, -0.58,  1.00,  0.88, -0.17, -0.73, -0.35, -0.44, -0.43,  0.33],
    "D": [ 0.24, -0.60,  0.03,  0.44,  0.19,  0.07,  0.34,  0.34, -0.54,  0.05],
    "E": [ 0.25, -0.62,  0.06,  0.61,  0.15,  0.09,  0.28,  0.17, -0.44, -0.01],
    "F": [-0.63, -1.30,  0.08, -0.16,  0.23, -0.11, -0.23,  0.21,  0.70,  0.30],
    "G": [ 0.50, -0.49,  0.45, -0.03, -0.30,  0.12,  0.43,  0.14, -0.19, -0.44],
    "H": [ 0.34, -0.68,  0.30,  0.64, -0.01, -0.05,  0.06,  0.16, -0.09,  0.07],
    "I": [-0.98, -1.69,  0.25, -0.41, -0.01, -0.12, -0.29,  0.06,  0.55,  0.12],
    "K": [ 0.40, -0.71,  0.06,  0.58,  0.00,  0.07,  0.27,  0.05, -0.38,  0.00],
    "L": [-0.92, -1.60,  0.22, -0.31, -0.03, -0.16, -0.23,  0.15,  0.48,  0.14],
    "M": [-0.76, -1.37,  0.21, -0.20,  0.03, -0.23, -0.24,  0.17,  0.54,  0.17],
    "N": [ 0.39, -0.60,  0.20,  0.50,  0.12,  0.06,  0.26,  0.18, -0.35,  0.00],
    "P": [ 0.29, -0.22,  0.45, -0.18, -0.48,  0.21,  0.46, -0.22, -0.21, -0.44],
    "Q": [ 0.34, -0.66,  0.15,  0.58,  0.07,  0.04,  0.23,  0.14, -0.39,  0.02],
    "R": [ 0.40, -0.69,  0.07,  0.62,  0.01,  0.07,  0.26,  0.07, -0.38,  0.00],
    "S": [ 0.55, -0.62,  0.43,  0.07, -0.12, -0.03,  0.26,  0.00, -0.19, -0.33],
    "T": [ 0.44, -0.82,  0.30,  0.11, -0.05, -0.22,  0.00,  0.01, -0.20, -0.21],
    "V": [-0.73, -1.51,  0.29, -0.30,  0.00, -0.07, -0.16,  0.06,  0.41,  0.09],
    "W": [-0.46, -1.02,  0.27,  0.00,  0.39, -0.17, -0.36,  0.21,  0.64,  0.29],
    "Y": [-0.38, -0.93,  0.23,  0.11,  0.30, -0.16, -0.24,  0.17,  0.57,  0.21],
}
KIDERA = {aa: np.array(v, dtype=np.float32) for aa, v in _KIDERA_RAW.items()}

# ---------------------------------------------------------------------------
# Charge at pH 7
# ---------------------------------------------------------------------------
CHARGE_PH7: dict[str, float] = {
    "K": +1.0, "R": +1.0, "H": +0.1,
    "D": -1.0, "E": -1.0,
}

# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity scale
# ---------------------------------------------------------------------------
HYDROPHOBICITY: dict[str, float] = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# ---------------------------------------------------------------------------
# Molecular volume (Å³, Darby & Creighton 1993 / standard reference)
# ---------------------------------------------------------------------------
MOL_VOLUME: dict[str, float] = {
    "A":  88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G":  60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S":  89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

_BIOPHYS_PROPS = ["kidera_0","kidera_1","kidera_2","kidera_3","kidera_4",
                  "kidera_5","kidera_6","kidera_7","kidera_8","kidera_9",
                  "charge","hydrophobicity","mol_volume"]
_N_BIOPHYS = len(_BIOPHYS_PROPS)  # 13


def _aa_biophys_vector(aa: str) -> np.ndarray:
    """Return the 13-dimensional biophysical property vector for one amino acid."""
    if aa not in AA_INDEX:
        return np.zeros(_N_BIOPHYS, dtype=np.float32)
    vec = np.empty(_N_BIOPHYS, dtype=np.float32)
    vec[:10] = KIDERA.get(aa, np.zeros(10, dtype=np.float32))
    vec[10]  = CHARGE_PH7.get(aa, 0.0)
    vec[11]  = HYDROPHOBICITY.get(aa, 0.0)
    vec[12]  = MOL_VOLUME.get(aa, 0.0)
    return vec


def _biophys_summary(seq: str) -> np.ndarray:
    """
    Summarise a sequence's biophysical properties as [mean, std, max, min]
    of each of the 13 per-residue properties.

    Returns array of shape (13 * 4,) = (52,).
    """
    if not seq:
        return np.zeros(_N_BIOPHYS * 4, dtype=np.float32)

    mat = np.stack([_aa_biophys_vector(aa) for aa in seq])  # (L, 13)
    summary = np.concatenate([
        mat.mean(axis=0),
        mat.std(axis=0),
        mat.max(axis=0),
        mat.min(axis=0),
    ])  # (52,)
    return summary.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FeatureType = Literal["sequence", "biophysical"]

_DEFAULT_MAX_CDR3B = 30
_DEFAULT_MAX_PEP   = 15


def extract_features(
    df: pd.DataFrame,
    feature_type: FeatureType = "sequence",
    max_cdr3b_len: int = _DEFAULT_MAX_CDR3B,
    max_pep_len:   int = _DEFAULT_MAX_PEP,
    kmer_k:        int = 3,
) -> np.ndarray:
    """
    Extract a numeric feature matrix from a TCR-pMHC DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'CDR3b' and 'peptide'.
    feature_type : {'sequence', 'biophysical'}
        Which feature family to use.
    max_cdr3b_len : int
        Padding length for BLOSUM62 CDR3b encoding (sequence mode only).
    max_pep_len : int
        Padding length for BLOSUM62 peptide encoding (sequence mode only).
    kmer_k : int
        k for k-mer frequency features (sequence mode only).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if feature_type == "sequence":
        return _extract_sequence_features(df, max_cdr3b_len, max_pep_len, kmer_k)
    elif feature_type == "biophysical":
        return _extract_biophysical_features(df)
    else:
        raise ValueError(
            f"Unknown feature_type '{feature_type}'. "
            "Choose 'sequence' or 'biophysical'."
        )


def _extract_sequence_features(
    df: pd.DataFrame,
    max_cdr3b_len: int,
    max_pep_len: int,
    kmer_k: int,
) -> np.ndarray:
    """
    Sequence-based features per row:
      - BLOSUM62 CDR3b  : max_cdr3b_len * 20
      - k-mer freq CDR3b: 20^k
      - BLOSUM62 peptide: max_pep_len * 20
    """
    rows = []
    for _, row in df.iterrows():
        cdr3b = str(row["CDR3b"]) if pd.notna(row["CDR3b"]) else ""
        pep   = str(row["peptide"]) if pd.notna(row["peptide"]) else ""

        b62_cdr3b = _blosum_encode(cdr3b, max_cdr3b_len)
        kmer_cdr3b = _kmer_freq(cdr3b, k=kmer_k)
        b62_pep   = _blosum_encode(pep, max_pep_len)

        rows.append(np.concatenate([b62_cdr3b, kmer_cdr3b, b62_pep]))

    X = np.vstack(rows).astype(np.float32)
    logger.debug(f"Sequence features shape: {X.shape}")
    return X


def _extract_biophysical_features(df: pd.DataFrame) -> np.ndarray:
    """
    Biophysical features per row:
      - CDR3b summary: 13 props × 4 stats = 52
      - peptide summary: 13 props × 4 stats = 52
      Total: 104 features
    """
    rows = []
    for _, row in df.iterrows():
        cdr3b = str(row["CDR3b"]) if pd.notna(row["CDR3b"]) else ""
        pep   = str(row["peptide"]) if pd.notna(row["peptide"]) else ""

        feat_cdr3b = _biophys_summary(cdr3b)
        feat_pep   = _biophys_summary(pep)

        rows.append(np.concatenate([feat_cdr3b, feat_pep]))

    X = np.vstack(rows).astype(np.float32)
    logger.debug(f"Biophysical features shape: {X.shape}")
    return X


def get_feature_dim(
    feature_type: FeatureType,
    max_cdr3b_len: int = _DEFAULT_MAX_CDR3B,
    max_pep_len:   int = _DEFAULT_MAX_PEP,
    kmer_k:        int = 3,
) -> int:
    """Return the number of features for a given configuration."""
    if feature_type == "sequence":
        return max_cdr3b_len * N_AA + N_AA ** kmer_k + max_pep_len * N_AA
    elif feature_type == "biophysical":
        return _N_BIOPHYS * 4 * 2  # CDR3b + peptide
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def get_feature_names(
    feature_type: FeatureType,
    max_cdr3b_len: int = _DEFAULT_MAX_CDR3B,
    max_pep_len:   int = _DEFAULT_MAX_PEP,
    kmer_k:        int = 3,
) -> list[str]:
    """Return human-readable feature names (useful for inspection/debugging)."""
    names: list[str] = []
    if feature_type == "sequence":
        for pos in range(max_cdr3b_len):
            for aa in AA:
                names.append(f"cdr3b_blosum_pos{pos}_{aa}")
        for km in product(AA, repeat=kmer_k):
            names.append(f"cdr3b_kmer_{''.join(km)}")
        for pos in range(max_pep_len):
            for aa in AA:
                names.append(f"pep_blosum_pos{pos}_{aa}")
    elif feature_type == "biophysical":
        for seq_name in ("cdr3b", "pep"):
            for prop in _BIOPHYS_PROPS:
                for stat in ("mean", "std", "max", "min"):
                    names.append(f"{seq_name}_{prop}_{stat}")
    return names
