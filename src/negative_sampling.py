"""
negative_sampling.py
====================
Five negative-sampling strategies for TCR-pMHC specificity studies.

All public functions accept a DataFrame of *positive* TCR-pMHC pairs
(label == 1, columns: CDR3a, CDR3b, peptide, mhc, label) and return a
balanced DataFrame containing the original positives plus an equal number
of synthetically generated negatives (label == 0).

Strategies
----------
1. random_swap          – pair each TCR with a randomly chosen non-cognate
                          peptide from the dataset
2. epitope_balanced     – sample negatives so every peptide contributes
                          equally to the negative set
3. within_cluster       – sample negatives from TCRs sharing the same
                          inferred V-gene cluster (harder negatives)
4. shuffled_cdr3        – shuffle CDR3b residues in-place to create synthetic
                          decoy sequences
5. leave_one_peptide_out – LOPO cross-validation generator: for each peptide
                          p, yield (train_df, test_df) where test_df contains
                          only pairs with peptide == p and train_df uses
                          random-swap negatives on all other peptides

All random operations accept an optional *random_state* argument for
reproducibility.
"""

from __future__ import annotations

import logging
import random
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_positives(df: pd.DataFrame) -> pd.DataFrame:
    """Return only positive rows; warn if negatives are already present."""
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column.")
    pos = df[df["label"] == 1].copy().reset_index(drop=True)
    if len(pos) < len(df):
        logger.warning(
            f"Input contained {len(df) - len(pos)} non-positive rows; "
            "only positives are used as the starting point."
        )
    if len(pos) == 0:
        raise ValueError("No positive examples found (label == 1).")
    return pos


def _set_seed(random_state: Optional[int]) -> None:
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)


def _combine(positives: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """Stack positives and negatives, shuffle, reset index."""
    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


def _infer_vgene(cdr3b: str) -> str:
    """
    Heuristically infer a V-gene cluster from CDR3b sequence prefix.

    Real datasets would use IMGT V-gene calls; here we approximate by the
    first 3 amino acids (a proxy that groups sequences sharing the same
    germline origin when proper V-gene annotations are unavailable).
    """
    return cdr3b[:3] if len(cdr3b) >= 3 else cdr3b


# ---------------------------------------------------------------------------
# Strategy 1 — Random swap
# ---------------------------------------------------------------------------

def random_swap(
    df_positives: pd.DataFrame,
    random_state: Optional[int] = 42,
    n_neg_per_pos: int = 1,
) -> pd.DataFrame:
    """
    Generate negatives by randomly pairing each TCR with a non-cognate
    peptide drawn uniformly from the dataset's peptide repertoire.

    Parameters
    ----------
    df_positives : pd.DataFrame
        Positive TCR-pMHC pairs (label == 1).
    random_state : int, optional
        Seed for reproducibility.
    n_neg_per_pos : int
        Number of negative samples per positive (default 1 → balanced).

    Returns
    -------
    pd.DataFrame
        Balanced DataFrame with columns identical to input plus 'label'.
    """
    _set_seed(random_state)
    pos = _validate_positives(df_positives)
    peptides = pos["peptide"].unique().tolist()

    neg_rows = []
    for _, row in pos.iterrows():
        cognate = row["peptide"]
        non_cognate_peps = [p for p in peptides if p != cognate]
        if not non_cognate_peps:
            logger.warning(
                f"Only one peptide in dataset — cannot generate negatives for {cognate}."
            )
            continue
        chosen = np.random.choice(non_cognate_peps, size=n_neg_per_pos, replace=True)
        for pep in chosen:
            neg_row = row.copy()
            neg_row["peptide"] = pep
            neg_row["label"] = 0
            neg_rows.append(neg_row)

    negatives = pd.DataFrame(neg_rows).reset_index(drop=True)
    logger.info(
        f"random_swap: {len(pos)} positives → {len(negatives)} negatives generated."
    )
    return _combine(pos, negatives)


# ---------------------------------------------------------------------------
# Strategy 2 — Epitope-balanced swap
# ---------------------------------------------------------------------------

def epitope_balanced(
    df_positives: pd.DataFrame,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate negatives ensuring each peptide contributes equally to the
    negative pool.

    For a dataset with P peptides and N total positives, each peptide
    provides floor(N / (P-1)) negatives (approximately equal contribution),
    sampling TCRs from *other* peptides' cognate TCRs.

    Parameters
    ----------
    df_positives : pd.DataFrame
    random_state : int, optional

    Returns
    -------
    pd.DataFrame
        Balanced DataFrame.
    """
    _set_seed(random_state)
    pos = _validate_positives(df_positives)
    peptides = pos["peptide"].unique().tolist()
    n_peptides = len(peptides)

    if n_peptides < 2:
        logger.warning("Fewer than 2 peptides — falling back to random_swap.")
        return random_swap(df_positives, random_state=random_state)

    target_neg = len(pos)
    per_peptide_quota = max(1, target_neg // (n_peptides - 1))

    neg_rows: list[dict] = []

    # For each peptide, sample TCRs from other peptides
    pep_to_tcrs: dict[str, list[dict]] = {
        p: pos[pos["peptide"] == p].to_dict("records") for p in peptides
    }

    for pep in peptides:
        other_tcr_pool: list[dict] = []
        for other_pep in peptides:
            if other_pep != pep:
                other_tcr_pool.extend(pep_to_tcrs[other_pep])

        if not other_tcr_pool:
            continue

        sampled = np.random.choice(
            len(other_tcr_pool),
            size=per_peptide_quota,
            replace=len(other_tcr_pool) < per_peptide_quota,
        )
        for idx in sampled:
            tcr_row = dict(other_tcr_pool[int(idx)])
            tcr_row["peptide"] = pep
            tcr_row["label"] = 0
            neg_rows.append(tcr_row)

    # Trim or pad to exactly target_neg
    np.random.shuffle(neg_rows)
    neg_rows = neg_rows[:target_neg]

    negatives = pd.DataFrame(neg_rows).reset_index(drop=True)
    logger.info(
        f"epitope_balanced: {len(pos)} positives → {len(negatives)} negatives generated."
    )
    return _combine(pos, negatives)


# ---------------------------------------------------------------------------
# Strategy 3 — Within-cluster (harder negatives)
# ---------------------------------------------------------------------------

def within_cluster(
    df_positives: pd.DataFrame,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate harder negatives by sampling TCRs from within the same
    inferred V-gene cluster.

    TCRs in the same V-gene cluster share germline-encoded CDR1/CDR2
    features, making them more similar to the true binder than random TCRs.
    This tests whether models can discriminate within closely related TCRs.

    If CDR3b prefix grouping yields singleton clusters, falls back to
    random_swap for those TCRs.

    Parameters
    ----------
    df_positives : pd.DataFrame
    random_state : int, optional

    Returns
    -------
    pd.DataFrame
    """
    _set_seed(random_state)
    pos = _validate_positives(df_positives)

    # Assign cluster labels
    pos = pos.copy()
    pos["_vgene"] = pos["CDR3b"].apply(_infer_vgene)

    neg_rows: list[dict] = []

    for _, row in pos.iterrows():
        cluster = row["_vgene"]
        cognate_pep = row["peptide"]

        # Candidate pool: same cluster, different peptide
        pool = pos[
            (pos["_vgene"] == cluster) & (pos["peptide"] != cognate_pep)
        ]

        if len(pool) == 0:
            # Fallback: any TCR with a different peptide
            pool = pos[pos["peptide"] != cognate_pep]

        if len(pool) == 0:
            continue

        chosen = pool.sample(n=1, random_state=random_state).iloc[0]
        neg_row = row.to_dict()
        neg_row["CDR3b"] = chosen["CDR3b"]
        neg_row["CDR3a"] = chosen.get("CDR3a", np.nan)
        neg_row["label"] = 0
        neg_rows.append(neg_row)

    negatives = pd.DataFrame(neg_rows)
    if "_vgene" in negatives.columns:
        negatives = negatives.drop(columns=["_vgene"])
    negatives = negatives.reset_index(drop=True)

    pos_clean = pos.drop(columns=["_vgene"])
    logger.info(
        f"within_cluster: {len(pos_clean)} positives → {len(negatives)} negatives generated."
    )
    return _combine(pos_clean, negatives)


# ---------------------------------------------------------------------------
# Strategy 4 — Shuffled CDR3
# ---------------------------------------------------------------------------

def shuffled_cdr3(
    df_positives: pd.DataFrame,
    random_state: Optional[int] = 42,
    n_attempts: int = 5,
) -> pd.DataFrame:
    """
    Generate synthetic negatives by shuffling the amino-acid residues of
    each CDR3b sequence.

    The shuffled sequence preserves amino-acid composition but destroys
    positional information.  Up to *n_attempts* shuffles are tried per
    sequence to avoid accidentally regenerating the original.

    Parameters
    ----------
    df_positives : pd.DataFrame
    random_state : int, optional
    n_attempts : int
        Max shuffle attempts before accepting (even if same as original).

    Returns
    -------
    pd.DataFrame
    """
    _set_seed(random_state)
    pos = _validate_positives(df_positives)
    rng = np.random.default_rng(random_state)

    def _shuffle_seq(seq: str) -> str:
        arr = list(seq)
        for _ in range(n_attempts):
            rng.shuffle(arr)
            shuffled = "".join(arr)
            if shuffled != seq:
                return shuffled
        return "".join(arr)  # accept even if same (very short seqs)

    neg = pos.copy()
    neg["CDR3b"] = neg["CDR3b"].apply(_shuffle_seq)
    neg["label"] = 0

    logger.info(
        f"shuffled_cdr3: {len(pos)} positives → {len(neg)} negatives generated."
    )
    return _combine(pos, neg)


# ---------------------------------------------------------------------------
# Strategy 5 — Leave-One-Peptide-Out generator
# ---------------------------------------------------------------------------

def leave_one_peptide_out(
    df_positives: pd.DataFrame,
    random_state: Optional[int] = 42,
) -> Generator[Tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generator for Leave-One-Peptide-Out (LOPO) cross-validation.

    For each unique peptide *p* in the dataset:
      - test set  : all positives with peptide == p (labels preserved as 1)
      - train set : random-swap negatives generated from all *other* peptides,
                    balanced 1:1 pos:neg

    Yields
    ------
    (peptide, train_df, test_df) : tuple
        peptide  – the held-out peptide string
        train_df – balanced training DataFrame (positives + random-swap negatives)
        test_df  – test DataFrame containing only positives for *peptide*

    Notes
    -----
    The test set contains only positives so that callers can add their own
    negatives strategy for evaluation, or score purely on positive ranking.
    For AUC calculation, ``evaluation.py`` pairs the test positives with
    random-swap negatives drawn from the *other* peptides.
    """
    pos = _validate_positives(df_positives)
    peptides = sorted(pos["peptide"].unique())

    if len(peptides) < 2:
        raise ValueError(
            "LOPO requires at least 2 distinct peptides in the dataset."
        )

    for pep in peptides:
        test_pos  = pos[pos["peptide"] == pep].copy().reset_index(drop=True)
        train_pos = pos[pos["peptide"] != pep].copy().reset_index(drop=True)

        if len(train_pos) == 0:
            logger.warning(f"LOPO: No training data for fold peptide={pep}; skipping.")
            continue

        # Build balanced training set with random-swap negatives from train peptides
        train_df = random_swap(train_pos, random_state=random_state)

        logger.debug(
            f"LOPO fold peptide={pep}: "
            f"train={len(train_df)} ({(train_df['label']==1).sum()} pos), "
            f"test_pos={len(test_pos)}"
        )

        yield pep, train_df, test_pos


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, callable] = {
    "random_swap":           random_swap,
    "epitope_balanced":      epitope_balanced,
    "within_cluster":        within_cluster,
    "shuffled_cdr3":         shuffled_cdr3,
    # leave_one_peptide_out is a generator and handled separately
}

STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys()) + ["leave_one_peptide_out"]
