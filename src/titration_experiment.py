"""
titration_experiment.py
=======================
Negative difficulty titration experiment for TCR-pMHC specificity prediction.

Constructs a continuum of 8 difficulty levels by partitioning candidate
negatives into quantile bins based on biophysical distance to nearest positive.
Runs 5-fold CV at each level x 2 feature families x 2 models x 5 seeds.
Generates Figure 14: AUC vs difficulty level for biophysical vs sequence features.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Live output
sys.stdout.reconfigure(line_buffering=True)

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features
from immrep_loader import load_training_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEVEL_NAMES = {
    1: "trivial_far",
    2: "very_easy",
    3: "easy",
    4: "moderate_easy",
    5: "moderate_hard",
    6: "hard",
    7: "very_hard",
    8: "extreme",
}

# Percentile boundaries — level 1 = farthest (easiest), level 8 = closest (hardest)
# Each level covers a 12.5 percentile bin
LEVEL_BOUNDARIES = [0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0]

N_SEEDS = 5
N_SPLITS = 5
N_NEGATIVES = 2445  # 1:1 with positives


# ---------------------------------------------------------------------------
# Model builders (matching evaluation.py patterns but with plan-specified params)
# ---------------------------------------------------------------------------

def _build_lr(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=0.1,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=1,
        )),
    ])


def _build_rf(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            n_jobs=1,
        )),
    ])


_MODEL_BUILDERS = {"lr": _build_lr, "rf": _build_rf}


# ---------------------------------------------------------------------------
# Helper: safe AUC
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    model_builder,
    n_splits: int = 5,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> list[float]:
    if groups is not None:
        # GroupKFold: no TCR leaks across folds
        # Shuffle groups deterministically before splitting
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(random_state)
        rng.shuffle(unique_groups)
        group_map = {g: i for i, g in enumerate(unique_groups)}
        shuffled_groups = np.array([group_map[g] for g in groups])
        gkf = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
        splitter = gkf.split(X, y, groups=shuffled_groups)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splitter = skf.split(X, y)

    aucs: list[float] = []
    for train_idx, test_idx in splitter:
        # Skip folds with single class in test
        if len(np.unique(y[test_idx])) < 2:
            continue
        model = model_builder(random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(_safe_auc(y[test_idx], proba))
    return aucs


# ---------------------------------------------------------------------------
# Step 1: Build candidate pool
# ---------------------------------------------------------------------------

def build_candidate_pool(positives: pd.DataFrame) -> pd.DataFrame:
    """
    Generate ~39K candidate negatives: each positive TCR × 16 non-cognate peptides.

    Returns DataFrame with columns:
        CDR3a, CDR3b, peptide, mhc, label, source_tcr_idx, cognate_peptide,
        dist_to_nearest_pos, difficulty_level (1-8)
    """
    print("  Extracting biophysical features for positives...")
    X_pos = extract_features(positives, feature_type="biophysical")

    all_peptides = sorted(positives["peptide"].unique())
    print(f"  Found {len(all_peptides)} unique peptides, {len(positives)} positives")

    # Build candidate rows
    print("  Building candidate negative pool...")
    rows = []
    for idx, row in positives.iterrows():
        cognate = row["peptide"]
        for pep in all_peptides:
            if pep == cognate:
                continue
            rows.append({
                "CDR3a": row["CDR3a"],
                "CDR3b": row["CDR3b"],
                "peptide": pep,
                "mhc": row["mhc"],
                "label": 0,
                "source_tcr_idx": idx,
                "cognate_peptide": cognate,
            })

    candidates = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  Candidate pool size: {len(candidates)}")

    # Extract biophysical features for all candidates
    print("  Extracting biophysical features for candidates (this may take ~1 min)...")
    X_cand = extract_features(candidates, feature_type="biophysical")

    # Compute distance to nearest positive, EXCLUDING the source TCR's cognate pair.
    # This prevents the distance from being trivially dominated by the peptide
    # difference between cognate and swapped peptide on the same TCR.
    print("  Computing distances to nearest non-cognate positive...")
    pos_indices = positives.index.values  # original indices in positives df
    source_tcr_indices = candidates["source_tcr_idx"].values

    chunk_size = 5000
    min_dists = np.empty(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), chunk_size):
        end = min(start + chunk_size, len(candidates))
        diff = X_cand[start:end, np.newaxis, :] - X_pos[np.newaxis, :, :]  # (chunk, n_pos, 104)
        dists = np.sqrt((diff ** 2).sum(axis=2))  # (chunk, n_pos)

        # Mask out same-TCR cognate: set distance to inf for the source positive
        for i in range(start, end):
            src_idx = source_tcr_indices[i]
            # Find which column in X_pos corresponds to the source TCR
            pos_col = np.where(pos_indices == src_idx)[0]
            if len(pos_col) > 0:
                dists[i - start, pos_col[0]] = np.inf

        min_dists[start:end] = dists.min(axis=1)
        if start % 20000 == 0 and start > 0:
            print(f"    ... processed {start}/{len(candidates)}")

    candidates["dist_to_nearest_pos"] = min_dists

    # Assign difficulty levels based on percentile bins.
    # Level 1 = trivial_far = FARTHEST from positives (largest distances) = EASIEST
    # Level 8 = extreme     = CLOSEST  to positives  (smallest distances) = HARDEST
    # Strategy: invert distances so that rank 1 = largest distance.
    boundaries = np.percentile(min_dists, LEVEL_BOUNDARIES)
    print(f"  Distance percentile boundaries (ascending): {boundaries.round(2)}")

    # Assign percentile rank bins, then invert: bin 8 (top distances) → level 1
    levels = np.zeros(len(candidates), dtype=int)
    for bin_idx in range(8):  # bins 0..7
        lo = boundaries[bin_idx]
        hi = boundaries[bin_idx + 1]
        if bin_idx == 7:
            mask = (min_dists >= lo) & (min_dists <= hi)
        else:
            mask = (min_dists >= lo) & (min_dists < hi)
        # Invert: bin 0 (smallest dists, closest, hardest) → level 8
        #         bin 7 (largest dists, farthest, easiest) → level 1
        levels[mask] = 8 - bin_idx

    # Fix any unassigned (edge cases at boundaries)
    levels[levels == 0] = 1
    candidates["difficulty_level"] = levels

    # Verify level counts and monotonicity
    # Level 1 = farthest → mean dist should be LARGEST
    # Level 8 = closest  → mean dist should be SMALLEST
    # So mean_dists should be DECREASING from level 1 to level 8
    print("\n  Level statistics:")
    mean_dists = []
    for lvl in range(1, 9):
        subset = candidates[candidates["difficulty_level"] == lvl]
        mean_d = subset["dist_to_nearest_pos"].mean()
        mean_dists.append(mean_d)
        print(f"    Level {lvl} ({LEVEL_NAMES[lvl]}): {len(subset)} candidates, "
              f"mean dist={mean_d:.3f}")

    # Verify monotonically decreasing (level 1 farthest, level 8 closest)
    monotone = all(mean_dists[i] > mean_dists[i + 1] for i in range(len(mean_dists) - 1))
    print(f"  Monotonically decreasing distances (L1->L8): {monotone}")
    if not monotone:
        print("  WARNING: non-monotonic ordering detected!")

    return candidates


def sample_negatives_at_level(
    candidate_pool: pd.DataFrame,
    level: int,
    n_negatives: int = N_NEGATIVES,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample n_negatives from the specified difficulty level.
    Uses replacement if pool is smaller than n_negatives.
    """
    pool = candidate_pool[candidate_pool["difficulty_level"] == level]
    n_unique = len(pool)
    replacement_needed = n_unique < n_negatives

    sampled = pool.sample(
        n=n_negatives,
        replace=replacement_needed,
        random_state=random_state,
    )
    sampled = sampled.reset_index(drop=True)

    return sampled, n_unique, replacement_needed


# ---------------------------------------------------------------------------
# Step 2: Run the titration benchmark
# ---------------------------------------------------------------------------

def run_titration_experiment(
    positives: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    n_seeds: int = N_SEEDS,
    n_splits: int = N_SPLITS,
) -> pd.DataFrame:
    """
    Run benchmark across all difficulty levels, feature types, models, and seeds.

    Returns long-format DataFrame with per-fold AUC values.
    """
    print("\n[Step 2] Running titration benchmark...")
    print(f"  Config: {8} levels × 2 features × 2 models × {n_seeds} seeds × {n_splits} folds")
    total_expected = 8 * 2 * 2 * n_seeds * n_splits
    print(f"  Expected rows: {total_expected}")

    # Pre-extract sequence features for positives (reused across levels/seeds)
    print("  Pre-extracting sequence features for positives...")
    X_pos_seq = extract_features(positives, feature_type="sequence")
    print("  Pre-extracting biophysical features for positives...")
    X_pos_bio = extract_features(positives, feature_type="biophysical")
    y_pos = positives["label"].values

    all_rows = []

    for level in range(1, 9):
        level_name = LEVEL_NAMES[level]
        print(f"\n  Level {level} ({level_name})...")

        level_pool = candidate_pool[candidate_pool["difficulty_level"] == level]
        n_unique = len(level_pool)

        # Compute mean biophysical distance for this level
        mean_neg_dist_bio = level_pool["dist_to_nearest_pos"].mean()

        for seed in range(n_seeds):
            # Sample negatives at this level for this seed
            neg_df, _n_unique, replacement_used = sample_negatives_at_level(
                candidate_pool, level, n_negatives=N_NEGATIVES, random_state=seed
            )

            # Build balanced dataset
            combined = pd.concat([positives, neg_df], ignore_index=True)
            y = combined["label"].values

            # Build group labels for GroupKFold: group by CDR3b so same
            # TCR never appears in both train and test
            groups = combined["CDR3b"].values
            # Map CDR3b strings to integer group IDs
            unique_cdr3 = {s: i for i, s in enumerate(np.unique(groups))}
            group_ids = np.array([unique_cdr3[s] for s in groups])

            # Feature extraction for negatives
            neg_seq = extract_features(neg_df, feature_type="sequence")
            neg_bio = extract_features(neg_df, feature_type="biophysical")

            X_seq = np.vstack([X_pos_seq, neg_seq])
            X_bio = np.vstack([X_pos_bio, neg_bio])

            feature_matrices = {
                "biophysical": X_bio,
                "sequence": X_seq,
            }

            for feat_type, X in feature_matrices.items():
                for model_name, builder in _MODEL_BUILDERS.items():
                    fold_aucs = _cv_auc(X, y, builder, n_splits=n_splits,
                                        random_state=seed, groups=group_ids)
                    for fold_idx, auc in enumerate(fold_aucs):
                        all_rows.append({
                            "difficulty_level": level,
                            "level_name": level_name,
                            "feature_type": feat_type,
                            "model": model_name,
                            "seed": seed,
                            "fold": fold_idx,
                            "auc": auc,
                            "mean_neg_dist_bio": mean_neg_dist_bio,
                            "n_unique_negatives": n_unique,
                            "n_sampled": N_NEGATIVES,
                            "replacement_used": replacement_used,
                            "is_anchor": False,
                        })

        # Progress report per level
        level_rows = [r for r in all_rows if r["difficulty_level"] == level]
        for feat_type in ["biophysical", "sequence"]:
            for model_name in ["lr", "rf"]:
                subset_aucs = [
                    r["auc"] for r in level_rows
                    if r["feature_type"] == feat_type
                    and r["model"] == model_name
                    and not np.isnan(r["auc"])
                ]
                mean_auc = np.mean(subset_aucs) if subset_aucs else float("nan")
                print(f"    [{feat_type:12s} {model_name:2s}] mean AUC = {mean_auc:.4f}")

    results_df = pd.DataFrame(all_rows)
    print(f"\n  Total rows generated: {len(results_df)}")
    return results_df


# ---------------------------------------------------------------------------
# Step 3: Crossover detection
# ---------------------------------------------------------------------------

def find_crossover(
    summary_df: pd.DataFrame,
    model: str = "rf",
) -> dict | None:
    """
    Find difficulty level where biophysical AUC drops below sequence AUC.
    Linear interpolation between adjacent levels if crossover falls between levels.

    Returns dict or None if curves don't cross.
    """
    bio_rows = summary_df[
        (summary_df["feature_type"] == "biophysical") &
        (summary_df["model"] == model)
    ].sort_values("difficulty_level")

    seq_rows = summary_df[
        (summary_df["feature_type"] == "sequence") &
        (summary_df["model"] == model)
    ].sort_values("difficulty_level")

    if len(bio_rows) == 0 or len(seq_rows) == 0:
        return None

    bio_aucs = bio_rows["auc_mean"].values
    seq_aucs = seq_rows["auc_mean"].values
    levels = bio_rows["difficulty_level"].values

    # Find sign change in (bio - seq)
    diff = bio_aucs - seq_aucs
    crossover_level = None
    crossover_dist = None
    bio_at_cross = None
    seq_at_cross = None

    for i in range(len(diff) - 1):
        if diff[i] >= 0 and diff[i + 1] < 0:
            # Linear interpolation between level[i] and level[i+1]
            # diff[i] * (1-t) + diff[i+1] * t = 0  => t = diff[i] / (diff[i] - diff[i+1])
            t = diff[i] / (diff[i] - diff[i + 1])
            crossover_level = levels[i] + t * (levels[i + 1] - levels[i])
            # Interpolate distance
            dist_i = bio_rows.iloc[i]["mean_neg_dist_bio"]
            dist_i1 = bio_rows.iloc[i + 1]["mean_neg_dist_bio"]
            crossover_dist = dist_i + t * (dist_i1 - dist_i)
            bio_at_cross = bio_aucs[i] + t * (bio_aucs[i + 1] - bio_aucs[i])
            seq_at_cross = seq_aucs[i] + t * (seq_aucs[i + 1] - seq_aucs[i])
            break

    if crossover_level is None:
        return None

    return {
        "crossover_level": float(crossover_level),
        "crossover_distance": float(crossover_dist),
        "bio_auc_at_crossover": float(bio_at_cross),
        "seq_auc_at_crossover": float(seq_at_cross),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Step 4: Generate Figure 14
# ---------------------------------------------------------------------------

def plot_titration_figure(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate Figure 14: two panels (LR | RF), biophysical vs sequence AUC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    colors = {"biophysical": "#d62728", "sequence": "#1f77b4"}  # red, blue
    labels = {"biophysical": "Biophysical (104-d)", "sequence": "Sequence (BLOSUM62+kmer)"}

    level_names_short = [LEVEL_NAMES[i] for i in range(1, 9)]
    x_levels = list(range(1, 9))

    for ax_idx, (ax, model_name) in enumerate(zip(axes, ["lr", "rf"])):
        for feat_type in ["biophysical", "sequence"]:
            sub = summary_df[
                (summary_df["feature_type"] == feat_type) &
                (summary_df["model"] == model_name)
            ].sort_values("difficulty_level")

            means = sub["auc_mean"].values
            ci_low = sub["auc_ci_low"].values
            ci_high = sub["auc_ci_high"].values
            levels = sub["difficulty_level"].values

            color = colors[feat_type]
            ax.plot(levels, means, "o-", color=color, linewidth=2,
                    label=labels[feat_type], zorder=3)
            ax.fill_between(levels, ci_low, ci_high, alpha=0.2, color=color)

        # Chance line
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Chance (0.5)")

        # Crossover annotation
        crossover = find_crossover(summary_df, model=model_name)
        if crossover is not None:
            cx = crossover["crossover_level"]
            cy = crossover["bio_auc_at_crossover"]
            ax.axvline(cx, color="purple", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.annotate(
                f"Crossover\nLevel {cx:.1f}\n(d={crossover['crossover_distance']:.2f})",
                xy=(cx, cy),
                xytext=(cx + 0.3, cy + 0.07),
                fontsize=8,
                color="purple",
                arrowprops=dict(arrowstyle="->", color="purple", lw=1),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lavender", alpha=0.8),
            )
        else:
            ax.text(0.05, 0.05, "No crossover found", transform=ax.transAxes,
                    fontsize=9, color="purple", style="italic")

        model_label = "Logistic Regression" if model_name == "lr" else "Random Forest"
        ax.set_title(f"{model_label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Difficulty Level", fontsize=11)
        ax.set_xticks(x_levels)
        ax.set_xticklabels(
            [f"{i}\n({level_names_short[i-1].replace('_', ' ')})" for i in x_levels],
            fontsize=7,
        )
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel("ROC-AUC", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

        # Secondary x-axis: mean biophysical distance
        bio_sub = summary_df[
            (summary_df["feature_type"] == "biophysical") &
            (summary_df["model"] == model_name)
        ].sort_values("difficulty_level")
        mean_dists = bio_sub["mean_neg_dist_bio"].values

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_levels)
        ax2.set_xticklabels([f"{d:.1f}" for d in mean_dists], fontsize=7)
        ax2.set_xlabel("Mean biophys. dist. to nearest positive", fontsize=8)

    fig.suptitle(
        "AUC vs Negative Difficulty: Biophysical vs Sequence Features",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to: {output_path}")


# ---------------------------------------------------------------------------
# Step 5: Compute summary statistics
# ---------------------------------------------------------------------------

def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-level/feature/model statistics with 95% CI."""
    rows = []
    for level in range(1, 9):
        for feat_type in ["biophysical", "sequence"]:
            for model_name in ["lr", "rf"]:
                subset = results_df[
                    (results_df["difficulty_level"] == level) &
                    (results_df["feature_type"] == feat_type) &
                    (results_df["model"] == model_name) &
                    (~results_df["is_anchor"])
                ]
                aucs = subset["auc"].dropna().values
                if len(aucs) == 0:
                    continue

                mean_dist = subset["mean_neg_dist_bio"].mean()
                n_unique = subset["n_unique_negatives"].iloc[0] if len(subset) > 0 else 0

                rows.append({
                    "difficulty_level": level,
                    "level_name": LEVEL_NAMES[level],
                    "feature_type": feat_type,
                    "model": model_name,
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "auc_ci_low": float(np.percentile(aucs, 2.5)),
                    "auc_ci_high": float(np.percentile(aucs, 97.5)),
                    "mean_neg_dist_bio": float(mean_dist),
                    "n_unique_negatives": int(n_unique),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Negative Difficulty Titration Experiment")
    print("=" * 70)

    # --- Load data ---
    print("\n[Step 0] Loading data...")
    all_data = load_training_data()
    positives = all_data[all_data["label"] == 1].reset_index(drop=True)
    print(f"  Positives: {len(positives)} rows")
    print(f"  Peptides: {sorted(positives['peptide'].unique())}")

    # --- Build candidate pool ---
    print("\n[Step 1] Building candidate pool...")
    candidate_pool = build_candidate_pool(positives)
    print(f"  Candidate pool: {len(candidate_pool)} rows")

    # Verify NaN-free
    nan_count = candidate_pool["dist_to_nearest_pos"].isna().sum()
    print(f"  NaN distances: {nan_count}")

    # --- Run benchmark ---
    results_df = run_titration_experiment(
        positives, candidate_pool, n_seeds=N_SEEDS, n_splits=N_SPLITS
    )

    # --- Save raw results ---
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "titration_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n  Raw results saved to: {results_csv}")

    # --- Compute summary ---
    print("\n[Step 3] Computing summary statistics...")
    summary_df = compute_summary(results_df)
    summary_csv = results_dir / "titration_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary saved to: {summary_csv}")

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Mean AUC per Level per Feature Type per Model")
    print("=" * 70)
    print(f"{'Level':<5} {'Name':<16} {'Feature':<14} {'LR AUC':>8} {'RF AUC':>8}")
    print("-" * 55)
    for level in range(1, 9):
        level_name = LEVEL_NAMES[level]
        for feat_type in ["biophysical", "sequence"]:
            lr_row = summary_df[
                (summary_df["difficulty_level"] == level) &
                (summary_df["feature_type"] == feat_type) &
                (summary_df["model"] == "lr")
            ]
            rf_row = summary_df[
                (summary_df["difficulty_level"] == level) &
                (summary_df["feature_type"] == feat_type) &
                (summary_df["model"] == "rf")
            ]
            lr_auc = f"{lr_row['auc_mean'].values[0]:.4f}" if len(lr_row) > 0 else "  N/A"
            rf_auc = f"{rf_row['auc_mean'].values[0]:.4f}" if len(rf_row) > 0 else "  N/A"
            print(f"  {level:<3} {level_name:<16} {feat_type:<14} {lr_auc:>8} {rf_auc:>8}")
        print()

    # --- Find crossover ---
    print("=" * 70)
    print("CROSSOVER ANALYSIS")
    print("=" * 70)
    for model_name in ["lr", "rf"]:
        result = find_crossover(summary_df, model=model_name)
        model_label = "Logistic Regression" if model_name == "lr" else "Random Forest"
        if result is not None:
            print(f"  {model_label}: Crossover at Level {result['crossover_level']:.2f} "
                  f"(distance={result['crossover_distance']:.3f})")
            print(f"    Bio AUC at crossover: {result['bio_auc_at_crossover']:.4f}")
            print(f"    Seq AUC at crossover: {result['seq_auc_at_crossover']:.4f}")
        else:
            # Determine direction (bio always better or seq always better)
            bio_sub = summary_df[
                (summary_df["feature_type"] == "biophysical") &
                (summary_df["model"] == model_name)
            ]["auc_mean"].values
            seq_sub = summary_df[
                (summary_df["feature_type"] == "sequence") &
                (summary_df["model"] == model_name)
            ]["auc_mean"].values
            if len(bio_sub) > 0 and len(seq_sub) > 0:
                bio_wins = (bio_sub > seq_sub).all()
                seq_wins = (seq_sub > bio_sub).all()
                if bio_wins:
                    print(f"  {model_label}: No crossover — biophysical AUC > sequence AUC at all levels")
                elif seq_wins:
                    print(f"  {model_label}: No crossover — sequence AUC > biophysical AUC at all levels")
                else:
                    print(f"  {model_label}: No clean crossover detected (non-monotonic or irregular pattern)")
            else:
                print(f"  {model_label}: No crossover — insufficient data")

    # --- Generate figure ---
    print("\n[Step 4] Generating Figure 14...")
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = str(fig_dir / "fig14_titration_crossover.png")
    plot_titration_figure(results_df, summary_df, fig_path)

    print("\n" + "=" * 70)
    print("Experiment complete.")
    print(f"  Results CSV : {results_csv}")
    print(f"  Summary CSV : {summary_csv}")
    print(f"  Figure      : {fig_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
