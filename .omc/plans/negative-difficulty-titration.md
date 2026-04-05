# Negative Difficulty Titration Experiment

**Tier 2, Item 4** | Created: 2026-04-04

## Context

We have shown (benchmark_results.csv) that random-swap negatives inflate biophysical AUC by +22.5pp (0.764 standard vs 0.539 LOPO) while deflating sequence AUC by -8.5pp (0.496 standard vs 0.581 LOPO). The geometric mechanism (manifold_analysis.py) confirms random-swap negatives land far from positives in biophysical space (easy boundary) but intermixed in sequence space (impossible boundary).

**The missing piece:** We don't know *where the crossover point is*. At what negative difficulty level does biophysical transition from "better than sequence" to "worse than sequence"? This experiment constructs a continuum of 8 difficulty levels and plots AUC as a function of difficulty for both feature families.

## Existing Infrastructure

- `immrep_loader.py` -- load_positives_only() returns 2445 rows, 17 peptides, HLA-A*02:01
- `negative_sampling.py` -- 5 strategies (random_swap, epitope_balanced, within_cluster, shuffled_cdr3, LOPO)
- `features.py` -- extract_features(df, "biophysical") -> 104-d; extract_features(df, "sequence") -> 8900-d
- `evaluation.py` -- _cv_auc(), _safe_auc(), _build_lr(), _build_rf() pipelines
- `manifold_analysis.py` -- min_distances_to_set() for vectorized nearest-neighbor distances
- CDR3 diversity data: 2067 unique CDR3b sequences across 17 peptides, median intra-peptide Hamming ~0.70

---

## Work Objectives

1. Define 8 discrete difficulty levels forming a monotonic continuum from "trivially easy" to "maximally hard" negatives
2. Implement a negative generator that produces balanced (1:1) datasets at each level
3. Run standard 5-fold CV at each level x 2 feature families x 2 models x 10 seeds
4. Produce the crossover figure (Figure 6) and supporting CSV
5. Validate that existing endpoints (Level 1 ~ random_swap, Level 7-8 ~ within_cluster/LOPO) reproduce known results

---

## Guardrails

### Must Have
- Each difficulty level must produce at least 2445 negatives (1:1 with positives) or document why not and use resampling
- Difficulty levels must be monotonically ordered by a single quantifiable proxy metric (mean biophysical distance to nearest positive)
- The two extreme levels must approximately reproduce our existing random_swap and within_cluster results as a sanity check
- All random operations must accept random_state for reproducibility
- Results CSV must include per-level, per-feature, per-model, per-seed AUC values (long format)

### Must NOT Have
- No new feature extraction methods -- use existing biophysical and sequence features only
- No new models -- use existing LR and RF pipelines only
- No changes to existing files in src/ -- all new code in new files
- No dependency on external databases (VDJdb, IEDB) -- use IMMREP positives only

---

## Difficulty Level Definitions

The difficulty axis is **biophysical proximity of negatives to positives**. We control this by constraining which TCR-peptide swaps are allowed at each level, using progressively tighter similarity filters.

### Operational Definition of "Difficulty"

For each candidate negative (TCR_i, peptide_j), compute its biophysical feature vector and find the Euclidean distance to the nearest positive in biophysical feature space. Negatives that are far away are easy (model can trivially separate them); negatives that are close are hard (model must learn fine-grained discrimination).

### The 8 Levels

We pre-compute the biophysical feature distance from every possible candidate negative to its nearest positive, then partition candidates into quantile bins. This gives us a principled, data-driven difficulty scale.

**Pre-computation step:** Generate the full candidate pool. For N=2445 positives across P=17 peptides, the candidate negative pool is all (TCR_i, peptide_j) pairs where peptide_j != cognate_peptide_i. This gives roughly N*(P-1) = 2445*16 = 39,120 candidate negatives. For each, extract biophysical features and compute distance to nearest positive.

| Level | Name | Selection Rule | Expected AUC (bio) | Expected AUC (seq) |
|-------|------|---------------|--------------------|--------------------|
| 1 | `trivial_far` | Bottom 12.5% of distances (farthest from positives) | ~0.95+ | ~0.50 |
| 2 | `very_easy` | 12.5-25th percentile of distances | ~0.85 | ~0.50 |
| 3 | `easy` | 25-37.5th percentile | ~0.75 | ~0.52 |
| 4 | `moderate_easy` | 37.5-50th percentile | ~0.65 | ~0.55 |
| 5 | `moderate_hard` | 50-62.5th percentile | ~0.58 | ~0.57 |
| 6 | `hard` | 62.5-75th percentile | ~0.52 | ~0.58 |
| 7 | `very_hard` | 75-87.5th percentile | ~0.48 | ~0.58 |
| 8 | `extreme` | Top 12.5% of distances (closest to positives) | ~0.45 | ~0.55 |

**Why 8 levels, not 6 or 10:** With 39,120 candidates and 8 levels, each level contains ~4,890 candidates. We need 2,445 negatives per level (1:1 ratio), so each bin has ~2x headroom. 10 levels would give ~3,912 per bin (still feasible but tighter). 8 is the sweet spot.

**Why percentile-based, not absolute-distance thresholds:** Absolute thresholds would be feature-scale-dependent and would produce unequal bin sizes. Percentile bins guarantee equal pool size at each level.

**Key insight:** We define difficulty in biophysical space because that is the space where we have the strongest prior (random-swap negatives are geometrically far). The figure will show whether sequence features are *invariant* to this biophysical difficulty axis (flat curve) or also respond to it (but with different slope/intercept).

### Fallback: Resampling with Replacement

If any level's bin contains fewer than 2,445 unique candidates (unlikely with 8 bins, but possible at extremes), sample with replacement. Record the effective unique count and flag it in the output CSV. If unique count drops below 50% of target (< 1,223), that level is unreliable -- still run it but add a `low_diversity` flag.

---

## Task Flow

### Step 1: Build the Candidate Pool and Distance Index
**File:** `src/titration_negatives.py`

**What to do:**
1. Load positives via `load_positives_only()`
2. Generate the full candidate negative pool: for each positive (TCR_i, peptide_cognate), create one row for each of the 16 non-cognate peptides, assigning the TCR to that peptide. Result: ~39,120 candidate rows, each with columns `[CDR3a, CDR3b, peptide, mhc, label=0, source_tcr_idx, cognate_peptide]`
3. Extract biophysical features for all positives (2,445 x 104) and all candidates (39,120 x 104) using `extract_features()`
4. For each candidate, compute Euclidean distance to nearest positive in biophysical feature space. Store as column `dist_to_nearest_pos`
5. Assign each candidate to a difficulty level (1-8) based on percentile rank of `dist_to_nearest_pos`. Use `np.percentile` with boundaries at [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
6. Verify monotonicity: `mean_dist[level_k] > mean_dist[level_k+1]` for all k (level 1 = farthest = easiest)

**Function signatures:**
```python
def build_candidate_pool(positives: pd.DataFrame) -> pd.DataFrame:
    """Returns ~39K candidate negatives with columns:
    CDR3a, CDR3b, peptide, mhc, label, source_tcr_idx, cognate_peptide,
    dist_to_nearest_pos, difficulty_level (1-8)
    """

def sample_negatives_at_level(
    candidate_pool: pd.DataFrame,
    level: int,
    n_negatives: int = 2445,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample n_negatives from the specified difficulty level.
    Returns DataFrame with same schema as positives + label=0.
    Uses replacement if pool is smaller than n_negatives.
    """
```

**Acceptance criteria:**
- candidate_pool has ~39,120 rows with no NaN in dist_to_nearest_pos
- Each of the 8 levels contains >= 2,445 candidates (or documented exception)
- mean distance is monotonically decreasing from level 1 to level 8
- Level 1 mean distance approximately matches the mean distance of random_swap negatives from manifold_analysis.py

### Step 2: Run the Titration Benchmark
**File:** `src/run_titration.py`

**What to do:**
1. For each of 8 difficulty levels:
   a. Sample 2,445 negatives at that level (10 different random seeds)
   b. Combine with the 2,445 positives to form a balanced dataset
   c. Extract both biophysical and sequence features
   d. Run 5-fold stratified CV with both LR and RF (reuse `_cv_auc` pattern from evaluation.py)
   e. Record per-fold AUC values
2. Also run the existing random_swap and within_cluster strategies as comparison anchors
3. Compute per-level summary statistics: mean AUC, std AUC, 95% CI

**Key design decisions:**
- Use 10 seeds (not 5) because the per-level variance will be higher with constrained sampling
- Use 5-fold CV (matching existing evaluation.py) for consistency
- Total experiment size: 8 levels x 2 features x 2 models x 10 seeds x 5 folds = 1,600 individual AUC measurements + 2 anchors x 2 x 2 x 10 x 5 = 200 anchor measurements

**Function signatures:**
```python
def run_titration_experiment(
    positives: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    n_seeds: int = 10,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Returns long-format DataFrame with columns:
    difficulty_level, level_name, feature_type, model, seed, fold, auc,
    mean_neg_dist, n_unique_negatives, low_diversity_flag
    """
```

**Acceptance criteria:**
- Output CSV has 1,600 rows (8 x 2 x 2 x 10 x 5) + 200 anchor rows
- AUC values at level 1 (biophysical) should be within 0.05 of random_swap biophysical AUC (0.764)
- No NaN AUC values (every fold must have both classes represented)
- Runtime estimate: ~15 minutes total (each CV fold is fast with LR/RF on 4,890 samples x 104 or 8,900 features)

### Step 3: Compute the Crossover Point and Generate Figure 6
**File:** `src/plot_titration.py`

**What to do:**
1. Load the titration results CSV
2. Compute per-level mean and 95% CI for each (feature_type, model) combination
3. Find the crossover point: the difficulty level where biophysical AUC drops below sequence AUC. Interpolate linearly between adjacent levels if crossover falls between levels.
4. Generate Figure 6 with the following design:

**Figure 6 Design: "Negative Difficulty Titration"**

```
Layout: 1 row x 2 columns (LR | RF), shared y-axis

X-axis: Difficulty Level (1-8), labeled with level names
        Secondary x-axis (top): Mean biophysical distance to nearest positive
Y-axis: AUC (0.3 to 1.0)

Curves per panel:
  - Solid red line:  Biophysical features (mean AUC +/- 95% CI shaded)
  - Solid blue line: Sequence features (mean AUC +/- 95% CI shaded)
  - Dashed gray horizontal line: AUC = 0.5 (chance level)

Annotations:
  - Vertical dashed line at crossover point (where red and blue curves intersect)
  - Text box: "Crossover at Level X (d = Y.YY)"
  - Arrow/markers at Level 1 and Level 7 connecting to existing benchmark AUCs
    as sanity check anchors

Title: "AUC vs Negative Difficulty: Biophysical Features Degrade Faster"
       (or whatever the data shows)
```

5. Also generate a supplementary panel: the same plot but using the RF model only, with individual seed traces shown as thin transparent lines behind the mean curve (to show variance structure).

**Output files:**
- `results/titration_results.csv` -- long-format raw data
- `results/titration_summary.csv` -- per-level aggregated statistics
- `results/figures/fig6_titration_crossover.png` -- main figure
- `results/figures/fig6_titration_crossover_supplement.png` -- variance structure

**Function signatures:**
```python
def find_crossover(
    summary_df: pd.DataFrame,
    model: str = "rf",
) -> dict:
    """Returns {crossover_level: float, crossover_distance: float,
                bio_auc_at_crossover: float, seq_auc_at_crossover: float}
    or None if curves don't cross.
    """

def plot_titration_figure(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate Figure 6."""
```

**Acceptance criteria:**
- Figure has two panels (LR, RF) with shaded confidence intervals
- Crossover point is annotated if it exists
- Biophysical curve is monotonically decreasing (or nearly so -- allow small non-monotonicity within CI)
- The figure clearly demonstrates whether the crossover exists or not
- Anchor points from existing benchmarks are shown and match within 0.05

### Step 4: Validate and Characterize
**File:** `src/titration_validation.py`

**What to do:**
1. **Sanity checks:** Compare Level 1 AUC to random_swap AUC and Level 7-8 AUC to within_cluster AUC from benchmark_results.csv. Print concordance report.
2. **Distance verification:** For each level, verify that mean biophysical distance is monotonically ordered. Plot histogram of distances per level (small multipanel figure) to visually confirm separation.
3. **Sequence-space distance at each level:** Compute the mean sequence-space distance (PCA-50, as in manifold_analysis.py) of negatives at each level to nearest positive. This answers whether biophysical difficulty and sequence difficulty are correlated or orthogonal.
4. **Per-peptide breakdown:** At each difficulty level, compute AUC separately per peptide (LOPO-style but simpler -- just filter test data to each peptide). This reveals whether the crossover point varies by peptide.
5. **Output:** Print a summary table and save `results/titration_validation.csv`

**Acceptance criteria:**
- Sanity check concordance within 0.05 AUC for both anchors
- Monotonicity confirmed or explained
- Sequence-vs-biophysical distance correlation reported (Pearson r)
- Per-peptide crossover points tabulated for at least the 5 largest peptides

---

## Output File Specifications

### `results/titration_results.csv` (long format, ~1800 rows)
| Column | Type | Description |
|--------|------|-------------|
| `difficulty_level` | int (1-8) | Difficulty level |
| `level_name` | str | Human-readable name (trivial_far, very_easy, ..., extreme) |
| `feature_type` | str | "biophysical" or "sequence" |
| `model` | str | "lr" or "rf" |
| `seed` | int | Random seed (0-9) |
| `fold` | int | CV fold (0-4) |
| `auc` | float | ROC-AUC for this fold |
| `mean_neg_dist_bio` | float | Mean biophysical distance of negatives to nearest positive |
| `mean_neg_dist_seq` | float | Mean sequence PCA distance of negatives to nearest positive |
| `n_unique_negatives` | int | Number of unique negatives available at this level |
| `n_sampled` | int | Number actually sampled (2445 or less) |
| `replacement_used` | bool | Whether sampling with replacement was needed |
| `is_anchor` | bool | True for random_swap/within_cluster comparison runs |

### `results/titration_summary.csv` (aggregated, 32 rows = 8 levels x 2 features x 2 models)
| Column | Type | Description |
|--------|------|-------------|
| `difficulty_level` | int | |
| `level_name` | str | |
| `feature_type` | str | |
| `model` | str | |
| `auc_mean` | float | Mean AUC across seeds and folds |
| `auc_std` | float | Std AUC |
| `auc_ci_low` | float | 2.5th percentile |
| `auc_ci_high` | float | 97.5th percentile |
| `mean_neg_dist_bio` | float | |
| `mean_neg_dist_seq` | float | |
| `n_unique_negatives` | int | |

---

## Success Criteria

1. **The crossover figure exists and is interpretable:** Either (a) a clear crossover point is found and annotated, or (b) the curves do not cross and we can explain why (e.g., sequence features are uniformly better or worse regardless of difficulty).
2. **Sanity checks pass:** Level 1 biophysical AUC is within 0.05 of the established random_swap biophysical AUC (0.764). Level 7-8 is within 0.10 of within_cluster biophysical AUC (0.602).
3. **The difficulty axis is real:** Biophysical AUC monotonically decreases from Level 1 to Level 8, confirming that our difficulty operationalization captures genuine signal.
4. **Statistical robustness:** 95% CIs at the crossover point do not overlap (if crossover exists), indicating the crossover is statistically significant with 10 seeds x 5 folds.
5. **Manuscript-ready figure:** Figure 6 is publication-quality (300 dpi, proper axis labels, legend, annotation).

---

## Estimated Complexity: MEDIUM

- 4 new files, 0 modifications to existing files
- ~400-500 lines of new code total
- ~15 min runtime for full experiment
- Primary risk: the crossover may not exist cleanly (sequence features may be uniformly flat across difficulty levels, in which case the story is still interesting but different)

---

## Open Questions (deferred to execution)

See `.omc/plans/open-questions.md`
