# Code Review: TCR-pMHC Negative Sampling Bias Study
## IMMREP_2022 Benchmark Analysis Pipeline

**Review date:** 2026-04-04  
**Reviewer:** Senior Code Review Agent  
**Scope:** All 13 scripts in `src/`; all produced output files in `results/`  
**Status of outputs:** All figures (fig0–fig12) and all CSV/JSON result files are present and match the scientific narrative.

---

## Summary of Findings

The codebase is scientifically coherent and the main findings are internally consistent with the output files. The core benchmark logic (LOPO implementation, AUC computation, feature extraction) is correct. The most important issues are reproducibility hazards from mixed seed practices and hardcoded absolute paths, and one statistical validity concern about how the LOPO AUC standard deviation is computed. None of the issues identified invalidate the headline findings, but several require disclosure or fixing before submission.

**Issue counts by severity:**

| Severity | Count |
|---|---|
| Critical (must fix before submission) | 3 |
| Important (should fix) | 9 |
| Suggestion (nice to have) | 7 |

---

## Per-Script Review

---

### 1. `immrep_loader.py`

**Purpose:** Load per-epitope IMMREP_2022 `.txt` files; remap labels from {1, -1} to {1, 0}; return a standardised DataFrame.

**Inputs:** Per-epitope tab-delimited files under `data/IMMREP_2022_TCRSpecificity/`.  
**Outputs:** DataFrame with columns `[CDR3a, CDR3b, peptide, mhc, label]`.

**What is correct:**
- Column aliasing (`TRA_CDR3` / `A3`, `TRB_CDR3` / `B3`) handles two known IMMREP file variants.
- Label remapping is double-checked for both int and string keys (line 64).
- CDR3b amino acid validity filter (`^[ACDEFGHIKLMNPQRSTVWY]+$`, line 75) correctly rejects non-standard characters and empty strings.
- `IMMREP_DATA_ROOT` is constructed relative to `__file__` (line 26-27), which is portable.

**Issues:**

**[Important]** Line 56-57: `cdr3a = df.get("TRA_CDR3", df.get("A3", pd.Series(dtype=str)))`. If neither column exists, `cdr3a` is an all-NaN Series indexed differently from `df`. Downstream concat into `out` at line 66 silently produces an all-NaN CDR3a column without raising an error or warning. CDR3a is not used in the current feature extractors, so this has no practical impact today, but it is a silent failure mode. A warning should be emitted if both column names are absent.

**[Important]** Line 59: `if cdr3b is None or cdr3b.isna().all()`. `df.get()` never returns `None` — it returns the default, which is `pd.Series(dtype=str)`. The `is None` branch is dead code. The check functions correctly because `.isna().all()` on an empty Series returns `True`, but the logic is misleading. Change the guard to `if cdr3b.isna().all()`.

**[Suggestion]** `load_test_data()` only loads from `true_dir` and ignores the `test_dir` argument after computing it (lines 125-137). The `test_dir` parameter is accepted but never used. Either use it or remove it to avoid confusion.

**[Suggestion]** `summarize()` (line 155) prints to stdout with no logging level option. This is appropriate for interactive use but could pollute `run_analysis.py` output; consider accepting a logger argument.

---

### 2. `negative_sampling.py`

**Purpose:** Implement 5 negative sampling strategies: `random_swap`, `epitope_balanced`, `within_cluster`, `shuffled_cdr3`, `leave_one_peptide_out`.

**Inputs:** DataFrame of positives (label == 1).  
**Outputs:** Balanced DataFrame (strategies 1-4) or a generator of `(peptide, train_df, test_df)` tuples (strategy 5).

**What is correct:**
- `_validate_positives()` robustly filters and warns if non-positive rows are passed.
- All four dataset-building strategies produce balanced 1:1 pos:neg sets.
- `leave_one_peptide_out()` correctly excludes the held-out peptide from training data before calling `random_swap`, so there is no data leakage in the train/test split.
- The docstring for `leave_one_peptide_out()` correctly documents that the test set contains only positives, with negatives added by the caller in `evaluation.py`.

**Issues:**

**[Critical]** Line 69: `_combine()` uses a hardcoded `random_state=42` for the final shuffle, regardless of the `random_state` argument passed to the calling strategy function. This means that even when `_standard_cv_benchmark()` iterates over seeds 0, 1, 2 (line 141 of `evaluation.py`), the dataset shuffle order is identical across all seeds. The variation between seeds comes only from the `np.random.choice` inside each strategy and from `StratifiedKFold`, not from dataset ordering. This understates the true variance of AUC estimates. The fix is to pass `random_state` through to `_combine()`.

**[Critical]** Line 267: `within_cluster()` calls `pool.sample(n=1, random_state=None)`. All other randomness in the pipeline is seeded, but this call is fully non-deterministic. Running the analysis twice will produce different `within_cluster` datasets. The `random_state` parameter received by `within_cluster()` must be passed here.

**[Important]** Line 62-63: `_set_seed()` calls `np.random.seed()` (legacy global RNG) and `random.seed()` (Python stdlib RNG). The `shuffled_cdr3()` function (line 316) additionally creates a `np.random.default_rng(random_state)` Generator object, bypassing the legacy global seed. This means two parallel RNG streams exist with the same seed but in different API families. The resulting shuffle behavior is deterministic but the co-mingling of legacy and Generator APIs creates confusion about which RNG controls which operations. Use `np.random.default_rng` throughout for new code.

**[Important]** The `within_cluster()` proxy for V-gene cluster assignment (first 3 amino acids of CDR3b, line 81) is a weak heuristic. CDR3b sequences begin with the conserved `C` anchor residue; the first three characters are almost always `CAS` or similar germline-conserved motifs. This means `_infer_vgene()` effectively groups by TRBV family in broad strokes, but the clustering will not meaningfully distinguish TRBV subfamilies. This is documented as an approximation, but the scientific paper should quantify how many clusters are singletons and what fraction fall back to random sampling.

**[Suggestion]** `epitope_balanced()` pads/trims to exactly `target_neg` samples at line 205 (`neg_rows[:target_neg]`). If `per_peptide_quota * (n_peptides - 1) > target_neg`, the trim is silent. A log message indicating how many negatives were trimmed would aid transparency.

---

### 3. `features.py`

**Purpose:** Extract biophysical (104-dim) and sequence (8900-dim) feature vectors from CDR3b and peptide sequences.

**Inputs:** DataFrame with `CDR3b` and `peptide` columns.  
**Outputs:** `np.ndarray` of shape `(n_samples, n_features)`.

**What is correct:**
- BLOSUM62 matrix entries match the standard NCBI matrix (verified by spot-check of A-row and W-row).
- Kidera factors match the 1985 Kidera et al. reference values for all 20 standard amino acids.
- Kyte-Doolittle values match the 1982 reference scale.
- Biophysical dimensionality is correctly computed: 2 sequences × 13 properties × 4 statistics = 104.
- Sequence dimensionality: 30×20 (BLOSUM CDR3b) + 8000 (3-mer) + 15×20 (BLOSUM peptide) = 600 + 8000 + 300 = 8900.
- `get_feature_dim()` returns the correct values for both feature types.
- Unknown amino acids are silently zero-coded — appropriate behaviour with a silent log warning at DEBUG level.

**Issues:**

**[Important]** `_biophys_summary()` uses `mat.std(axis=0)` with default `ddof=0` (population std). This is the statistically correct choice given that the sequence is the complete object being summarised (not a sample), but it means single-residue sequences (length 1) return `std=0` for all properties. This is technically correct but should be documented. No peptide in IMMREP_2022 has length 1, so there is no practical impact.

**[Important]** The `_kmer_freq()` function (line 97) normalises by `n = len(seq) - k + 1`, the number of k-mer windows. This is the correct relative frequency. However, `features.py` and `kmer_audit.py` both independently implement k-mer frequency extraction with identical logic (lines 81-90 of `kmer_audit.py`). This duplication means a bug fix in one would not automatically propagate to the other. Both produce numerically identical output, but `kmer_audit.py` should import from `features.py`.

**[Suggestion]** `_blosum_encode()` with `max_len=30` zero-pads all sequences to length 30. CDR3b sequences in IMMREP_2022 range from approximately 10 to 25 amino acids; the 5-10 trailing zero rows encode nothing and slightly inflate dimensionality. Consider a data-driven `max_len` set to the 99th percentile sequence length. This is a minor efficiency concern, not a correctness issue.

---

### 4. `evaluation.py`

**Purpose:** Core benchmark engine exposing `run_benchmark()`. Runs standard stratified 5-fold CV and LOPO CV for any (sampling strategy, feature type) combination.

**Inputs:** DataFrame of positives, strategy name, feature type, n_repeats.  
**Outputs:** Dict with AUC metrics.

**What is correct:**
- LOPO is implemented correctly: the held-out peptide is excluded from training; test negatives are drawn exclusively from other peptides' random-swap pool. This is the right construction for zero-shot generalisation evaluation.
- `_safe_auc()` handles the single-class edge case.
- `StandardScaler` is included inside the Pipeline, so test-set statistics do not leak into the scaler fit.
- Per-model results (LR, RF) are reported separately in addition to the combined mean.

**Issues:**

**[Critical]** Lines 244-245 in `_lopo_benchmark()`: `all_lopo_aucs` is extended with all individual model AUC values across all folds. The reported `auc_lopo_std` is therefore the standard deviation of raw per-model per-fold AUC values, not the standard error of the mean LOPO AUC. With 17 peptides × 2 models = 34 values, the std will appear artificially tight and does not correctly represent fold-level variance. For error bars in a publication figure, the correct quantity is either the std of the 17 per-peptide mean AUCs, or the std of the 2 per-model mean AUCs. The existing std is not wrong as a descriptive statistic but is misleading as an uncertainty estimate. This affects the error bars in `fig2_standard_vs_lopo_bar.png`.

**[Important]** Line 192-200: The test negatives for each LOPO fold are generated by calling `random_swap(other_positives, random_state=0)` with a fixed seed of `0`. This means the test negative set is identical across all 8 experimental conditions (4 strategies × 2 feature types), because `_lopo_benchmark()` does not accept a strategy argument and always uses `random_swap` for test negatives. This is intentional and documented in the docstring (line 26-27 of `evaluation.py`), but it means the LOPO AUC for all conditions within a feature type is identical — as confirmed in the actual results CSV where all 4 strategies for `biophysical` have the same `auc_lopo` value (0.5390) and all 4 for `sequence` have the same `auc_lopo` (0.5810). This is not a bug but must be explicitly stated in the paper's methods. The `auc_inflation` column is therefore not a property of the sampling strategy but a property of the standard CV component only.

**[Important]** Line 215: `model = builder(random_state=42)` uses a fixed seed for all LOPO fold model fits. This is reproducible but means that LOPO fold variance estimates cannot separate sampling variance from optimisation variance.

**[Important]** Line 343-344: The `run_benchmark()` return dict references `std_results.get("model_lr_mean_auc")` but the key set by `_standard_cv_benchmark()` at line 155 is `f"model_{model_name}_mean_auc"` — which for `model_name="lr"` is `"model_lr_mean_auc"`. These match, but the construction is fragile: the key format is set in one function and consumed by string concatenation in another with no shared constant. Define key name constants at module level.

---

### 5. `run_analysis.py`

**Purpose:** Orchestrator. Loads IMMREP training data, runs all 8 conditions (4 strategies × 2 feature types), prints results tables, saves CSV/JSON.

**What is correct:**
- Runs all 8 conditions; handles per-condition exceptions gracefully without aborting the entire run.
- Saves both a flat CSV (for downstream analysis) and a full JSON with per-peptide AUC data.
- The headline-findings block correctly identifies the max/min inflation condition.
- The differential inflation test (line 261: `if diff > 0.02`) is a reasonable magnitude threshold.

**Issues:**

**[Important]** `N_REPEATS = 3` (line 75) is set in `run_analysis.py` but not passed through to `run_benchmark()` as a default; the `run_benchmark()` signature defaults to `n_repeats=5`. Because `run_analysis.py` explicitly passes `n_repeats=N_REPEATS`, the actual value used is 3, giving 3×5=15 AUC estimates per standard CV condition. This should be documented clearly. The discrepancy between the default (5) and the used value (3) could cause confusion if `run_benchmark()` is called directly.

**[Important]** Line 215: The JSON serialisation uses `default=lambda x: None if np.isnan(x) else x`. This converts NaN floats to `null` in JSON, which is correct. However, `np.float32` values (returned by the feature extractors) will cause a `TypeError` in `json.dump` on some platforms. The lambda should cast to `float` before testing: `default=lambda x: None if isinstance(x, float) and np.isnan(x) else float(x) if isinstance(x, np.floating) else x`.

**[Suggestion]** The comment on line 31 says "Key output: auc_inflation = auc_standard(strategy) - auc_lopo(random_swap)". This is accurate but subtly important: the LOPO component is always computed with random_swap negatives regardless of the `strategy` parameter (see evaluation.py issue above). Making this explicit in the docstring would aid readers.

---

### 6. `visualize_results.py`

**Purpose:** Generate 4 publication figures (heatmap, bar chart, per-peptide, LR vs RF) from `benchmark_results.csv`.

**What is correct:**
- Matplotlib backend set to `Agg` before import (line 32) — correct for headless environments.
- Graceful degradation if matplotlib is not installed (lines 28-37).
- Figure 3 correctly averages LOPO AUC over sampling strategies before plotting, which is valid since all strategies produce identical LOPO AUC (as noted above).

**Issues:**

**[Important]** `fig3_per_peptide()` averages `lopo_auc` over strategies in line 187-190. The comment says "averaged over strategies (since LOPO is strategy-independent)". This is correct, but a reader unfamiliar with the code might not understand why all strategies produce the same LOPO AUC. This average currently collapses 4 identical values per peptide per feature type — it would be worth an assertion that the per-strategy values are indeed equal (within floating-point tolerance) to detect any future regression.

**[Suggestion]** `fig1_inflation_heatmap()` at line 89 uses `abs(val) > 0.15` to decide white vs black annotation text. The threshold of 0.15 is a magic number; it should be named (e.g., `TEXT_CONTRAST_THRESHOLD = 0.15`).

**[Suggestion]** All 4 figure functions accept `patches` as a parameter but only `fig2` and `fig4` use it. The unused parameter is harmless but untidy.

---

### 7. `manifold_analysis.py`

**Purpose:** Geometric investigation of why biophysical features inflate AUC (+0.225) while sequence features deflate it (-0.086). UMAP visualisations, per-sample proximity ratios, polarity inversion test.

**What is correct:**
- PCA reduction of 8900-dim sequence features to 50 PCs before distance computation is appropriate and the explained variance is reported (line 104).
- Joint scaling of pos/RS/WC before splitting (lines 99-121) is correct — the scaler must be fit on all data to avoid information leakage between partitions.
- Wilcoxon signed-rank test (lines 204-205) is the right non-parametric test for paired distance distributions.
- Polarity inversion test (Section 7) correctly implements LOPO: trains on other-peptide data, evaluates on held-out peptide positives paired with RS negatives from other peptides.

**Issues:**

**[Important]** Lines 181-192: The file executes all analysis at module level with no `if __name__ == "__main__":` guard. If any other script imports `manifold_analysis` (unlikely but possible), the full UMAP computation (~several minutes) would execute. Wrap the module-level execution in a guard.

**[Important]** Section 6 (per-peptide analysis, line 326): `pd.read_csv(LOPO_CSV)` uses the string path `LOPO_CSV = os.path.join(RESULTS_DIR, "per_peptide_lopo_auc.csv")`. This file is produced by `run_analysis.py`. If `manifold_analysis.py` is run before `run_analysis.py`, it will fail with a `FileNotFoundError`. There is no explicit dependency check or error message. Add a check with a clear error like `"Run run_analysis.py first to generate per_peptide_lopo_auc.csv"`.

**[Important]** Line 219-225: Normalised proximity ratios `rs_relative_bio` and `rs_relative_seq` are computed as per-sample ratios `d(pos→RS) / d(pos→WC)`. The Wilcoxon test on these (line 226) compares the per-sample ratio distributions between spaces. However, the sizes of `d_pos_rs_bio` and `d_pos_rs_seq` are equal (both equal to `n_pos`) because they index the same positive samples. The Wilcoxon signed-rank test is valid here as a paired test on the same positives evaluated in two different spaces. This is methodologically sound.

**[Suggestion]** The `min_distances_to_set()` function (line 127) uses O(n_A × n_B) memory per batch. For the IMMREP dataset sizes (~2,400 positives × ~2,400 negatives), each batch of 256 allocates a 256 × 2400 × 104 float32 array ≈ 256 MB. This is acceptable but worth documenting as a memory bound.

---

### 8. `kmer_audit.py`

**Purpose:** Compare 3-mer CDR3b frequency features (LR and RF) against 22 published IMMREP_2022 models using the official test set with ground-truth labels.

**What is correct:**
- Uses the actual IMMREP test set paired with true labels via merge on TRA_CDR3 + TRB_CDR3 (lines 129-136). This is the correct evaluation protocol matching the IMMREP challenge.
- Random baseline correctly bypasses model training (lines 157-167).
- `micro_auc()` is correctly documented as the average AUC per epitope, matching the IMMREP `evaluate.py` implementation.

**Issues:**

**[Critical — Hardcoded Path]** Line 31: `BASE = Path("C:/Users/Jack/Documents/Research/TCR-pMHC Specificity Prediction")`. This absolute Windows path will fail on any other machine. All downstream paths inherit from `BASE`. This is the most severe reproducibility hazard in the entire codebase. The same issue appears in `shap_attribution.py` (line 50), `shap_fast.py` (line 15), and `structural_analysis.py` (line 45). The scripts that use `Path(__file__).parent.parent` (e.g., `immrep_loader.py`, `ceiling_analysis.py`) are correctly portable.

**[Important]** Lines 193-197: Epitope discovery uses `TRAIN_DIR.iterdir()` without sorting guarantee. Python 3.6+ `Path.iterdir()` does not guarantee alphabetical order. The code wraps in `sorted()`, which is correct.

**[Important]** Lines 228-235: The `microaucs.csv` column parsing logic assumes a specific column ordering (model name at index 0, "MicroAUC" label at index 1, epitopes thereafter, metadata at end). This will silently produce wrong results if the CSV format changes. The parsing should be documented with the actual expected CSV structure, and a column-count assertion should be added.

**[Important]** `load_test_with_labels()` (lines 118-137) merges test and true sets on both `TRA_CDR3` and `TRB_CDR3`. Some IMMREP test entries have missing TRA_CDR3 (CDR3a). A left merge with `how="left"` followed by `dropna(subset=["Label", "TRB_CDR3"])` will silently drop any rows where the merge failed because TRA_CDR3 was NaN in the test set but not the true set (or vice versa). The merge should be on TRB_CDR3 alone, or the merge-failure rate should be logged.

---

### 9. `ceiling_analysis.py`

**Purpose:** Three analyses: (1) label noise sensitivity; (2) mutual information / 1-NN Bayes error ceiling; (3) sample size confound; (4) CDR3b sequence diversity.

**What is correct:**
- Analysis 1 correctly re-runs the full LOPO pipeline on noise-corrupted positive sets, showing flat AUC across 0–30% noise (matching the reported finding).
- Analysis 2 1-NN LOO ceiling is methodologically sound as an upper-bound estimate on discriminability.
- Analysis 3 correctly distinguishes n_test_positives (this peptide's count) from n_train_positives (all other peptides' count).
- `observed_overall = 0.539` (line 148) matches the actual result in `benchmark_results.csv` (row 1: `auc_lopo = 0.5390`).

**Issues:**

**[Important]** Analysis 2, line 329-333: The per-peptide 1-NN LOO accuracy loop re-calls `knn.fit()` inside a `sum()` generator expression. This is legal Python but extremely inefficient and hard to read. More importantly, the `knn.fit()` call is in the condition of the `and` expression: `1 for i in pep_indices if (knn.fit(...) and knn.predict(...)[0] == y_pep[i])`. `knn.fit()` returns the fitted KNN object, which is always truthy. The condition is always triggered. This is correct behaviour, but the logic looks like a predicate when it is actually a side-effectful fit call. The reader may mistake this for a bug. Refactor into an explicit loop.

**[Important]** Line 149: `observed_overall = 0.539` is a literal constant derived from a prior run. If the benchmark is re-run with different hyperparameters or a corrected seed, this constant becomes stale and the crossover analysis (lines 196-206) will silently use an incorrect reference value. This constant should be loaded from `per_peptide_lopo_auc.csv` or `benchmark_results.csv` rather than hardcoded.

**[Suggestion]** Analysis 4 CDR3b diversity uses `CDR3b` column (line 479) but `immrep_loader.py` canonicalises the column as `"CDR3b"`. This is consistent, but note that ceiling_analysis.py uses `df_pos["CDR3b"]` while structural_analysis.py uses `"TRB_CDR3"` — the two scripts load data differently and use different column names for the same field.

---

### 10. `structural_analysis.py`

**Purpose:** HLA-A*02:01 anchor residue analysis; RCSB PDB structure search; statistical comparison of failure vs success peptides; CDR3b 3-mer motif analysis.

**What is correct:**
- Anchor deviation score formula (line 119: `max(0.0, 1.8 - p2_hydro) + max(0.0, 1.8 - end_hydro)`) correctly penalises only sub-threshold hydrophobicity, not supra-threshold.
- `HLA_A0201_P2_PREFERRED = set("LMVIA")` matches published HLA-A*02:01 binding motif (Sette 1994, confirmed by NetMHCPan4.1).
- Mann-Whitney U test (two-sided) is appropriate for the small group sizes (failure n=5, success n=6).
- RCSB search uses a polite rate limit (`time.sleep(0.3)`, line 226).

**Issues:**

**[Critical — Hardcoded Path]** Line 45: `PROJECT_ROOT = Path("C:/Users/Jack/Documents/Research/TCR-pMHC Specificity Prediction")`. Same issue as `kmer_audit.py`.

**[Important]** Lines 56-81: `FAILURE_PEPTIDES`, `SUCCESS_PEPTIDES`, and `MID_PEPTIDES` have LOPO AUC values hardcoded as dictionary literals. These values must match the output of `run_analysis.py`. If the benchmark is re-run, these constants become stale with no warning. The script should load AUC values from `per_peptide_lopo_auc.csv` and compute failure/success groupings dynamically (e.g., failure = bottom quartile, success = top quartile).

**[Important]** The Mann-Whitney U test at line 276-279 tests 7 metrics across the failure-vs-success comparison without any multiple-testing correction. With 7 comparisons, the expected number of false positives at α=0.05 under the null is 0.35. The reported p=0.034 for anchor deviation score is the only significant result; applying Bonferroni correction gives α'=0.0071, and the result becomes non-significant at that threshold. This does not necessarily invalidate the finding (the anchor deviation score has strong biological prior support), but the paper must acknowledge the multiple-comparisons issue.

**[Important]** Lines 299-322: `load_cdr3b_data()` reads training files using `usecols=["Label", "TRB_CDR3"]` and filters `Label == -1` for negatives. This is the raw IMMREP label, which is correct. However, this function in `structural_analysis.py` duplicates data-loading logic from `immrep_loader.py` without reusing it. The column name used here is `"TRB_CDR3"` (raw IMMREP format) whereas the pipeline-standard column is `"CDR3b"`. This is consistent within the script but creates a divergence: if a future IMMREP data release changes column names, structural_analysis.py will break while immrep_loader.py (which handles both `TRB_CDR3` and `B3`) will not.

---

### 11. `shap_attribution.py`

**Purpose:** Compute SHAP values for 3-mer RF models; test whether Spearman ρ(SHAP rank, MI rank) confirms k-mer fingerprinting hypothesis.

**What is correct:**
- SHAP API version handling (lines 184-190) correctly addresses both old list-of-arrays API and new 3D array API.
- Subsampling to 300 background samples (line 173) is a reasonable computational trade-off for `TreeExplainer` with interventional feature perturbation.
- Using `feature_perturbation="interventional"` with a background dataset is the correct choice for correlated features (3-mer frequencies are correlated).
- Global ρ is computed over all 8000 features, not just the top-N, which is the correct test of the fingerprinting hypothesis.

**Issues:**

**[Critical — Hardcoded Path]** Line 50: `BASE_DIR = Path("C:/Users/Jack/Documents/Research/TCR-pMHC Specificity Prediction")`. Same issue as `kmer_audit.py`.

**[Important]** Lines 164-206: The script executes SHAP computation at module level with no `if __name__ == "__main__":` guard. Any accidental import triggers the full computation.

**[Important]** Line 284-285: `BEST_EPS` and `WORST_EPS` are hardcoded lists: `["LLWNGPMAV", "GLCTLVAML", "NYNYLYRLF"]` and `["GPRLGVRAT", "TPRVTGGGAM", "RAQAPPPSW"]`. These match the LOPO AUC rankings in `structural_analysis.py` and the results CSV. If the benchmark is re-run with a different result, these lists will not auto-update. Load from the results CSV.

**[Important]** Line 178: `explainer = shap.TreeExplainer(rf, data=X_bg, model_output="probability", feature_perturbation="interventional")`. Then line 181: `shap_vals = explainer.shap_values(X_bg)`. The SHAP values are computed on the same background dataset used to fit the explainer. This computes SHAP values for the background distribution, which is the correct interpretation under interventional perturbation but should be explicitly noted. Computing SHAP on the same data used as the reference distribution is a deliberate choice, not a mistake, but it means SHAP values measure the feature attribution relative to the background mean rather than to any specific reference.

**[Suggestion]** The scientific claim (reported ρ = 0.037 globally) aligns with the conclusion that models do NOT learn k-mer fingerprints at the global level (ρ < 0.3, conclusion = "REJECTED"). The per-epitope analysis showing CAS as top SHAP feature across 12/17 epitopes is the stronger evidence for the V-gene bias finding. These two findings point in different directions and deserve careful framing in the paper.

---

### 12. `shap_fast.py`

**Purpose:** Faster version of `shap_attribution.py` (100 background samples vs 300; 50 RF estimators vs 100).

**What is correct:**
- `n_estimators=50` and background n=100 produce faster runtime with some accuracy loss.
- `sys.stdout.reconfigure(line_buffering=True)` is appropriate for long-running scripts piped to log files.

**Issues:**

**[Critical — Hardcoded Path]** Line 15: `BASE_DIR = Path("C:/Users/Jack/Documents/Research/TCR-pMHC Specificity Prediction")`.

**[Important]** Both `shap_attribution.py` and `shap_fast.py` write to the same output files (`results/shap_attribution.csv`, `results/figures/fig12_*.png`). The last script run wins. There is no versioning or naming convention to distinguish which output came from which script. This must be resolved before submission — either retire one script or write to distinct output paths.

**[Important]** Lines 84-88 in `shap_fast.py`: The SHAP API version handling has a bug relative to `shap_attribution.py`. Line 87 calls `expl.shap_values(bg)` a second time inside the `elif isinstance(expl.shap_values(bg), list)` check. This triggers a second full SHAP computation for the `elif` branch. The result is discarded; the branch only checks the type. The fix: `sv = np.array(expl.shap_values(bg))` once, then branch on `sv.ndim`.

**[Suggestion]** The two SHAP scripts should be consolidated into one with a `--fast` flag.

---

### 13. `figure1_proximity_ratio.py`

**Purpose:** Generate the primary publication figure (Panel A: log10 RS/WC ratio distributions; Panel B: ratio vs AUC inflation dual-axis bar).

**What is correct:**
- Uses `Path(__file__).parent.parent` for path construction — fully portable.
- Output at 300 DPI (vs 150 DPI in other scripts) is appropriate for the primary figure.
- Panel B uses a log-scale y-axis for the proximity ratio, which is necessary given the 1862× vs 1.116× range.
- The AUC inflation values (+0.225, -0.086) are hardcoded from the paper results and match the values in `benchmark_results.csv`.

**Issues:**

**[Important]** Lines 140-147: `AUC_INFLATION` and `MEAN_RATIO` dictionaries use hardcoded values from a prior analysis run. The `MEAN_RATIO` values are re-computed live in this script (lines 125-126: `mean_ratio_bio`, `mean_ratio_seq`) but `AUC_INFLATION` is not. If the benchmark results change, `AUC_INFLATION` must be updated manually. At minimum, load these from `benchmark_results.csv`.

**[Suggestion]** Line 191-192: `fig.canvas.draw()` is called to force layout before reading `get_ylim()`. This is a known matplotlib pattern but fragile — the annotation placement for mean-line labels (lines 194-205) depends on the y-axis limits computed at render time. Consider computing annotation y positions from the data rather than from axis limits.

---

## Cross-Cutting Issues

### Hardcoded Absolute Paths

Four scripts (`kmer_audit.py`, `shap_attribution.py`, `shap_fast.py`, `structural_analysis.py`) use hardcoded Windows absolute paths of the form `Path("C:/Users/Jack/Documents/Research/TCR-pMHC Specificity Prediction")`. The remaining 9 scripts use `Path(__file__).parent.parent` or equivalent. The fix is uniform: replace all four instances with:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

### Scripts Without `if __name__ == "__main__":` Guard

`manifold_analysis.py` and `shap_attribution.py` execute all their analysis code at module import time. This is a reproducibility anti-pattern: accidental imports (e.g., during testing or interactive exploration) silently run expensive computations. These scripts should be refactored to call a `main()` function inside an `if __name__ == "__main__":` block.

### Inconsistent Random Seed Architecture

The pipeline mixes three seeding mechanisms:
1. Legacy global `np.random.seed()` in `_set_seed()` (negative_sampling.py line 62)
2. `np.random.default_rng(seed)` Generator objects in `shuffled_cdr3()` and `ceiling_analysis.py`
3. `random_state` parameters passed to sklearn objects

These three mechanisms do not interact. A paper reviewer attempting to verify reproducibility may be confused. The pipeline should standardise on `np.random.default_rng` throughout, passing explicit Generator objects to each function rather than relying on global state.

### LOPO AUC Standard Deviation Is Pooled Across Models

As noted under `evaluation.py`, `auc_lopo_std` is the std of 34 individual AUC values (17 peptides × 2 models), not the std of the 17 peptide-level AUC estimates. This affects error bars in Figure 2. The correct per-peptide-level standard deviation is available in `per_peptide_lopo_auc.csv` and should be used for publication figures.

### Duplicate Implementation of k-mer Frequency

`features.py::_kmer_freq()` and `kmer_audit.py::trimer_freq()` implement the same normalised 3-mer frequency computation. `shap_attribution.py::kmer_freq_matrix()` and `shap_fast.py::kmer_freq()` implement a fourth and fifth version. All produce identical output, but a bug fix must be applied to all four independently. Consolidate into `features.py`.

### Hardcoded Scientific Constants From Prior Runs

Multiple scripts embed numeric results from prior runs as literals:
- `ceiling_analysis.py` line 148: `observed_overall = 0.539`
- `structural_analysis.py` lines 56-81: all LOPO AUC values
- `figure1_proximity_ratio.py` lines 140-143: AUC inflation values
- `shap_attribution.py` lines 284-285: best/worst epitope lists

These create a reproducibility trap: if the benchmark is re-run with any change (different `N_REPEATS`, corrected seeds, new data), the downstream scripts will silently use stale reference values. All these constants should be loaded dynamically from `benchmark_results.csv` or `per_peptide_lopo_auc.csv`.

### `per_peptide_lopo_auc.csv` Averages Identical Values

Since all 4 strategies produce the same LOPO AUC (as explained in the evaluation.py review), averaging over strategies in `visualize_results.py::fig3_per_peptide()` and `manifold_analysis.py` section 6 collapses 4 identical values. The average is numerically correct but the operation is uninformative. The sampling strategy dimension should be collapsed by selecting a single strategy (e.g., `random_swap`) rather than averaging.

---

## Statistical Validity Assessment

### AUC Estimation

**Standard CV AUC:** Estimated from 3 seeds × 5 folds = 15 AUC values per model per condition. This is adequate for a benchmark characterisation. The standard deviation correctly reflects fold-level variation.

**LOPO AUC:** One AUC value per peptide per model, averaged across 17 peptides and 2 models = 34 values. The averaging is correct, but as noted, the reported std conflates fold variance and model variance. For a 17-peptide dataset, the effective sample size for the mean LOPO AUC is n=17, not n=34.

**AUC inflation:** Defined as `auc_standard - auc_lopo`. The `auc_lopo` denominator is constant across all 4 strategies within a feature type (all use the same random-swap test negatives). This means inflation differences between strategies are entirely driven by differences in `auc_standard`. This is correct by design but must be stated in the paper.

### Class Balance

All datasets are balanced 1:1 positive:negative by construction in every strategy. ROC-AUC is invariant to class balance for a fixed threshold, so the balanced design does not bias AUC estimates. The biophysical LOPO AUC std of 0.292 (from `benchmark_results.csv`) reflects genuine per-peptide variation, not class imbalance artefacts.

### Multiple Comparisons

The structural analysis tests 7 metrics across 2 groups without correction (noted above). The k-mer audit comparison (3-mer LR vs 22 published models) is a descriptive ranking, not a formal hypothesis test, so no correction is needed there.

### Label Quality

The label noise analysis (ceiling_analysis.py) correctly shows LOPO AUC is flat across 0–30% noise. This is a genuine finding but also means the LOPO floor cannot be explained by label noise alone. The paper should also consider the CDR3b diversity analysis as a complementary explanation (low-diversity peptide repertoires are inherently harder to generalise from).

---

## Reproducibility Assessment

| Component | Reproducible? | Notes |
|---|---|---|
| Data loading | Yes | Relative paths, deterministic sort |
| random_swap | Yes (given seed) | Seeds 0, 1, 2 passed explicitly |
| epitope_balanced | Yes (given seed) | |
| within_cluster | **No** | `random_state=None` at line 267 of negative_sampling.py |
| shuffled_cdr3 | Yes | Uses `default_rng` |
| _combine shuffle | Partially | Always uses seed 42 regardless of caller seed |
| LOPO benchmark | Yes | Fixed seed=0 for test negatives |
| Standard CV | Yes | Seeded StratifiedKFold |
| UMAP (manifold_analysis) | Yes | `random_state=42` passed to UMAP |
| SHAP | Yes | Background sample fixed with `default_rng(42)` |
| kmer_audit | Yes | Fixed RF seed=42 |

The `within_cluster` non-reproducibility is the most significant gap. All other components are reproducible given the fixed seeds.

---

## Potential Bugs That Could Affect Findings

1. **`within_cluster` non-determinism (negative_sampling.py line 267):** The within_cluster results in Figures 1 and 2 may differ from a fresh run. The `auc_standard` for `within_cluster,biophysical` is reported as 0.6024 with std=0.1286 (the highest std in the dataset), which is consistent with non-deterministic negatives. This affects the within-cluster inflation estimate of +0.063.

2. **LOPO std misrepresentation (evaluation.py lines 244-245):** The error bars on Figure 2 LOPO bars are too narrow because they pool 34 values. The visual impression of tight LOPO estimates is misleading; the actual per-peptide std is 0.292 (as stored in `auc_lopo_std` in the results CSV, which is itself the pooled std). The per-peptide variation is visible in Figure 3 and is the more informative display.

3. **shap_fast.py double SHAP call (line 87):** A redundant `expl.shap_values(bg)` call in the isinstance check. This is a performance bug, not a correctness bug.

4. **shuffled_cdr3 AUC = 1.000 for sequence features:** The results CSV shows `auc_standard = 0.9999` for `shuffled_cdr3, sequence`. This is expected: sequence features (which include BLOSUM62 encoding of CDR3b) will perfectly discriminate shuffled CDR3b from real CDR3b because shuffling destroys positional structure. This is scientifically correct behaviour and should be noted explicitly in the paper as a positive control.

---

## Recommendations by Priority

### Before Submission (Critical)

1. Fix `within_cluster` non-determinism: pass `random_state` to `pool.sample()` in `negative_sampling.py` line 267.
2. Fix hardcoded absolute paths in `kmer_audit.py`, `shap_attribution.py`, `shap_fast.py`, `structural_analysis.py`. Replace with `Path(__file__).resolve().parent.parent`.
3. Replace or clarify `auc_lopo_std` in the output dict and Figure 2 error bars. Use per-peptide std (std of the 17 per-peptide mean AUCs) rather than the pooled 34-value std.

### Before Submission (Important)

4. Fix `_combine()` hardcoded `random_state=42` to respect the caller's seed.
5. Add `if __name__ == "__main__":` guard to `manifold_analysis.py` and `shap_attribution.py`.
6. Consolidate the 4 k-mer frequency implementations into `features.py`.
7. Load hardcoded AUC constants (`observed_overall`, failure/success peptide lists, AUC inflation values) from the results CSV at runtime.
8. Document in the paper that `auc_lopo` is strategy-independent within a feature type (constant across all 4 strategy rows of the same feature type).
9. Address multiple comparisons in `structural_analysis.py` Step 3.
10. Resolve the output file conflict between `shap_attribution.py` and `shap_fast.py`.
11. Fix the `shap_fast.py` double SHAP call bug in lines 84-88.

### Nice to Have

12. Replace legacy `np.random.seed()` global seeding with `np.random.default_rng` throughout.
13. Add assertion in `fig3_per_peptide()` that LOPO AUC values are equal across strategies.
14. Add `if cdr3b.isna().all()` guard in `immrep_loader.py` (remove dead `is None` branch).
15. Unify data loading in `structural_analysis.py` to use `immrep_loader.py` rather than duplicate file parsing.
16. Add module-level constants for magic numbers (text contrast threshold 0.15 in `visualize_results.py`; anchor threshold 1.8 in `structural_analysis.py`).
17. Document expected memory bounds for the O(n_A × n_B) batched distance computation in `manifold_analysis.py`.
