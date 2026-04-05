# Open Questions

## Negative Difficulty Titration — 2026-04-04

- [ ] Should the distance metric be Euclidean on raw biophysical features or on StandardScaler-transformed features? — Manifold_analysis.py uses StandardScaler before distance computation; we should match that for consistency, but this means the difficulty axis is scale-dependent. The plan assumes scaled features (matching manifold_analysis.py precedent).

- [ ] Is 8 levels the right granularity, or should we also run a finer 16-level version as supplementary? — 8 levels gives ~4,890 candidates per bin (2x headroom over 2,445 needed). 16 levels would give ~2,445 per bin (no headroom, requiring replacement sampling at some levels). Decision: start with 8, add 16 as supplementary only if the 8-level curve shows interesting non-linearity.

- [ ] Should the candidate pool include self-peptide pairs where TCR_i is paired with its own cognate peptide but a different TCR's CDR3? — Current plan only swaps peptides (keeping TCR fixed, changing peptide). An alternative is to also swap TCRs (keeping peptide fixed, changing TCR from a different peptide's pool). The current approach matches random_swap semantics. Decision: keep current approach for consistency with existing negative_sampling.py.

- [ ] What if the biophysical AUC curve is not monotonically decreasing? — This could happen if the biophysical feature space has non-convex structure (easy negatives in some directions, hard in others). If non-monotonicity exceeds CI bounds, we should investigate per-peptide effects. Decision: deferred to validation step.

- [ ] Per-peptide crossover points: should these go into the main figure or supplementary? — The main figure shows aggregate crossover. Per-peptide crossover variation could be a separate supplementary panel or table. Decision: deferred to after seeing the data.
