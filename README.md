Peptide-MHC binding prediction is solved (NetMHCPan). But the next question - does a specific T-cell receptor actually recognize a given peptide-MHC complex? - is what the field calls "a holy grail of systems immunology" (Hudson et al., 2023), and it's still mostly broken.

ML methods report AUC 0.85-0.95 on standard benchmarks but crater on unseen peptides. This gap is largely a measurement artifact caused by the near-universal practice of generating negative training examples via random TCR-peptide swaps.

## Key Findings

- **Random-swap negatives inflate AUC by +0.21 to +0.62** relative to leave-one-peptide-out (LOPO) evaluation across all models tested (LR, RF, MLP)
- **Models learn distributional shortcuts** (V-gene usage, CDR3 anchor motifs) rather than binding logic -- confirmed by SHAP analysis and V-gene ablation
- **Deeper models don't fix a data problem** -- MLP shows *more* inflation (+0.41) than RF (+0.21)
- **Biophysical features overtake sequence features** as negative difficulty increases, mechanistically explaining why physics-based methods outperform sequence methods on novel peptides
- **Experimental negatives recover real signal**: training on dextramer-sorted negatives achieves LOPO AUC 0.710 vs 0.184 for random-swap-trained models (3.9x improvement)
- Results replicate across three independent benchmarks (IMMREP22, TChard, ITRAP)

## Repository Structure

```
src/
  main.py                    # Core benchmark: 4 strategies x 2 features x 2 models
  data_loader.py             # IMMREP22 dataset loading and preprocessing
  immrep_loader.py           # ITRAP and TChard dataset loaders
  features.py                # Biophysical and sequence (3-mer) feature extraction
  negative_sampling.py       # Four negative sampling strategies
  evaluation.py              # Standard CV and LOPO evaluation
  titration_experiment.py    # Difficulty titration across 8 negative hardness levels
  shap_attribution.py        # SHAP analysis of learned features
  vgene_ablation.py          # V-gene contribution ablation study
  experimental_neg_transfer.py  # ITRAP experimental vs swapped negative comparison
  tchard_comparison.py       # TChard cross-benchmark validation
  immrep23_replication.py    # IMMREP23 replication
  dl_baseline_test.py        # MLP deep learning baseline
  generate_paper_figures.py  # Publication figures (seaborn)
results/
  *.csv                      # All experimental results
  figures/paper/             # Publication-quality figures (PNG + PDF)
manuscript.tex               # Full manuscript (LaTeX)
manuscript.pdf               # Compiled manuscript
```

## Reproducing Results

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Download IMMREP22, TChard, and ITRAP data into data/
# Core benchmark
python src/main.py

# Difficulty titration
python src/titration_experiment.py

# SHAP + V-gene ablation
python src/shap_attribution.py
python src/vgene_ablation.py

# Cross-benchmark validation
python src/tchard_comparison.py
python src/immrep23_replication.py

# Experimental negatives (requires ITRAP data)
python src/experimental_neg_transfer.py

# MLP baseline
python src/dl_baseline_test.py

# Generate figures
python src/generate_paper_figures.py
```
