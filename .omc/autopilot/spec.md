# Autopilot Spec: TCR-pMHC Negative Sampling Bias Study

## Research Question
How much do different negative sampling strategies inflate AUC in TCR-pMHC specificity prediction, and does this confound the apparent advantage of sequence-based over physics-based features?

## Novel Contributions
1. **Primary**: First systematic quantification of AUC inflation across negative sampling strategies (random swap, epitope-balanced, external, within-donor) using IMMREP_2022 benchmark
2. **Secondary**: Leave-one-peptide-out evaluation comparing sequence embeddings vs biophysical features under each sampling regime

## Data
- IMMREP_2022 benchmark dataset (primary)
- VDJdb (supplementary enrichment)
- IEDB (supplementary enrichment)

## Methods
### Negative Sampling Strategies to Compare
1. **Random swap** (standard field practice): randomly pair TCRs with non-cognate peptides
2. **Epitope-balanced**: ensure equal neg/pos ratio per epitope
3. **Within-donor**: use TCRs from same donor as negatives
4. **Shuffled CDR3**: shuffle amino acids in CDR3 loops
5. **External DB**: pull non-cognate TCRs from orthogonal database

### Features
- **Sequence-based**: ESM2 embeddings of CDR3 loops, BLOSUM62 encoding
- **Biophysical**: amino acid physicochemical properties (charge, hydrophobicity, volume, SASA), pairwise interaction scores

### Classifier
- Logistic regression + Random forest (interpretable baselines)
- Leave-one-peptide-out cross-validation

### Primary Metric
- AUC-ROC per sampling strategy and feature type
- Delta-AUC (inflation = AUC_strategy - AUC_leave_one_out)

## Expected Novel Finding
That AUC inflation from random negative sampling is substantial (>0.15 AUC units) and that this inflation differentially benefits sequence-based models over biophysical ones — explaining the apparent ML superiority on known peptides.
