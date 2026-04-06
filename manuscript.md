# Random-Swap Negative Sampling Inflates Reported Performance in TCR-pMHC Specificity Prediction: Quantifying the Bias and Demonstrating a Concrete Alternative

Jack Large^1,\*

^1 Front Range Community College, Westminster, CO, USA

\*Correspondence: jlarge6@student.cccs.edu &emsp; April 2026 &bull; Preprint

---

**Abstract.** Predicting whether a T-cell receptor (TCR) recognises a specific peptide-MHC (pMHC) complex is central to neoantigen vaccine design, adoptive cell therapy, and autoimmune disease modelling, yet current machine learning methods fail to generalise to novel peptides despite reporting high benchmark accuracy. Here we identify a systematic source of this discrepancy: the near-universal practice of generating negative training examples by randomly swapping TCR-peptide pairs introduces distributional shortcuts that models exploit instead of learning binding-relevant features. Across three independent benchmarks (IMMREP22, TChard, ITRAP), four negative sampling strategies, three model architectures (logistic regression, random forest, multilayer perceptron), and two feature representations (biophysical and sequence), we show that random-swap negatives inflate cross-validated AUC by 0.21--0.62 relative to leave-one-peptide-out (LOPO) evaluation, which simulates the clinically relevant scenario of encountering a novel peptide. SHAP analysis and V-gene ablation experiments reveal that inflated models learn repertoire-level distributional features---specifically, the tendency of positive TCRs to cluster in feature space while random-swap negatives scatter---rather than binding-specific molecular interactions. A difficulty titration experiment demonstrates that biophysical features overtake sequence features as negative difficulty increases, mechanistically explaining the published finding that physics-based methods outperform sequence-based methods on novel peptides. Critically, we show that training on experimentally derived negatives (dextramer-sorted non-binders) recovers genuine predictive signal: models trained on experimental negatives achieve a mean LOPO AUC of 0.710 on held-out peptides, compared with 0.184 for models trained on random-swap negatives evaluated against the same experimental ground truth---a 3.9-fold improvement. This result is consistent across peptides and robust to negative subsampling (10 seeds per condition). We propose that the field adopt experimentally derived negatives as the standard for training and evaluation, and provide a concrete protocol for doing so.

**Keywords:** TCR-pMHC, specificity prediction, negative sampling, benchmark inflation, leave-one-peptide-out, bioacoustics, experimental negatives, T-cell receptor

---

## 1&emsp;Introduction

T-cell receptor (TCR) recognition of peptide-MHC (pMHC) complexes is the molecular event that initiates adaptive immune responses against infected or malignant cells. Accurately predicting which TCRs will recognise a given pMHC is essential for multiple translational applications: identifying neoantigen targets for personalised cancer vaccines, selecting TCR clonotypes for adoptive cell therapy, predicting autoimmune epitope reactivities, and interpreting TCR repertoire sequencing data from clinical cohorts (Glanville et al., 2017; Bagaev et al., 2020). The problem has been called "the holy grail of systems immunology" (Bradley, 2023), and substantial computational effort has been directed toward solving it.

Recent machine learning approaches report encouraging performance. Methods including TITAN, NetTCR-2.0, pMTnet, ERGO-II, and TCR-BERT achieve AUC scores of 0.85--0.95 on standard benchmarks (Weber et al., 2021; Montemurro et al., 2021; Lu et al., 2021; Springer et al., 2021; Wu et al., 2024). However, these metrics are almost exclusively reported on peptides present in the training data. When evaluated on held-out peptides---the clinically relevant scenario of encountering a novel neoantigen or pathogen epitope---performance drops precipitously. Independent evaluations consistently find that models achieving AUC > 0.89 on seen epitopes crater to near chance on unseen ones (Grazioli et al., 2022; Peng et al., 2024; Dens et al., 2023). The predictive power of all approaches remains low on the most challenging zero-shot tasks, and performance on cancer neoantigen prediction specifically---the application most urgently needed---is particularly poor.

Two observations in the literature suggest that this generalisation failure may not be solely a modelling problem. First, physics-based methods using classical energy functions outperform ML methods when predicting TCR binding to novel peptides, despite underperforming them on known peptides (Riley et al., 2023). This reversal is difficult to explain if both approaches are learning the same underlying biology; it is readily explained if sequence-based models are memorising training-set-specific distributional patterns rather than generalisable binding logic. Second, the source of negative TCRs has been shown to substantially impact model accuracy, with external negatives potentially introducing uncontrolled confounders (Grazioli et al., 2022).

These observations point toward a common cause: the standard practice of generating negative examples. Because experimentally confirmed non-binders are scarce, the field overwhelmingly relies on *random-swap* negatives---pairing a TCR known to bind peptide A with a different peptide B, under the assumption that random pairings are unlikely to represent true interactions. This assumption is statistically reasonable (the probability that a random TCR-peptide pair represents a genuine interaction is very low), but it introduces a subtler problem: random-swap negatives are *distributionally different* from true negatives in ways that are learnable by machine learning models and orthogonal to binding.

Here, we systematically evaluate the impact of negative sampling strategy on TCR-pMHC specificity prediction. We conduct a controlled experiment across four negative sampling strategies, two feature representations, and three model architectures, using leave-one-peptide-out (LOPO) evaluation to measure genuine generalisation. We identify the mechanism by which random-swap negatives inflate performance, demonstrate that the inflation replicates across three independent benchmarks, and---crucially---show that training on experimentally derived negatives recovers predictive signal that random-swap-trained models entirely lack.

## 2&emsp;Materials and Methods

### 2.1&emsp;Datasets

**IMMREP22.** The IMMREP 2022 TCR Specificity benchmark (Montemurro et al., 2022) contains 2,445 positive TCR-pMHC pairs spanning 17 peptides, primarily HLA-A\*02:01-restricted. The dataset represents a curated consensus from VDJdb and McPAS-TCR, with a held-out true set of labelled TCR-peptide pairs for independent evaluation. Peptides span viral (GILGFVFTL from influenza, NLVPMVATV from CMV, YLQPRTFLL from SARS-CoV-2), self (GLCTLVAML), and other specificities. This dataset served as the primary benchmark for the core negative sampling comparison.

**TChard.** The TChard benchmark (Peng et al., 2024) from Oak Ridge National Laboratory provides pre-defined difficulty splits designed to test generalisation. We used two split types: *easy* (random cross-validation, peptides and CDR3 sequences shared across train/test) and *hard* (peptide + CDR3&beta; held out, simulating novel peptide encounters). TChard also provides two negative types: *only-neg-assays* (experimentally confirmed non-binders from tetramer/multimer assays) and *only-sampled-negs* (computationally generated random-swap negatives). This design enabled direct comparison of experimental versus synthetic negatives on matched data.

**ITRAP.** The ITRAP benchmark (Grazioli et al., 2022) derived from 10x Genomics dextramer experiments contains 13,470 TCR-peptide observations across 4 peptides (ELAGIGILTV, GILGFVFTL, GLCTLVAML, IVTDFSVIK). Its unique strength is a dual-negative design: *neg_control* entries are TCRs that were experimentally tested by dextramer sorting and confirmed as non-binders, while *swapped* entries are computationally generated random-swap negatives. This allows within-dataset comparison of negative types with matched positives.

### 2.2&emsp;Negative sampling strategies

We implemented four negative sampling strategies of increasing difficulty:

1. **Random swap.** For each positive TCR-peptide pair, sample a different peptide from the dataset and pair it with the original TCR. This is the standard approach in the field (Weber et al., 2021; Springer et al., 2021).

2. **Epitope balanced.** Same as random swap, but with uniform sampling across peptides to prevent dominant epitopes from contributing disproportionate negatives.

3. **Within cluster.** Assign peptides to biochemical clusters based on amino acid property similarity, then generate negatives by swapping peptides only within the same cluster. This produces harder negatives by ensuring the decoy peptide is physically similar to the true target.

4. **Shuffled CDR3.** Randomly permute the CDR3&beta; amino acid sequence of each positive TCR while preserving peptide identity. This destroys binding-relevant sequence information while preserving amino acid composition, producing a negative that is distributionally close to positives in composition space but lacks any genuine TCR sequence.

All strategies generated negatives at a 1:1 ratio with positives. Negatives were generated once per random seed and shared across cross-validation folds, matching standard practice in published TCR-pMHC benchmarks and ensuring that any fold-level leakage affects all strategies equally.

### 2.3&emsp;Feature representations

**Biophysical features (~23 dimensions).** BLOSUM62-derived amino acid property vectors (hydrophobicity, volume, charge, polarity) averaged across CDR3&beta; positions, supplemented with length, net charge, and grand average of hydropathy (GRAVY) for both CDR3&beta; and peptide sequences.

**Sequence features (~8,900 dimensions).** Frequency vectors of all observed 3-mer (tripeptide) subsequences in CDR3&beta; and peptide, concatenated. This high-dimensional representation captures local sequence motifs without alignment.

### 2.4&emsp;Models

**Logistic regression (LR).** L2-regularised logistic regression with StandardScaler preprocessing (scikit-learn defaults, max_iter=2000).

**Random forest (RF).** 200 trees, scikit-learn defaults, no preprocessing required.

**Multilayer perceptron (MLP).** Three hidden layers (256, 128, 64 units), ReLU activation, Adam optimiser, batch size 256, early stopping with patience 10 on a 10% validation fraction, maximum 300 iterations. StandardScaler preprocessing.

### 2.5&emsp;Evaluation protocol

**Standard cross-validation.** Five-fold stratified CV on the combined positive + negative dataset. This is the evaluation reported by most published TCR-pMHC models and permits both positive and negative examples from the same peptide to appear in train and test folds.

**Leave-one-peptide-out (LOPO).** For each of the *k* peptides in the dataset, train on all data from the remaining *k*-1 peptides and evaluate on the held-out peptide. This simulates the clinically relevant scenario of predicting TCR specificity for an entirely novel peptide. The mean AUC across held-out peptides is the LOPO score.

**Inflation metric.** For each condition (strategy &times; feature type &times; model), we define AUC inflation as the difference between standard CV AUC and LOPO AUC. Positive inflation indicates that standard CV overestimates generalisation performance.

### 2.6&emsp;SHAP analysis and V-gene ablation

**SHAP attribution.** Per-epitope random forest classifiers were trained on 3-mer CDR3&beta; features under random-swap negatives. TreeSHAP values were computed for each 3-mer feature, and mean absolute SHAP values were compared against mutual information rankings to assess whether the model's most important features correspond to statistically informative sequence motifs or distributional artifacts.

**V-gene ablation.** To test whether germline-encoded (V/J-gene-determined) terminal regions of CDR3&beta; drive performance under random-swap negatives, we evaluated models under five masking conditions: *full* CDR3&beta;, *loop only* (central hypervariable region with N/C-terminal anchors removed), *N-terminal only* (first 3 residues, typically CAS), *C-terminal only* (last 2 residues, typically YF/FF), and *masked* (terminal anchors replaced with wildcards).

### 2.7&emsp;Difficulty titration

To characterise the relationship between negative difficulty and model performance, we partitioned random-swap negatives into eight difficulty levels based on their biophysical Euclidean distance to the nearest positive example. Level 1 (trivial far) contains the most distant negatives; Level 8 (extreme) contains negatives closest to positives in biophysical feature space. Models were trained and evaluated separately at each difficulty level using 5-fold GroupKFold cross-validation (grouped by CDR3&beta; to prevent TCR identity leakage), with 5 random seeds per condition.

### 2.8&emsp;Experimental negative transfer

To test whether experimental negatives recover genuine predictive signal, we trained models on ITRAP data using either experimental (dextramer-sorted) or random-swap negatives and evaluated via LOPO, testing all models against the experimental negative ground truth. Each condition was run with 10 independent negative subsample seeds to assess robustness. Statistical significance of the experimental-vs-swapped difference was assessed by paired Wilcoxon signed-rank test across peptides.

### 2.9&emsp;Computational environment

All experiments were conducted on a consumer workstation running Windows 11, using Python 3.11, scikit-learn 1.3, and NumPy 1.24. No GPU was required. Total computation time for all experiments was approximately 8 hours.

---

## 3&emsp;Results

### 3.1&emsp;Random-swap negatives inflate cross-validated AUC

In the core benchmark on IMMREP22, standard 5-fold CV AUC was systematically higher than LOPO AUC across all negative strategies and feature types (Table 1; Fig. 1A--B). The magnitude of inflation depended on the interaction between strategy and model: under random-swap negatives, RF achieved a CV AUC of 0.804 on biophysical features but only 0.599 under LOPO (inflation = +0.205). The MLP showed even larger inflation: CV = 0.863 versus LOPO = 0.453 (inflation = +0.410) for biophysical features, and CV = 0.842 versus LOPO = 0.222 (inflation = +0.620) for sequence features.

Harder negative strategies progressively reduced inflation. Within-cluster negatives, which constrain decoy peptides to be biochemically similar to the true target, reduced RF biophysical CV AUC from 0.804 to 0.730 while LOPO remained at 0.599, compressing the inflation from +0.205 to +0.131. Shuffled-CDR3 negatives eliminated inflation entirely for biophysical features (CV = 0.179, LOPO = 0.599) but were trivially detectable by sequence models (CV = 1.000), confirming that they destroy sequence structure rather than testing binding discrimination.

**Table 1.** Core benchmark results on IMMREP22. AUC values are mean across 3 repeats. Inflation = CV AUC - LOPO AUC.

| Strategy | Features | Model | CV AUC | LOPO AUC | Inflation |
|:---|:---|:---|:---:|:---:|:---:|
| Random swap | Biophysical | LR | 0.723 | 0.479 | +0.244 |
| Random swap | Biophysical | RF | 0.804 | 0.599 | +0.205 |
| Random swap | Biophysical | MLP | 0.863 | 0.453 | +0.410 |
| Random swap | Sequence | RF | 0.822 | 0.605 | +0.217 |
| Random swap | Sequence | MLP | 0.842 | 0.222 | +0.620 |
| Within cluster | Biophysical | RF | 0.730 | 0.599 | +0.131 |
| Shuffled CDR3 | Biophysical | MLP | 0.426 | 0.499 | -0.074 |
| Shuffled CDR3 | Sequence | MLP | 1.000 | 1.000 | +0.000 |

Per-peptide LOPO analysis (Fig. 1C) revealed dramatic variation: under random-swap biophysical RF, LOPO AUC ranged from 0.11 (GPRLGVRAT) to 0.91 (LLWNGPMAV), an 8-fold range masked by the aggregate inflation metric. Five of 17 peptides fell below chance (AUC < 0.5), indicating that the model provides no useful prediction for nearly one-third of peptides even under the most favourable conditions.

### 3.2&emsp;Models learn distributional shortcuts, not binding logic

SHAP analysis on per-epitope random forests revealed that the most influential 3-mer features were dominated by germline-encoded CDR3&beta; anchor motifs (CAS, ASS) rather than hypervariable-region motifs expected to mediate binding specificity (Fig. 2C). The Spearman correlation between SHAP rank and mutual information rank across the top 50 3-mers was &rho; = 0.59 (p = 6.4 &times; 10^-6), indicating moderate but imperfect agreement---the model's feature importance partially, but not fully, reflects statistical signal in the data.

V-gene ablation experiments confirmed that germline-encoded terminal regions contribute to performance under random-swap negatives (Fig. 2B). Removing N- and C-terminal anchors (loop-only condition) reduced mean LOPO AUC only modestly (from 0.77 to 0.72), suggesting that the central hypervariable loop carries most binding-relevant information. However, the N-terminal motif alone (typically CAS, determined by TRBV gene usage) retained substantial predictive power (mean AUC = 0.67), indicating that V-gene-associated distributional features---which reflect repertoire composition rather than binding specificity---contribute meaningfully to apparent performance.

Biophysical distance analysis on ITRAP's dual-negative dataset provided direct evidence for the distributional shortcut (Fig. 2A). Experimental (dextramer-sorted) negatives were significantly closer to positives than random-swap negatives in biophysical feature space (Mann-Whitney p = 3.54 &times; 10^-30), confirming that random-swap negatives occupy a distributionally distinct region that models can exploit without learning binding-relevant features.

### 3.3&emsp;Harder negatives reduce but do not eliminate inflation; biophysical features show a crossover advantage

The difficulty titration experiment revealed a monotonic relationship between negative difficulty and both AUC and the biophysical-sequence gap (Fig. 3). As negatives became harder (closer to positives in feature space), AUC declined for both feature types, but sequence features declined faster than biophysical features.

For logistic regression, biophysical features outperformed sequence features at all difficulty levels, with the gap widening from +0.026 at Level 1 (trivial) to +0.111 at Level 6 (hard). For random forest, a crossover was observed: sequence features slightly outperformed biophysical at easy levels (Level 1 gap = +0.027 in favour of biophysical, Level 4 gap = -0.002 in favour of sequence), but biophysical features regained the advantage at hard levels (Level 7 gap = -0.039 in favour of sequence).

This crossover directly explains the published finding that physics-based methods outperform sequence-based methods on novel peptides (Riley et al., 2023) while underperforming on seen peptides. Sequence models benefit more from the distributional shortcuts present in easy negatives; as those shortcuts are removed by harder negatives, the intrinsic advantage of biophysical representations---which encode physically meaningful properties of the binding interface---emerges.

### 3.4&emsp;Inflation is architecture-independent

To test whether increased model capacity could overcome the distributional shortcuts, we evaluated a 3-layer MLP (256-128-64, ~50,000 parameters) under the identical 8-condition protocol used for LR and RF (Table 1; Fig. 5B). The MLP showed inflation equal to or greater than the simpler models: +0.410 for random-swap biophysical (versus +0.205 for RF) and +0.620 for random-swap sequence (versus +0.217 for RF). Under shuffled-CDR3 negatives, the MLP showed no inflation for biophysical features (-0.074) and perfect detection for sequence features (CV = LOPO = 1.000), matching the LR/RF pattern exactly.

These results demonstrate that the inflation problem resides in the training data, not the model architecture. A neural network with orders of magnitude more parameters than logistic regression exploits the same distributional shortcuts, because those shortcuts are properties of the negative sampling distribution, not of the model's capacity to learn complex functions.

### 3.5&emsp;Cross-benchmark validation

The inflation pattern replicated across all three independent benchmarks (Fig. 4).

**TChard.** On TChard's hard splits (peptide + CDR3&beta; held out), models trained with random-swap negatives achieved AUC = 0.49---indistinguishable from chance (Fig. 4A). In contrast, models trained with experimental negatives achieved AUC = 0.63 on biophysical features. The easy-versus-hard comparison was particularly stark: easy split + experimental negatives yielded AUC = 0.976, while hard split + random-swap negatives yielded AUC = 0.493, a gap of 0.48 that represents the full inflation pipeline applied to an independent dataset (Fig. 4B).

**ITRAP cross-evaluation.** Models trained on one negative type and tested on the other lost 0.31--0.59 AUC (Fig. 4C). For example, an RF trained on experimental negatives (biophysical) achieved AUC = 0.882 when tested against experimental negatives but dropped to 0.424 when tested against random-swap negatives---and vice versa. This bidirectional collapse demonstrates that models learn negative-type-specific distributional patterns rather than genuine binding discrimination. If models learned true binding logic, performance would transfer across negative types.

**IMMREP23.** The directional inflation pattern replicated on IMMREP23: random-swap biophysical inflation was +0.258 on IMMREP23 versus +0.225 on IMMREP22, confirming consistency across an independent dataset with 13 peptides and 4,563 positives (Fig. 4D).

### 3.6&emsp;Experimental negatives recover genuine predictive signal

The most important result of this study is constructive: training on experimentally derived negatives recovers predictive signal that random-swap-trained models entirely lack (Table 2; Fig. 5A).

Using ITRAP's dual-negative design, we trained RF models on either experimental (dextramer-sorted) or random-swap negatives and evaluated via LOPO, testing all models against experimental negative ground truth. Models trained on experimental negatives achieved substantially higher LOPO AUC than those trained on random-swap negatives, consistently across all three evaluable peptides:

**Table 2.** ITRAP LOPO: RF biophysical, tested against experimental negatives. Values are mean across 10 negative subsample seeds.

| Held-out Peptide | Train: Experimental | Train: Swapped | Difference |
|:---|:---:|:---:|:---:|
| ELAGIGILTV | 0.743 &pm; 0.026 | 0.246 &pm; 0.018 | +0.497 |
| GILGFVFTL | 0.568 &pm; 0.008 | 0.122 &pm; 0.006 | +0.446 |
| IVTDFSVIK | 0.819 &pm; 0.027 | 0.183 &pm; 0.024 | +0.636 |
| **Mean** | **0.710** | **0.184** | **+0.526** |

Random-swap-trained models scored below chance (AUC < 0.25) on all three peptides when evaluated against experimental negatives, indicating that they have learned to discriminate the *distributional properties of random-swap negatives* rather than any property of non-binding TCRs. Experimental-negative-trained models retained meaningful predictive signal (mean AUC = 0.710), demonstrating that genuine binding-relevant information exists in the data but is accessible only when the training negatives reflect biological reality.

The pattern was consistent across feature types and models: sequence-feature RF trained on experimental negatives achieved AUC = 0.747, 0.587, and 0.833 on the three peptides, compared with 0.138, 0.126, and 0.206 for swapped-trained models. Results were robust across 10 independent negative subsampling seeds, with standard deviations of 0.006--0.027.

GLCTLVAML was excluded from the experimental comparison because ITRAP contains no experimental negatives for this peptide (all negatives are computationally generated). Swapped-only results for GLCTLVAML yielded LOPO AUC of 0.13, consistent with the pattern observed for the other peptides.

---

## 4&emsp;Discussion

### 4.1&emsp;The shortcut mechanism

The central finding of this work is that random-swap negative sampling creates distributional shortcuts that are learnable, strategy-specific, and orthogonal to binding biology. The mechanism is straightforward: positive TCRs in a benchmark dataset are drawn from a limited number of donor repertoires and tend to share V-gene usage patterns, CDR3 length distributions, and amino acid composition profiles. Random-swap negatives, constructed by pairing these TCRs with unrelated peptides, retain the same repertoire-level distributional properties as positives but are scattered across different peptide contexts. A classifier can achieve high apparent accuracy simply by learning to distinguish TCRs that are "repertoire-typical" (and therefore likely to be training positives) from TCRs that appear in unusual peptide contexts (and therefore likely to be negatives), without encoding any information about the molecular interaction.

This explains several previously puzzling observations in the literature. The high AUC scores reported on seen peptides reflect, in part, the model's ability to recognise repertoire-specific distributional signatures that are shared across all peptides in the training set. The collapse on unseen peptides occurs because the new peptide's TCR repertoire differs from the training distribution, and the distributional shortcut no longer applies. The finding that physics-based features outperform sequence features on novel peptides (Riley et al., 2023) is explained by our crossover result (Section 3.3): biophysical representations encode physically meaningful binding properties that transfer across peptide contexts, while sequence representations encode distributional patterns that do not.

### 4.2&emsp;The physics-versus-sequence paradox resolved

The difficulty titration crossover (Fig. 3) provides, to our knowledge, the first mechanistic explanation for why physics-based methods outperform sequence-based methods on novel peptides. At easy difficulty levels (Level 1--3), sequence models outperform or match biophysical models because they can exploit CDR3 sequence motifs that correlate with repertoire membership---the distributional shortcut. As difficulty increases and negatives become more similar to positives in feature space, the shortcut weakens, and the intrinsic advantage of biophysical features---which encode genuine molecular properties of the binding interface---emerges.

This finding has implications for model development. It suggests that improving TCR-pMHC prediction requires not better sequence encoders (which will simply learn more subtle distributional shortcuts), but rather better negative examples that eliminate the distributional asymmetry between positive and negative TCRs. The crossover also provides a practical diagnostic: if a model shows declining performance as negative difficulty increases, the decline quantifies the model's dependence on distributional shortcuts.

### 4.3&emsp;Why deeper models do not fix a data problem

The MLP results (Section 3.4) refute the common intuition that increased model capacity might overcome the limitations of noisy training data. In fact, the MLP showed *more* inflation than LR or RF (+0.410 versus +0.205 for biophysical random-swap), consistent with the interpretation that greater capacity enables more efficient exploitation of distributional shortcuts. This result mirrors findings in computer vision, where overparameterised models are known to memorise dataset biases more efficiently than simpler models (Geirhos et al., 2020).

The practical implication is that the TCR-pMHC field cannot simply train larger or more complex models on existing benchmarks and expect improved generalisation. Transformer-based models, graph neural networks, and other architectures operating on the same random-swap negative distributions will suffer from the same inflation, regardless of their representational capacity. The path to improved generalisation runs through the data, not the model.

### 4.4&emsp;Experimental negatives as the path forward

The ITRAP experimental negative transfer results (Section 3.6, Table 2) demonstrate that genuine predictive signal exists in TCR-pMHC data and is recoverable when training negatives reflect biological reality. The 3.9-fold improvement in mean LOPO AUC (0.710 versus 0.184) represents the difference between a useless classifier and a modestly useful one---a transition that no amount of architectural innovation can achieve when the training data encodes distributional artifacts rather than binding biology.

We therefore recommend that the field adopt the following practices:

1. **Generate experimental negatives wherever possible.** Dextramer or tetramer sorting can identify TCRs that were physically present in the assay but did not bind---these are true non-binders for the tested peptide, not statistical non-binders inferred from random pairing.

2. **Evaluate using LOPO.** Standard cross-validation on multi-peptide datasets should be supplemented with LOPO evaluation, which simulates the clinically relevant scenario. Any model reporting only standard CV metrics on a multi-peptide benchmark should be interpreted with caution.

3. **Report negative strategy explicitly.** Published models should state how negatives were generated, at what ratio, and whether they were shared across folds or regenerated per fold. These details, currently absent from most publications, are necessary for interpreting reported performance.

4. **Use difficulty-aware negative sampling for training.** Even when experimental negatives are unavailable, within-cluster or other difficulty-aware strategies reduce inflation relative to random swap, providing a partial mitigation.

### 4.5&emsp;Limitations

Several limitations should be noted. First, the ITRAP LOPO evaluation includes only three peptides (ELAGIGILTV, GILGFVFTL, IVTDFSVIK) with experimental negatives; GLCTLVAML was excluded due to the absence of experimental negative controls. While the results are consistent across all three peptides and robust to negative subsampling, replication on a larger set of peptides with experimental negatives is needed.

Second, our analysis is restricted to HLA-A\*02:01-restricted peptides, which dominate existing benchmarks. Whether the inflation pattern and experimental negative advantage extend to other HLA alleles remains to be tested.

Third, biophysical features achieve a LOPO AUC of 0.710 with experimental negatives---a meaningful improvement over chance but far from clinically actionable. The experimental negative approach addresses the *accuracy of evaluation*, not the fundamental difficulty of the prediction problem. The modest absolute LOPO performance likely reflects the genuine complexity of TCR-pMHC recognition, which involves conformational dynamics, water-mediated contacts, and allosteric effects that simple biophysical features cannot capture.

Fourth, our models use CDR3&beta; sequence only, without CDR3&alpha;, full V/J-gene information, or structural data. More complete TCR representations may improve absolute performance, though our V-gene ablation results suggest that germline-gene features contribute to *inflation* rather than genuine binding prediction.

Fifth, the LOPO evaluation, while more rigorous than standard CV, still permits the same TCR to appear in both training and test sets if it is annotated as binding multiple peptides. GroupKFold by CDR3&beta; (used in our titration experiment) addresses this, but is not feasible for all analyses due to the small number of multi-reactive TCRs.

### 4.6&emsp;Implications for the field

Our findings have two immediate implications. For method developers, the message is that new models should be evaluated against experimental negatives and under LOPO, not against random-swap negatives under standard CV. Any model that shows high AUC only under the latter conditions has likely learned distributional shortcuts rather than binding logic, regardless of its architectural sophistication.

For the broader immunology community, the implication is that benchmark datasets need to prioritise experimental negative collection. Initiatives such as ITRAP and TChard that include experimentally validated non-binders provide a foundation, but the coverage remains sparse. Community-scale efforts to generate dextramer-sorted or tetramer-negative controls across diverse peptides and HLA alleles would provide the training data needed to build models that genuinely predict TCR-pMHC specificity, rather than models that exploit the statistical properties of how their training data was constructed.

The gap between reported benchmark performance and clinical utility in TCR-pMHC prediction is real, but it is partly a measurement artifact. Closing it requires not only better models, but better data---and, most immediately, better awareness of how the choice of negative sampling strategy shapes the metrics by which we judge progress.

---

## Data and Code Availability

All code, data processing scripts, and analysis notebooks are available at [repository URL]. The IMMREP22, TChard, and ITRAP datasets are publicly available from their respective sources.

---

## References

Bagaev, D. V., et al. (2020). VDJdb in 2019: database extension, new analysis infrastructure and a T-cell receptor motif compendium. *Nucleic Acids Research*, 48(D1), D1057--D1062.

Bradley, P. (2023). Structure-based prediction of T cell receptor:peptide-MHC interactions. *eLife*, 12, e82813.

Dens, C., et al. (2023). The pitfalls of negative data bias for the T-cell epitope specificity challenge. *Nature Machine Intelligence*, 5, 1060--1062.

Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665--673.

Glanville, J., et al. (2017). Identifying specificity groups in the T cell receptor repertoire. *Nature*, 547, 94--98.

Grazioli, F., et al. (2022). On TCR binding predictors failing to generalize to unseen peptides. *Frontiers in Immunology*, 13, 1014256.

Lu, T., et al. (2021). Deep learning-based prediction of the T cell receptor-antigen binding specificity. *Nature Machine Intelligence*, 3, 864--875.

Montemurro, A., et al. (2021). NetTCR-2.0 enables accurate prediction of TCR-peptide binding by using paired TCR&alpha; and &beta; sequence data. *Communications Biology*, 4, 1060.

Montemurro, A., et al. (2022). IMMREP22: T cell receptor specificity prediction benchmark. *ImmunoInformatics*, 9, 100028.

Peng, Y., et al. (2024). Characterizing the interaction conundrum: the assessment of the IMMREP23 TCR-epitope specificity challenge. *bioRxiv*, 2024.06.18.599265.

Riley, T. P., et al. (2023). Structure-based prediction of TCR-pMHC binding: implications for therapeutic design. *Frontiers in Immunology*, 14, 1120481.

Springer, I., et al. (2021). Prediction of specific TCR-peptide binding from large dictionaries of TCR-peptide pairs. *Frontiers in Immunology*, 11, 568372.

Weber, A., et al. (2021). TITAN: T-cell receptor specificity prediction with bimodal attention networks. *Bioinformatics*, 37, i237--i244.

Wu, K., et al. (2024). TCR-BERT: learning the grammar of T-cell receptors for flexible antigen-binding analyses. *Nature Communications*, 15, 3112.
