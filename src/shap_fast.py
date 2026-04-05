"""Fast SHAP attribution — proves k-mer fingerprinting hypothesis."""
import sys, warnings, itertools
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

import shap, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

BASE_DIR   = Path(__file__).resolve().parent.parent
TRAIN_DIR  = BASE_DIR / "data/IMMREP_2022_TCRSpecificity/IMMREP_2022_TCRSpecificity-main/training_data"
TEST_DIR   = BASE_DIR / "data/IMMREP_2022_TCRSpecificity/IMMREP_2022_TCRSpecificity-main/true_set"
FIG_DIR    = BASE_DIR / "results/figures"
RESULTS_DIR = BASE_DIR / "results"

AA    = list("ACDEFGHIKLMNPQRSTVWY")
K     = 3
KMERS = ["".join(p) for p in itertools.product(AA, repeat=K)]
KMER_INDEX = {km: i for i, km in enumerate(KMERS)}
N_FEAT = len(KMERS)
print(f"3-mer features: {N_FEAT}", flush=True)

def kmer_freq(seqs):
    X = np.zeros((len(seqs), N_FEAT), dtype=np.float32)
    for i, seq in enumerate(seqs):
        n = len(seq) - K + 1
        if n <= 0: continue
        for j in range(n):
            idx = KMER_INDEX.get(seq[j:j+K])
            if idx is not None: X[i, idx] += 1.0
        X[i] /= n
    return X

def load_ep(ep):
    def _l(p):
        df = pd.read_csv(p, sep="\t", usecols=["Label","TRB_CDR3"]).dropna(subset=["TRB_CDR3"])
        df["y"] = (df["Label"].astype(int) == 1).astype(int)
        return df
    return _l(TRAIN_DIR / f"{ep}.txt"), _l(TEST_DIR / f"{ep}.txt")

EPITOPES = sorted([p.stem for p in TRAIN_DIR.glob("*.txt")])
print(f"Epitopes: {EPITOPES}", flush=True)

rf_models, X_trains, y_trains, auc_scores = {}, {}, {}, {}

for ep in EPITOPES:
    try:
        tr, te = load_ep(ep)
        if tr["y"].nunique() < 2 or te["y"].nunique() < 2:
            print(f"  SKIP {ep}: single class", flush=True); continue
        X_tr = kmer_freq(tr["TRB_CDR3"].tolist())
        y_tr = tr["y"].values
        X_te = kmer_freq(te["TRB_CDR3"].tolist())
        y_te = te["y"].values
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=2, random_state=42)
        rf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
        rf_models[ep] = rf; X_trains[ep] = X_tr; y_trains[ep] = y_tr; auc_scores[ep] = auc
        print(f"  {ep:20s}  n={len(y_tr):5d}  AUC={auc:.4f}", flush=True)
    except Exception as e:
        print(f"  FAIL {ep}: {e}", flush=True)

print(f"\nMean AUC: {np.mean(list(auc_scores.values())):.4f}", flush=True)
print("Computing SHAP ...", flush=True)

shap_sum = np.zeros(N_FEAT)
mi_sum   = np.zeros(N_FEAT)
n_ep     = 0
per_ep_top = {}

rng = np.random.default_rng(42)
for ep in EPITOPES:
    if ep not in rf_models: continue
    rf, X_tr, y_tr = rf_models[ep], X_trains[ep], y_trains[ep]
    n_bg = min(100, len(X_tr))
    bg   = X_tr[rng.choice(len(X_tr), n_bg, replace=False)]
    expl = shap.TreeExplainer(rf, data=bg, model_output="probability",
                               feature_perturbation="interventional")
    raw_sv = expl.shap_values(bg)
    sv = np.array(raw_sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    elif isinstance(raw_sv, list):
        sv = raw_sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    shap_sum += mean_abs
    mi = mutual_info_classif(X_tr, y_tr, discrete_features=False, random_state=42)
    mi_sum += mi
    top10 = [(KMERS[i], mean_abs[i]) for i in np.argsort(mean_abs)[::-1][:10]]
    per_ep_top[ep] = top10
    n_ep += 1
    print(f"  SHAP done: {ep}  top3={[k for k,_ in top10[:3]]}", flush=True)

mean_shap = shap_sum / n_ep
mean_mi   = mi_sum   / n_ep

shap_rank = stats.rankdata(-mean_shap)
mi_rank   = stats.rankdata(-mean_mi)
rho, pval = stats.spearmanr(shap_rank, mi_rank)

print("\n" + "="*60, flush=True)
print("FINGERPRINTING HYPOTHESIS", flush=True)
print(f"Spearman rho = {rho:.4f}  p = {pval:.2e}", flush=True)
if rho > 0.7:
    verdict = "CONFIRMED (rho > 0.7)"
elif rho < 0.3:
    verdict = "REJECTED (rho < 0.3)"
else:
    verdict = f"INTERMEDIATE (rho={rho:.3f})"
print(f"Verdict: {verdict}", flush=True)
print("="*60, flush=True)

# Top 20
top20 = np.argsort(mean_shap)[::-1][:20]
print("\nTop 20 3-mers by mean |SHAP|:", flush=True)
rows = []
for rank, idx in enumerate(top20, 1):
    ep_lst = [ep for ep, lst in per_ep_top.items() if any(k==KMERS[idx] for k,_ in lst)]
    print(f"  {rank:2d}  {KMERS[idx]}  shap={mean_shap[idx]:.6f}  mi={mean_mi[idx]:.6f}  eps={ep_lst}", flush=True)
    rows.append({"rank":rank,"kmer":KMERS[idx],"mean_abs_shap":mean_shap[idx],"mean_mi":mean_mi[idx]})

# Per-epitope
for label, eps in [("BEST", ["LLWNGPMAV","GLCTLVAML","NYNYLYRLF"]),
                   ("WORST", ["GPRLGVRAT","TPRVTGGGAM","RAQAPPPSW"])]:
    print(f"\n--- {label} PEPTIDES ---", flush=True)
    for ep in eps:
        if ep not in per_ep_top: continue
        print(f"  {ep} AUC={auc_scores.get(ep,float('nan')):.4f}  top3={[k for k,_ in per_ep_top[ep][:3]]}", flush=True)

# Save results
df_full = pd.DataFrame({
    "kmer": KMERS, "mean_abs_shap": mean_shap, "mean_mi": mean_mi,
    "shap_rank": shap_rank, "mi_rank": mi_rank,
}).sort_values("mean_abs_shap", ascending=False)
df_full.to_csv(RESULTS_DIR / "shap_attribution.csv", index=False)
print(f"\nSaved shap_attribution.csv", flush=True)

# Scatter plot
fig, ax = plt.subplots(figsize=(7,6))
idx_s = rng.choice(N_FEAT, min(2000, N_FEAT), replace=False)
ax.scatter(mi_rank[idx_s], shap_rank[idx_s], alpha=0.3, s=8, color="steelblue")
for idx in top20[:5]:
    ax.annotate(KMERS[idx], xy=(mi_rank[idx], shap_rank[idx]), fontsize=8, color="crimson")
ax.set_xlabel("MI rank"); ax.set_ylabel("SHAP rank")
ax.set_title(f"CDR3b 3-mer: MI vs SHAP rank  (rho={rho:.3f}, p={pval:.1e})")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig12_shap_mi_correlation.png", dpi=150)
plt.close(fig)
print("Saved fig12_shap_mi_correlation.png", flush=True)

# Bar chart top 20
fig, ax = plt.subplots(figsize=(10,5))
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=mi_rank[top20].min(), vmax=mi_rank[top20].max())
cmap = plt.cm.RdYlGn_r
colors = [cmap(norm(mi_rank[i])) for i in top20]
ax.bar(range(20), mean_shap[top20], color=colors)
ax.set_xticks(range(20)); ax.set_xticklabels([KMERS[i] for i in top20], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Mean |SHAP|")
ax.set_title("Top-20 CDR3b 3-mers by SHAP  (green=high MI, red=low MI)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
plt.colorbar(sm, ax=ax, label="MI rank")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig12_top_kmers.png", dpi=150)
plt.close(fig)
print("Saved fig12_top_kmers.png", flush=True)

print("\nDONE", flush=True)
