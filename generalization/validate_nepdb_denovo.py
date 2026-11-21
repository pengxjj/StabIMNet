#!/usr/bin/env python3
"""
De-novo (deployable) feature predictor external validation on NEPdb.

NEPdb (Xia et al. 2021, Front. Immunol.) is a database of T-cell
experimentally-validated neoantigens (response 'P') and ineffective
neopeptides ('N'), with the mutant peptide sequence and restricting HLA
allele. It is the closest external analogue to the TESLA task: given a
neoantigen (mutated peptide) + its HLA allele, predict whether it elicits a
T-cell response.

This mirrors compute_denovo_features.py, but on the *external* NEPdb cohort:
  - binding_score     = -log10(mhcflurry pan-allelic binding IC50) of the
                        best-binding 8-11mer sub-peptide of the mutant peptide
                        (the standard neoantigen-presentation reduction)
  - frac_hydrophobic  = fraction of residues with Kyte-Doolittle score > 0
                        (of the best-binding epitope)
  - net_charge        = (#R+#K+#H) - (#D+#E)  (of the best-binding epitope)
  - pep_len           = length of the best-binding epitope

Only the mutant peptide sequence + HLA allele are used -> fully deployable
from WES (no consortium table, no TCR-seq, no label leak).

Outputs data/NEPdb/nepdb_denovo_features.csv and prints:
  (a) cohort statistics (P/N, dedup, ambiguous-pair counts)
  (b) univariate oriented ROC per feature
  (c) 5-fold CV logistic / GBM on Tier-1 sequence-only features.

Usage:
    python validate_nepdb_denovo.py
"""
import os
import csv
import re
import collections

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.realpath(__file__))
NEPDB_CSV = os.path.join(HERE, "data", "NEPdb", "nepdb_validated.csv")
OUT_CSV = os.path.join(HERE, "data", "NEPdb", "nepdb_denovo_features.csv")
OUT_LOG = os.path.join(HERE, "data", "NEPdb", "nepdb_denovo.log")

KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
      'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
      'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
      'Y': -1.3, 'V': 4.2}
STD_AA = set(KD)


def log(msg):
    print(msg, flush=True)


def frac_hydrophobic(pep):
    return float(np.mean([1.0 if KD.get(a, 0.0) > 0 else 0.0 for a in pep]))


def net_charge(pep):
    return float(sum(1 for a in pep if a in "RKH") - sum(1 for a in pep if a in "DE"))


def sub_peptides(seq, lo=8, hi=11):
    out = []
    for L in range(lo, hi + 1):
        for i in range(len(seq) - L + 1):
            out.append(seq[i:i + L])
    return out


def load_nepdb(path):
    """Return list of (mut_peptide, allele, response) for HLA-I entries."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["locus"] not in ("A", "B", "C"):
                continue
            mp = (r["mut_peptide"] or "").strip().upper()
            a = (r["alleleA"] or "").strip().upper()
            resp = (r["response"] or "").strip().upper()
            if resp not in ("P", "N"):
                continue
            rows.append((mp, a, resp))
    return rows


def clean(rows):
    """Filter to standard-AA peptides of length>=8, well-formed alleles; dedup."""
    kept = []
    for mp, a, resp in rows:
        if len(mp) < 8:
            continue
        if not all(c in STD_AA for c in mp):
            continue
        if not re.fullmatch(r"[ABC]\*\d{2}:\d{2}", a):
            continue
        kept.append((mp, a, resp))

    # dedup (peptide, allele); drop ambiguous pairs (mixed P/N)
    grouped = collections.defaultdict(set)
    for mp, a, resp in kept:
        grouped[(mp, a)].add(resp)
    clean_rows, n_amb = [], 0
    for (mp, a), resps in grouped.items():
        if len(resps) > 1:
            n_amb += 1
            continue
        clean_rows.append((mp, a, next(iter(resps))))
    return clean_rows, n_amb


def main():
    log("=" * 78)
    log("  NEPdb de-novo feature validation (mhcflurry binding + physchem)")
    log("=" * 78)

    raw = load_nepdb(NEPDB_CSV)
    log(f"  raw HLA-I rows: {len(raw)}")
    rows, n_amb = clean(raw)
    log(f"  after clean+dedup: {len(rows)} (ambiguous mixed-P/N pairs dropped: {n_amb})")

    y = np.array([1.0 if r == "P" else 0.0 for _, _, r in rows], dtype=float)
    peptides = [mp for mp, _, _ in rows]
    alleles = [a for _, a, _ in rows]
    n_pos = int(y.sum())
    log(f"  n={len(rows)}  pos={n_pos} ({n_pos / len(rows) * 100:.1f}%)  "
        f"baseline PR={n_pos / len(rows):.4f}")

    # ---- mhcflurry binding: best 8-11mer sub-peptide per (mut_peptide, allele) ----
    from mhcflurry import Class1AffinityPredictor
    log("  loading mhcflurry Class1AffinityPredictor ...")
    pred = Class1AffinityPredictor.load()

    # build (sub_peptide, hla-allele) pairs
    log("  building sliding-window sub-peptides ...")
    sub_to_antigen = []       # index -> antigen idx
    sub_seqs, sub_alleles = [], []
    for i, (mp, a) in enumerate(zip(peptides, alleles)):
        for s in sub_peptides(mp):
            sub_seqs.append(s)
            sub_alleles.append("HLA-" + a)
            sub_to_antigen.append(i)
    log(f"  total sub-peptide predictions: {len(sub_seqs)}")

    aff_all = np.full(len(sub_seqs), np.nan, dtype=float)
    CH = 20000
    for start in range(0, len(sub_seqs), CH):
        end = min(start + CH, len(sub_seqs))
        a = pred.predict(sub_seqs[start:end],
                         alleles=sub_alleles[start:end]).astype(float)
        aff_all[start:end] = a
        log(f"    predicted {end}/{len(sub_seqs)}")
    log("  mhcflurry prediction done")

    # reduce to best (strongest) binder per antigen
    best_ic50 = np.full(len(rows), np.nan)
    best_epi = [None] * len(rows)
    for j, i in enumerate(sub_to_antigen):
        a = aff_all[j]
        if not np.isfinite(a) or a <= 0:
            continue
        if np.isnan(best_ic50[i]) or a < best_ic50[i]:
            best_ic50[i] = a
            best_epi[i] = sub_seqs[j]

    bs = -np.log10(best_ic50)
    frac_h = np.array([frac_hydrophobic(e) if e else np.nan for e in best_epi])
    nc = np.array([net_charge(e) if e else np.nan for e in best_epi])
    plen = np.array([len(e) if e else np.nan for e in best_epi], dtype=float)

    feats = {"binding_score": bs, "frac_hydrophobic": frac_h,
             "net_charge": nc, "pep_len": plen}

    # save CSV
    with open(OUT_CSV, "w") as f:
        f.write("mut_peptide,allele,label,binding_score,frac_hydrophobic,net_charge,pep_len,best_epitope\n")
        for i, ((mp, a, r), ic50) in enumerate(zip(rows, best_ic50)):
            f.write(f"{mp},{a},{r},{bs[i]:.6f},{frac_h[i]:.6f},{nc[i]:.1f},"
                    f"{plen[i]:.0f},{best_epi[i] or ''}\n")
    log(f"  saved -> {OUT_CSV}")

    # ---- evaluation ----
    tier1 = ["binding_score", "frac_hydrophobic", "net_charge", "pep_len"]

    log("\n" + "=" * 78)
    log(f"  NEPdb de-novo features — n={len(rows)} pos={n_pos}")
    log("=" * 78)

    # univariate oriented ROC (finite only)
    log("\n  [univariate] oriented ROC:")
    for n in tier1:
        v = feats[n]
        m = np.isfinite(v)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        roc = roc_auc_score(y[m], v[m])
        log(f"    {n:18s}: n={m.sum():6d}  ROC={max(roc, 1 - roc):.4f} "
            f"({'orient +' if roc >= 0.5 else 'orient -'})")

    # multivariate CV
    X = np.stack([feats[n] for n in tier1], axis=1)
    keep_col = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep_col]
    kept_names = [n for n, k in zip(tier1, keep_col) if k]
    fin = np.all(np.isfinite(X), axis=1)
    Xf, yf = X[fin], y[fin]
    log(f"\n  [sequence-only] features={kept_names} finite n={fin.sum()} pos={int(yf.sum())}")

    def cv5(clf):
        oof = np.zeros(len(yf))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, yf):
            Xs = StandardScaler().fit(Xf[tr])
            clf.fit(Xs.transform(Xf[tr]), yf[tr])
            oof[te] = clf.predict_proba(Xs.transform(Xf[te]))[:, 1]
        return oof

    for name, clf in [("logistic", LogisticRegression(max_iter=3000)),
                      ("GBM", GradientBoostingClassifier(n_estimators=200, max_depth=2, random_state=0))]:
        oof = cv5(clf)
        log(f"  [5-fold CV {name}] ROC={roc_auc_score(yf, oof):.4f}  "
            f"PR={average_precision_score(yf, oof):.4f}  n={len(yf)} pos={int(yf.sum())}")

    log("\n  done.")


if __name__ == "__main__":
    main()
