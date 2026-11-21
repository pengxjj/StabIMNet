#!/usr/bin/env python3
"""
De-novo (deployable) immunogenicity feature computation for TESLA.

Recomputes features from RAW inputs only — peptide + MHC allele sequence, plus
routine RNA-seq / WES measurements — independent of the TESLA consortium's
precomputed feature tables (which is where tesla_{split}_features.csv came from).

Tier 1 — sequence-only (recomputable from peptide + HLA sequence):
    binding_score    = -log10(mhcflurry pan-allelic binding affinity, nM)
    frac_hydrophobic = fraction of residues with Kyte-Doolittle score > 0
    net_charge       = (#R + #K + #H) - (#D + #E)  (peptide net charge)
    pep_len          = len(peptide)
Tier 2 — routine measurements (RNA-seq / WES, carried from the raw measurement,
         NOT the consortium's predictor columns):
    tumor_abundance  = RNA-seq TPM
    mutation_position= WES mutation position

Outputs tesla_{split}_denovo_features.csv and prints:
  (a) validation: correlation mhcflurry affinity vs TESLA NetMHCpan affinity
  (b) LOO-CV logistic / GBM on Tier 1 vs Tier 1+2 features.

Usage:
    python compute_denovo_features.py --gpu -1
"""
import os
import csv
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.realpath(__file__))
TESLA_DIR = os.path.join(HERE, "data", "TESLA")

# Kyte & Doolittle (1982) hydropathy index
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
      'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
      'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
      'Y': -1.3, 'V': 4.2}


def frac_hydrophobic(pep):
    return np.mean([1.0 if KD.get(a, 0.0) > 0 else 0.0 for a in pep])


def net_charge(pep):
    return sum(1 for a in pep if a in "RKH") - sum(1 for a in pep if a in "DE")


def load_tesla(split):
    rows = []
    with open(os.path.join(TESLA_DIR, f"tesla_{split}.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({"hla": r["hla"].strip(), "peptide": r["peptide"].strip(),
                         "label": int(r["label"])})
    # dedup (hla, peptide) — mirror eval_v6_tesla.py (608 -> 605)
    seen, deduped = {}, []
    for r in rows:
        k = (r["hla"], r["peptide"])
        if k in seen:
            continue
        seen[k] = r
        deduped.append(r)
    return deduped


def mhcflurry_affinities(peptides, alleles):
    from mhcflurry import Class1AffinityPredictor
    pred = Class1AffinityPredictor.load()
    # returns numpy array of IC50 (nM), one per (peptide, allele); NaN for
    # lengths outside 8-11 supported by the pan-allelic model.
    return pred.predict(peptides, alleles=alleles).astype(float)


def tofloat(v):
    if v == "" or v is None:
        return np.nan
    return float(v)


def load_netmhcpan_affinity(split):
    """TESLA's precomputed NetMHCpan affinity (for validation correlation)."""
    path = os.path.join(TESLA_DIR, f"tesla_{split}_features.csv")
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["hla"], r["peptide"])] = tofloat(r["binding_affinity"])
    return d


def load_routine(split):
    """tumor_abundance (TPM) + mutation_position from the raw-measurement columns."""
    path = os.path.join(TESLA_DIR, f"tesla_{split}_features.csv")
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["hla"], r["peptide"])] = (
                tofloat(r["tumor_abundance"]), tofloat(r["mutation_position"]))
    return d


def loo_cv(X, y, clf):
    if len(np.unique(y)) < 2:
        return None
    Xs = StandardScaler().fit_transform(X)
    oof = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(Xs):
        clf.fit(Xs[tr], y[tr])
        oof[te] = clf.predict_proba(Xs[te])[0, 1]
    return oof


def report(name, y, score):
    if score is None or len(np.unique(y)) < 2:
        print(f"  {name:40s}: insufficient")
        return
    print(f"  {name:40s}: n={len(y)} pos={int(y.sum())} "
          f"ROC={roc_auc_score(y, score):.3f} "
          f"PR={average_precision_score(y, score):.3f}")


def run(split, device):
    rows = load_tesla(split)
    peptides = [r["peptide"] for r in rows]
    alleles = [r["hla"] for r in rows]
    y = np.array([r["label"] for r in rows], dtype=float)

    print(f"\n  predicting mhcflurry affinity for {len(rows)} (peptide, allele) pairs ...")
    aff = mhcflurry_affinities(peptides, alleles)

    # build de-novo feature matrix
    feats = {
        "binding_score": -np.log10(np.where(np.isfinite(aff) & (aff > 0), aff, np.nan)),
        "frac_hydrophobic": np.array([frac_hydrophobic(p) for p in peptides]),
        "net_charge": np.array([net_charge(p) for p in peptides], dtype=float),
        "pep_len": np.array([len(p) for p in peptides], dtype=float),
    }
    routine = load_routine(split)
    feats["tumor_abundance"] = np.array([routine[(r["hla"], r["peptide"])][0] for r in rows])
    feats["mutation_position"] = np.array([routine[(r["hla"], r["peptide"])][1] for r in rows])

    tier1 = ["binding_score", "frac_hydrophobic", "net_charge", "pep_len"]
    tier2 = tier1 + ["tumor_abundance", "mutation_position"]

    # save CSV
    out = os.path.join(TESLA_DIR, f"tesla_{split}_denovo_features.csv")
    with open(out, "w") as f:
        f.write("hla,peptide,label,binding_score,frac_hydrophobic,net_charge,pep_len,"
                "tumor_abundance,mutation_position\n")
        for r, a in zip(rows, aff):
            bs = -np.log10(a) if np.isfinite(a) and a > 0 else ""
            f.write(f"{r['hla']},{r['peptide']},{r['label']},"
                    f"{bs:.6f},{frac_hydrophobic(r['peptide']):.6f},{net_charge(r['peptide']):.1f},"
                    f"{len(r['peptide'])},{routine[(r['hla'], r['peptide'])][0]:.6f},"
                    f"{routine[(r['hla'], r['peptide'])][1]:.6f}\n")
    print(f"  saved -> {out}")

    # (a) validation: mhcflurry affinity vs TESLA NetMHCpan affinity
    nm = load_netmhcpan_affinity(split)
    tesla_aff = np.array([nm[(r["hla"], r["peptide"])] for r in rows])
    m = np.isfinite(aff) & np.isfinite(tesla_aff) & (aff > 0) & (tesla_aff > 0)
    if m.sum() > 10:
        c = np.corrcoef(-np.log10(aff[m]), -np.log10(tesla_aff[m]))[0, 1]
        print(f"\n  [validation] mhcflurry vs TESLA NetMHCpan -log10(IC50): "
              f"n={m.sum()} Pearson r={c:.3f}")

    def X_of(names):
        cols = [feats[n] for n in names]
        X = np.stack(cols, axis=1)
        keep = ~np.all(np.isnan(X), axis=0)
        return X[:, keep], [n for n, k in zip(names, keep) if k]

    print(f"\n{'=' * 74}")
    print(f"  TESLA {split} de-novo features — n={len(y)} pos={int(y.sum())}")
    print(f"{'=' * 74}")

    for tier_name, names in [("Tier 1 sequence-only", tier1),
                             ("Tier 1+2 (+routine)", tier2)]:
        X, kept = X_of(names)
        fin = np.all(np.isfinite(X), axis=1)
        Xf, yf = X[fin], y[fin]
        print(f"\n  [{tier_name}] features={kept} finite n={fin.sum()} pos={int(yf.sum())}")
        for clf_name, clf in [("logistic", LogisticRegression(max_iter=2000)),
                              ("GBM", GradientBoostingClassifier(n_estimators=200, max_depth=2, random_state=0))]:
            report(f"{tier_name} {clf_name}", yf, loo_cv(Xf, yf, clf))

    # univariate oriented ROC
    print("\n  [univariate] oriented ROC:")
    for n in tier2:
        v = feats[n]
        mm = np.isfinite(v)
        if mm.sum() < 5 or len(np.unique(y[mm])) < 2:
            continue
        roc = roc_auc_score(y[mm], v[mm])
        oroc = max(roc, 1 - roc)
        sign = "+" if roc >= 0.5 else "-"
        print(f"    {n:18s}: n={mm.sum():4d}  ROC={oroc:.3f} ({sign})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["608", "310"])
    ap.add_argument("--gpu", type=int, default=-1, help="unused; kept for parity")
    args = ap.parse_args()
    for s in args.split:
        run(s, args.gpu)


if __name__ == "__main__":
    main()
