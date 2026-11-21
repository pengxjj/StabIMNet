#!/usr/bin/env python3
"""
Seen-vs-OOD allele analysis of the de-novo benchmark.

Splits data/allele_balanced_benchmark.csv into
  - "seen"   alleles present in IEDB training (the 27 A+B alleles), and
  - "OOD"    alleles absent from IEDB (all HLA-C, plus non-IEDB A/B alleles),
then reports the de-novo binding/4-feature ROC in each stratum, per locus and
overall. This answers the honest generalization question: does the de-novo
binding prior degrade on alleles never seen in IEDB training?

Usage:
    python analyze_allele_seen_ood.py
"""
import os
import csv
import collections

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, "data")
BENCH = os.path.join(DATA, "allele_balanced_benchmark.csv")
IEDB = os.path.join(DATA, "ImmunoStruct_IEDB_data.csv")

FEATURES = ["binding_score", "frac_hydrophobic", "net_charge", "pep_len"]


def iedb_alleles():
    s = set()
    with open(IEDB) as f:
        for r in csv.DictReader(f):
            s.add(r["allele"].strip())
    return s


def load_bench():
    rows = []
    with open(BENCH) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def oriented_roc(y, s):
    roc = roc_auc_score(y, s)
    return max(roc, 1 - roc)


def binding_roc(rows):
    y = np.array([int(r["label"]) for r in rows], dtype=float)
    s = np.array([float(r["binding_score"]) if r["binding_score"] != "" else np.nan
                  for r in rows])
    m = np.isfinite(s)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return None, len(rows), int(y.sum())
    return oriented_roc(y[m], s[m]), len(rows), int(y.sum())


def logistic_roc(rows):
    if len(rows) == 0:
        return None, 0, 0
    X = np.array([[float(r[f]) if r[f] != "" else np.nan for f in FEATURES]
                  for r in rows], dtype=float)
    y = np.array([int(r["label"]) for r in rows], dtype=float)
    fin = np.all(np.isfinite(X), axis=1)
    Xf, yf = X[fin], y[fin]
    if len(np.unique(yf)) < 2 or len(yf) < 10:
        return None, len(yf), int(yf.sum())
    oof = np.zeros(len(yf))
    clf = LogisticRegression(max_iter=3000)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, yf):
        Xs = StandardScaler().fit(Xf[tr])
        clf.fit(Xs.transform(Xf[tr]), yf[tr])
        oof[te] = clf.predict_proba(Xs.transform(Xf[te]))[:, 1]
    return roc_auc_score(yf, oof), len(yf), int(yf.sum())


def main():
    seen = iedb_alleles()
    rows = load_bench()
    print(f"IEDB seen alleles: {len(seen)}")
    print(f"benchmark rows: {len(rows)}")

    # strata
    by_seen = {"seen": [], "ood": []}
    by_locus_seen = collections.defaultdict(lambda: {"seen": [], "ood": []})
    for r in rows:
        key = "seen" if r["allele"] in seen else "ood"
        by_seen[key].append(r)
        by_locus_seen[r["locus"]][key].append(r)

    def fmt(roc, n, pos):
        return (f"{roc:.3f}" if roc is not None else "  n/a"), n, pos

    print("\n" + "=" * 78)
    print("  De-novo binding_score ROC by allele stratum (seen vs OOD)")
    print("=" * 78)
    print(f"\n  {'stratum':8s} {'locus':6s} {'n':>6s} {'pos':>5s} {'bindROC':>8s}")
    print("  " + "-" * 42)
    for key in ("seen", "ood"):
        for loc in "ABC":
            rs = by_locus_seen[loc][key]
            roc, n, pos = binding_roc(rs)
            rstr, n, pos = fmt(roc, n, pos)
            print(f"  {key:8s} {loc:6s} {n:6d} {pos:5d} {rstr:>8s}")
        roc, n, pos = binding_roc(by_seen[key])
        rstr, n, pos = fmt(roc, n, pos)
        print(f"  {key:8s} {'ALL':6s} {n:6d} {pos:5d} {rstr:>8s}")

    print("\n" + "=" * 78)
    print("  De-novo 4-feature 5-fold-CV logistic ROC by stratum")
    print("=" * 78)
    print(f"\n  {'stratum':8s} {'locus':6s} {'n':>6s} {'pos':>5s} {'ROC':>7s}")
    print("  " + "-" * 40)
    for key in ("seen", "ood"):
        for loc in "ABC":
            rs = by_locus_seen[loc][key]
            roc, n, pos = logistic_roc(rs)
            rstr, n, pos = fmt(roc, n, pos)
            print(f"  {key:8s} {loc:6s} {n:6d} {pos:5d} {rstr:>7s}")
        roc, n, pos = logistic_roc(by_seen[key])
        rstr, n, pos = fmt(roc, n, pos)
        print(f"  {key:8s} {'ALL':6s} {n:6d} {pos:5d} {rstr:>7s}")

    # allele census: which benchmark alleles are OOD
    bench_alleles = sorted(set(r["allele"] for r in rows))
    ood_alleles = [a for a in bench_alleles if a not in seen]
    print(f"\n  benchmark alleles total: {len(bench_alleles)}")
    print(f"  seen alleles: {len(bench_alleles) - len(ood_alleles)}")
    print(f"  OOD alleles: {len(ood_alleles)}")
    ood_by_locus = collections.defaultdict(list)
    for a in ood_alleles:
        ood_by_locus[a.split('*')[0].replace('HLA-', '')].append(a)
    for loc in "ABC":
        print(f"    OOD HLA-{loc}: {len(ood_by_locus[loc])} -> {ood_by_locus[loc]}")


if __name__ == "__main__":
    main()
