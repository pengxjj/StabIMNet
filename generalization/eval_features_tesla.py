#!/usr/bin/env python3
"""
TESLA feature-only baseline + deep-score fusion (the "positive solution" C4).

Loads tesla_{608,310}_features.csv, then:
  1. univariate oriented ROC per feature (higher = more immunogenic)
  2. feature-only logistic / GBM with leave-one-out CV  (the "feature ceiling")
  3. fusion: v6 deep score + features (LOO logistic) — does the deep model add
     anything on top of the features? (v6 score read from *_foreignness_diag.csv)

Metrics: ROC / PR / mean_PPVn (matching the paper's PPVn convention).

Usage:
    python eval_features_tesla.py            # both splits
    python eval_features_tesla.py --split 608
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

# feature -> (column, deployable?)
FEATURES = [
    ("binding_score",   True),   # -log10(IC50), higher = stronger
    ("tumor_abundance", True),
    ("binding_stability", True),
    ("frac_hydrophobic", True),
    ("agretopicity",    True),
    ("foreignness",     True),
    ("mutation_position", True),
    ("pep_len",         True),
    ("number_predicting", False),  # consortium meta-feature, NOT clinical
]


def load(split):
    path = os.path.join(TESLA_DIR, f"tesla_{split}_features.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def tofloat(v):
    if v == "" or v is None:
        return np.nan
    return float(v)


def build_matrix(rows, deployable_only=False):
    y = np.array([int(r["label"]) for r in rows], dtype=float)
    Xcols = []
    names = []
    for name, deployable in FEATURES:
        if deployable_only and not deployable:
            continue
        if name == "binding_score":
            aff = np.array([tofloat(r["binding_affinity"]) for r in rows])
            # IC50 nM -> -log10; higher = stronger binding. Guard <=0.
            aff = np.where(np.isfinite(aff) & (aff > 0), aff, np.nan)
            col = -np.log10(aff)
        else:
            col = np.array([tofloat(r[name]) for r in rows])
        Xcols.append(col)
        names.append(name)
    X = np.stack(Xcols, axis=1)
    # drop columns that are entirely NaN in this split (e.g. frac_hydrophobic /
    # number_predicting are absent from mmc7=310)
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    return X, y, names


def loo_cv(X, y, clf, scaler=True):
    """Leave-one-out OOF predictions; returns nan array if class-imbalanced."""
    if len(np.unique(y)) < 2:
        return None
    Xs = StandardScaler().fit_transform(X) if scaler else X
    oof = np.zeros(len(y))
    loo = LeaveOneOut()
    for tr, te in loo.split(Xs):
        clf.fit(Xs[tr], y[tr])
        oof[te] = clf.predict_proba(Xs[te])[0, 1]
    return oof


def mean_ppvn(y, score, topk=None):
    order = np.argsort(score)[::-1]
    ys = y[order]
    cum = np.cumsum(ys)
    ppvn = cum / np.arange(1, len(ys) + 1)
    npos = int(ys.sum())
    if npos == 0:
        return np.nan
    if topk is None:
        return np.mean(ppvn[:npos])
    return np.mean(ppvn[:min(npos, topk)])


def report(name, y, score):
    if score is None or len(np.unique(y)) < 2:
        print(f"  {name:38s}: insufficient")
        return None
    return (roc_auc_score(y, score), average_precision_score(y, score),
            mean_ppvn(y, score), mean_ppvn(y, score, topk=30))


def run(split):
    rows = load(split)
    # dedup identical (hla, peptide) pairs — mirror eval_v6_tesla.py (608 -> 605)
    seen, deduped = {}, []
    for r in rows:
        k = (r["hla"], r["peptide"])
        if k in seen:
            continue
        seen[k] = r
        deduped.append(r)
    rows = deduped
    X, y, names = build_matrix(rows)

    print(f"\n{'=' * 74}")
    print(f"  TESLA {split} — n={len(y)} pos={int(y.sum())}")
    print(f"{'=' * 74}")

    # 1. univariate oriented ROC
    print("  [univariate] oriented ROC (higher score = more immunogenic):")
    for j, name in enumerate(names):
        v = X[:, j]
        m = np.isfinite(v)
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            continue
        roc = roc_auc_score(y[m], v[m])
        oroc = max(roc, 1 - roc)
        sign = "+" if roc >= 0.5 else "-"
        print(f"    {name:18s}: n={m.sum():4d}  ROC={oroc:.3f} ({sign} orient)")

    # 2. feature-only multivariate (all features)
    fin = np.all(np.isfinite(X), axis=1)
    Xf, yf = X[fin], y[fin]
    print(f"\n  [feature-only] finite-intersection n={fin.sum()} pos={int(yf.sum())}")
    for label, clf in [("logistic", LogisticRegression(max_iter=2000)),
                       ("GBM", GradientBoostingClassifier(n_estimators=200, max_depth=2, random_state=0))]:
        oof = loo_cv(Xf, yf, clf)
        r = report(f"feature-only {label} (all)", yf, oof)
        if r:
            print(f"    ROC={r[0]:.3f}  PR={r[1]:.3f}  PPVn={r[2]:.3f}  PPVn@30={r[3]:.3f}")

    # 3. deployable-only (drop number_predicting)
    Xd, yd, nd = build_matrix(rows, deployable_only=True)
    fin = np.all(np.isfinite(Xd), axis=1)
    Xd, yd = Xd[fin], yd[fin]
    oof = loo_cv(Xd, yd, LogisticRegression(max_iter=2000))
    r = report("feature-only logistic (deployable)", yd, oof)
    if r:
        print(f"    ROC={r[0]:.3f}  PR={r[1]:.3f}  PPVn={r[2]:.3f}  PPVn@30={r[3]:.3f}")
    oof = loo_cv(Xd, yd, GradientBoostingClassifier(n_estimators=200, max_depth=2, random_state=0))
    r = report("feature-only GBM (deployable)", yd, oof)
    if r:
        print(f"    ROC={r[0]:.3f}  PR={r[1]:.3f}  PPVn={r[2]:.3f}  PPVn@30={r[3]:.3f}")

    # 4. fusion with deep-model scores (v6 + v2)
    deep_srcs = [
        ("v6", f"tesla_{split}_foreignness_diag.csv", "v6_pred"),
        ("v2", f"tesla_{split}_v2_diag.csv", "v2_pred"),
    ]
    for tag, diag_name, col in deep_srcs:
        diag = os.path.join(TESLA_DIR, diag_name)
        if not os.path.exists(diag):
            continue
        d = {}
        with open(diag) as f:
            for r in csv.DictReader(f):
                d[(r["hla"], r["peptide"])] = float(r[col])
        score = np.array([d.get((r["hla"], r["peptide"]), np.nan) for r in rows])
        Xf = np.column_stack([X, score])
        fin = np.all(np.isfinite(Xf), axis=1)
        Xf = Xf[fin]
        yf = y[fin]
        print(f"\n  [fusion] {tag}-score + features, finite n={fin.sum()} pos={int(yf.sum())}")
        oof = loo_cv(Xf, yf, LogisticRegression(max_iter=2000))
        r = report(f"{tag} + features (LOO)", yf, oof)
        if r:
            print(f"    ROC={r[0]:.3f}  PR={r[1]:.3f}  PPVn={r[2]:.3f}  PPVn@30={r[3]:.3f}")
        # deep score alone on same finite subset (apples-to-apples)
        m = np.isfinite(score)
        r = report(f"{tag} deep score alone", y[m], score[m])
        if r:
            print(f"    ROC={r[0]:.3f}  PR={r[1]:.3f}  PPVn={r[2]:.3f}  PPVn@30={r[3]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["608", "310"])
    args = ap.parse_args()
    for s in args.split:
        run(s)


if __name__ == "__main__":
    main()
