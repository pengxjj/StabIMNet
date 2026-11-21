#!/usr/bin/env python3
"""Evaluate the trained TCR-pMHC binding model on the held-out test fold.

Loads data/tcr_pmhc/best_model.pt + data/tcr_pmhc/test.csv, reports:
  - test ROC-AUC and PR-AUC (auc(precision_recall_curve) + average_precision_score)
  - per-allele ROC (only for alleles with >=3 pos AND >=3 neg in test)

Run:
    python eval_tcr_pmhc.py [--data data/tcr_pmhc] [--device cuda:0]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from tcr_model import TCRpMHCBindingModel
from train_tcr_pmhc import TCRDataset, collate

HERE = os.path.dirname(os.path.realpath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tcr_pmhc"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(args.data, "best_model.pt"), map_location=device,
                      weights_only=False)
    vocab = ckpt["vocab"]

    model = TCRpMHCBindingModel(
        d_model=64, n_heads=2, dropout=0.1, max_len=vocab["max_len"],
        n_v_alpha=len(vocab["v.alpha"]), n_j_alpha=len(vocab["j.alpha"]),
        n_v_beta=len(vocab["v.beta"]), n_j_beta=len(vocab["j.beta"]),
        n_allele=len(vocab["allele"]), ablate=ckpt.get("ablate"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = TCRDataset(os.path.join(args.data, "test.csv"), vocab)
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate,
                        num_workers=4)

    y_true, y_score, alleles = [], [], []
    with torch.no_grad():
        for batch in loader:
            alleles += batch["allele"].tolist()
            for k in ("cdr3a", "cdr3b", "peptide", "va", "ja", "vb", "jb", "allele"):
                batch[k] = batch[k].to(device)
            logits = model(batch).squeeze(-1)
            y_true.append(batch["label"].cpu().numpy())
            y_score.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    alleles = np.array(alleles)
    allele_names = np.array(vocab["allele"])[alleles]

    roc = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)
    ap = average_precision_score(y_true, y_score)

    print("=" * 60)
    print("  TCR-pMHC binding model — test fold")
    print("=" * 60)
    print(f"  n={len(y_true)}  pos={int(y_true.sum())} "
          f"({y_true.mean()*100:.1f}%)")
    print(f"  ROC-AUC            = {roc:.4f}")
    print(f"  PR-AUC (auc)       = {pr_auc:.4f}")
    print(f"  avg-precision      = {ap:.4f}")
    print(f"  (val ROC at ckpt)  = {ckpt.get('val_roc', float('nan')):.4f}")

    # per-allele ROC
    print("\n  per-allele ROC (min 3 pos AND 3 neg):")
    print(f"  {'allele':16s} {'n':>6s} {'pos':>5s} {'ROC':>7s}")
    print("  " + "-" * 38)
    rows = []
    for a in sorted(set(allele_names)):
        m = allele_names == a
        ya = y_true[m]
        sa = y_score[m]
        n_pos = int(ya.sum())
        n_neg = int(len(ya) - n_pos)
        if n_pos >= 3 and n_neg >= 3:
            r = roc_auc_score(ya, sa)
            r = max(r, 1 - r)
            rows.append((a, len(ya), n_pos, r))
            print(f"  {a:16s} {len(ya):6d} {n_pos:5d} {r:7.3f}")
    if not rows:
        print("  (none — test fold too small per allele)")


if __name__ == "__main__":
    main()
