#!/usr/bin/env python3
"""Run the trained TCR-pMHC binding model on the IMMREP26-style test set.

Maps the benchmark's column schema (cdr3_alpha_aa / cdr3_beta_aa / epitope_aa /
TRAV_IMGT / TRAJ_IMGT / TRBV_IMGT / TRBJ_IMGT / hla_long) onto the model's inputs
(char-encoded CDR3+peptide, gene vocab indices, HLA allele index), runs the model,
and writes a per-record activation score (sigmoid of the logit; higher = more
likely TCR-pMHC reactivity).

Gene names are matched to the model's VDJdb-derived vocab by exact match first,
then by IMGT-family normalisation (strip allele / -NN / _DVx / /DVx / S-NN
suffixes). Alleles are prefixed with "HLA-" to match the training vocab.

Run:
    PYTHONPATH=.. python predict_tcr_pmhc.py \
        --data data/tcr_pmhc \
        --test /path/to/test.csv \
        --out data/tcr_pmhc/test_predictions.csv [--device cuda:0]
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from tcr_model import CHAR2IDX, TCRpMHCBindingModel, UNK_IDX

HERE = os.path.dirname(os.path.realpath(__file__))


def gene_family(g):
    """Reduce an IMGT gene name to its family (TRAV14_DV4*01 -> TRAV14)."""
    g = str(g).strip()
    g = re.sub(r"\*.*$", "", g)          # strip allele *01
    g = re.sub(r"[-/]\d+.*$", "", g)     # strip -1 / -2 / -2-01 suffixes
    g = re.sub(r"[_/]DV\d+.*$", "", g)   # strip _DVx / /DVx
    g = re.sub(r"S\d+$", "", g)          # strip S1 (orphon) suffix
    return g


def build_gene_mapper(vocab_key, vocab):
    """Return a function mapping a raw gene name to a vocab index.

    Exact match first, then family match, else 0 (arbitrary but consistent
    fallback so out-of-vocab genes still run through the embedding).
    """
    names = vocab[vocab_key]
    idx = {g: i for i, g in enumerate(names)}
    family_idx = {gene_family(g): i for i, g in enumerate(names) if gene_family(g) != g}

    def map_gene(g):
        g = str(g).strip()
        if g in idx:
            return idx[g]
        if gene_family(g) in family_idx:
            return family_idx[gene_family(g)]
        return 0
    return map_gene


def encode_all(seqs, max_len):
    """Encode a list of AA strings to a [N, L] LongTensor of char indices."""
    out = []
    for s in seqs:
        ids = [CHAR2IDX.get(c, UNK_IDX) for c in str(s)[:max_len]]
        ids += [0] * (max_len - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tcr_pmhc"))
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path; defaults to <data>/best_model.pt")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "tcr_pmhc", "test_predictions.csv"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_path = args.ckpt or os.path.join(args.data, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    max_len = vocab["max_len"]

    model = TCRpMHCBindingModel(
        d_model=64, n_heads=2, dropout=0.1, max_len=max_len,
        n_v_alpha=len(vocab["v.alpha"]), n_j_alpha=len(vocab["j.alpha"]),
        n_v_beta=len(vocab["v.beta"]), n_j_beta=len(vocab["j.beta"]),
        n_allele=len(vocab["allele"]), ablate=ckpt.get("ablate"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[model] loaded val_roc={ckpt.get('val_roc', float('nan')):.4f} "
          f"ablate={ckpt.get('ablate')}")

    df = pd.read_csv(args.test)
    print(f"[test] {len(df)} records")

    map_va = build_gene_mapper("v.alpha", vocab)
    map_ja = build_gene_mapper("j.alpha", vocab)
    map_vb = build_gene_mapper("v.beta", vocab)
    map_jb = build_gene_mapper("j.beta", vocab)

    allele_names = vocab["allele"]
    allele_idx = {a: i for i, a in enumerate(allele_names)}
    missing_allele = 0
    allele_ids = []
    for a in df["hla_long"].astype(str).str.strip():
        key = "HLA-" + a
        if key in allele_idx:
            allele_ids.append(allele_idx[key])
        else:
            allele_ids.append(0)
            missing_allele += 1
    if missing_allele:
        print(f"[warn] {missing_allele} records with allele not in vocab -> index 0")

    cdr3a = encode_all(df["cdr3_alpha_aa"].tolist(), max_len)
    cdr3b = encode_all(df["cdr3_beta_aa"].tolist(), max_len)
    peptide = encode_all(df["epitope_aa"].tolist(), max_len)
    va = torch.tensor([map_va(g) for g in df["TRAV_IMGT"]], dtype=torch.long)
    ja = torch.tensor([map_ja(g) for g in df["TRAJ_IMGT"]], dtype=torch.long)
    vb = torch.tensor([map_vb(g) for g in df["TRBV_IMGT"]], dtype=torch.long)
    jb = torch.tensor([map_jb(g) for g in df["TRBJ_IMGT"]], dtype=torch.long)
    allele = torch.tensor(allele_ids, dtype=torch.long)

    n = len(df)
    scores = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            sl = slice(i, i + args.batch_size)
            batch = {
                "cdr3a": cdr3a[sl].to(device), "cdr3b": cdr3b[sl].to(device),
                "peptide": peptide[sl].to(device),
                "va": va[sl].to(device), "ja": ja[sl].to(device),
                "vb": vb[sl].to(device), "jb": jb[sl].to(device),
                "allele": allele[sl].to(device),
            }
            logits = model(batch).squeeze(-1)
            scores[sl] = torch.sigmoid(logits).cpu().numpy()

    out = df[["tcr_id"]].copy()
    out["prediction"] = scores
    out.to_csv(args.out, index=False)
    print(f"[save] {args.out}")

    # ---- optional evaluation against vdjdb_score (if present) ----
    if "vdjdb_score" in df.columns:
        y = df["vdjdb_score"].astype(float).values
        rho, _ = spearmanr(scores, y)
        print("\n[metric] vs vdjdb_score (0-3):")
        print(f"  Spearman rho        = {rho:.4f}")
        for thr in (1, 2, 3):
            if (y >= thr).sum() > 0 and (y < thr).sum() > 0:
                roc = roc_auc_score((y >= thr).astype(int), scores)
                roc = max(roc, 1 - roc)
                print(f"  ROC (score>={thr} pos) = {roc:.4f}  "
                      f"(pos={int((y >= thr).sum())}/{len(y)})")


if __name__ == "__main__":
    main()
