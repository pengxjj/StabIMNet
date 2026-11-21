#!/usr/bin/env python3
"""Multi-task: contrastive binding (InfoNCE on VDJdb positives) + functional
supervision (BCE on TCRvdb validating/non-validating, on the cosine score).

The functional term teaches the model to rank *validating* TCRs above
*non-validating* TCRs within an epitope, directly in the same cosine space used
for zero-shot scoring. Proper leave-one-epitope-out CV answers whether this
functional signal transfers to unseen epitopes (the held-out CIN/LLW/LTD case).

Run (LOEO):
    python train_tcr_pmhc_sup.py --tcrvdb <path> --ckpt <contrastive ckpt> --cv
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from finetune_tcr_pmhc import load_tcrvdb, build_inputs
from predict_tcr_pmhc import build_gene_mapper, encode_all
from tcr_model_contrastive import TCRpMHCContrastiveModel, multi_positive_infonce

HERE = os.path.dirname(os.path.realpath(__file__))


def build_vdjdb_positives(data_dir, vocab):
    """VDJdb train.csv positives -> (tensors, pmhc_id)."""
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df = df[df["label"] == 1].reset_index(drop=True)
    max_len = vocab["max_len"]
    gene_idx = {
        "va": {g: i for i, g in enumerate(vocab["v.alpha"])},
        "ja": {g: i for i, g in enumerate(vocab["j.alpha"])},
        "vb": {g: i for i, g in enumerate(vocab["v.beta"])},
        "jb": {g: i for i, g in enumerate(vocab["j.beta"])},
    }
    allele_idx = {a: i for i, a in enumerate(vocab["allele"])}
    from tcr_model import encode_seq
    tensors = {
        "cdr3a": torch.stack([encode_seq(s, max_len) for s in df["cdr3a"]]),
        "cdr3b": torch.stack([encode_seq(s, max_len) for s in df["cdr3b"]]),
        "peptide": torch.stack([encode_seq(s, max_len) for s in df["peptide"]]),
        "va": torch.tensor([gene_idx["va"][g] for g in df["va"]], dtype=torch.long),
        "ja": torch.tensor([gene_idx["ja"][g] for g in df["ja"]], dtype=torch.long),
        "vb": torch.tensor([gene_idx["vb"][g] for g in df["vb"]], dtype=torch.long),
        "jb": torch.tensor([gene_idx["jb"][g] for g in df["jb"]], dtype=torch.long),
        "allele": torch.tensor([allele_idx[a] for a in df["allele"]], dtype=torch.long),
    }
    pmhc_id = torch.from_numpy(pd.factorize((df["peptide"] + "|" + df["allele"]).astype(str))[0])
    return tensors, pmhc_id


def build_model(vocab, device, ckpt=None):
    m = TCRpMHCContrastiveModel(
        d_model=64, n_heads=2, dropout=0.1, max_len=vocab["max_len"],
        n_v_alpha=len(vocab["v.alpha"]), n_j_alpha=len(vocab["j.alpha"]),
        n_v_beta=len(vocab["v.beta"]), n_j_beta=len(vocab["j.beta"]),
        n_allele=len(vocab["allele"]),
    )
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        m.load_state_dict(sd)
    return m.to(device)


def functional_bce(model, batch, y, func_scale):
    """BCE on the cosine score, scaled into a logit."""
    z_tcr, z_pmhc = model(batch)
    score = (z_tcr * z_pmhc).sum(-1)
    return nn.functional.binary_cross_entropy_with_logits(score * func_scale, y)


def score_batch(model, tensors, idx, device, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            batch = {k: tensors[k][b].to(device) for k in tensors}
            out.append(model.score(batch).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tcr_pmhc"))
    ap.add_argument("--tcrvdb", required=True)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "data", "tcr_pmhc", "best_model_contrastive_s42.pt.epoch3.pt"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--func-scale", type=float, default=8.0)
    ap.add_argument("--lambda-func", type=float, default=1.0)
    ap.add_argument("--oversample", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "tcr_pmhc", "best_model_sup.pt"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.data, "vocab.json")) as f:
        vocab = json.load(f)

    vdj_t, vdj_pid = build_vdjdb_positives(args.data, vocab)
    print(f"[vdjdb] positives={len(vdj_pid)}")

    tdf = load_tcrvdb(args.tcrvdb)
    func_t, func_y, func_ep = build_inputs(tdf, vocab)
    print(f"[tcrvdb] n={len(func_y)} pos={int(func_y.sum())} neg={int(len(func_y)-func_y.sum())}")

    if args.cv:
        print("\n=== leave-one-epitope-out (functional supervision) ===")
        rocs = []
        for ep in sorted(set(func_ep)):
            val_idx = np.where(func_ep == ep)[0]
            tr_idx = np.where(func_ep != ep)[0]
            vlabels = func_y[val_idx].numpy()
            if vlabels.sum() == 0 or vlabels.sum() == len(vlabels):
                print(f"  {ep:12s} skip (no minority)")
                continue
            model = build_model(vocab, device, ckpt=args.ckpt)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            # oversample functional train indices
            tr_oversamp = np.tile(tr_idx, args.oversample)
            best_roc, best_state, patience = 0.0, None, 0
            n_vdj = len(vdj_pid)
            for e in range(args.epochs):
                model.train()
                # binding pass
                perm = np.random.permutation(n_vdj)
                for s in range(0, n_vdj, 512):
                    b = perm[s:s + 512]
                    batch = {k: vdj_t[k][b].to(device) for k in vdj_t}
                    pid = vdj_pid[b].to(device)
                    opt.zero_grad()
                    z_t, z_p = model(batch)
                    loss = multi_positive_infonce(z_t, z_p, model.logit_scale, pid)
                    loss.backward(); opt.step()
                # functional pass
                fp = np.random.permutation(len(tr_oversamp))
                for s in range(0, len(fp), 256):
                    b = tr_oversamp[fp[s:s + 256]]
                    batch = {k: func_t[k][b].to(device) for k in func_t}
                    y = func_y[b].to(device)
                    opt.zero_grad()
                    loss = args.lambda_func * functional_bce(model, batch, y, args.func_scale)
                    loss.backward(); opt.step()
                roc = roc_auc_score(vlabels, score_batch(model, func_t, val_idx, device))
                if roc > best_roc:
                    best_roc, patience = roc, 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 5:
                        break
            rocs.append(best_roc)
            print(f"  {ep:12s} n={len(val_idx):4d} pos={int(vlabels.sum()):4d} ROC={best_roc:.4f}")
        rocs = [r for r in rocs if not np.isnan(r)]
        print(f"\n  mean ROC = {np.mean(rocs):.4f} ± {np.std(rocs):.4f} (n={len(rocs)})")
        print("  (zero-shot contrastive baseline ~0.770)")
        return

    # ---- final: train on all functional data ----
    model = build_model(vocab, device, ckpt=args.ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_all = np.tile(np.arange(len(func_y)), args.oversample)
    n_vdj = len(vdj_pid)
    for e in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_vdj)
        for s in range(0, n_vdj, 512):
            b = perm[s:s + 512]
            batch = {k: vdj_t[k][b].to(device) for k in vdj_t}
            pid = vdj_pid[b].to(device)
            opt.zero_grad()
            z_t, z_p = model(batch)
            loss = multi_positive_infonce(z_t, z_p, model.logit_scale, pid)
            loss.backward(); opt.step()
        fp = np.random.permutation(len(tr_all))
        for s in range(0, len(fp), 256):
            b = tr_all[fp[s:s + 256]]
            batch = {k: func_t[k][b].to(device) for k in func_t}
            y = func_y[b].to(device)
            opt.zero_grad()
            loss = args.lambda_func * functional_bce(model, batch, y, args.func_scale)
            loss.backward(); opt.step()
        print(f"[epoch {e+1}/{args.epochs}] done", flush=True)
    torch.save({"state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "vocab": vocab, "mode": "vdjdb-contrastive-sup"}, args.out)
    print(f"[save] -> {args.out}")


if __name__ == "__main__":
    main()
