#!/usr/bin/env python3
"""Train TCRpMHCContrastiveModel on VDJdb *positive* pairs with multi-positive InfoNCE.

Same data and hyperparameters as train_tcr_pmhc.py (lr 1e-3, batch 512, AdamW
wd 1e-4, early-stop on val binding ROC), so the only difference vs the baseline
binding model is the objective (contrastive on positive pairs vs BCE on
swap-decoys). Val ROC is computed on val.csv (pos + swap-decoy neg) using the
cosine-similarity score, so it is directly comparable to the BCE val ROC ~0.848.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from tcr_model import encode_seq
from tcr_model_contrastive import TCRpMHCContrastiveModel, multi_positive_infonce

HERE = os.path.dirname(os.path.realpath(__file__))


def build_tensors(csv_path, vocab, positives_only=False):
    """train.csv/val.csv format -> (tensor_dict, pmhc_id, label)."""
    df = pd.read_csv(csv_path)
    if positives_only:
        df = df[df["label"] == 1].reset_index(drop=True)
    max_len = vocab["max_len"]
    gene_idx = {
        "va": {g: i for i, g in enumerate(vocab["v.alpha"])},
        "ja": {g: i for i, g in enumerate(vocab["j.alpha"])},
        "vb": {g: i for i, g in enumerate(vocab["v.beta"])},
        "jb": {g: i for i, g in enumerate(vocab["j.beta"])},
    }
    allele_idx = {a: i for i, a in enumerate(vocab["allele"])}

    cdr3a = np.stack([encode_seq(s, max_len).numpy() for s in df["cdr3a"]])
    cdr3b = np.stack([encode_seq(s, max_len).numpy() for s in df["cdr3b"]])
    peptide = np.stack([encode_seq(s, max_len).numpy() for s in df["peptide"]])
    va = np.array([gene_idx["va"][g] for g in df["va"]], dtype=np.int64)
    ja = np.array([gene_idx["ja"][g] for g in df["ja"]], dtype=np.int64)
    vb = np.array([gene_idx["vb"][g] for g in df["vb"]], dtype=np.int64)
    jb = np.array([gene_idx["jb"][g] for g in df["jb"]], dtype=np.int64)
    allele = np.array([allele_idx[a] for a in df["allele"]], dtype=np.int64)

    tensors = {
        "cdr3a": torch.from_numpy(cdr3a), "cdr3b": torch.from_numpy(cdr3b),
        "peptide": torch.from_numpy(peptide),
        "va": torch.from_numpy(va), "ja": torch.from_numpy(ja),
        "vb": torch.from_numpy(vb), "jb": torch.from_numpy(jb),
        "allele": torch.from_numpy(allele),
    }
    pmhc_id = torch.from_numpy(pd.factorize((df["peptide"] + "|" + df["allele"]).astype(str))[0])
    label = torch.from_numpy(df["label"].to_numpy(dtype=np.float32))
    return tensors, pmhc_id, label


def evaluate_roc(model, tensors, labels, idx, device):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            batch = {k: tensors[k][b].to(device) for k in tensors}
            y_true.append(labels[b].cpu().numpy())
            y_score.append(model.score(batch).cpu().numpy())
    return roc_auc_score(np.concatenate(y_true), np.concatenate(y_score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tcr_pmhc"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "tcr_pmhc", "best_model_contrastive.pt"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.data, "vocab.json")) as f:
        vocab = json.load(f)

    tr_t, tr_pid, _ = build_tensors(os.path.join(args.data, "train.csv"), vocab, positives_only=True)
    va_t, _, va_y = build_tensors(os.path.join(args.data, "val.csv"), vocab)
    print(f"[data] train_pos={len(tr_pid)} val={len(va_y)} device={device}")

    model = TCRpMHCContrastiveModel(
        d_model=64, n_heads=2, dropout=0.1, max_len=vocab["max_len"],
        n_v_alpha=len(vocab["v.alpha"]), n_j_alpha=len(vocab["j.alpha"]),
        n_v_beta=len(vocab["v.beta"]), n_j_beta=len(vocab["j.beta"]),
        n_allele=len(vocab["allele"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_idx = np.arange(len(tr_pid))

    best_roc, best_state, patience = 0.0, None, 0
    for ep in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        perm = np.random.permutation(len(tr_idx))
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            batch = {k: tr_t[k][b].to(device) for k in tr_t}
            pid = tr_pid[b].to(device)
            opt.zero_grad()
            z_tcr, z_pmhc = model(batch)
            loss = multi_positive_infonce(z_tcr, z_pmhc, model.logit_scale, pid)
            loss.backward(); opt.step()
            total += loss.item() * len(b); n += len(b)
        val_roc = evaluate_roc(model, va_t, va_y, np.arange(len(va_y)), device)
        print(f"[epoch {ep+1}/{args.epochs}] loss={total/max(n,1):.4f} valROC={val_roc:.4f}", flush=True)
        torch.save({"state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "vocab": vocab, "val_roc": val_roc, "mode": "vdjdb-contrastive"},
                   f"{args.out}.epoch{ep+1}.pt")
        if val_roc > best_roc:
            best_roc, patience = val_roc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print("[early stop]")
                break

    torch.save({"state_dict": best_state, "vocab": vocab, "val_roc": best_roc,
                "mode": "vdjdb-contrastive"}, args.out)
    print(f"[save] best val ROC={best_roc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
