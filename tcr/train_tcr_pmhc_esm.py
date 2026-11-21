#!/usr/bin/env python3
"""Train TCRpMHCBindingModelESM (frozen ESM2 TCR encodings) on VDJdb swap-decoy.

Same objective, data and hyperparameters as train_tcr_pmhc.py (BCE, lr 1e-3,
batch 512, early-stop on val ROC) so the only difference vs the baseline binding
model is the CDR3 sequence encoder (ESM2 mean-pool + projection vs char-embedding
transformer).
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

from tcr_model import encode_seq
from tcr_model_esm import TCRpMHCBindingModelESM

HERE = os.path.dirname(os.path.realpath(__file__))


def load_emb(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["seqs"], d["emb"]  # list[str], [N, D]


def seq2gather(seqs, emb):
    idx = {s: i for i, s in enumerate(seqs)}
    return idx, emb.numpy().astype(np.float32)


def build_tensors(csv_path, vocab, a_idx, a_arr, b_idx, b_arr):
    """train.csv / val.csv format -> (tensor_dict, labels)."""
    df = pd.read_csv(csv_path)
    max_len = vocab["max_len"]
    gene_idx = {
        "va": {g: i for i, g in enumerate(vocab["v.alpha"])},
        "ja": {g: i for i, g in enumerate(vocab["j.alpha"])},
        "vb": {g: i for i, g in enumerate(vocab["v.beta"])},
        "jb": {g: i for i, g in enumerate(vocab["j.beta"])},
    }
    allele_idx = {a: i for i, a in enumerate(vocab["allele"])}

    ra = np.array([a_idx[s] for s in df["cdr3a"]], dtype=np.int64)
    rb = np.array([b_idx[s] for s in df["cdr3b"]], dtype=np.int64)
    cdr3a_esm = a_arr[ra]
    cdr3b_esm = b_arr[rb]
    peptide = np.stack([encode_seq(s, max_len).numpy() for s in df["peptide"]])
    va = np.array([gene_idx["va"][g] for g in df["va"]], dtype=np.int64)
    ja = np.array([gene_idx["ja"][g] for g in df["ja"]], dtype=np.int64)
    vb = np.array([gene_idx["vb"][g] for g in df["vb"]], dtype=np.int64)
    jb = np.array([gene_idx["jb"][g] for g in df["jb"]], dtype=np.int64)
    allele = np.array([allele_idx[a] for a in df["allele"]], dtype=np.int64)
    label = df["label"].to_numpy(dtype=np.float32)

    tensors = {
        "cdr3a_esm": torch.from_numpy(cdr3a_esm),
        "cdr3b_esm": torch.from_numpy(cdr3b_esm),
        "peptide": torch.from_numpy(peptide),
        "va": torch.from_numpy(va), "ja": torch.from_numpy(ja),
        "vb": torch.from_numpy(vb), "jb": torch.from_numpy(jb),
        "allele": torch.from_numpy(allele),
    }
    return tensors, torch.from_numpy(label)


def evaluate(model, tensors, labels, idx, device):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            batch = {k: tensors[k][b].to(device) for k in tensors}
            logits = model(batch).squeeze(-1)
            y_true.append(labels[b].cpu().numpy())
            y_score.append(torch.sigmoid(logits).cpu().numpy())
    return roc_auc_score(np.concatenate(y_true), np.concatenate(y_score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tcr_pmhc"))
    ap.add_argument("--esm-dir", default=os.path.join(HERE, "data", "tcr_pmhc", "esm"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "tcr_pmhc", "best_model_esm.pt"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.data, "vocab.json")) as f:
        vocab = json.load(f)

    a_seqs, a_emb = load_emb(os.path.join(args.esm_dir, "cdr3a_emb.pt"))
    b_seqs, b_emb = load_emb(os.path.join(args.esm_dir, "cdr3b_emb.pt"))
    a_idx, a_arr = seq2gather(a_seqs, a_emb)
    b_idx, b_arr = seq2gather(b_seqs, b_emb)
    esm_dim = a_arr.shape[1]
    print(f"[esm] dim={esm_dim} unique cdr3a={len(a_seqs)} cdr3b={len(b_seqs)}")

    tr_t, tr_y = build_tensors(os.path.join(args.data, "train.csv"), vocab, a_idx, a_arr, b_idx, b_arr)
    va_t, va_y = build_tensors(os.path.join(args.data, "val.csv"), vocab, a_idx, a_arr, b_idx, b_arr)
    print(f"[data] train={len(tr_y)} val={len(va_y)} device={device}")

    model = TCRpMHCBindingModelESM(
        esm_dim=esm_dim, d_model=64, n_heads=2, dropout=0.1, max_len=vocab["max_len"],
        n_v_alpha=len(vocab["v.alpha"]), n_j_alpha=len(vocab["j.alpha"]),
        n_v_beta=len(vocab["v.beta"]), n_j_beta=len(vocab["j.beta"]),
        n_allele=len(vocab["allele"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    tr_idx = np.arange(len(tr_y))

    best_roc, best_state, patience = 0.0, None, 0
    for ep in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        perm = np.random.permutation(len(tr_idx))
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            batch = {k: tr_t[k][b].to(device) for k in tr_t}
            y = tr_y[b].to(device)
            opt.zero_grad()
            logits = model(batch).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            total += loss.item() * len(y); n += len(y)
        val_roc = evaluate(model, va_t, va_y, np.arange(len(va_y)), device)
        print(f"[epoch {ep+1}/{args.epochs}] loss={total/max(n,1):.4f} valROC={val_roc:.4f}", flush=True)
        if val_roc > best_roc:
            best_roc, patience = val_roc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                print("[early stop]")
                break

    torch.save({"state_dict": best_state, "vocab": vocab, "val_roc": best_roc,
                "esm_dim": esm_dim, "mode": "vdjdb-swapdecoy-esm"}, args.out)
    print(f"[save] best val ROC={best_roc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
