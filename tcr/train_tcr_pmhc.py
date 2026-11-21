#!/usr/bin/env python3
"""Train the TCR-pMHC binding model on the preprocessed VDJdb dataset.

Loads data/tcr_pmhc/{train,val,test}.csv + vocab.json (built by
preprocess_vdjdb_tcr.py), trains TCRpMHCBindingModel with BCEWithLogitsLoss,
early-stops on validation ROC-AUC, and writes the best checkpoint.

Run:
    python train_tcr_pmhc.py [--data data/tcr_pmhc] [--epochs 10] [--device cuda:0] \
        [--batch-size 512] [--lr 1e-3] [--smoke]
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from tcr_model import TCRpMHCBindingModel, encode_seq

HERE = os.path.dirname(os.path.realpath(__file__))


class TCRDataset(Dataset):
    def __init__(self, csv_path, vocab):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.max_len = vocab["max_len"]
        self.gene_idx = {
            "va": {g: i for i, g in enumerate(vocab["v.alpha"])},
            "ja": {g: i for i, g in enumerate(vocab["j.alpha"])},
            "vb": {g: i for i, g in enumerate(vocab["v.beta"])},
            "jb": {g: i for i, g in enumerate(vocab["j.beta"])},
        }
        self.allele_idx = {a: i for i, a in enumerate(vocab["allele"])}

        self.cdr3a = [encode_seq(s, self.max_len) for s in df["cdr3a"]]
        self.cdr3b = [encode_seq(s, self.max_len) for s in df["cdr3b"]]
        self.peptide = [encode_seq(s, self.max_len) for s in df["peptide"]]
        self.va = [self.gene_idx["va"][g] for g in df["va"]]
        self.ja = [self.gene_idx["ja"][g] for g in df["ja"]]
        self.vb = [self.gene_idx["vb"][g] for g in df["vb"]]
        self.jb = [self.gene_idx["jb"][g] for g in df["jb"]]
        self.allele = [self.allele_idx[a] for a in df["allele"]]
        self.label = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return {
            "cdr3a": self.cdr3a[i], "cdr3b": self.cdr3b[i], "peptide": self.peptide[i],
            "va": self.va[i], "ja": self.ja[i], "vb": self.vb[i], "jb": self.jb[i],
            "allele": self.allele[i], "label": self.label[i],
        }


def collate(batch):
    out = {}
    for k in ("cdr3a", "cdr3b", "peptide"):
        out[k] = torch.stack([b[k] for b in batch])
    for k in ("va", "ja", "vb", "jb", "allele"):
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    out["label"] = torch.stack([b["label"] for b in batch])
    return out


def evaluate(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for batch in loader:
            for k in ("cdr3a", "cdr3b", "peptide", "va", "ja", "vb", "jb", "allele"):
                batch[k] = batch[k].to(device)
            logits = model(batch).squeeze(-1)
            y_true.append(batch["label"].cpu().numpy())
            y_score.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    return roc_auc_score(y_true, y_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "tcr_pmhc"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ablate", choices=["tcr", "pmhc"], default=None,
                    help="ablation: 'tcr' trains pMHC-only, 'pmhc' trains TCR-only")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.epochs = 2

    with open(os.path.join(args.data, "vocab.json")) as f:
        vocab = json.load(f)

    train_ds = TCRDataset(os.path.join(args.data, "train.csv"), vocab)
    val_ds = TCRDataset(os.path.join(args.data, "val.csv"), vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=4)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} device={device}")

    model = TCRpMHCBindingModel(
        d_model=64, n_heads=2, dropout=0.1, max_len=vocab["max_len"],
        n_v_alpha=len(vocab["v.alpha"]), n_j_alpha=len(vocab["j.alpha"]),
        n_v_beta=len(vocab["v.beta"]), n_j_beta=len(vocab["j.beta"]),
        n_allele=len(vocab["allele"]), ablate=args.ablate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_roc, best_state, patience = 0.0, None, 0
    for ep in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            for k in ("cdr3a", "cdr3b", "peptide", "va", "ja", "vb", "jb", "allele"):
                batch[k] = batch[k].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            logits = model(batch).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        val_roc = evaluate(model, val_loader, device)
        print(f"[epoch {ep+1}/{args.epochs}] loss={total_loss/max(n,1):.4f} "
              f"valROC={val_roc:.4f}")
        if val_roc > best_roc:
            best_roc, patience = val_roc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3 and not args.smoke:
                print("[early stop] no val ROC improvement for 3 epochs")
                break

    out = os.path.join(args.data, "best_model.pt")
    torch.save({"state_dict": best_state, "vocab": vocab, "val_roc": best_roc,
                "ablate": args.ablate}, out)
    print(f"[save] best val ROC={best_roc:.4f} -> {out}")


if __name__ == "__main__":
    main()
