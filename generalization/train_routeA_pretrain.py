#!/usr/bin/env python3
"""Route A — stage 1: sequence-only pan-allelic presentation pretraining.

Pretrain the v6 *sequence tower* (pep_proj + hla_proj + pep_hla_attn) on the
108-allele BigMHC eluted-ligand (EL) presentation data (label = eluted ligand
0/1). Export ONLY those three modules' weights so stage 2 (finetune) can
initialize HybridModelv6's sequence tower from them — the EGNN structure tower
and the immunogenicity classifier stay randomly initialized.

No structure tower here (EL data has no ESMFold2 graphs). The inductive bias is
exactly v6's: peptide residues cross-attend to the canonical 34-res pseudo-seq.

The EL ``mhc_seq`` column is a stale pseudo-seq convention; we regenerate the
canonical pseudo-seq from ``full_mhc_seq`` (see precompute_esmc_el_pretrain.py)
and look up ``esmc_el_pseudo_perres.pt`` by that string.

Usage:
    python train_routeA_pretrain.py --gpu 1 --seeds 42
"""
import os
import sys
import argparse
import types
import logging
import json
import gc

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_orig_torch_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_load

from sklearn.metrics import roc_auc_score, average_precision_score

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from models.layers import CrossModalAttentionBlock

SEQ_EMB_DIM = 2560
PSEUDO_POSITIONS = [
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
    84, 95, 97, 99, 114, 116, 118, 143, 147, 152, 156, 158, 159, 163,
    167, 171, 180,
]
assert len(PSEUDO_POSITIONS) == 34


def canonical_pseudo(full_seq):
    """34-aa canonical pseudo-seq from a full MHC sequence (signal peptide stripped)."""
    idx = full_seq.find("SHSM")
    if idx < 0:
        return None
    mature = "G" + full_seq[idx:]
    if len(mature) < max(PSEUDO_POSITIONS):
        return None
    return "".join(mature[p - 1] for p in PSEUDO_POSITIONS)


# ────────────────────────────────────────────────────────────────────
# Sequence-only presentation model (pep_proj + hla_proj + cross-attn)
# ────────────────────────────────────────────────────────────────────

class ELPresentationModel(nn.Module):
    """Peptide residues cross-attend to HLA pseudo-seq → presentation logits.

    The three sequence-tower modules share names with HybridModelv6 so their
    weights transfer directly (load_state_dict(strict=False)).
    """

    def __init__(self, seq_emb_dim=2560, proj_dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.pep_proj = nn.Linear(seq_emb_dim, proj_dim)
        self.hla_proj = nn.Linear(seq_emb_dim, proj_dim)
        self.pep_hla_attn = CrossModalAttentionBlock(proj_dim, n_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * proj_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _masked_mean(x, mask):
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)

    def forward(self, pep_emb, hla_emb, pep_mask=None, hla_mask=None):
        B = pep_emb.shape[0]
        dev = pep_emb.device
        pep = self.pep_proj(pep_emb)   # [B, Lp, d]
        hla = self.hla_proj(hla_emb)   # [B, 34, d]
        if pep_mask is None:
            pep_mask = torch.ones(B, pep.shape[1], dtype=torch.bool, device=dev)
        if hla_mask is None:
            hla_mask = torch.ones(B, hla.shape[1], dtype=torch.bool, device=dev)
        pep_hla = self.pep_hla_attn(pep, hla, key_padding_mask=hla_mask)
        pep_vec = self._masked_mean(pep_hla, pep_mask)   # [B, d]
        hla_vec = self._masked_mean(hla, hla_mask)       # [B, d]
        return self.classifier(torch.cat([pep_vec, hla_vec], dim=1))

    def export_seq_tower(self):
        """State dict with ONLY the three sequence-tower modules (for v6 init)."""
        return {k: v for k, v in self.state_dict().items()
                if k.startswith(("pep_proj", "hla_proj", "pep_hla_attn"))}


# ────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────

class ELDataset(Dataset):
    def __init__(self, rows, pep_perres, pseudo_perres):
        self.peps = rows["peptide"].tolist()
        self.pseudos = rows["canon_pseudo"].tolist()
        self.labels = rows["label"].astype(float).tolist()
        self.pep_perres = pep_perres
        self.pseudo_perres = pseudo_perres

    def __len__(self):
        return len(self.peps)

    def __getitem__(self, idx):
        return (self.pep_perres[self.peps[idx]],
                self.pseudo_perres[self.pseudos[idx]],
                self.labels[idx])


def el_collate(items):
    peps = [it[0] for it in items]
    hlas = [it[1] for it in items]
    labels = torch.tensor([it[2] for it in items], dtype=torch.float32)

    B = len(items)
    Lp_max = max(p.shape[0] for p in peps)
    pep = torch.zeros(B, Lp_max, SEQ_EMB_DIM)
    pep_mask = torch.zeros(B, Lp_max, dtype=torch.bool)
    for i, p in enumerate(peps):
        L = p.shape[0]
        pep[i, :L] = p
        pep_mask[i, :L] = True

    hla = torch.stack(list(hlas), dim=0)                # [B, 34, 2560]
    hla_mask = torch.ones(B, hla.shape[1], dtype=torch.bool)
    return pep, pep_mask, hla, hla_mask, labels


# ────────────────────────────────────────────────────────────────────
# Train / eval
# ────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss, n = 0.0, 0
    for pep, pep_mask, hla, hla_mask, target in loader:
        pep, pep_mask = pep.to(device), pep_mask.to(device)
        hla, hla_mask = hla.to(device), hla_mask.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        out = model(pep, hla, pep_mask, hla_mask).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            out, target, pos_weight=pos_weight.to(device), reduction="mean")
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = len(target)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n if n else 0.0


@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    total_loss, n = 0.0, 0
    preds, labels = [], []
    for pep, pep_mask, hla, hla_mask, target in loader:
        pep, pep_mask = pep.to(device), pep_mask.to(device)
        hla, hla_mask = hla.to(device), hla_mask.to(device)
        target = target.to(device)
        out = model(pep, hla, pep_mask, hla_mask).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            out, target, pos_weight=pos_weight.to(device), reduction="mean")
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item() * len(target)
            n += len(target)
        preds.append(torch.sigmoid(out).cpu().numpy())
        labels.append(target.cpu().numpy())
    if n == 0:
        return 0.0, 0.5, 0.0
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    roc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    pr = average_precision_score(labels, preds)
    return total_loss / n, roc, pr


def run_seed(seed, config, rows, pep_perres, pseudo_perres, device, logger):
    g = torch.Generator().manual_seed(seed)
    n = len(rows)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(config.val_frac * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    logger.info(f"[seed {seed}] EL pretrain: train {len(train_idx)} / val {len(val_idx)}")

    train_ds = ELDataset(rows.iloc[train_idx].reset_index(drop=True),
                         pep_perres, pseudo_perres)
    val_ds = ELDataset(rows.iloc[val_idx].reset_index(drop=True),
                       pep_perres, pseudo_perres)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              collate_fn=el_collate, shuffle=True,
                              num_workers=config.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            collate_fn=el_collate, shuffle=False,
                            num_workers=config.num_workers)

    pos_count = int(rows.iloc[train_idx]["label"].astype(float).sum())
    neg_count = len(train_idx) - pos_count
    pos_weight = torch.tensor(neg_count / max(pos_count, 1))

    model = ELPresentationModel(seq_emb_dim=SEQ_EMB_DIM, proj_dim=config.proj_dim,
                                n_heads=config.n_heads, dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    seq_path = os.path.join(config.save_dir, f"el_pretrain_seed{seed}_seqtower.pt")
    best_val_roc, best_epoch, no_improve = 0.0, 0, 0
    early_stop = config.early_stop if config.early_stop > 0 else config.epochs + 1

    for epoch in range(1, config.epochs + 1):
        tl = train_epoch(model, train_loader, optimizer, device, pos_weight)
        scheduler.step()
        vl, vr, vp = evaluate(model, val_loader, device, pos_weight)
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[seed {seed}] Ep {epoch:3d}/{config.epochs} | Train {tl:.4f} | "
                        f"Val ROC {vr:.4f} | PR {vp:.4f}")
        if vr > best_val_roc:
            best_val_roc, best_epoch, no_improve = vr, epoch, 0
            torch.save(model.export_seq_tower(), seq_path)
        else:
            no_improve += 1
            if no_improve >= early_stop:
                logger.info(f"[seed {seed}] Early stop @ ep {epoch} (best Val ROC "
                            f"{best_val_roc:.4f} @ ep {best_epoch})")
                break

    logger.info(f"[seed {seed}] saved best-val seq tower -> {seq_path} "
                f"(best Val ROC {best_val_roc:.4f} @ ep {best_epoch})")
    return {"seed": seed, "best_val_roc": float(best_val_roc),
            "best_epoch": int(best_epoch)}


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description="Route A stage 1: EL presentation pretrain")
    parser.add_argument("--el-csv", default=f"{_PROJ_ROOT}/../data/mhc1_el_data/processed/mhc1_fullseq_train.csv")
    parser.add_argument("--pep-emb-path", default=f"{_PROJ_ROOT}/data/esmc_el_peptide_perres.pt")
    parser.add_argument("--pseudo-emb-path", default=f"{_PROJ_ROOT}/data/esmc_el_pseudo_perres.pt")
    parser.add_argument("--proj-dim", default=256, type=int)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-6, type=float)
    parser.add_argument("--early-stop", default=10, type=int)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/routeA_pretrain", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--dry-run", action="store_true")
    config = parser.parse_args()

    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(os.path.join(config.save_dir, "pretrain.log")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(f"Device {device} | save-dir {config.save_dir} | seeds {config.seeds}")

    df = pd.read_csv(config.el_csv)
    df["canon_pseudo"] = df["full_mhc_seq"].astype(str).apply(canonical_pseudo)
    n_bad = int(df["canon_pseudo"].isna().sum())
    if n_bad:
        logger.warning(f"{n_bad} rows without extractable pseudo-seq dropped")
        df = df.dropna(subset=["canon_pseudo"])
    logger.info(f"EL rows: {len(df)} | unique canonical pseudo: {df['canon_pseudo'].nunique()}")

    pep_perres = torch.load(config.pep_emb_path, map_location="cpu")
    pseudo_perres = torch.load(config.pseudo_emb_path, map_location="cpu")
    logger.info(f"peptide perres: {len(pep_perres)} | pseudo perres: {len(pseudo_perres)}")

    # drop rows whose peptide or pseudo is missing from the precomputed dicts
    n0 = len(df)
    df = df[df["peptide"].isin(pep_perres) & df["canon_pseudo"].isin(pseudo_perres)]
    logger.info(f"rows after embedding-coverage filter: {len(df)} (dropped {n0 - len(df)})")

    if config.dry_run:
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        config.epochs = 1
        config.early_stop = 0

    results = []
    for seed in config.seeds:
        r = run_seed(seed, config, df, pep_perres, pseudo_perres, device, logger)
        results.append(r)
        with open(os.path.join(config.save_dir, f"seed_{seed}_metrics.json"), "w") as f:
            json.dump(r, f, indent=2)
        del r
        gc.collect()
        torch.cuda.empty_cache()
    logger.info(f"DONE. {len(results)} seed(s).")


if __name__ == "__main__":
    main()
