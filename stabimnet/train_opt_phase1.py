#!/usr/bin/env python3
"""ImmunoStruct-Opt Phase 1: ConvVAE sequence pretraining on peptide foreignness.

Trains the 1D-CNN VAE (``ConvVAE``) on the IEDB sequence-only foreignness
regression task, matching the paper's transfer-learning stage 1 but with the
CNN encoder replacing the flatten-then-linear VAE. The saved ``opt_phase1_vae.pt``
is a plain ``ConvVAE`` state_dict, directly loadable into ``HybridModelOpt.vae``
for Phase 2 via ``load_state_dict(strict=False)``.

Usage:
    python train_opt_phase1.py \
        --property-path data/ImmunoStruct_IEDB_data.csv \
        --hla-path data/HLA_allele_sequences.csv \
        --epochs 40 --batch-size 128 --lr 1e-3 \
        --save-dir results/opt_phase1_foreignness
"""

import os
import sys
import argparse
import time
import types

_dgl_gb = types.ModuleType("dgl.graphbolt")
_dgl_gb.load_graphbolt = lambda: None
sys.modules["dgl.graphbolt"] = _dgl_gb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from data_loading.constants import AMINO_ACIDS, PADDING_CHAR
from data_loading.data_utils import one_hot_encode_sequence
from models.optimized_models import MLPVAE

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False

# 272-residue HLA pseudo-sequence + 11-mer max peptide == 283, matching the
# dynamic max_full_length computed by ImmunoPredDataset (phase 2/3) so the VAE
# decoder `dec_fc2` transfers cleanly across all three stages.
UNIFIED_MAX_LEN = 283
N_AMINO_ACIDS = len(AMINO_ACIDS) + 1  # 20 AA + padding char
VAE_INPUT_DIM = UNIFIED_MAX_LEN * N_AMINO_ACIDS
MHC_TRUNC = 272
DEFAULT_PI, DEFAULT_GRAVY = 6.5, -0.5


def compute_pi_gravy(seq):
    if not _HAS_BIOPYTHON:
        return DEFAULT_PI, DEFAULT_GRAVY
    clean = seq.replace(PADDING_CHAR, "")
    if len(clean) < 3:
        return DEFAULT_PI, DEFAULT_GRAVY
    try:
        a = PA(clean)
        return max(3.0, min(12.0, a.isoelectric_point())), max(-3.0, min(3.0, a.gravy()))
    except Exception:
        return DEFAULT_PI, DEFAULT_GRAVY


def prepare_sequence(full_mhc_seq, peptide):
    combined = full_mhc_seq[:MHC_TRUNC] + peptide
    if len(combined) < UNIFIED_MAX_LEN:
        combined = combined.ljust(UNIFIED_MAX_LEN, PADDING_CHAR)
    else:
        combined = combined[:UNIFIED_MAX_LEN]
    return one_hot_encode_sequence(combined, AMINO_ACIDS, PADDING_CHAR)


class ForeignnessDataset(Dataset):
    def __init__(self, csv_path, hla_path, foreignness_col="Foreignness_Score"):
        df = pd.read_csv(csv_path).dropna(subset=[foreignness_col]).reset_index(drop=True)
        hla_df = pd.read_csv(hla_path)
        hla_dict = dict(zip(hla_df["allele"], hla_df["seqs"]))

        raw = df[foreignness_col].values.astype(np.float32)
        self.t_min, self.t_max = float(raw.min()), float(raw.max())
        span = self.t_max - self.t_min
        self.targets = torch.tensor((raw - self.t_min) / span if span > 0 else np.zeros_like(raw), dtype=torch.float32)

        self.encoded, self.properties = [], []
        for _, row in df.iterrows():
            hla_seq = hla_dict[row["allele"]]
            pep = row["peptide"]
            oh = prepare_sequence(hla_seq, pep)
            pi, gravy = compute_pi_gravy(hla_seq[:MHC_TRUNC] + pep)
            self.encoded.append(torch.from_numpy(oh).float())
            self.properties.append(torch.tensor([pi, gravy], dtype=torch.float32))
        self.encoded = torch.stack(self.encoded)
        self.properties = torch.stack(self.properties)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.encoded[idx], self.properties[idx], self.targets[idx]


def collate(batch):
    encoded, props, targets = zip(*batch)
    return torch.stack(encoded), torch.stack(props), torch.stack(targets)


def main():
    p = argparse.ArgumentParser(description="ImmunoStruct-Opt Phase 1: ConvVAE foreignness pretraining")
    p.add_argument("--property-path", default=f"{_PROJ_ROOT}/data/ImmunoStruct_IEDB_data.csv")
    p.add_argument("--hla-path", default=f"{_PROJ_ROOT}/data/HLA_allele_sequences.csv")
    p.add_argument("--foreignness-col", default="Foreignness_Score")
    p.add_argument("--vae-hidden-dim", default=768, type=int)
    p.add_argument("--vae-latent-dim", default=64, type=int)
    p.add_argument("--property-emb-dim", default=16, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--epochs", default=40, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight-decay", default=1e-6, type=float)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--split-seed", default=42, type=int)
    p.add_argument("--task-weight", default=2.0, type=float)
    p.add_argument("--mse-weight", default=0.5, type=float)
    p.add_argument("--kld-weight", default=0.5, type=float)
    p.add_argument("--kl-warmup", default=5, type=int)
    p.add_argument("--kl-max", default=0.1, type=float)
    p.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/opt_phase1_foreignness")
    p.add_argument("--device", default="cuda:0")
    config = p.parse_args()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    os.makedirs(config.save_dir, exist_ok=True)

    full_ds = ForeignnessDataset(config.property_path, config.hla_path, config.foreignness_col)
    g = torch.Generator().manual_seed(config.split_seed)
    n = len(full_ds)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    perm = torch.randperm(n, generator=g).tolist()
    train_ds = Subset(full_ds, perm[:n_train])
    valid_ds = Subset(full_ds, perm[n_train:n_train + n_val])
    test_ds = Subset(full_ds, perm[n_train + n_val:])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=collate,
                              drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=collate, pin_memory=True)

    print(f"Phase1 Opt: ConvVAE foreignness | train={n_train} val={n_val} test={n-n_train-n_val}", flush=True)
    print(f"target range [{full_ds.t_min:.3f},{full_ds.t_max:.3f}] -> [0,1]", flush=True)

    vae = MLPVAE(VAE_INPUT_DIM, vae_hidden_dim=config.vae_hidden_dim,
                 vae_latent_dim=config.vae_latent_dim, property_embedding_dim=config.property_emb_dim).to(device)
    prop_emb = nn.Sequential(nn.Linear(2, 32), nn.ReLU(True), nn.Dropout(0.1),
                             nn.Linear(32, config.property_emb_dim), nn.ReLU(True)).to(device)
    head = nn.Sequential(nn.Linear(config.vae_latent_dim, 32), nn.ReLU(True),
                         nn.Dropout(0.1), nn.Linear(32, 1)).to(device)

    all_params = list(vae.parameters()) + list(prop_emb.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.lr, weight_decay=config.weight_decay)

    def forward_vae(x, props):
        x_flat = x.view(x.shape[0], -1)
        mu, logvar = vae.encode(x_flat)
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z, prop_emb(props))
        return recon, mu, logvar, z

    def kl_beta(ep):
        if config.kl_warmup <= 1:
            return config.kl_max
        return config.kl_max * min(1.0, (ep - 1) / (config.kl_warmup - 1))

    @torch.no_grad()
    def evaluate(loader):
        vae.eval(); head.eval(); prop_emb.eval()
        preds, tgts, mse = [], [], 0.0
        for enc, props, t in loader:
            x = enc.to(device); props = props.to(device); t = t.to(device)
            recon, mu, logvar, z = forward_vae(x, props)
            pred = head(z).view(-1)
            mse += F.mse_loss(pred, t, reduction="sum").item()
            preds.append(pred.cpu().numpy()); tgts.append(t.cpu().numpy())
        preds = np.concatenate(preds); tgts = np.concatenate(tgts)
        r = pearsonr(preds, tgts)[0] if len(tgts) > 1 and preds.std() > 0 else 0.0
        return mse / len(tgts), r

    best_r, no_improve = -1.0, 0
    vae_path = os.path.join(config.save_dir, "opt_phase1_vae.pt")
    best_path = os.path.join(config.save_dir, "opt_phase1_vae_best.pt")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        vae.train(); head.train(); prop_emb.train()
        tl = 0.0
        for enc, props, t in train_loader:
            x = enc.to(device); props = props.to(device); t = t.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = forward_vae(x, props)
            pred = head(z).view(-1)
            task = F.mse_loss(pred, t)
            mse = F.mse_loss(recon, x.view(x.shape[0], -1))
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = config.task_weight * task + config.mse_weight * mse + kl_beta(epoch) * kld
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            tl += loss.item() * len(t)
        val_mse, val_r = evaluate(valid_loader)
        print(f"Ep {epoch:3d}/{config.epochs} | loss {tl/n_train:.4f} | val MSE {val_mse:.4f} | val r {val_r:.4f} | {time.time()-t0:.0f}s", flush=True)
        if val_r > best_r:
            best_r, no_improve = val_r, 0
            torch.save({"vae": vae.state_dict(), "prop_emb": prop_emb.state_dict(), "head": head.state_dict()}, best_path)
            torch.save(vae.state_dict(), vae_path)
        else:
            no_improve += 1

    best_ckpt = torch.load(best_path)
    vae.load_state_dict(best_ckpt["vae"])
    prop_emb.load_state_dict(best_ckpt["prop_emb"])
    head.load_state_dict(best_ckpt["head"])
    test_mse, test_r = evaluate(test_loader)
    print(f"\nTEST | MSE {test_mse:.4f} | Pearson r {test_r:.4f}", flush=True)
    print(f"VAE saved -> {vae_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
