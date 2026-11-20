#!/usr/bin/env python3
"""Evaluate multiple Phase-2 checkpoints on the SAME held-out test split in one pass.

Loads the IEDB dataset once, reconstructs the seed-42 80/10/10 split, then evals
each ``name:ckpt:decoder_layers`` triple with mc-averaging. Prints a full metric
row (ROC / PR / mean-PPVn / accuracy / F1 / precision / recall) per model so
models trained under different decoder depths are compared apples-to-apples.

Usage:
    python eval_opt_phase2_multi.py \
        --ckpt opt_focal:results/opt_phase2_opt_focal/opt_phase2_best.pt:1 \
               dec3:results/opt_phase2_dec3/opt_phase2_best.pt:3 \
        --device cuda:2
"""

import os
import sys
import argparse
import types

_dgl_gb = types.ModuleType("dgl.graphbolt")
_dgl_gb.load_graphbolt = lambda: None
sys.modules["dgl.graphbolt"] = _dgl_gb

import torch

_orig_torch_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_load

import numpy as np
from torch.utils.data import Subset
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             accuracy_score, f1_score, precision_score, recall_score)

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from dgl.dataloading import GraphDataLoader
from models.optimized_models import HybridModelOpt
from data_loading import ImmunoPredDataset, collate, SplitDataset
from utils import seed_everything, update_paths


def paper_mean_ppvn(y_true, y_pred, thresh):
    y_true = np.asarray(y_true)
    binpred = (np.asarray(y_pred) >= thresh).astype(float)
    idx = np.argsort(binpred)[::-1]
    y = y_true[idx]
    ppvn = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(np.mean(ppvn[:int(y.sum())]))


def opt_thresh(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return thr[np.argmax(tpr - fpr)]


def evaluate(model, loader, device, mc=20):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            graph_data, sequence_data, target, peptide_property = batch[:4]
            sequence_data = sequence_data.to(device)
            peptide_property = peptide_property.to(device)
            probs_sum = None
            for _ in range(mc):
                _, _, _, output = model(graph_data, sequence_data, peptide_property)
                p = torch.sigmoid(output.squeeze(-1))
                probs_sum = p if probs_sum is None else probs_sum + p
            ps.append((probs_sum / mc).cpu().numpy())
            ys.append(target.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    roc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
    pr = average_precision_score(ys, ps)
    th = opt_thresh(ys, ps)
    ppvn = paper_mean_ppvn(ys, ps, th)
    pred = (ps >= th).astype(int)
    return dict(roc=roc, pr=pr, ppvn=ppvn, acc=accuracy_score(ys, pred),
                f1=f1_score(ys, pred), prec=precision_score(ys, pred),
                rec=recall_score(ys, pred))


def main():
    p = argparse.ArgumentParser(description="Eval multiple Phase-2 checkpoints on one test split")
    p.add_argument("--ckpt", nargs="+", required=True, type=str,
                   help="one or more 'name:path:decoder_layers' triples")
    p.add_argument("--graph-dir", default=f"{_PROJ_ROOT}/data/graph_pyg_IEDB_esmfold2/")
    p.add_argument("--property-path", default=f"{_PROJ_ROOT}/data/ImmunoStruct_IEDB_data.csv")
    p.add_argument("--hla-path", default=f"{_PROJ_ROOT}/data/HLA_allele_sequences.csv")
    p.add_argument("--egnn-type", default="opt", type=str, choices=["opt", "torch"])
    p.add_argument("--vae-hidden-dim", default=768, type=int)
    p.add_argument("--vae-latent-dim", default=64, type=int)
    p.add_argument("--gat-hidden", default=64, type=int)
    p.add_argument("--gnn-layers", default=5, type=int)
    p.add_argument("--property-emb-dim", default=16, type=int)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--feature-size", default=23, type=int)
    p.add_argument("--coord-size", default=3, type=int)
    p.add_argument("--sequence-pad-count", default=0, type=int)
    p.add_argument("--structure-pad-count", default=0, type=int)
    p.add_argument("--mc", default=20, type=int)
    p.add_argument("--device", default="cuda:0")
    config = p.parse_args()
    update_paths(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)

    dataset = ImmunoPredDataset(config, graph_directory=config.graph_dir,
                                property_path=config.property_path, hla_path=config.hla_path)
    input_dim = dataset.encoded_full_sequence.shape[1] * dataset.encoded_full_sequence.shape[2]
    print(f"IEDB n={len(dataset)} input_dim={input_dim}", flush=True)

    g = torch.Generator().manual_seed(config.seed)
    n = len(dataset)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    perm = torch.randperm(n, generator=g).tolist()
    test_idx = perm[n_train + n_val:]

    test_loader = GraphDataLoader(
        SplitDataset(Subset(dataset, test_idx), "test", binary=True, full=True,
                     comparative=False, return_amino_acid=False),
        batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)

    print(f"{'model':<18}{'ROC':>8}{'PR':>8}{'PPVn':>8}{'ACC':>8}{'F1':>8}{'Prec':>8}{'Rec':>8}")
    for spec in config.ckpt:
        name, ckpt_path, dl = spec.rsplit(":", 2)
        dl = int(dl)
        model = HybridModelOpt(input_dim, device, gcn_layers=config.gnn_layers,
                               vae_hidden_dim=config.vae_hidden_dim,
                               vae_latent_dim=config.vae_latent_dim,
                               gat_hidden_channels=config.gat_hidden,
                               property_embedding_dim=config.property_emb_dim,
                               dropout=config.dropout,
                               egnn_type=config.egnn_type,
                               decoder_layers=dl).to(device)
        missing, unexpected = model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        m = evaluate(model, test_loader, device, mc=config.mc)
        print(f"{name:<18}{m['roc']:>8.4f}{m['pr']:>8.4f}{m['ppvn']:>8.4f}"
              f"{m['acc']:>8.4f}{m['f1']:>8.4f}{m['prec']:>8.4f}{m['rec']:>8.4f}"
              f"   [{len(missing)} missing/{len(unexpected)} unexpected]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
