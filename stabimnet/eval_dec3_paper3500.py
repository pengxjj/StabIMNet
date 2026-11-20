#!/usr/bin/env python3
"""Evaluate dec3 (HybridModelOpt, decoder_layers=3) on the paper-aligned fixed
3500-sample IEDB test set, using the EXACT protocol of train_v2_multiseed.py:
stratified_holdout(n_test=3500, seed=42) + 4 metrics (roc_auc /
average_precision / mean_PPVn(continuous) / precision@0.5).

This makes dec3 directly comparable to (a) the paper's IEDB numbers
(AUROC 0.882 / AUPRC 0.696 / mean PPVn 0.514) and (b) the prior baseline table
(foreignness arm 0.859 / 0.636 / 0.723 / 0.485 on the same split).

Usage:
    python eval_dec3_paper3500.py \
        --ckpt results/opt_phase2_dec3/opt_phase2_best.pt \
        --decoder-layers 3 --egnn-type opt --device cuda:0
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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from dgl.dataloading import GraphDataLoader
from models.optimized_models import HybridModelOpt
from data_loading import ImmunoPredDataset, collate, SplitDataset
from utils import seed_everything, update_paths


def mean_PPVn(values_true, values_pred, topk=None):
    """BigMHC continuous PPV@n (identical to procedures/metric.py:mean_PPVn)."""
    sorting_idx = np.argsort(values_pred)[::-1]
    values_true = values_true[sorting_idx]
    cum_true_positives = np.cumsum(values_true)
    total_predictions = np.arange(1, len(values_true) + 1)
    ppvn = cum_true_positives / total_predictions
    num_positives = int(values_true.sum())
    if topk is None:
        mean_ppvn = np.mean(ppvn[:num_positives])
    elif topk >= len(ppvn[:num_positives]):
        mean_ppvn = np.mean(ppvn[:num_positives])
    else:
        mean_ppvn = np.mean(ppvn[:num_positives][:topk])
    return mean_ppvn


def stratified_holdout(dataset, n_test, seed):
    """Fixed stratified holdout of n_test samples (identical to train_v2_multiseed.py)."""
    labels = dataset.immunogenicity.numpy()
    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    pos_rate = len(pos_idx) / len(labels)
    n_test_pos = int(round(n_test * pos_rate))
    n_test_neg = n_test - n_test_pos
    rng = np.random.default_rng(seed)
    test_pos = rng.choice(pos_idx, size=n_test_pos, replace=False)
    test_neg = rng.choice(neg_idx, size=n_test_neg, replace=False)
    test_idx = np.sort(np.concatenate([test_pos, test_neg]))
    return test_idx.astype(np.int64)


@torch.no_grad()
def get_predictions(model, loader, device, mc=1):
    """Single-forward (mc=1, matches train_v2_multiseed.py) or mc-averaged preds."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        graph_data, sequence_data, target, peptide_property = batch[:4]
        sequence_data = sequence_data.to(device)
        peptide_property = peptide_property.to(device)
        probs_sum = None
        for _ in range(mc):
            _, _, _, output = model(graph_data, sequence_data, peptide_property)
            p = torch.sigmoid(output.squeeze(-1))
            probs_sum = p if probs_sum is None else probs_sum + p
        probs = probs_sum / mc
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(target.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
    return preds, labels


def main():
    p = argparse.ArgumentParser(description="dec3 on paper 3500 test set (train_v2_multiseed protocol)")
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--graph-dir", default=f"{_PROJ_ROOT}/data/graph_pyg_IEDB_esmfold2/")
    p.add_argument("--property-path", default=f"{_PROJ_ROOT}/data/ImmunoStruct_IEDB_data.csv")
    p.add_argument("--hla-path", default=f"{_PROJ_ROOT}/data/HLA_allele_sequences.csv")
    p.add_argument("--vae-hidden-dim", default=768, type=int)
    p.add_argument("--vae-latent-dim", default=64, type=int)
    p.add_argument("--gat-hidden", default=64, type=int)
    p.add_argument("--gnn-layers", default=5, type=int)
    p.add_argument("--egnn-type", default="opt", type=str, choices=["opt", "torch"])
    p.add_argument("--property-emb-dim", default=16, type=int)
    p.add_argument("--decoder-layers", default=3, type=int)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--mc", default=1, type=int, help="1 = single forward (matches baseline table); 20 = mc20")
    p.add_argument("--eval-seed", default=0, type=int)
    p.add_argument("--test-size", default=3500, type=int)
    p.add_argument("--test-seed", default=42, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--feature-size", default=23, type=int)
    p.add_argument("--coord-size", default=3, type=int)
    p.add_argument("--sequence-pad-count", default=0, type=int)
    p.add_argument("--structure-pad-count", default=0, type=int)
    p.add_argument("--device", default="cuda:0")
    config = p.parse_args()
    update_paths(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seed_everything(config.eval_seed)

    dataset = ImmunoPredDataset(config, graph_directory=config.graph_dir,
                                property_path=config.property_path, hla_path=config.hla_path)
    input_dim = dataset.encoded_full_sequence.shape[1] * dataset.encoded_full_sequence.shape[2]
    print(f"IEDB n={len(dataset)} input_dim={input_dim}", flush=True)

    test_idx = stratified_holdout(dataset, config.test_size, config.test_seed)
    n_pos = int((dataset.immunogenicity.numpy()[test_idx] > 0.5).sum())
    print(f"test (stratified holdout n={len(test_idx)}, seed={config.test_seed}): "
          f"pos={n_pos} neg={len(test_idx)-n_pos}", flush=True)

    model = HybridModelOpt(input_dim, device, gcn_layers=config.gnn_layers,
                           vae_hidden_dim=config.vae_hidden_dim,
                           vae_latent_dim=config.vae_latent_dim,
                           gat_hidden_channels=config.gat_hidden,
                           property_embedding_dim=config.property_emb_dim,
                           dropout=config.dropout,
                           egnn_type=config.egnn_type,
                           decoder_layers=config.decoder_layers,
                           mask_ratio=0.0).to(device)
    ck = torch.load(config.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(ck, strict=False)
    print(f"ckpt={config.ckpt}", flush=True)
    print(f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()

    test_loader = GraphDataLoader(
        SplitDataset(Subset(dataset, test_idx.tolist()), "test", binary=True,
                     full=True, comparative=False, return_amino_acid=False),
        batch_size=config.batch_size, collate_fn=collate, shuffle=False,
        num_workers=config.num_workers)

    preds, labels = get_predictions(model, test_loader, device, mc=config.mc)
    roc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    pr = average_precision_score(labels, preds)
    ppvn_cont = mean_PPVn(labels, preds)
    ppvn_prec = precision_score(labels, preds >= 0.5, zero_division=0)

    print(f"\n=== dec3 (mc={config.mc}) on paper 3500 test set ===", flush=True)
    print(f"ROC       {roc:.4f}   (paper 0.882)", flush=True)
    print(f"PR        {pr:.4f}   (paper 0.696)", flush=True)
    print(f"PPVn_cont {ppvn_cont:.4f}   (foreignness 0.723)", flush=True)
    print(f"PPVn_prec {ppvn_prec:.4f}   (paper ~0.514)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
