#!/usr/bin/env python3
"""Train dec3 (HybridModelOpt) with a stratified per-seed split and seeds 42-46,
evaluated under the paper's full metric protocol.

Mirrors train_opt_phase2.py (same model / losses / optimizer / early-stop), with
three changes:

1. Split -- per seed, a STRATIFIED 80/10/10 split (positives & negatives split
   separately via sklearn train_test_split(stratify=...)) so every train/val/
   test fold keeps the dataset's ~22% positive rate.  This is deliberate: the
   paper's uniform `random_split` does NOT stratify, so fold class ratios drift
   by sampling noise (~±0.8% on test).
2. Metric -- the FULL paper evaluate_metrics (procedures/metric.py): ROC AUC,
   PR AUC (auc of precision_recall_curve, NOT average_precision_score),
   Accuracy/F1/Precision/Recall @0.5 and @op, and the binarized-sort bug PPVn
   (mean PPVn + PPVn n=30, each @0.5 and @op). All 14 metrics reported.
3. Aggregation -- N seeds, reported as mean +/- s.d. for all metrics, plus the
   paper Fig.3 headline (AUROC/AUPRC as s.d., mean PPVn as s.e.m.).

Note: the Phase-1 VAE is warm-started from a fixed checkpoint (as dec3 already
does). Loss coefficients remain dec3's (bce 5.0 / mse 0.1 / kld 0.1, focal
gamma=1.0), not the paper's lambda_pred/lambda_recon/lambda_normal.

Usage:
    python train_opt_phase2_paper.py --seeds 42 43 44 45 46 --device cuda:0
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

_orig_torch_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_load

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve)

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from dgl.dataloading import GraphDataLoader
from models.optimized_models import HybridModelOpt, transfer_phase1_vae
from data_loading import ImmunoPredDataset, collate, SplitDataset
from utils import seed_everything, update_paths
from utils.optimized_loss import OptimizedLosses


def mean_PPVn(values_true, values_pred, topk=None):
    """Exact copy of procedures/metric.py:mean_PPVn (paper's BigMHC PPV@n)."""
    assert len(values_true) == len(values_pred)
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


def find_optimal_threshold(y_true, y_prob):
    """Youden's J (exact copy of procedures/metric.py:find_optimal_threshold)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def stratified_split(dataset, fractions=(0.8, 0.1, 0.1), seed=0):
    """Stratified 80/10/10 split: positives and negatives are split separately so
    every fold keeps the dataset's class ratio (~22% positive).  Unlike the
    paper's uniform `random_split`, this guarantees the train/val/test positive
    rate matches the full dataset.  Deterministic in `seed`."""
    labels = dataset.immunogenicity.numpy()
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(idx, test_size=1.0 - fractions[0],
                                           stratify=labels, random_state=seed)
    val_frac = fractions[1] / (fractions[1] + fractions[2])  # 0.5 for 0.1/0.1
    val_idx, test_idx = train_test_split(rest_idx, test_size=1.0 - val_frac,
                                         stratify=labels[rest_idx], random_state=seed)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


@torch.no_grad()
def inference(model, loader, device):
    """Single forward (mc=1, matches procedures/infer.py:inference)."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        graph_data, sequence_data, target, peptide_property = batch[:4]
        sequence_data = sequence_data.to(device)
        peptide_property = peptide_property.to(device)
        _, _, _, output = model(graph_data, sequence_data, peptide_property)
        p = torch.sigmoid(output.squeeze(-1))
        p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
        ps.append(p.cpu().numpy())
        ys.append(target.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate_metrics(true_targets, predicted_probs, optimal_threshold):
    """Exact copy of procedures/metric.py:evaluate_metrics (paper's full 14-metric set).

    Note the paper computes PR AUC as auc(precision_recall_curve) (NOT
    average_precision_score), and mean PPVn / PPVn(n=30) use the BINARIZED
    predictions `predicted_probs >= threshold` fed into the ranking metric.
    """
    roc_auc = roc_auc_score(true_targets, predicted_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(true_targets, predicted_probs)
    pr_auc = auc(recall_curve, precision_curve)
    accuracy = accuracy_score(true_targets, predicted_probs >= 0.5)
    accuracy_op = accuracy_score(true_targets, predicted_probs >= optimal_threshold)
    f1 = f1_score(true_targets, predicted_probs >= 0.5)
    f1_op = f1_score(true_targets, predicted_probs >= optimal_threshold)
    precision = precision_score(true_targets, predicted_probs >= 0.5)
    precision_op = precision_score(true_targets, predicted_probs >= optimal_threshold)
    recall = recall_score(true_targets, predicted_probs >= 0.5)
    recall_op = recall_score(true_targets, predicted_probs >= optimal_threshold)
    ppvn = mean_PPVn(true_targets, predicted_probs >= 0.5)
    ppvn_op = mean_PPVn(true_targets, predicted_probs >= optimal_threshold)
    ppv30 = mean_PPVn(true_targets, predicted_probs >= 0.5, topk=30)
    ppv30_op = mean_PPVn(true_targets, predicted_probs >= optimal_threshold, topk=30)
    ppvn_cont = mean_PPVn(true_targets, predicted_probs)
    ppv30_cont = mean_PPVn(true_targets, predicted_probs, topk=30)
    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "accuracy": accuracy, "accuracy_op": accuracy_op,
        "f1": f1, "f1_op": f1_op,
        "precision": precision, "precision_op": precision_op,
        "recall": recall, "recall_op": recall_op,
        "ppvn": ppvn, "ppvn_op": ppvn_op,
        "ppv30": ppv30, "ppv30_op": ppv30_op,
        "ppvn_cont": ppvn_cont, "ppv30_cont": ppv30_cont,
    }


def evaluate(model, train_loader, test_loader, device):
    """Paper's inference flow: optimal_threshold from TRAIN (Youden's J), then
    the full evaluate_metrics on TEST."""
    train_y, train_p = inference(model, train_loader, device)
    test_y, test_p = inference(model, test_loader, device)
    opt = find_optimal_threshold(train_y, train_p)
    stats = evaluate_metrics(test_y, test_p, opt)
    stats["optimal_threshold"] = opt
    return stats


def main():
    p = argparse.ArgumentParser(description="dec3 under paper metric protocol (stratified split per seed)")
    p.add_argument("--graph-dir", default=f"{_PROJ_ROOT}/data/graph_pyg_IEDB_esmfold2/")
    p.add_argument("--property-path", default=f"{_PROJ_ROOT}/data/ImmunoStruct_IEDB_data.csv")
    p.add_argument("--hla-path", default=f"{_PROJ_ROOT}/data/HLA_allele_sequences.csv")
    p.add_argument("--phase1-ckpt", default=f"{_PROJ_ROOT}/model_weights/vae_presentation_vae.pt", type=str)
    p.add_argument("--vae-hidden-dim", default=768, type=int)
    p.add_argument("--vae-latent-dim", default=64, type=int)
    p.add_argument("--gat-hidden", default=64, type=int)
    p.add_argument("--gnn-layers", default=5, type=int)
    p.add_argument("--egnn-type", default="opt", type=str, choices=["opt", "torch"])
    p.add_argument("--property-emb-dim", default=16, type=int)
    p.add_argument("--decoder-layers", default=3, type=int)
    p.add_argument("--mask-ratio", default=0.0, type=float)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--lr-vae", default=3e-5, type=float)
    p.add_argument("--weight-decay", default=1e-6, type=float)
    p.add_argument("--bce-weight", default=5.0, type=float)
    p.add_argument("--mse-weight", default=0.1, type=float)
    p.add_argument("--kld-weight", default=0.1, type=float)
    p.add_argument("--focal-gamma", default=1.0, type=float)
    p.add_argument("--free-bits", default=0.0, type=float)
    p.add_argument("--vae-type", default="kld", type=str, choices=["kld", "mmd"])
    p.add_argument("--mmd-weight", default=1.0, type=float)
    p.add_argument("--early-stop", default=20, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"), help="stratified train/val/test fractions")
    p.add_argument("--feature-size", default=23, type=int)
    p.add_argument("--coord-size", default=3, type=int)
    p.add_argument("--sequence-pad-count", default=0, type=int)
    p.add_argument("--structure-pad-count", default=0, type=int)
    p.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/opt_phase2_paper")
    p.add_argument("--device", default="cuda:0")
    config = p.parse_args()
    update_paths(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    dataset = ImmunoPredDataset(config, graph_directory=config.graph_dir,
                                property_path=config.property_path, hla_path=config.hla_path)
    input_dim = dataset.encoded_full_sequence.shape[1] * dataset.encoded_full_sequence.shape[2]
    n_total = len(dataset)
    print(f"IEDB n={n_total} input_dim={input_dim}", flush=True)

    metric_names = ["roc_auc", "pr_auc", "accuracy", "accuracy_op", "f1", "f1_op",
                    "precision", "precision_op", "recall", "recall_op",
                    "ppvn", "ppvn_op", "ppv30", "ppv30_op",
                    "ppvn_cont", "ppv30_cont"]
    results = {k: [] for k in metric_names}

    for seed in config.seeds:
        seed_everything(seed)
        # --- Stratified 80/10/10 split (class ratio preserved per fold) ---
        train_idx, val_idx, test_idx = stratified_split(dataset, fractions=tuple(config.split), seed=seed)
        print(f"\n=== seed {seed} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} ===", flush=True)
        labels = dataset.immunogenicity.numpy()
        for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            n_pos = int((labels[idx] > 0.5).sum())
            print(f"  {name}: n={len(idx)} pos={n_pos} neg={len(idx) - n_pos} "
                  f"pos_rate={n_pos / len(idx):.4f}", flush=True)

        model = HybridModelOpt(input_dim, device, gcn_layers=config.gnn_layers,
                               vae_hidden_dim=config.vae_hidden_dim,
                               vae_latent_dim=config.vae_latent_dim,
                               gat_hidden_channels=config.gat_hidden,
                               property_embedding_dim=config.property_emb_dim,
                               dropout=config.dropout,
                               egnn_type=config.egnn_type,
                               decoder_layers=config.decoder_layers,
                               mask_ratio=config.mask_ratio).to(device)
        if config.phase1_ckpt and os.path.exists(config.phase1_ckpt):
            transfer_phase1_vae(model, config.phase1_ckpt, map_location=device)

        losses = OptimizedLosses(input_dim, dataset.class_weights,
                                 bce_weight=config.bce_weight, mse_weight=config.mse_weight,
                                 kld_weight=config.kld_weight, focal_gamma=config.focal_gamma,
                                 free_bits=config.free_bits,
                                 vae_type=config.vae_type, mmd_weight=config.mmd_weight)

        vae_params = [p for k, p in model.named_parameters() if k.startswith("vae.")]
        other_params = [p for k, p in model.named_parameters() if not k.startswith("vae.")]
        optimizer = torch.optim.AdamW([
            {"params": vae_params, "lr": config.lr_vae, "weight_decay": config.weight_decay},
            {"params": other_params, "lr": config.lr, "weight_decay": config.weight_decay},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        train_loader = GraphDataLoader(
            SplitDataset(Subset(dataset, train_idx), "train", binary=True, full=True,
                         comparative=False, return_amino_acid=False),
            batch_size=config.batch_size, collate_fn=collate, shuffle=True,
            num_workers=config.num_workers, drop_last=True)
        val_loader = GraphDataLoader(
            SplitDataset(Subset(dataset, val_idx), "val", binary=True, full=True,
                         comparative=False, return_amino_acid=False),
            batch_size=config.batch_size, collate_fn=collate, shuffle=False,
            num_workers=config.num_workers)
        test_loader = GraphDataLoader(
            SplitDataset(Subset(dataset, test_idx), "test", binary=True, full=True,
                         comparative=False, return_amino_acid=False),
            batch_size=config.batch_size, collate_fn=collate, shuffle=False,
            num_workers=config.num_workers)

        os.makedirs(config.save_dir, exist_ok=True)
        best_path = os.path.join(config.save_dir, f"opt_phase2_seed{seed}_best.pt")
        best_val_roc, no_improve = 0.0, 0

        for epoch in range(1, config.epochs + 1):
            t0 = time.time()
            model.train()
            tl = 0.0
            for batch in train_loader:
                graph_data, sequence_data, target, peptide_property = batch[:4]
                sequence_data = sequence_data.to(device)
                target = target.to(device)
                peptide_property = peptide_property.to(device)
                optimizer.zero_grad()
                recon, mu, logvar, output = model(graph_data, sequence_data, peptide_property)
                loss = config.bce_weight * losses.bce(output, target) + \
                    losses.vae_terms(recon, sequence_data, mu, logvar, mask=model.vae.mask)
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tl += loss.item() * len(target)
            scheduler.step()
            val_stats = evaluate(model, val_loader, val_loader, device)
            val_roc = val_stats["roc_auc"]
            print(f"seed {seed} Ep {epoch:3d}/{config.epochs} | loss {tl/len(train_idx):.4f} | "
                  f"val ROC {val_roc:.4f} | {time.time()-t0:.0f}s", flush=True)
            if val_roc > best_val_roc:
                best_val_roc, no_improve = val_roc, 0
                torch.save(model.state_dict(), best_path)
            else:
                no_improve += 1
                if config.early_stop > 0 and no_improve >= config.early_stop:
                    print(f"seed {seed} early stop @ ep {epoch}", flush=True)
                    break

        model.load_state_dict(torch.load(best_path, map_location=device))
        test_stats = evaluate(model, train_loader, test_loader, device)
        print(f"seed {seed} TEST | ROC {test_stats['roc_auc']:.4f} | PR {test_stats['pr_auc']:.4f} | "
              f"Acc@0.5 {test_stats['accuracy']:.4f} | F1@0.5 {test_stats['f1']:.4f} | "
              f"PPVn@0.5(bug) {test_stats['ppvn']:.4f} | PPV30@0.5(bug) {test_stats['ppv30']:.4f}",
              flush=True)
        for k in metric_names:
            results[k].append(test_stats[k])

    print("\n=== AGGREGATE (paper protocol, full evaluate_metrics) ===", flush=True)
    print(f"seeds: {config.seeds}", flush=True)
    for k in metric_names:
        arr = np.array(results[k])
        print(f"{k:12s} {arr.mean():.4f} +/- {arr.std(ddof=1):.4f} (s.d.)", flush=True)

    print("\n--- headline (paper Fig.3 style) ---", flush=True)
    roc_arr = np.array(results["roc_auc"]); pr_arr = np.array(results["pr_auc"])
    ppvn_arr = np.array(results["ppvn"]); ppv30_arr = np.array(results["ppv30"])
    n_seeds = len(config.seeds)
    print(f"AUROC     {roc_arr.mean():.4f} +/- {roc_arr.std(ddof=1):.4f} (s.d.)   [paper 0.882 +/- 0.005]", flush=True)
    print(f"AUPRC     {pr_arr.mean():.4f} +/- {pr_arr.std(ddof=1):.4f} (s.d.)   [paper 0.696 +/- 0.020]", flush=True)
    print(f"mean PPVn {ppvn_arr.mean():.4f} +/- {ppvn_arr.std(ddof=1)/np.sqrt(n_seeds):.4f} (s.e.m.)   [paper ~0.514 +/- 0.020]", flush=True)
    print(f"PPVn n=30 {ppv30_arr.mean():.4f} +/- {ppv30_arr.std(ddof=1)/np.sqrt(n_seeds):.4f} (s.e.m.)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
