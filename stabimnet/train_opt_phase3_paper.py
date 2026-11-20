#!/usr/bin/env python3
"""CEDAR (Stage-3) cancer/wildtype CONTRASTIVE finetuning, mirroring the
Phase-2 paper protocol (train_opt_phase2_paper.py).

Each seed (42-46) warm-starts the shared ConvVAE+EGNN backbone of
``HybridModelOpt_Comparative`` from the matching Phase-2 checkpoint
``results/opt_phase2_paper/opt_phase2_seed{seed}_best.pt``, then finetunes the
mutation-delta classifier + a temperature-scaled contrastive objective on CEDAR.

Training strategy matches train_opt_phase2_paper.py exactly:
  * stratified 80/10/10 split PER SEED (sklearn train_test_split, stratify=labels,
    random_state=seed), so every train/val/test fold keeps the ~19% positive rate.
  * warmup(10) + cosine scheduler, AdamW, early-stop 20 on val ROC.
  * full paper metric protocol on TEST, PLUS the corrected CONTINUOUS PPVn
    alongside the paper's binarized "bug" PPVn.

Model dims MUST match the Phase-2 checkpoint (vae_hidden_dim=768, vae_latent_dim=64,
property_emb_dim=16, gat_hidden=64, decoder_layers=3).  decoder_layers=3 is
critical: the Phase-2 checkpoint has vae.dec_hidden.0/.1 (decoder_layers=3), so
the comparative model must use the same value or transfer_shared_from_opt will
drop those tensors.

Usage:
    python train_opt_phase3_paper.py --seeds 42 43 44 45 46 --device cuda:0
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

import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve)

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from dgl.dataloading import GraphDataLoader
from models.optimized_models import HybridModelOpt_Comparative, transfer_shared_from_opt
from data_loading import ImmunoPredDatasetComparative, collate, SplitDataset
from utils import seed_everything, update_paths
from utils.optimized_loss import OptimizedLosses


def mean_PPVn(values_true, values_pred, topk=None):
    """Exact copy of procedures/metric.py:mean_PPVn (BigMHC PPV@n ranking metric)."""
    assert len(values_true) == len(values_pred)
    sorting_idx = np.argsort(values_pred)[::-1]
    values_true = values_true[sorting_idx]
    cum_true_positives = np.cumsum(values_true)
    total_predictions = np.arange(1, len(values_true) + 1)
    ppvn = cum_true_positives / total_predictions
    num_positives = int(values_true.sum())
    if num_positives == 0:
        return 0.0
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
    return thresholds[optimal_idx]


def stratified_split(labels, fractions=(0.8, 0.1, 0.1), seed=0):
    """Stratified 80/10/10 split on the label array (positives/negatives split
    separately so every fold keeps the dataset class ratio).  Deterministic in seed."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(idx, test_size=1.0 - fractions[0],
                                           stratify=labels, random_state=seed)
    val_frac = fractions[1] / (fractions[1] + fractions[2])
    val_idx, test_idx = train_test_split(rest_idx, test_size=1.0 - val_frac,
                                         stratify=labels[rest_idx], random_state=seed)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


@torch.no_grad()
def inference(model, loader, device, mc=1):
    """Comparative forward with Monte-Carlo averaging over the VAE latent z."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        graph_pair, seq_pair, target, prop_pair = batch[:4]
        gc, gw = graph_pair
        sc, sw = seq_pair
        pc, pw = prop_pair
        sc, sw = sc.to(device), sw.to(device)
        pc, pw = pc.to(device), pw.to(device)
        probs_sum = None
        for _ in range(mc):
            _, _, _, _, output = model.forward_comparative((gc, gw), (sc, sw), (pc, pw))
            p = torch.sigmoid(output.squeeze(-1))
            probs_sum = p if probs_sum is None else probs_sum + p
        p = (probs_sum / mc)
        p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
        ps.append(p.cpu().numpy())
        ys.append(target.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    ps = np.nan_to_num(ps, nan=0.5, posinf=1.0, neginf=0.0)
    return ys, ps


def evaluate_metrics(y_true, y_prob, opt):
    """Full paper 14-metric battery (binarized "bug" PPVn) + corrected continuous PPVn."""
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    y05 = y_prob >= 0.5
    yop = y_prob >= opt

    acc, acc_op = accuracy_score(y_true, y05), accuracy_score(y_true, yop)
    f1, f1_op = f1_score(y_true, y05), f1_score(y_true, yop)
    prec, prec_op = precision_score(y_true, y05, zero_division=0), precision_score(y_true, yop, zero_division=0)
    rec, rec_op = recall_score(y_true, y05), recall_score(y_true, yop)

    # paper bug version (binarized before ranking)
    ppvn_bug = mean_PPVn(y_true, y05.astype(float))
    ppvn_bug_op = mean_PPVn(y_true, yop.astype(float))
    ppv30_bug = mean_PPVn(y_true, y05.astype(float), topk=30)
    ppv30_bug_op = mean_PPVn(y_true, yop.astype(float), topk=30)

    # corrected continuous version (continuous probabilities before ranking)
    ppvn_cont = mean_PPVn(y_true, y_prob)
    ppv30_cont = mean_PPVn(y_true, y_prob, topk=30)

    return dict(roc_auc=roc_auc, pr_auc=pr_auc,
                acc=acc, acc_op=acc_op, f1=f1, f1_op=f1_op,
                prec=prec, prec_op=prec_op, rec=rec, rec_op=rec_op,
                ppvn_bug=ppvn_bug, ppvn_bug_op=ppvn_bug_op,
                ppv30_bug=ppv30_bug, ppv30_bug_op=ppv30_bug_op,
                ppvn_cont=ppvn_cont, ppv30_cont=ppv30_cont)


def main():
    p = argparse.ArgumentParser(description="CEDAR contrastive finetune under paper protocol (stratified split per seed)")
    p.add_argument("--graph-dir-cancer", default=f"{_PROJ_ROOT}/data/graph_pyg_CEDAR_cancer_esmfold2/")
    p.add_argument("--graph-dir-wildtype", default=f"{_PROJ_ROOT}/data/graph_pyg_CEDAR_wildtype_esmfold2/")
    p.add_argument("--property-path-cancer", default=f"{_PROJ_ROOT}/data/ImmunoStruct_CEDAR_data_cancer.csv")
    p.add_argument("--property-path-wildtype", default=f"{_PROJ_ROOT}/data/ImmunoStruct_CEDAR_data_wildtype.csv")
    p.add_argument("--hla-path", default=f"{_PROJ_ROOT}/data/HLA_allele_sequences.csv")
    p.add_argument("--phase2-dir", default=f"{_PROJ_ROOT}/results/opt_phase2_paper", type=str,
                   help="dir containing opt_phase2_seed{seed}_best.pt checkpoints")
    p.add_argument("--vae-hidden-dim", default=768, type=int)
    p.add_argument("--vae-latent-dim", default=64, type=int)
    p.add_argument("--gat-hidden", default=64, type=int)
    p.add_argument("--gnn-layers", default=5, type=int)
    p.add_argument("--property-emb-dim", default=16, type=int)
    p.add_argument("--egnn-type", default="opt", type=str, choices=["opt", "torch"])
    p.add_argument("--decoder-layers", default=3, type=int,
                   help="MUST be 3 to match the Phase-2 checkpoint (vae.dec_hidden.0/.1)")
    p.add_argument("--mask-ratio", default=0.0, type=float)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--epochs", default=40, type=int)
    p.add_argument("--warmup-epochs", default=10, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--weight-decay", default=1e-6, type=float)
    p.add_argument("--bce-weight", default=2.0, type=float)
    p.add_argument("--mse-weight", default=0.5, type=float)
    p.add_argument("--kld-weight", default=0.5, type=float)
    p.add_argument("--focal-gamma", default=1.0, type=float)
    p.add_argument("--free-bits", default=0.0, type=float)
    p.add_argument("--coeff-contrastive", default=0.01, type=float)
    p.add_argument("--eval-mc", default=100, type=int, help="MC averaging for final TEST eval")
    p.add_argument("--early-stop", default=20, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="train/val/test fractions (must sum to 1)")
    p.add_argument("--feature-size", default=23, type=int)
    p.add_argument("--coord-size", default=3, type=int)
    p.add_argument("--sequence-pad-count", default=0, type=int)
    p.add_argument("--structure-pad-count", default=0, type=int)
    p.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/opt_phase3_paper")
    p.add_argument("--device", default="cuda:0")
    config = p.parse_args()
    update_paths(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    dataset = ImmunoPredDatasetComparative(config,
                                           graph_directory_cancer=config.graph_dir_cancer,
                                           graph_directory_wt=config.graph_dir_wildtype,
                                           property_path_cancer=config.property_path_cancer,
                                           property_path_wt=config.property_path_wildtype,
                                           hla_path=config.hla_path)
    vae_input_dim = dataset.encoded_full_seq_cancer.shape[1] * dataset.encoded_full_seq_cancer.shape[2]
    labels_all = dataset.immunogenicity_cancer.numpy()
    n_total = len(dataset)
    print(f"CEDAR n={n_total} input_dim={vae_input_dim}", flush=True)

    metric_names = ["roc_auc", "pr_auc", "acc", "acc_op", "f1", "f1_op",
                    "prec", "prec_op", "rec", "rec_op",
                    "ppvn_bug", "ppvn_bug_op", "ppv30_bug", "ppv30_bug_op",
                    "ppvn_cont", "ppv30_cont"]
    results = {k: [] for k in metric_names}
    os.makedirs(config.save_dir, exist_ok=True)

    for seed in config.seeds:
        seed_everything(seed)
        train_idx, val_idx, test_idx = stratified_split(labels_all, fractions=config.split, seed=seed)
        print(f"\n=== seed {seed} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} ===", flush=True)
        for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            n_pos = int(labels_all[idx].sum())
            print(f"  {name}: n={len(idx)} pos={n_pos} neg={len(idx) - n_pos} "
                  f"pos_rate={n_pos / len(idx):.4f}", flush=True)

        model = HybridModelOpt_Comparative(vae_input_dim, device,
                                           gcn_layers=config.gnn_layers,
                                           vae_hidden_dim=config.vae_hidden_dim,
                                           vae_latent_dim=config.vae_latent_dim,
                                           gat_hidden_channels=config.gat_hidden,
                                           property_embedding_dim=config.property_emb_dim,
                                           dropout=config.dropout,
                                           egnn_type=config.egnn_type,
                                           decoder_layers=config.decoder_layers,
                                           mask_ratio=config.mask_ratio).to(device)
        phase2_ckpt = os.path.join(config.phase2_dir, f"opt_phase2_seed{seed}_best.pt")
        if os.path.exists(phase2_ckpt):
            transfer_shared_from_opt(model, phase2_ckpt, map_location=device)
        else:
            print(f"[warn] missing phase2 checkpoint {phase2_ckpt}; training from scratch", flush=True)

        losses = OptimizedLosses(vae_input_dim, dataset.class_weights,
                                 bce_weight=config.bce_weight, mse_weight=config.mse_weight,
                                 kld_weight=config.kld_weight, focal_gamma=config.focal_gamma,
                                 free_bits=config.free_bits,
                                 contrastive_embed_dim=config.gat_hidden + config.vae_latent_dim)
        losses.contrastive = losses.contrastive.to(device)

        optimizer = torch.optim.AdamW(list(model.parameters()) + list(losses.contrastive.parameters()),
                                      lr=config.lr, weight_decay=config.weight_decay)
        if config.warmup_epochs > 0:
            cosine_epochs = max(config.epochs - config.warmup_epochs, 1)
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        def make_loader(idx, name, shuffle):
            return GraphDataLoader(
                SplitDataset(Subset(dataset, idx), name, binary=True, full=True,
                             comparative=True, return_amino_acid=False),
                batch_size=config.batch_size, collate_fn=collate, shuffle=shuffle,
                num_workers=config.num_workers)

        train_loader = make_loader(train_idx, "train", True)
        val_loader = make_loader(val_idx, "val", False)
        test_loader = make_loader(test_idx, "test", False)

        best_path = os.path.join(config.save_dir, f"opt_phase3_seed{seed}_best.pt")
        best_val_roc, no_improve = 0.0, 0

        for epoch in range(1, config.epochs + 1):
            t0 = time.time()
            model.train()
            tl = 0.0
            for batch in train_loader:
                graph_pair, seq_pair, target, prop_pair = batch[:4]
                gc, gw = graph_pair
                sc, sw = seq_pair
                pc, pw = prop_pair
                sc, sw = sc.to(device), sw.to(device)
                pc, pw = pc.to(device), pw.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                (emb_c, emb_w), (recon_c, recon_w), (mu_c, mu_w), (logvar_c, logvar_w), output = \
                    model.forward_comparative((gc, gw), (sc, sw), (pc, pw))

                bce = losses.bce(output, target)
                recon = 0.5 * (losses.recon(recon_c, sc) + losses.recon(recon_w, sw))
                kld = 0.5 * (losses.kld(mu_c, logvar_c) + losses.kld(mu_w, logvar_w))
                loss = config.bce_weight * bce + config.mse_weight * recon + config.kld_weight * kld
                if config.coeff_contrastive > 0:
                    loss = loss + config.coeff_contrastive * losses.contrastive(emb_c, emb_w, target)

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tl += loss.item() * len(target)
            scheduler.step()

            val_y, val_p = inference(model, val_loader, device, mc=1)
            val_roc = roc_auc_score(val_y, val_p) if len(np.unique(val_y)) > 1 else 0.5
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
        # optimal threshold from TRAIN (Youden's J, mc=1), then full metrics on TEST (mc=eval_mc)
        train_y, train_p = inference(model, train_loader, device, mc=1)
        opt = find_optimal_threshold(train_y, train_p)
        test_y, test_p = inference(model, test_loader, device, mc=config.eval_mc)
        m = evaluate_metrics(test_y, test_p, opt)
        print(f"seed {seed} TEST | ROC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} | "
              f"PPVn BUG@0.5 {m['ppvn_bug']:.4f} | PPV30 BUG@0.5 {m['ppv30_bug']:.4f} | "
              f"PPVn CONT {m['ppvn_cont']:.4f} | PPV30 CONT {m['ppv30_cont']:.4f}", flush=True)
        for k in metric_names:
            results[k].append(m[k])

    print("\n=== AGGREGATE (CEDAR contrastive, paper protocol, 5 seeds) ===", flush=True)
    print(f"seeds: {config.seeds}", flush=True)
    for k in metric_names:
        arr = np.array(results[k])
        print(f"{k:12s} {arr.mean():.4f} +/- {arr.std(ddof=1):.4f} (s.d.)", flush=True)

    print("\n--- headline (bug PPVn vs continuous PPVn) ---", flush=True)
    for k in ["ppvn_bug", "ppvn_bug_op", "ppv30_bug", "ppv30_bug_op", "ppvn_cont", "ppv30_cont"]:
        arr = np.array(results[k])
        print(f"{k:14s} {arr.mean():.4f} +/- {arr.std(ddof=1)/np.sqrt(len(config.seeds)):.4f} (s.e.m.)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
