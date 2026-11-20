#!/usr/bin/env python3
"""Evaluate the OFFICIAL CEDAR checkpoint (CEDAR_model_seed2.pt) using the
ORIGINAL HybridModel_Comparative architecture (extracted verbatim from the
released code), on the local ESMFold2 graph data.

Reports BOTH the paper's (buggy, binarized) PPVn AND the corrected continuous
PPVn, so the numbers are directly comparable to the paper's CEDAR protocol.

The CEDAR checkpoint was trained with:
  * HybridModel_Comparative (v1), use_wt_for_downstream=False (classifier.1=(32,104))
  * self_attention = SelfAttention (query/key/value), NO combined_attention
  * full_sequence=True  -> input_dim = 283*21 = 5943
  * gcn_layers=5, vae_hidden_dim=512, vae_latent_dim=32, gat_hidden=64, property_emb=8
  * split = torch.random_split([0.8,0.1,0.1], generator seed=2)

Two split protocols:
  * --protocol paper   : torch random_split(0.8/0.1/0.1, seed=2)  (matches the paper)
  * --protocol strat   : stratified holdout, seed 42 (matches our dec3 eval)

Usage:
    python eval_original_cedar.py --ckpt CEDAR_model_seed2.pt --protocol paper
    python eval_original_cedar.py --ckpt CEDAR_model_seed2.pt --protocol strat
"""

import os
import sys
import argparse
import types

# dgl.graphbolt stub (torch 2.5 / dgl 2.1 import guard).
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
from torch.utils.data import Subset, random_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve)

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
_ORIG_CODE = os.path.join(_PROJ_ROOT, "orig_code")
# Insert base first, then orig_code, so orig_code ends up at sys.path[0]:
# `models.*` must resolve to orig_code/models (which has HybridModel_Comparative),
# while data_loading/utils only exist under base.
for _p in (_BASE, _ORIG_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dgl.dataloading import GraphDataLoader
from models.comparative_models import HybridModel_Comparative
from data_loading import ImmunoPredDatasetComparative, collate, SplitDataset
from utils import seed_everything, update_paths


def mean_PPVn(values_true, values_pred, topk=None):
    """BigMHC continuous PPV@n (procedures/metric.py:mean_PPVn, verbatim)."""
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
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def stratified_holdout(labels, n_test, seed):
    labels = np.asarray(labels)
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
def get_predictions_comparative(model, loader, device, mc=1):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        graph_data_pair, sequence_data_pair, target, peptide_property_pair = batch[:4]
        graph_cancer, graph_wt = graph_data_pair
        seq_cancer, seq_wt = sequence_data_pair
        prop_cancer, prop_wt = peptide_property_pair
        seq_cancer, seq_wt = seq_cancer.to(device), seq_wt.to(device)
        prop_cancer, prop_wt = prop_cancer.to(device), prop_wt.to(device)
        probs_sum = None
        for _ in range(mc):
            _, _, _, _, final_output = model.forward_comparative(
                (graph_cancer, graph_wt), (seq_cancer, seq_wt), (prop_cancer, prop_wt))
            p = torch.sigmoid(final_output.squeeze(-1))
            probs_sum = p if probs_sum is None else probs_sum + p
        probs = probs_sum / mc
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(target.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
    return preds, labels


def evaluate_all(true_targets, predicted_probs):
    """Full metric battery: paper's buggy (binarized) PPVn + corrected continuous PPVn."""
    opt = find_optimal_threshold(true_targets, predicted_probs)
    roc_auc = roc_auc_score(true_targets, predicted_probs) if len(np.unique(true_targets)) > 1 else 0.5
    pr = average_precision_score(true_targets, predicted_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(true_targets, predicted_probs)
    pr_auc = auc(recall_curve, precision_curve)

    y05 = predicted_probs >= 0.5
    yop = predicted_probs >= opt

    acc, acc_op = accuracy_score(true_targets, y05), accuracy_score(true_targets, yop)
    f1, f1_op = f1_score(true_targets, y05), f1_score(true_targets, yop)
    prec, prec_op = precision_score(true_targets, y05, zero_division=0), precision_score(true_targets, yop, zero_division=0)
    rec, rec_op = recall_score(true_targets, y05), recall_score(true_targets, yop)

    ppvn_bug = mean_PPVn(true_targets, y05.astype(float))
    ppvn_bug_op = mean_PPVn(true_targets, yop.astype(float))
    ppv30_bug = mean_PPVn(true_targets, y05.astype(float), topk=30)
    ppv30_bug_op = mean_PPVn(true_targets, yop.astype(float), topk=30)

    ppvn_cont = mean_PPVn(true_targets, predicted_probs)
    ppv30_cont = mean_PPVn(true_targets, predicted_probs, topk=30)

    return dict(opt=opt, roc_auc=roc_auc, pr=pr, pr_auc=pr_auc,
                acc=acc, acc_op=acc_op, f1=f1, f1_op=f1_op,
                prec=prec, prec_op=prec_op, rec=rec, rec_op=rec_op,
                ppvn_bug=ppvn_bug, ppvn_bug_op=ppvn_bug_op,
                ppv30_bug=ppv30_bug, ppv30_bug_op=ppv30_bug_op,
                ppvn_cont=ppvn_cont, ppv30_cont=ppv30_cont)


def main():
    p = argparse.ArgumentParser(description="Evaluate official CEDAR checkpoint (original HybridModel_Comparative)")
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--graph-dir-cancer", default=f"{_PROJ_ROOT}/data/graph_pyg_CEDAR_cancer_esmfold2/")
    p.add_argument("--graph-dir-wt", default=f"{_PROJ_ROOT}/data/graph_pyg_CEDAR_wildtype_esmfold2/")
    p.add_argument("--property-path-cancer", default=f"{_PROJ_ROOT}/data/ImmunoStruct_CEDAR_data_cancer.csv")
    p.add_argument("--property-path-wt", default=f"{_PROJ_ROOT}/data/ImmunoStruct_CEDAR_data_wildtype.csv")
    p.add_argument("--hla-path", default=f"{_PROJ_ROOT}/data/HLA_allele_sequences.csv")
    p.add_argument("--vae-hidden-dim", default=512, type=int)
    p.add_argument("--vae-latent-dim", default=32, type=int)
    p.add_argument("--gat-hidden", default=64, type=int)
    p.add_argument("--gnn-layers", default=5, type=int)
    p.add_argument("--property-emb-dim", default=8, type=int)
    p.add_argument("--protocol", default="paper", choices=["paper", "strat"])
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--mc", default=1, type=int, help="1 = paper single-forward; 20 = mc20 stable")
    p.add_argument("--eval-seed", default=0, type=int)
    p.add_argument("--test-seed", default=42, type=int)
    p.add_argument("--test-size", default=300, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--feature-size", default=23, type=int)
    p.add_argument("--coord-size", default=3, type=int)
    p.add_argument("--sequence-pad-count", default=0, type=int)
    p.add_argument("--structure-pad-count", default=0, type=int)
    p.add_argument("--device", default="cpu")
    config = p.parse_args()
    update_paths(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seed_everything(config.eval_seed)

    dataset = ImmunoPredDatasetComparative(config,
                                           graph_directory_cancer=config.graph_dir_cancer,
                                           graph_directory_wt=config.graph_dir_wt,
                                           property_path_cancer=config.property_path_cancer,
                                           property_path_wt=config.property_path_wt,
                                           hla_path=config.hla_path)
    input_dim = 283 * 21
    labels_all = dataset.immunogenicity_cancer.numpy()
    print(f"CEDAR n={len(dataset)} input_dim={input_dim}", flush=True)

    # ---- test split ----
    if config.protocol == "paper":
        gen = torch.Generator().manual_seed(2)
        _, _, test_ds = random_split(dataset, [0.8, 0.1, 0.1], generator=gen)
        test_idx = np.array(test_ds.indices)
        n_pos = int((labels_all[test_idx] > 0.5).sum())
        print(f"split: paper random_split(seed=2)  test n={len(test_idx)} pos={n_pos} "
              f"neg={len(test_idx)-n_pos} pos_rate={n_pos/len(test_idx):.4f}", flush=True)
    else:
        test_idx = stratified_holdout(labels_all, config.test_size, config.test_seed)
        n_pos = int((labels_all[test_idx] > 0.5).sum())
        print(f"split: stratified holdout(seed={config.test_seed}) test n={len(test_idx)} "
              f"pos={n_pos} neg={len(test_idx)-n_pos} pos_rate={n_pos/len(test_idx):.4f}", flush=True)

    # ---- model (original HybridModel_Comparative; checkpoint self_attention uses
    #      query/key/value + no combined_attention, i.e. the v1 comparative model) ----
    model = HybridModel_Comparative(input_dim, device,
                                    gcn_layers=config.gnn_layers,
                                    vae_hidden_dim=config.vae_hidden_dim,
                                    vae_latent_dim=config.vae_latent_dim,
                                    gat_hidden_channels=config.gat_hidden,
                                    property_embedding_dim=config.property_emb_dim,
                                    use_wt_for_downstream=False).to(device)
    ck = torch.load(config.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(ck, strict=False)
    print(f"ckpt={config.ckpt}", flush=True)
    print(f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("  missing keys:", missing[:10], flush=True)
    if unexpected:
        print("  unexpected keys:", unexpected[:10], flush=True)

    test_loader = GraphDataLoader(
        SplitDataset(Subset(dataset, test_idx.tolist()), "test", binary=True,
                     full=True, comparative=True, return_amino_acid=False),
        batch_size=config.batch_size, collate_fn=collate, shuffle=False,
        num_workers=config.num_workers)

    preds, labels = get_predictions_comparative(model, test_loader, device, mc=config.mc)
    m = evaluate_all(labels, preds)

    print(f"\n=== original HybridModel_Comparative ({os.path.basename(config.ckpt)}) mc={config.mc} "
          f"protocol={config.protocol} ===", flush=True)
    print(f"ROC-AUC    {m['roc_auc']:.4f}", flush=True)
    print(f"PR (avg)   {m['pr']:.4f}", flush=True)
    print(f"PR-AUC     {m['pr_auc']:.4f}", flush=True)
    print(f"Acc@0.5    {m['acc']:.4f}", flush=True)
    print(f"F1@0.5     {m['f1']:.4f}", flush=True)
    print(f"Prec@0.5   {m['prec']:.4f}", flush=True)
    print(f"Rec@0.5    {m['rec']:.4f}", flush=True)
    print(f"opt-thr    {m['opt']:.4f}", flush=True)
    print(f"PPVn BUG @0.5  {m['ppvn_bug']:.4f}", flush=True)
    print(f"PPV30 BUG @0.5 {m['ppv30_bug']:.4f}", flush=True)
    print(f"PPVn CONT      {m['ppvn_cont']:.4f}   (corrected)", flush=True)
    print(f"PPV30 CONT     {m['ppv30_cont']:.4f}   (corrected)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
