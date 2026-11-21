#!/usr/bin/env python3
"""
Allele-generalization breakdown on the ITSNdb external benchmark.

Splits samples by whether their HLA allele was SEEN (in the IEDB training set,
27 alleles) or OOD (unseen, incl. all HLA-C and e.g. B*27:05), then reports
ROC/PR/PPVn per group for both v2 and v6.

This directly tests whether v6's cross-attention actually improved *allele*
generalization (the OOD group), as opposed to *peptide* generalization
(seen-allele group) which Val_dataset mostly measures.

Usage:
    python analyze_itsndb_allele_ood.py --seeds 42 123 456 789 --gpu 0
"""
import os
import sys
import csv
import types
import argparse
import logging

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
from sklearn.metrics import roc_auc_score, average_precision_score

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from models.hybrid_models import HybridModelv6
from procedures.metric import mean_PPVn
from eval_v6_itsndb import (load_itsndb_val, load_itsndb_full, load_graphs,
                            preprocess_graph_to_dgl, ITSNdbV6Dataset, v6_collate,
                            normalize_allele, SEQ_EMB_DIM)

IEDB_HLA = os.path.join(_PROJ_ROOT, "data", "HLA_allele_sequences.csv")
V2_PRED_CSV = os.path.join(_PROJ_ROOT, "..", "ImmunoStruct_Phase2_Repro",
                           "results", "eval_itsndb_predictions.csv")


def iedb_alleles():
    with open(IEDB_HLA) as f:
        return {r["allele"].strip() for r in csv.DictReader(f)}


def load_v2_preds():
    rows = list(csv.DictReader(open(V2_PRED_CSV)))
    preds = {f"{r['hla']}_{r['peptide']}": float(r["prediction"]) for r in rows}
    return preds


def load_v6_preds(samples, graph_map, pep_perres, hla_perres, seeds, save_dir, device):
    dataset = ITSNdbV6Dataset(samples, graph_map, pep_perres, hla_perres)
    idx = list(range(len(dataset)))
    # sum of sigmoid probs across seeds (ensemble)
    acc = np.zeros(len(dataset))
    for seed in seeds:
        ckpt = os.path.join(save_dir, f"seed_{seed}", "v6_phase2_best.pt")
        model = HybridModelv6(
            seq_emb_dim=SEQ_EMB_DIM, proj_dim=256, gnn_dim=64, gnn_layers=6,
            n_heads=4, dropout=0.1, device=device, property_embedding_dim=16,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(idx), 150):
                items = [(dataset.graphs[j], dataset.peps[j], dataset.hlas[j],
                          dataset.properties[j], dataset.labels[j]) for j in idx[i:i + 150]]
                g, pep, pep_mask, hla, hla_mask, prop, labels = v6_collate(items)
                out = model(g, pep.to(device), hla.to(device), pep_mask.to(device),
                            hla_mask.to(device), prop.to(device)).squeeze(-1)
                acc[idx[i:i + 150]] += torch.sigmoid(out).cpu().numpy()
        del model
        torch.cuda.empty_cache()
    acc = acc / len(seeds)
    return {f"{s['hla']}_{s['peptide']}": acc[i] for i, s in enumerate(samples)}


def report(name, labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    n = len(labels)
    n_pos = int(labels.sum())
    if n == 0:
        print(f"  {name:14s} n=0")
        return
    if n_pos == 0 or n_pos == n:
        print(f"  {name:14s} n={n:3d} pos={n_pos:3d}  ROC=--  PR=--  (degenerate)")
        return
    roc = roc_auc_score(labels, preds)
    pr = average_precision_score(labels, preds)
    ppvn = mean_PPVn(labels, preds)
    print(f"  {name:14s} n={n:3d} pos={n_pos:3d}  ROC={roc:.4f}  PR={pr:.4f}  PPVn={ppvn:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/v6_full")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger(__name__)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    iedb = iedb_alleles()

    # samples (deduped both)
    samples = load_itsndb_val() + load_itsndb_full()
    seen = set(); dedup = []
    for s in samples:
        k = (s["hla"], s["peptide"])
        if k not in seen:
            seen.add(k); dedup.append(s)
    samples = dedup

    graph_map = load_graphs(os.path.join(_PROJ_ROOT, "..", "ImmunoStruct_Phase2_Repro",
                                         "data", "graph_pyg_ITSNdb_esmfold2"))
    pep_perres = torch.load(f"{_PROJ_ROOT}/data/esmc_peptide_perres_itsndb.pt", map_location="cpu")
    hla_perres = torch.load(f"{_PROJ_ROOT}/data/esmc_hla_pseudo_perres_itsndb.pt", map_location="cpu")

    valid = [s for s in samples
             if f"{s['hla']}_{s['peptide']}" in graph_map
             and s["peptide"] in pep_perres and s["hla"] in hla_perres]
    logger.info(f"valid {len(valid)} samples")

    v2 = load_v2_preds()
    logger.info("computing v6 ensemble predictions ...")
    v6 = load_v6_preds(valid, graph_map, pep_perres, hla_perres, args.seeds,
                       args.save_dir, device)

    # assemble per-sample aligned arrays
    keys = [f"{s['hla']}_{s['peptide']}" for s in valid]
    labels = np.array([s["label"] for s in valid])
    ood = np.array([s["hla"] not in iedb for s in valid])
    v6p = np.array([v6[k] for k in keys])
    v2p = np.array([v2[k] for k in keys])

    print("\n================ ITSNdb ALLELE-GENERALIZATION BREAKDOWN ================")
    print(f"  IEDB seen alleles: {len(iedb)} | OOD samples: {int(ood.sum())} / {len(valid)}")
    print(f"  {'':14s} {'v2':>34s}  {'v6':>34s}")
    for grp_name, grp_mask in [("ALL", np.ones(len(valid), bool)),
                               ("SEEN-allele", ~ood),
                               ("OOD-allele", ood)]:
        l = labels[grp_mask]
        r2 = v2p[grp_mask]
        r6 = v6p[grp_mask]
        n_pos = int(l.sum())
        r2s = f"ROC={roc_auc_score(l,r2):.4f} PR={average_precision_score(l,r2):.4f}" if 0 < n_pos < len(l) else "--"
        r6s = f"ROC={roc_auc_score(l,r6):.4f} PR={average_precision_score(l,r6):.4f}" if 0 < n_pos < len(l) else "--"
        print(f"  {grp_name:14s} n={len(l):3d} pos={n_pos:3d}  {r2s:34s}  {r6s}")

    # OOD sub-breakdown by allele
    print("\n  --- OOD samples per allele (v6 ensemble) ---")
    for a in sorted(set(s["hla"] for s, m in zip(valid, ood) if m)):
        m = np.array([(s["hla"] == a and o) for s, o in zip(valid, ood)])
        l = labels[m]; r6 = v6p[m]
        tag = f"{a}"
        if 0 < int(l.sum()) < len(l):
            print(f"    {tag:16s} n={len(l):3d} pos={int(l.sum()):3d}  "
                  f"ROC={roc_auc_score(l, r6):.4f} PR={average_precision_score(l, r6):.4f}")
        else:
            print(f"    {tag:16s} n={len(l):3d} pos={int(l.sum()):3d}  (degenerate)")

    print("=" * 72)


if __name__ == "__main__":
    main()
