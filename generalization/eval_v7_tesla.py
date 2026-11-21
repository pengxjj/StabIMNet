#!/usr/bin/env python3
"""
Evaluate HybridModelv7 (full-MHC-sequence cross-attention) on the TESLA
external benchmark (Wells et al. 2020 Cell).

Same as eval_v6_tesla.py but the HLA key is the FULL MHC class-I sequence
(~272-276 aa) per-residue ESMC embedding instead of the 34-residue pseudo-seq:
peptide [Lp,2560], HLA [Lm,2560] (padded to per-batch Lm_max).

Reports 608 (tested) and 310 (validation) separately + seen/unseen breakdown.

Usage:
    python eval_v7_tesla.py                          # both splits, all 4 seeds
    python eval_v7_tesla.py --split 608
    python eval_v7_tesla.py --seeds 42
"""

import os
import sys
import csv
import types
import argparse
import logging

# DGL compatibility patches (mirror training scripts)
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
import dgl
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, f1_score, precision_score,
                              recall_score, matthews_corrcoef)

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from models.hybrid_models import HybridModelv7
from procedures.metric import mean_PPVn

TESLA_DIR = os.path.join(_PROJ_ROOT, "data", "TESLA")
GRAPH_DIR = os.path.join(TESLA_DIR, "graph_pyg_tesla_esmfold2")
DATA_DIR = os.path.join(_PROJ_ROOT, "data")
IEDB_PSEUDO_CSV = os.path.join(DATA_DIR, "HLA_pseudo_sequences.csv")  # 27 seen alleles

SEQ_EMB_DIM = 2560
DEFAULT_SEEDS = [42, 123, 456, 789]
DEFAULT_MPROP1 = 0.38
DEFAULT_MPROP2 = 0.42


def load_seen_alleles():
    with open(IEDB_PSEUDO_CSV) as f:
        return {r["allele"].strip() for r in csv.DictReader(f)}


def load_tesla(split):
    samples = []
    with open(os.path.join(TESLA_DIR, f"tesla_{split}.csv")) as f:
        for row in csv.DictReader(f):
            samples.append({
                "hla": row["hla"].strip(),
                "peptide": row["peptide"].strip(),
                "label": float(row["label"]),
                "patient": row["patient"].strip(),
            })
    return samples


def load_graphs(graph_dir):
    graph_map = {}
    for fname in os.listdir(graph_dir):
        if not fname.endswith(".pt"):
            continue
        g = torch.load(os.path.join(graph_dir, fname))
        graph_map[g.name] = g
    return graph_map


def preprocess_graph_to_dgl(graph, max_nodes, feature_size=23):
    if graph.x.shape[1] == 22:
        graph.x = graph.x[:, :-2]                       # strip 2 H-bond cols
    graph.x = torch.cat([graph.x, graph.coords], dim=-1)
    graph.x = graph.x.to(dtype=torch.float32)

    n_pad = max_nodes - graph.num_nodes
    if n_pad > 0:
        zero_feat = torch.zeros(n_pad, feature_size, dtype=torch.float32)
        graph.x = torch.cat([graph.x, zero_feat], dim=0)
        graph.num_nodes = max_nodes

    num_edges = graph.edge_index.size(1)
    src, dst = graph.edge_index
    dgl_graph = dgl.graph((src, dst), num_nodes=graph.num_nodes)
    dgl_graph.ndata["x"] = graph.x
    dgl_graph.edata["edge_attr"] = torch.ones(num_edges, 1)
    return dgl_graph


class TeslaV7Dataset:
    def __init__(self, samples, graph_map, pep_perres, hla_perres):
        self.samples = samples
        self.graphs = []
        self.peps = []
        self.hlas = []
        self.properties = []
        self.labels = []

        max_nodes = max(g.num_nodes for g in graph_map.values())

        for s in samples:
            graph_name = f"{s['hla']}_{s['peptide']}"
            pyg = graph_map.get(graph_name)
            if pyg is None:
                raise ValueError(f"No graph for {graph_name}")
            self.graphs.append(preprocess_graph_to_dgl(pyg, max_nodes))
            self.peps.append(pep_perres[s["peptide"]])
            self.hlas.append(hla_perres[s["hla"]])
            self.properties.append(torch.tensor([DEFAULT_MPROP1, DEFAULT_MPROP2],
                                                dtype=torch.float32))
            self.labels.append(s["label"])

    def __len__(self):
        return len(self.samples)


def v7_collate(items):
    graphs = [it[0] for it in items]
    peps = [it[1] for it in items]
    hlas = [it[2] for it in items]
    props = torch.stack([it[3] for it in items], dim=0)
    labels = torch.tensor([it[4] for it in items], dtype=torch.float32)

    batched_graph = dgl.batch(graphs)

    B = len(items)
    Lp_max = max(p.shape[0] for p in peps)
    pep = torch.zeros(B, Lp_max, SEQ_EMB_DIM)
    pep_mask = torch.zeros(B, Lp_max, dtype=torch.bool)
    for i, p in enumerate(peps):
        L = p.shape[0]
        pep[i, :L] = p
        pep_mask[i, :L] = True

    Lm_max = max(h.shape[0] for h in hlas)              # variable-length full MHC
    hla = torch.zeros(B, Lm_max, SEQ_EMB_DIM)
    hla_mask = torch.zeros(B, Lm_max, dtype=torch.bool)
    for i, h in enumerate(hlas):
        L = h.shape[0]
        hla[i, :L] = h
        hla_mask[i, :L] = True

    return batched_graph, pep, pep_mask, hla, hla_mask, props, labels


def evaluate(model, dataset, device, batch_size=150):
    model.eval()
    all_preds, all_labels = [], []
    idx = list(range(len(dataset)))
    for i in range(0, len(idx), batch_size):
        items = [(dataset.graphs[j], dataset.peps[j], dataset.hlas[j],
                  dataset.properties[j], dataset.labels[j]) for j in idx[i:i + batch_size]]
        g, pep, pep_mask, hla, hla_mask, prop, labels = v7_collate(items)
        with torch.no_grad():
            out = model(g, pep.to(device), hla.to(device),
                        pep_mask.to(device), hla_mask.to(device),
                        prop.to(device)).squeeze(-1)
            preds = torch.sigmoid(out).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)

    if len(np.unique(labels)) < 2:
        return {"preds": preds, "labels": labels,
                "roc_auc": float("nan"), "pr_auc": float("nan"),
                "ppvn_cont": float("nan"), "ppvn_prec": float("nan")}

    pred_bin = (preds >= 0.5).astype(int)
    return {
        "preds": preds, "labels": labels,
        "roc_auc": roc_auc_score(labels, preds),
        "pr_auc": average_precision_score(labels, preds),
        "accuracy": accuracy_score(labels, pred_bin),
        "f1": f1_score(labels, pred_bin, zero_division=0),
        "precision": precision_score(labels, pred_bin, zero_division=0),
        "recall": recall_score(labels, pred_bin, zero_division=0),
        "mcc": matthews_corrcoef(labels, pred_bin),
        "ppvn_cont": mean_PPVn(labels, preds),
        "ppvn_prec": precision_score(labels, pred_bin, zero_division=0),
    }


def summarize_metrics(per_seed):
    def agg(key):
        vals = np.array([m[key] for m in per_seed.values()], dtype=float)
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        return mu, sd
    out = {}
    for key in ["roc_auc", "pr_auc", "ppvn_cont", "ppvn_prec"]:
        mu, sd = agg(key)
        out[key] = f"{mu:.4f} +/- {sd:.4f}"
    return out


def run_split(split, samples, graph_map, pep_perres, hla_perres, seen_alleles,
              seeds, save_dir, ckpt_name, device, batch_size, logger):
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  TESLA {split} — {len(samples)} samples "
                f"({int(sum(s['label'] for s in samples))} pos)")
    logger.info(f"{'=' * 64}")

    # Deduplicate identical (hla, peptide) pairs (see eval_v6_tesla.py).
    seen_keys, deduped = {}, []
    for s in samples:
        key = (s["hla"], s["peptide"])
        if key in seen_keys:
            if seen_keys[key]["label"] != s["label"]:
                logger.warning(f"dup {key} conflicts: "
                               f"{seen_keys[key]['label']} vs {s['label']} — kept first")
            continue
        seen_keys[key] = s
        deduped.append(s)
    if len(deduped) < len(samples):
        logger.info(f"Deduped {len(samples)} -> {len(deduped)} unique (hla, peptide) pairs")
    samples = deduped

    valid, skipped = [], 0
    for s in samples:
        gname = f"{s['hla']}_{s['peptide']}"
        if gname not in graph_map or s["peptide"] not in pep_perres or s["hla"] not in hla_perres:
            logger.warning(f"Skip {gname} (graph={gname in graph_map}, "
                           f"pep={s['peptide'] in pep_perres}, hla={s['hla'] in hla_perres})")
            skipped += 1
            continue
        valid.append(s)
    logger.info(f"Valid: {len(valid)} (skipped {skipped})")

    dataset = TeslaV7Dataset(valid, graph_map, pep_perres, hla_perres)

    per_seed = {}
    for seed in seeds:
        ckpt = os.path.join(save_dir, f"seed_{seed}", ckpt_name)
        if not os.path.exists(ckpt):
            logger.warning(f"[seed {seed}] checkpoint not found: {ckpt} — skip")
            continue
        model = HybridModelv7(
            seq_emb_dim=SEQ_EMB_DIM, proj_dim=256, gnn_dim=64, gnn_layers=6,
            n_heads=4, dropout=0.1, device=device, property_embedding_dim=16,
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

        m = evaluate(model, dataset, device, batch_size)
        per_seed[seed] = m
        logger.info(f"[seed {seed}] ROC {m['roc_auc']:.4f} | PR {m['pr_auc']:.4f} | "
                    f"PPVn_cont {m['ppvn_cont']:.4f} | PPVn_prec {m['ppvn_prec']:.4f} "
                    f"(n={len(m['labels'])})")
        del model
        torch.cuda.empty_cache()

    if not per_seed:
        logger.error("No checkpoint evaluated — is training finished?")
        return

    logger.info(f"\n  TESLA {split} — aggregate ({len(per_seed)} seeds):")
    for key, val in summarize_metrics(per_seed).items():
        logger.info(f"    {key:10s}: {val}")

    preds = np.mean([per_seed[s]["preds"] for s in per_seed], axis=0)
    labels = per_seed[next(iter(per_seed))]["labels"]
    for group, mask in [("seen", [s["hla"] in seen_alleles for s in valid]),
                        ("unseen", [s["hla"] not in seen_alleles for s in valid])]:
        idxs = [i for i, b in enumerate(mask) if b]
        if not idxs or len(set(labels[idxs])) < 2:
            logger.info(f"    [{group}] n={len(idxs)} pos={int(sum(labels[idxs]))} "
                        f"(insufficient positives for ROC)")
            continue
        sl, sp = labels[idxs], preds[idxs]
        logger.info(f"    [{group}] n={len(idxs)} pos={int(sum(sl))} "
                    f"ROC={roc_auc_score(sl, sp):.4f} "
                    f"PR={average_precision_score(sl, sp):.4f} "
                    f"PPVn={mean_PPVn(sl, sp):.4f}")


def main():
    parser = argparse.ArgumentParser(description="HybridModelv7 TESLA eval")
    parser.add_argument("--split", nargs="+", default=["608", "310"],
                        choices=["608", "310"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/v7_full")
    parser.add_argument("--ckpt-name", default="v7_phase2_best.pt")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pep-emb-path",
                        default=f"{DATA_DIR}/esmc_peptide_perres_tesla.pt")
    parser.add_argument("--hla-emb-path",
                        default=f"{DATA_DIR}/esmc_hla_fullmhc_perres_tesla.pt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger(__name__)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    seen_alleles = load_seen_alleles()
    pep_perres = torch.load(args.pep_emb_path, map_location="cpu")
    hla_perres = torch.load(args.hla_emb_path, map_location="cpu")
    graph_map = load_graphs(GRAPH_DIR)
    logger.info(f"seen alleles: {len(seen_alleles)} | pep emb: {len(pep_perres)} | "
                f"hla emb: {len(hla_perres)} | graphs: {len(graph_map)}")

    for split in args.split:
        samples = load_tesla(split)
        run_split(split, samples, graph_map, pep_perres, hla_perres, seen_alleles,
                  args.seeds, args.save_dir, args.ckpt_name, device,
                  args.batch_size, logger)


if __name__ == "__main__":
    main()
