#!/usr/bin/env python3
"""
Evaluate HybridModelv6 (sequence-primary cross-attention) on the ITSNdb
external benchmark — the final arbiter for personalized-neoantigen
generalization (unseen HLA alleles + unseen tumor neoantigens).

Differs from ../ImmunoStruct_Phase2_Repro/eval_itsndb.py in that v6 has no
VAE: it consumes per-residue ESMC embeddings (peptide [Lp,2560], HLA
pseudo-seq [34,2560]) + pooled EGNN structure + [Mprop1, Mprop2] property.

Usage:
    python eval_v6_itsndb.py                          # all seeds, both datasets
    python eval_v6_itsndb.py --dataset val            # Val_dataset only
    python eval_v6_itsndb.py --seeds 42               # single seed
    python eval_v6_itsndb.py --save-dir results/v6_full
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

from models.hybrid_models import HybridModelv6, HybridModelv8
from procedures.metric import mean_PPVn

ITSNB_DIR = os.path.join(_PROJ_ROOT, "..", "data", "ITSNdb")
GRAPH_DIR = os.path.join(_PROJ_ROOT, "..", "ImmunoStruct_Phase2_Repro",
                         "data", "graph_pyg_ITSNdb_esmfold2")
DATA_DIR = os.path.join(_PROJ_ROOT, "data")

SEQ_EMB_DIM = 2560
DEFAULT_SEEDS = [42, 123, 456, 789]
DEFAULT_MPROP1 = 0.38
DEFAULT_MPROP2 = 0.42


# ────────────────────────────────────────────────────────────────────
# Data loading (mirrors eval_itsndb.py)
# ────────────────────────────────────────────────────────────────────

def normalize_allele(hla_str):
    hla_str = hla_str.strip().strip('"')
    if "HLA-" in hla_str and "*" not in hla_str:
        prefix, rest = hla_str.split("-", 1)
        return f"HLA-{rest[0]}*{rest[1:]}"
    return hla_str


def load_itsndb_val():
    samples = []
    with open(os.path.join(ITSNB_DIR, "data", "Val_dataset.csv")) as f:
        for row in csv.DictReader(f):
            samples.append({
                "hla": normalize_allele(row["HLA"]),
                "peptide": row["Neoantigen"].strip().strip('"'),
                "label": 1.0 if row["Sample"].strip().strip('"') == "Pos" else 0.0,
                "source": "Val_dataset",
            })
    return samples


def load_itsndb_full():
    samples = []
    with open(os.path.join(ITSNB_DIR, "data", "ITSNdb.csv")) as f:
        for row in csv.DictReader(f):
            samples.append({
                "hla": normalize_allele(row["HLA"]),
                "peptide": row["Neoantigen"].strip().strip('"'),
                "label": 1.0 if row["NeoType"].strip().strip('"') == "Positive" else 0.0,
                "source": "ITSNdb",
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


class ITSNdbV6Dataset:
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


def v6_collate(items):
    # items: list of (dgl_graph, pep, hla, prop, label)
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

    hla = torch.stack(list(hlas), dim=0)                # [B, 34, 2560]
    hla_mask = torch.ones(B, hla.shape[1], dtype=torch.bool)

    return batched_graph, pep, pep_mask, hla, hla_mask, props, labels


def evaluate(model, dataset, device, batch_size=150):
    model.eval()
    all_preds, all_labels = [], []
    idx = list(range(len(dataset)))
    for i in range(0, len(idx), batch_size):
        items = [(dataset.graphs[j], dataset.peps[j], dataset.hlas[j],
                  dataset.properties[j], dataset.labels[j]) for j in idx[i:i + batch_size]]
        g, pep, pep_mask, hla, hla_mask, prop, labels = v6_collate(items)
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


def main():
    parser = argparse.ArgumentParser(description="HybridModelv6 ITSNdb eval")
    parser.add_argument("--dataset", default="both", choices=["val", "full", "both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--save-dir", default=f"{_PROJ_ROOT}/results/v6_full")
    parser.add_argument("--ckpt-name", default="v6_phase2_best.pt")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pep-emb-path",
                        default=f"{DATA_DIR}/esmc_peptide_perres_itsndb.pt")
    parser.add_argument("--hla-emb-path",
                        default=f"{DATA_DIR}/esmc_hla_pseudo_perres_itsndb.pt")
    parser.add_argument("--model-version", default="v6", choices=["v6", "v8"],
                        help="which HybridModel variant the checkpoints belong to")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger(__name__)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── samples ─────────────────────────────────────────────────────
    all_samples = []
    if args.dataset in ("val", "both"):
        all_samples.extend(load_itsndb_val())
    if args.dataset in ("full", "both"):
        all_samples.extend(load_itsndb_full())
    if args.dataset == "both":
        seen, deduped = set(), []
        for s in all_samples:
            key = (s["hla"], s["peptide"])
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        all_samples = deduped
    logger.info(f"Dataset={args.dataset}: {len(all_samples)} samples")

    # ── embeddings ──────────────────────────────────────────────────
    pep_perres = torch.load(args.pep_emb_path, map_location="cpu")
    hla_perres = torch.load(args.hla_emb_path, map_location="cpu")
    logger.info(f"peptide perres: {len(pep_perres)} | HLA perres: {len(hla_perres)}")

    # ── graphs ──────────────────────────────────────────────────────
    graph_map = load_graphs(GRAPH_DIR)
    logger.info(f"Loaded {len(graph_map)} graphs")

    valid, skipped = [], 0
    for s in all_samples:
        gname = f"{s['hla']}_{s['peptide']}"
        if gname not in graph_map or s["peptide"] not in pep_perres or s["hla"] not in hla_perres:
            logger.warning(f"Skip {gname} (graph={gname in graph_map}, "
                           f"pep={s['peptide'] in pep_perres}, hla={s['hla'] in hla_perres})")
            skipped += 1
            continue
        valid.append(s)
    logger.info(f"Valid: {len(valid)} (skipped {skipped})")

    dataset = ITSNdbV6Dataset(valid, graph_map, pep_perres, hla_perres)

    # ── per-seed eval ───────────────────────────────────────────────
    per_seed = {}
    for seed in args.seeds:
        ckpt = os.path.join(args.save_dir, f"seed_{seed}", args.ckpt_name)
        if not os.path.exists(ckpt):
            logger.warning(f"[seed {seed}] checkpoint not found: {ckpt} — skip")
            continue
        model_cls = HybridModelv8 if args.model_version == "v8" else HybridModelv6
        model = model_cls(
            seq_emb_dim=SEQ_EMB_DIM, proj_dim=256, gnn_dim=64, gnn_layers=6,
            n_heads=4, dropout=0.1, device=device, property_embedding_dim=16,
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

        m = evaluate(model, dataset, device, args.batch_size)
        per_seed[seed] = m
        logger.info(f"[seed {seed}] ROC {m['roc_auc']:.4f} | PR {m['pr_auc']:.4f} | "
                    f"PPVn_cont {m['ppvn_cont']:.4f} | PPVn_prec {m['ppvn_prec']:.4f} "
                    f"(n={len(m['labels'])})")
        del model
        torch.cuda.empty_cache()

    if not per_seed:
        logger.error("No checkpoint evaluated — is training finished?")
        return

    def agg(key):
        vals = np.array([m[key] for m in per_seed.values()], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

    n_pos = int(sum(per_seed[next(iter(per_seed))]["labels"]))
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  HybridModel{args.model_version} ITSNdb ({args.dataset}, {len(per_seed)} seeds)")
    logger.info(f"  Samples: {len(valid)} ({n_pos} pos / {len(valid) - n_pos} neg)")
    for key in ["roc_auc", "pr_auc", "ppvn_cont", "ppvn_prec"]:
        mu, sd = agg(key)
        logger.info(f"  {key:10s}: {mu:.4f} +/- {sd:.4f}")
    logger.info(f"{'=' * 60}")

    # per-source breakdown on the last seed's predictions
    if args.dataset == "both":
        last = per_seed[max(per_seed)]
        for source in ["Val_dataset", "ITSNdb"]:
            idxs = [i for i, s in enumerate(valid) if s["source"] == source]
            if idxs and len(set(last["labels"][idxs])) >= 2:
                sl = last["labels"][idxs]
                sp = last["preds"][idxs]
                logger.info(f"  [{source}] n={len(idxs)} pos={int(sum(sl))} "
                            f"ROC={roc_auc_score(sl, sp):.4f} "
                            f"PR={average_precision_score(sl, sp):.4f} "
                            f"PPVn={mean_PPVn(sl, sp):.4f}")


if __name__ == "__main__":
    main()
