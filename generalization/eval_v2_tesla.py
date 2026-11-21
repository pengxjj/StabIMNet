#!/usr/bin/env python3
"""
Evaluate the ORIGINAL (non-generalization) ImmunoStruct model — HybridModelv2 —
on the TESLA external benchmark (Wells et al. 2020 Cell).

v2 = VAE sequence tower (full MHC + peptide one-hot) + EGNN structure tower +
2-dim peptide property, fused by concatenation + self-attention. This is the
model whose weights live at StructIM/model_weights/v2_phase2_best.pt.

Reports 608 (tested) and 310 (validation) separately + seen/unseen-HLA breakdown.

Usage:
    python eval_v2_tesla.py                 # both splits
    python eval_v2_tesla.py --split 608
    python eval_v2_tesla.py --gpu 1
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
from sklearn.metrics import (roc_auc_score, average_precision_score)

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))          # StructIM-hla
_STRUCTIM = os.path.join(os.path.dirname(_PROJ_ROOT), "StructIM")  # sibling repo
sys.path.insert(0, os.path.join(_STRUCTIM, "base"))

from models.hybrid_models import HybridModelv2
from procedures.metric import mean_PPVn

TESLA_DIR = os.path.join(_PROJ_ROOT, "data", "TESLA")
GRAPH_DIR = os.path.join(TESLA_DIR, "graph_pyg_tesla_esmfold2")
IEDB_HLA_CSV = os.path.join(_STRUCTIM, "data", "HLA_allele_sequences.csv")
CKPT = os.path.join(_STRUCTIM, "model_weights", "v2_phase2_best.pt")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PADDING_CHAR = "J"
MHC_LEN = 272          # IEDB mature HLA sequence length
MAX_PEP = 11           # max peptide length in IEDB training (283 - 272)
VAE_INPUT_DIM = (MHC_LEN + MAX_PEP) * (len(AMINO_ACIDS) + 1)   # 283 * 21 = 5943

DEFAULT_MPROP1 = 0.38
DEFAULT_MPROP2 = 0.42

# 3 TESLA alleles absent from IEDB's 27. 272-aa mature sequences (= mature[:272],
# drops the trailing "RWEP" so they end "LTLR" like the IEDB CSV).
UNSEEN_SEQS = {
    "HLA-B*27:05": ("SHSMRYFHTSVSRPGRGEPRFITVGYVDDTLFVRFDSDAASPREEPRAPWIEQEGPEYWDRETQICKAKAQTDREDLRTLLRYYNQSEAGSHTLQNM"
                    "YGCDVGPDGRLLRGYHQDAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGECVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
                    "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLR"),
    "HLA-C*05:01": ("SHSMRYFYTAVSRPGRGEPRFIAVGYVDDTQFVQFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVNLRKLRGYYNQSEAGSHTLQRM"
                    "YGCDLGPDGRLLRGYNQFAYDGKDYIALNEDLRSWTAADKAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKKTLQRAEHPKTHVTHHPVS"
                    "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLR"),
    "HLA-C*06:02": ("SHSMRYFDTAVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQADRVNLRKLRGYYNQSEDGSHTLQWM"
                    "YGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQWRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVS"
                    "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLR"),
}


def load_hla_seqs():
    seqs = {}
    with open(IEDB_HLA_CSV) as f:
        for r in csv.DictReader(f):
            seqs[r["allele"].strip()] = r["seqs"].strip()
    seqs.update(UNSEEN_SEQS)
    return seqs


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
    graph = graph.clone()
    if graph.x.shape[1] == 22:
        graph.x = graph.x[:, :-2]                        # strip 2 H-bond cols
    graph.x = torch.cat([graph.x, graph.coords], dim=-1)  # 20 one-hot + 3 coords
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


def one_hot(seq):
    char_to_int = {c: i for i, c in enumerate(AMINO_ACIDS + PADDING_CHAR)}
    mat = np.zeros((len(seq), len(char_to_int)), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in char_to_int:
            mat[i, char_to_int[c]] = 1.0
    return mat


def build_full_sequence(mhc_seq, peptide):
    """mhc(272) + peptide(truncated/padded to 11) -> one-hot [283,21] -> [5943]."""
    pep = peptide[:MAX_PEP].ljust(MAX_PEP, PADDING_CHAR)
    return one_hot(mhc_seq + pep).flatten()


def evaluate(model, samples, graph_map, hla_seqs, device, batch_size=150):
    seen_alleles = set(hla_seqs.keys()) - set(UNSEEN_SEQS.keys())
    max_nodes = max(g.num_nodes for g in graph_map.values())

    graphs, seqs, props, labels, hlas, peptides = [], [], [], [], [], []
    for s in samples:
        gname = f"{s['hla']}_{s['peptide']}"
        if gname not in graph_map or s["hla"] not in hla_seqs:
            continue
        graphs.append(preprocess_graph_to_dgl(graph_map[gname], max_nodes))
        seqs.append(build_full_sequence(hla_seqs[s["hla"]], s["peptide"]))
        props.append([DEFAULT_MPROP1, DEFAULT_MPROP2])
        labels.append(s["label"])
        hlas.append(s["hla"])
        peptides.append(s["peptide"])

    seq_t = torch.tensor(np.stack(seqs), dtype=torch.float32)
    prop_t = torch.tensor(np.array(props), dtype=torch.float32)
    labels = np.array(labels, dtype=float)

    all_preds = []
    idx = list(range(len(graphs)))
    for i in range(0, len(idx), batch_size):
        batch_idx = idx[i:i + batch_size]
        bg = dgl.batch([graphs[j] for j in batch_idx])
        with torch.no_grad():
            recon, mu, logvar, out = model(
                bg,
                seq_t[batch_idx].to(device),
                prop_t[batch_idx].to(device),
            )
            all_preds.append(torch.sigmoid(out.squeeze(-1)).cpu().numpy())
    preds = np.concatenate(all_preds)
    preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)

    return preds, labels, hlas, peptides, seen_alleles


def report(name, labels, score):
    if len(np.unique(labels)) < 2:
        print(f"  {name:28s}: n={len(labels)} pos={int(labels.sum())} (insufficient)")
        return
    print(f"  {name:28s}: n={len(labels)} pos={int(labels.sum())} "
          f"ROC={roc_auc_score(labels, score):.4f} "
          f"PR={average_precision_score(labels, score):.4f} "
          f"PPVn={mean_PPVn(labels, score):.4f} "
          f"PPVn@30={mean_PPVn(labels, score, topk=30):.4f}")


def dedup_samples(samples):
    seen_keys, deduped = {}, []
    for s in samples:
        k = (s["hla"], s["peptide"])
        if k in seen_keys:
            continue
        seen_keys[k] = s
        deduped.append(s)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="HybridModelv2 TESLA eval (single or ensemble)")
    parser.add_argument("--split", nargs="+", default=["608", "310"], choices=["608", "310"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--ckpts", nargs="+", default=[CKPT],
                        help="checkpoint(s); if multiple, predictions are averaged (ensemble)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger(__name__)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} | n_ckpts={len(args.ckpts)}")

    hla_seqs = load_hla_seqs()
    graph_map = load_graphs(GRAPH_DIR)
    logger.info(f"hla seqs: {len(hla_seqs)} | graphs: {len(graph_map)}")

    samples_by_split = {}
    for split in args.split:
        samples = dedup_samples(load_tesla(split))
        samples_by_split[split] = samples
        if len(samples) < len(load_tesla(split)):
            logger.info(f"{split}: deduped -> {len(samples)} unique")
    # (reload was wasteful; note original count once)
    samples_by_split = {s: dedup_samples(load_tesla(s)) for s in args.split}

    # per-split accumulator
    results = {s: {"preds": [], "labels": None, "hlas": None,
                   "peptides": None, "seen": None} for s in args.split}

    for ci, ckpt in enumerate(args.ckpts):
        model = HybridModelv2(
            vae_input_dim=VAE_INPUT_DIM, device=device,
            gcn_layers=5, vae_hidden_dim=768, vae_latent_dim=64,
            gat_hidden_channels=64, property_embedding_dim=16,
        )
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        logger.info(f"[{ci}] loaded {os.path.basename(ckpt)} (strict=True)")

        for split in args.split:
            preds, labels, hlas, peptides, seen_alleles = evaluate(
                model, samples_by_split[split], graph_map, hla_seqs, device, args.batch_size)
            r = results[split]
            r["preds"].append(preds)
            r["labels"] = labels
            r["hlas"] = hlas
            r["peptides"] = peptides
            r["seen"] = seen_alleles

    n_seeds = len(args.ckpts)
    for split in args.split:
        r = results[split]
        labels, hlas, peptides = r["labels"], r["hlas"], r["peptides"]
        seen_alleles = r["seen"]
        P = np.stack(r["preds"])          # (n_seeds, n_samples)
        ens = P.mean(axis=0)

        out_csv = os.path.join(TESLA_DIR, f"tesla_{split}_v2_diag.csv")
        with open(out_csv, "w") as f:
            f.write("hla,peptide,label,v2_pred\n")
            for h, p, l, pr in zip(hlas, peptides, labels, ens):
                f.write(f"{h},{p},{int(l)},{pr:.6f}\n")
        logger.info(f"saved ensemble -> {out_csv}")

        print(f"\n{'=' * 64}")
        print(f"  HybridModelv2 — TESLA {split}: n={len(labels)} pos={int(labels.sum())} "
              f"({n_seeds}-seed ensemble)")
        print(f"{'=' * 64}")
        report("ensemble", labels, ens)

        if n_seeds > 1:
            rocs = [roc_auc_score(labels, p) for p in P]
            prs = [average_precision_score(labels, p) for p in P]
            print(f"  per-seed ROC: {np.mean(rocs):.4f} ± {np.std(rocs):.4f}   "
                  f"PR: {np.mean(prs):.4f} ± {np.std(prs):.4f}")

        seen_mask = np.array([h in seen_alleles for h in hlas])
        unseen_mask = ~seen_mask
        for grp, m in [("seen", seen_mask), ("unseen", unseen_mask)]:
            if m.sum() == 0:
                continue
            report(f"{grp} (HLA)", labels[m], ens[m])


if __name__ == "__main__":
    main()
