#!/usr/bin/env python3
"""Precompute per-residue ESMC-6B embeddings for the ITSNdb external benchmark.

Mirrors precompute_esmc_perres.py (same ESMCEncoder.forward(pool=False) slicing),
but reads ITSNdb neoantigens + the ITSNdb HLA pseudo-sequences instead of the
IEDB training set. Needed to run HybridModelv6 (per-residue cross-attention) on
the external benchmark.

Outputs (keyed by string):
  data/esmc_peptide_perres_itsndb.pt   {peptide: tensor[L_aa, 2560]}
  data/esmc_hla_pseudo_perres_itsndb.pt {allele:  tensor[34, 2560]}

Usage:
  python precompute_esmc_perres_itsndb.py --smoke
  python precompute_esmc_perres_itsndb.py --gpu 0
"""
import os
import sys
import csv
import argparse
import types

import numpy as np

# DGL compatibility patch (mirrors the training scripts)
_dgl_gb = types.ModuleType("dgl.graphbolt")
_dgl_gb.load_graphbolt = lambda: None
sys.modules["dgl.graphbolt"] = _dgl_gb

import torch

_orig_torch_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(f, *a, **kw)
torch.load = _patched_load

_PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
_BASE = os.path.join(_PROJ_ROOT, "base")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from models.hybrid_models import ESMCEncoder

ESMC_DIR = "/public/home/lifex/pengxj/models/esmc-6b"
DATA_DIR = os.path.join(_PROJ_ROOT, "data")
ITSNB_DIR = os.path.join(_PROJ_ROOT, "..", "data", "ITSNdb", "data")
PSEUDO_CSV = os.path.join(DATA_DIR, "HLA_pseudo_sequences_itsndb.csv")
OUT_PEPTIDE = os.path.join(DATA_DIR, "esmc_peptide_perres_itsndb.pt")
OUT_HLA = os.path.join(DATA_DIR, "esmc_hla_pseudo_perres_itsndb.pt")


def load_itsndb_peptides():
    """Unique neoantigen peptides across ITSNdb.csv + Val_dataset.csv."""
    seqs = set()
    for fn in ["ITSNdb.csv", "Val_dataset.csv"]:
        with open(os.path.join(ITSNB_DIR, fn)) as f:
            for r in csv.DictReader(f):
                p = r["Neoantigen"].strip().strip('"')
                if p:
                    seqs.add(p)
    return sorted(seqs)


def load_pseudo_sequences(csv_path):
    with open(csv_path) as f:
        return {r["allele"].strip(): r["pseudo_sequence"].strip()
                for r in csv.DictReader(f)}


@torch.no_grad()
def encode_perres(encoder, sequences, batch_size=32):
    feats = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        tokens = encoder.tokenizer.tokenize(batch_seqs)
        embs, valid = encoder.forward(tokens, pool=False)  # [B, L, 2560], [B, L]
        for j in range(embs.shape[0]):
            feats.append(embs[j][valid[j]].float().cpu())
    return feats


def main():
    parser = argparse.ArgumentParser(description="Precompute ITSNdb per-residue ESMC")
    parser.add_argument("--esmc-dir", default=ESMC_DIR)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default=DATA_DIR)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[itsndb-perres] loading ESMC-6B from {args.esmc_dir} onto {device} (fp16) ...",
          flush=True)
    encoder = ESMCEncoder(args.esmc_dir, device=device, dtype=torch.float16)
    print("[itsndb-perres] encoder ready.", flush=True)

    if args.smoke:
        seqs = ["GRIAFFLKY", "FLLGKTSSV", "ACDEFGHIK", "NLVPMVATV"]
        feats = encode_perres(encoder, seqs, batch_size=args.batch_size)
        for s, f in zip(seqs, feats):
            print(f"[smoke] {s:<12} -> {tuple(f.shape)} dtype {f.dtype}")
        print("[smoke] OK.")
        return

    peptides = load_itsndb_peptides()
    print(f"[itsndb-perres] {len(peptides)} unique ITSNdb peptides to encode", flush=True)
    pep_feats = encode_perres(encoder, peptides, batch_size=args.batch_size)
    pep_dict = {p: f.clone() for p, f in zip(peptides, pep_feats)}
    lens = [f.shape[0] for f in pep_feats]
    print(f"[itsndb-perres] peptide residue lengths: {sorted(set(lens))}", flush=True)

    pseudo_map = load_pseudo_sequences(PSEUDO_CSV)
    alleles = list(pseudo_map.keys())
    pseudo_seqs = [pseudo_map[a] for a in alleles]
    print(f"[itsndb-perres] {len(alleles)} HLA pseudo-sequences to encode", flush=True)
    hla_feats = encode_perres(encoder, pseudo_seqs, batch_size=args.batch_size)
    hla_dict = {a: f.clone() for a, f in zip(alleles, hla_feats)}
    print(f"[itsndb-perres] HLA residue lengths: {sorted(set(f.shape[0] for f in hla_feats))}",
          flush=True)

    torch.save(pep_dict, os.path.join(args.out_dir, "esmc_peptide_perres_itsndb.pt"))
    torch.save(hla_dict, os.path.join(args.out_dir, "esmc_hla_pseudo_perres_itsndb.pt"))
    print(f"[itsndb-perres] wrote {OUT_PEPTIDE} ({len(pep_dict)} entries)")
    print(f"[itsndb-perres] wrote {OUT_HLA} ({len(hla_dict)} entries)")
    print("[itsndb-perres] DONE.")


if __name__ == "__main__":
    main()
