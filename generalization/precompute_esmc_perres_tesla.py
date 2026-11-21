#!/usr/bin/env python3
"""Precompute per-residue ESMC-6B embeddings for the TESLA external benchmark.

Mirrors precompute_esmc_perres_itsndb.py, but:
  - peptides come from data/TESLA/tesla_{608,310}.csv (no peptide overlap),
  - the HLA side is MERGED from the existing IEDB + ITSNdb embedding dicts
    (no new ESMC encoding needed): TESLA's 16 alleles are all covered by the
    IEDB-27 (seen) and ITSNdb-35 (unseen: B*27:05, C*05:01, C*06:02) dicts.
    For seen alleles we prefer the IEDB embedding (identical pseudo-seq the
    model was trained on); unseen alleles come from the ITSNdb dict.

Outputs (keyed by string):
  data/esmc_peptide_perres_tesla.pt   {peptide: tensor[L_aa, 2560]}
  data/esmc_hla_pseudo_perres_tesla.pt {allele:  tensor[34, 2560]}

Usage:
  python precompute_esmc_perres_tesla.py --smoke
  python precompute_esmc_perres_tesla.py --gpu 0
  python precompute_esmc_perres_tesla.py --hla-only   # skip peptide encoding
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
TESLA_DIR = os.path.join(DATA_DIR, "TESLA")
IEDB_HLA_EMB = os.path.join(DATA_DIR, "esmc_hla_pseudo_perres.pt")
ITSNB_HLA_EMB = os.path.join(DATA_DIR, "esmc_hla_pseudo_perres_itsndb.pt")
OUT_PEPTIDE = os.path.join(DATA_DIR, "esmc_peptide_perres_tesla.pt")
OUT_HLA = os.path.join(DATA_DIR, "esmc_hla_pseudo_perres_tesla.pt")


def load_tesla_alleles():
    """Union of alleles across tesla_608.csv + tesla_310.csv (16 distinct)."""
    alleles = set()
    for fn in ["tesla_608.csv", "tesla_310.csv"]:
        with open(os.path.join(TESLA_DIR, fn)) as f:
            for r in csv.DictReader(f):
                alleles.add(r["hla"].strip())
    return sorted(alleles)


def load_tesla_peptides():
    """Unique peptides across tesla_608.csv + tesla_310.csv."""
    seqs = set()
    for fn in ["tesla_608.csv", "tesla_310.csv"]:
        with open(os.path.join(TESLA_DIR, fn)) as f:
            for r in csv.DictReader(f):
                p = r["peptide"].strip()
                if p:
                    seqs.add(p)
    return sorted(seqs)


def merge_hla_dicts(alleles):
    """Build {allele: [34,2560]} for the given TESLA alleles, preferring the
    IEDB-27 embedding (seen) over the ITSNdb-35 embedding (unseen)."""
    iedb = torch.load(IEDB_HLA_EMB, map_location="cpu")
    itsndb = torch.load(ITSNB_HLA_EMB, map_location="cpu")
    merged, missing = {}, []
    for a in alleles:
        if a in iedb:
            merged[a] = iedb[a].clone()
        elif a in itsndb:
            merged[a] = itsndb[a].clone()
        else:
            missing.append(a)
    return merged, missing


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
    parser = argparse.ArgumentParser(description="Precompute TESLA per-residue ESMC")
    parser.add_argument("--esmc-dir", default=ESMC_DIR)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hla-only", action="store_true")
    args = parser.parse_args()

    # ── HLA side: merge existing dicts (cheap, no encoder) ──────────────
    alleles = load_tesla_alleles()
    hla_dict, missing = merge_hla_dicts(alleles)
    print(f"[tesla-perres] {len(alleles)} TESLA alleles, "
          f"{len(hla_dict)} covered, missing={missing}", flush=True)
    assert not missing, f"uncovered TESLA alleles: {missing}"
    torch.save(hla_dict, OUT_HLA)
    print(f"[tesla-perres] wrote {OUT_HLA} ({len(hla_dict)} entries)", flush=True)

    if args.hla_only:
        print("[tesla-perres] DONE (hla-only).", flush=True)
        return

    # ── peptide side ────────────────────────────────────────────────────
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[tesla-perres] loading ESMC-6B from {args.esmc_dir} onto {device} (fp16) ...",
          flush=True)
    encoder = ESMCEncoder(args.esmc_dir, device=device, dtype=torch.float16)
    print("[tesla-perres] encoder ready.", flush=True)

    peptides = load_tesla_peptides()
    print(f"[tesla-perres] {len(peptides)} unique TESLA peptides to encode", flush=True)

    if args.smoke:
        peptides = peptides[:4]
        feats = encode_perres(encoder, peptides, batch_size=args.batch_size)
        for s, f in zip(peptides, feats):
            print(f"[smoke] {s:<14} -> {tuple(f.shape)} dtype {f.dtype}", flush=True)
        print("[smoke] OK.", flush=True)
        return

    pep_feats = encode_perres(encoder, peptides, batch_size=args.batch_size)
    pep_dict = {p: f.clone() for p, f in zip(peptides, pep_feats)}
    lens = sorted(set(f.shape[0] for f in pep_feats))
    print(f"[tesla-perres] peptide residue lengths: {lens}", flush=True)

    torch.save(pep_dict, OUT_PEPTIDE)
    print(f"[tesla-perres] wrote {OUT_PEPTIDE} ({len(pep_dict)} entries)", flush=True)
    print("[tesla-perres] DONE.", flush=True)


if __name__ == "__main__":
    main()
