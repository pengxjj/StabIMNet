#!/usr/bin/env python3
"""Precompute per-residue ESMC-6B embeddings for (peptide, HLA pseudo-sequence).

HybridModelv6 replaces v5's pooled-FiLM fusion with per-residue cross-attention,
so it needs token-level (per-amino-acid) embeddings instead of mean-pooled ones.
This script mirrors precompute_esmc_embeddings.py but calls ESMCEncoder.forward
with ``pool=False`` and slices out CLS/EOS/PAD tokens, storing one [L_aa, 2560]
tensor per unique peptide and per HLA pseudo-sequence.

Outputs (keyed by string):
  data/esmc_peptide_perres.pt        {peptide: tensor[L_aa, 2560]}   (L_aa 9-11)
  data/esmc_hla_pseudo_perres.pt     {allele:   tensor[34, 2560]}    (34 residues)

Usage:
  python precompute_esmc_perres.py --smoke      # 5-seq sanity check
  python precompute_esmc_perres.py              # full precompute
"""
import os
import sys
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
IEDB_CSV = os.path.join(DATA_DIR, "ImmunoStruct_IEDB_data.csv")
PSEUDO_CSV = os.path.join(DATA_DIR, "HLA_pseudo_sequences.csv")
OUT_PEPTIDE = os.path.join(DATA_DIR, "esmc_peptide_perres.pt")
OUT_HLA = os.path.join(DATA_DIR, "esmc_hla_pseudo_perres.pt")


def load_unique_sequences(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["peptide"])
    df = df.dropna(subset=["peptide"])
    return sorted(set(df["peptide"].astype(str).tolist()))


def load_pseudo_sequences(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return dict(zip(df["allele"].astype(str), df["pseudo_sequence"].astype(str)))


@torch.no_grad()
def encode_perres(encoder, sequences, batch_size=32):
    """Encode sequences → list of [L_aa, 2560] (special tokens removed)."""
    feats = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        tokens = encoder.tokenizer.tokenize(batch_seqs)
        embs, valid = encoder.forward(tokens, pool=False)  # [B, L, 2560], [B, L]
        for j in range(embs.shape[0]):
            feats.append(embs[j][valid[j]].float().cpu())
    return feats


def main():
    parser = argparse.ArgumentParser(description="Precompute per-residue ESMC embeddings")
    parser.add_argument("--esmc-dir", default=ESMC_DIR)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="encode 5 dummy sequences only (sanity check)")
    parser.add_argument("--out-dir", default=DATA_DIR)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[perres] loading ESMC-6B from {args.esmc_dir} onto {device} (fp16) ...",
          flush=True)
    encoder = ESMCEncoder(args.esmc_dir, device=device, dtype=torch.float16)
    print("[perres] encoder ready.", flush=True)

    if args.smoke:
        seqs = ["SIINFEKL", "NLVPMVATV", "ACDEFGHIK", "GLCTLVAML",
                "YFAMYGEKVAHTHVDTLYVRYHYYTWVLAYTWYQ"]
        feats = encode_perres(encoder, seqs, batch_size=args.batch_size)
        for s, f in zip(seqs, feats):
            print(f"[smoke] {s[:12]:<14} -> {tuple(f.shape)} dtype {f.dtype}")
        print("[smoke] OK — per-residue ESMC encoder runs.")
        return

    # ── unique peptides ─────────────────────────────────────────────
    peptides = load_unique_sequences(IEDB_CSV)
    print(f"[perres] {len(peptides)} unique peptides to encode", flush=True)
    pep_feats = encode_perres(encoder, peptides, batch_size=args.batch_size)
    pep_dict = {p: f.clone() for p, f in zip(peptides, pep_feats)}
    lens = [f.shape[0] for f in pep_feats]
    print(f"[perres] peptide residue lengths: min {min(lens)} max {max(lens)}", flush=True)

    # ── HLA pseudo-sequences ────────────────────────────────────────
    pseudo_map = load_pseudo_sequences(PSEUDO_CSV)
    alleles = list(pseudo_map.keys())
    pseudo_seqs = [pseudo_map[a] for a in alleles]
    print(f"[perres] {len(alleles)} HLA pseudo-sequences to encode", flush=True)
    hla_feats = encode_perres(encoder, pseudo_seqs, batch_size=args.batch_size)
    hla_dict = {a: f.clone() for a, f in zip(alleles, hla_feats)}
    print(f"[perres] HLA residue lengths: {sorted(set(f.shape[0] for f in hla_feats))}",
          flush=True)

    # ── save ────────────────────────────────────────────────────────
    out_pep = os.path.join(args.out_dir, "esmc_peptide_perres.pt")
    out_hla = os.path.join(args.out_dir, "esmc_hla_pseudo_perres.pt")
    torch.save(pep_dict, out_pep)
    torch.save(hla_dict, out_hla)
    print(f"[perres] wrote {out_pep} ({len(pep_dict)} entries)")
    print(f"[perres] wrote {out_hla} ({len(hla_dict)} entries)")
    print("[perres] DONE.")


if __name__ == "__main__":
    main()
