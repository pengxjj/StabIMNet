#!/usr/bin/env python3
"""Precompute per-residue ESMC-6B embeddings for Route A (pan-allelic EL pretrain).

Route A = pretrain the v6-style cross-attention tower (sequence-only, NO
structure) on the 108-allele BigMHC eluted-ligand (EL) presentation data, then
finetune on the 27-allele IEDB immunogenicity data.

CRITICAL convention fix: the EL CSV's ``mhc_seq`` column is a stale pseudo-seq
convention that has ZERO string overlap with the IEDB pseudo-seqs (verified: the
same physical allele HLA-A*02:01 gives ``...TWAVLAYTWY`` in EL vs
``...TWVLAYTWYQ`` in IEDB). The ground truth is ``full_mhc_seq``; extracting the
34 NetMHCpan positions from it (with the leading Gly restored, exactly as
extract_hla_pseudosequence.py does) reproduces the IEDB pseudo-seq EXACTLY, and
covers 27/27 IEDB alleles. So we IGNORE ``mhc_seq`` and regenerate the canonical
pseudo-seq from ``full_mhc_seq``.

Outputs (keyed by string):
  data/esmc_el_pseudo_perres.pt     {pseudo_seq: tensor[34, 2560]}   (108)
  data/esmc_el_peptide_perres.pt    {peptide:    tensor[L_aa, 2560]} (--peptides)

Usage:
  python precompute_esmc_el_pretrain.py --smoke
  python precompute_esmc_el_pretrain.py --gpu 1            # pseudo-seq only (cheap)
  python precompute_esmc_el_pretrain.py --gpu 1 --peptides # + ~50k peptides (expensive)
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
_EL_CSV = os.path.join(_PROJ_ROOT, "..", "data", "mhc1_el_data", "processed",
                       "mhc1_fullseq_train.csv")
OUT_PSEUDO = os.path.join(DATA_DIR, "esmc_el_pseudo_perres.pt")
OUT_PEPTIDE = os.path.join(DATA_DIR, "esmc_el_peptide_perres.pt")

# NetMHCpan MHC-I pseudo-sequence positions (1-indexed, mature chain, leading
# Gly restored). MUST match extract_hla_pseudosequence.py so EL and IEDB agree.
PSEUDO_POSITIONS = [
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
    84, 95, 97, 99, 114, 116, 118, 143, 147, 152, 156, 158, 159, 163,
    167, 171, 180,
]
assert len(PSEUDO_POSITIONS) == 34


def canonical_pseudo(full_seq):
    """Regenerate the 34-aa canonical pseudo-seq from a full MHC sequence.

    The EL ``full_mhc_seq`` includes the signal peptide; locate the mature start
    ``SHSM`` and restore the leading Gly, exactly like
    extract_hla_pseudosequence.extract().
    """
    idx = full_seq.find("SHSM")
    if idx < 0:
        return None
    mature = "G" + full_seq[idx:]
    if len(mature) < max(PSEUDO_POSITIONS):
        return None
    return "".join(mature[p - 1] for p in PSEUDO_POSITIONS)


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
    import pandas as pd
    parser = argparse.ArgumentParser(description="Route A EL ESMC precompute")
    parser.add_argument("--esmc-dir", default=ESMC_DIR)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--peptides", action="store_true",
                        help="also encode unique EL peptides (expensive)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[el-pretrain] loading ESMC-6B from {args.esmc_dir} onto {device} (fp16) ...",
          flush=True)
    encoder = ESMCEncoder(args.esmc_dir, device=device, dtype=torch.float16)
    print("[el-pretrain] encoder ready.", flush=True)

    if args.smoke:
        seqs = ["YFAMYGEKVAHTHVDTLYVRYHYYTWVLAYTWYQ",
                "YYAMYQENVAQTDVDTLYIIYRDYTWAQAYRWYQ"]
        feats = encode_perres(encoder, seqs, batch_size=args.batch_size)
        for s, f in zip(seqs, feats):
            print(f"[smoke] {s[:12]:<14} -> {tuple(f.shape)} dtype {f.dtype}")
        print("[smoke] OK.")
        return

    df = pd.read_csv(_EL_CSV)
    print(f"[el-pretrain] EL rows: {len(df)}", flush=True)

    # ── canonical pseudo-seq from full_mhc_seq (IGNORE mhc_seq col) ──────
    df["canon_pseudo"] = df["full_mhc_seq"].astype(str).apply(canonical_pseudo)
    bad = df["canon_pseudo"].isna().sum()
    if bad:
        print(f"[el-pretrain] WARNING: {bad} rows with no extractable pseudo-seq "
              f"(dropped)", flush=True)
        df = df.dropna(subset=["canon_pseudo"])
    pseudo_seqs = sorted(set(df["canon_pseudo"]))
    print(f"[el-pretrain] {len(pseudo_seqs)} unique canonical pseudo-seqs", flush=True)

    # cross-check against IEDB
    iedb_csv = os.path.join(_PROJ_ROOT, "data", "HLA_pseudo_sequences.csv")
    iedb = pd.read_csv(iedb_csv)
    iedb_set = set(iedb["pseudo_sequence"].astype(str))
    cov = len(set(pseudo_seqs) & iedb_set)
    print(f"[el-pretrain] IEDB pseudo-seq coverage: {cov}/{len(iedb_set)}", flush=True)

    pseudo_feats = encode_perres(encoder, pseudo_seqs, batch_size=args.batch_size)
    pseudo_dict = {p: f.clone() for p, f in zip(pseudo_seqs, pseudo_feats)}
    lens = sorted(set(f.shape[0] for f in pseudo_feats))
    print(f"[el-pretrain] pseudo-seq residue lengths: {lens}", flush=True)
    torch.save(pseudo_dict, OUT_PSEUDO)
    print(f"[el-pretrain] wrote {OUT_PSEUDO} ({len(pseudo_dict)} entries)", flush=True)

    if args.peptides:
        peptides = sorted(set(df["peptide"].astype(str).tolist()))
        print(f"[el-pretrain] {len(peptides)} unique EL peptides to encode "
              f"(expensive)", flush=True)
        pep_feats = encode_perres(encoder, peptides, batch_size=args.batch_size)
        pep_dict = {p: f.clone() for p, f in zip(peptides, pep_feats)}
        torch.save(pep_dict, OUT_PEPTIDE)
        print(f"[el-pretrain] wrote {OUT_PEPTIDE} ({len(pep_dict)} entries)", flush=True)

    print("[el-pretrain] DONE.")


if __name__ == "__main__":
    main()
