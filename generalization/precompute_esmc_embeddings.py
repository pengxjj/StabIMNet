#!/usr/bin/env python3
"""Precompute ESMC-6B embeddings for (peptide, HLA pseudo-sequence).

Used by HybridModelv5 (HLA-conditioned three-tower model). The ESMC-6B
encoder is run ONCE here and its 2560-dim mean-pooled embeddings are cached
per unique peptide string and per HLA allele, so the training loop never
touches the 6B model.

Outputs (keyed by string, so lookup is robust to dataset ordering):
  data/esmc_peptide_emb.pt       {peptide_str: tensor[2560]}
  data/esmc_hla_pseudo_emb.pt    {allele:      tensor[2560]}

Usage:
  python precompute_esmc_embeddings.py --smoke       # 5 seq sanity check
  python precompute_esmc_embeddings.py               # full precompute
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
OUT_PEPTIDE = os.path.join(DATA_DIR, "esmc_peptide_emb.pt")
OUT_HLA = os.path.join(DATA_DIR, "esmc_hla_pseudo_emb.pt")


def load_unique_sequences(csv_path):
    """Return (unique_peptides: list[str], ) from the IEDB CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["peptide"])
    df = df.dropna(subset=["peptide"])
    uniq = sorted(set(df["peptide"].astype(str).tolist()))
    return uniq


def load_pseudo_sequences(csv_path):
    """Return {allele: pseudo_sequence} from HLA_pseudo_sequences.csv."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return dict(zip(df["allele"].astype(str), df["pseudo_sequence"].astype(str)))


def main():
    parser = argparse.ArgumentParser(description="Precompute ESMC-6B embeddings")
    parser.add_argument("--esmc-dir", default=ESMC_DIR)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="encode 5 dummy sequences only (sanity check)")
    parser.add_argument("--out-dir", default=DATA_DIR)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[esmc] loading ESMC-6B from {args.esmc_dir} onto {device} (fp16) ...",
          flush=True)
    encoder = ESMCEncoder(args.esmc_dir, device=device, dtype=torch.float16)
    print("[esmc] encoder ready.", flush=True)

    if args.smoke:
        seqs = ["SIINFEKL", "NLVPMVATV", "YFAMYGEKVAHTHVDTLYVRYHYYTWVLAYTWYQ",
                "ACDEFGHIK", "GLCTLVAML"]
        embs = encoder.encode_batch(seqs, batch_size=args.batch_size)
        print(f"[smoke] embeddings shape: {tuple(embs.shape)} dtype {embs.dtype}")
        print(f"[smoke] row0[:8] = {embs[0][:8].tolist()}")
        print("[smoke] OK — ESMCEncoder loads and runs.")
        return

    # ── unique peptides ─────────────────────────────────────────────
    peptides = load_unique_sequences(IEDB_CSV)
    print(f"[esmc] {len(peptides)} unique peptides to encode", flush=True)

    pep_embs = encoder.encode_batch(peptides, batch_size=args.batch_size)
    pep_dict = {p: pep_embs[i].clone() for i, p in enumerate(peptides)}

    # ── HLA pseudo-sequences ────────────────────────────────────────
    pseudo_map = load_pseudo_sequences(PSEUDO_CSV)
    alleles = list(pseudo_map.keys())
    pseudo_seqs = [pseudo_map[a] for a in alleles]
    print(f"[esmc] {len(alleles)} HLA pseudo-sequences to encode", flush=True)

    hla_embs = encoder.encode_batch(pseudo_seqs, batch_size=args.batch_size)
    hla_dict = {a: hla_embs[i].clone() for i, a in enumerate(alleles)}

    # ── save ────────────────────────────────────────────────────────
    out_pep = os.path.join(args.out_dir, "esmc_peptide_emb.pt")
    out_hla = os.path.join(args.out_dir, "esmc_hla_pseudo_emb.pt")
    torch.save(pep_dict, out_pep)
    torch.save(hla_dict, out_hla)
    print(f"[esmc] wrote {out_pep} ({len(pep_dict)} entries)")
    print(f"[esmc] wrote {out_hla} ({len(hla_dict)} entries)")
    print("[esmc] DONE.")


if __name__ == "__main__":
    main()
