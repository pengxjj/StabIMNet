#!/usr/bin/env python3
"""Precompute per-residue ESMC-6B embeddings for FULL MHC class-I sequences.

HybridModelv7 replaces v6's 34-residue NetMHCpan pseudo-sequence with the
FULL MHC extracellular domain (~276 aa, α1+α2+α3) so peptide residues can
cross-attend over the entire groove + flanking domain (MHCRoBERTa /
SoloPepHLAPan recipe) instead of 34 hand-picked pocket positions.

Sources (mature α1+α2+α3, starting "SHSM", truncated to 276 aa):
  - data/HLA_allele_sequences.csv   27 IEDB alleles (``seqs`` column)
  - MISSING_HLA_SEQS                12 extra alleles (mirrors eval_itsndb.py)
  - data/mhc1_el_data/hla_sequences/{A,B,C}_prot.fasta  (FASTA fallback)

Outputs (keyed by allele string):
  data/esmc_hla_fullmhc_perres.pt        {allele: tensor[~276, 2560]}  IEDB 27
  data/esmc_hla_fullmhc_perres_itsndb.pt {allele: tensor[~276, 2560]}  ITSNdb 35

Usage:
  python precompute_esmc_fullmhc_perres.py --smoke
  python precompute_esmc_fullmhc_perres.py
"""
import os
import re
import sys
import csv
import types
import argparse

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
IEDB_HLA_CSV = os.path.join(DATA_DIR, "HLA_allele_sequences.csv")
# StructIM-hla/data -> ../StructIM/data (has HLA CSV + graphs), but the
# pan-allelic FASTA + ITSNdb live in the sibling ImmunoStruct-main/data.
_REPO_DATA = os.path.join(_PROJ_ROOT, "..", "data")
FASTA_DIR = os.path.join(_REPO_DATA, "mhc1_el_data", "hla_sequences")
ITSNdb_DIR = os.path.join(_REPO_DATA, "ITSNdb")

MHC_TRUNC = 276  # mature α1+α2+α3 domain length

# HLA sequences not in HLA_allele_sequences.csv (mirrors eval_itsndb.py)
MISSING_HLA_SEQS = {
    "HLA-C*04:01": (
        "SHSMRYFSTSVSWPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPREPWVEQEGPEYWDRETQKYKRQAQADRVNLRKLRGYYNQSEAGSHTLQRM"
        "YGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP"),
    "HLA-C*12:03": (
        "SHSMRYFYTAVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQADRVSLRNLRGYYNQSEAGSHTLQWM"
        "YGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQWRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP"),
    "HLA-B*27:05": (
        "SHSMRYFHTSVSRPGRGEPRFITVGYVDDTLFVRFDSDAASPREEPRAPWIEQEGPEYWDRETQICKAKAQTDREDLRTLLRYYNQSEAGSHTLQNM"
        "YGCDVGPDGRLLRGYHQDAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGECVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-B*35:03": (
        "SHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRTEPRAPWIEQEGPEYWDRNTQIFKTNTQTYRENLRTALRYYNQSEAGSHTLQNM"
        "YGCDVGPDGRLLRGYNQDAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGLCVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-B*38:01": (
        "SHSMRYFYTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPREEPRAPWIEQEGPEYWDRNTQIFKTNTQTYRENLRTALRYYNQSEAGSHTLQRM"
        "YGCDVGPDGRLLRGHNQFAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGTCVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-B*39:06": (
        "SHSMRYFYTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPREEPRAPWIEQEGPEYWDRNTQIFKTNTQTYRENLRTALRYYNQSEAGSHTLQRM"
        "YGCDVGPDGRLLRGHNQFAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGTCVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-B*40:02": (
        "SHSMRYFHTSVSRPGRGEPRFITVGYVDDTLFVRFDSDATSPRKEPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRM"
        "YGCDVGPDGRLLRGHNQFAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGECVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-B*41:02": (
        "SHSMRYFHTAMSRPGRGEPRFITVGYVDDTLFVRFDSDATSPRKEPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRM"
        "YGCDVGPDGRLLRGHNQFAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGECVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-B*37:01": (
        "SHSMRYFHTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPRTEPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRM"
        "YGCDVGPDGRLLRGHNQFAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGLCVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-B*56:01": (
        "SHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPREEPRAPWIEQEGPEYWDRNTQIFKTNTQTYRENLRTALRYYNQSEAGSHTLQSM"
        "YGCDVGPDGRLLRGHNQFAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGTCVEWLRRYLENGKETLQRADPPKTHVTHHPIS"
        "DHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-A*02:05": (
        "SHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASRRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRM"
        "YGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVS"
        "DHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEP"),
    "HLA-A*25:01": (
        "SHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDRNTRNVKAHSQIDRVDLGTLRGYYNQSEAGSHTIQMM"
        "YGCDVGSDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITQRKWETAHEAEQWRAYLEGRCVEWLRRYLENGKETLQRTDAPKTHMTHHAVS"
        "DHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEP"),
}


def load_iedb_fullseq():
    """{allele: mature_seq} for the 27 IEDB alleles."""
    return {r["allele"].strip(): r["seqs"].strip()
            for r in csv.DictReader(open(IEDB_HLA_CSV))}


def load_fasta_map():
    """{allele (2-digit + 4-digit): full_seq} parsed from A/B/C_prot.fasta.

    A 2-digit key (e.g. HLA-A*02:17) aggregates many 4-digit entries; some are
    truncated/partial (181 aa, no signal peptide) while others are full-length
    (365 aa). Keep the LONGEST sequence per key so the 2-digit alias resolves
    to the full-length allele, not a partial fragment.
    """
    allele_map = {}
    for gene in ["A", "B", "C"]:
        path = os.path.join(FASTA_DIR, f"{gene}_prot.fasta")
        current_alleles = []
        current_seq = ""
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    if current_seq and current_alleles:
                        for a in current_alleles:
                            if a not in allele_map or len(current_seq) > len(allele_map[a]):
                                allele_map[a] = current_seq
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        full_name = parts[1]
                        current_alleles = [f"HLA-{full_name}"]
                        m = re.match(r"([ABC])\*(\d+):(\d+)", full_name)
                        if m:
                            current_alleles.append(
                                f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}")
                        current_seq = ""
                    else:
                        current_alleles = []
                else:
                    current_seq += line.strip()
        if current_seq and current_alleles:
            for a in current_alleles:
                if a not in allele_map or len(current_seq) > len(allele_map[a]):
                    allele_map[a] = current_seq
    return allele_map


def mature_seq(full_seq):
    """Strip signal peptide: take from first 'SHSM' motif, truncate to 276."""
    idx = full_seq.find("SHSM")
    if idx < 0:
        return full_seq[:MHC_TRUNC]
    return full_seq[idx:idx + MHC_TRUNC]


def resolve_itsndb_alleles():
    """{allele: mature_seq} for all ITSNdb alleles (CSV + MISSING + FASTA)."""
    seq_map = load_iedb_fullseq()
    seq_map.update(MISSING_HLA_SEQS)
    fasta = load_fasta_map()

    alleles = set()
    for f in ["Val_dataset.csv", "ITSNdb.csv"]:
        path = os.path.join(ITSNdb_DIR, "data", f)
        for r in csv.DictReader(open(path)):
            h = r["HLA"].strip().strip('"')
            if "HLA-" in h and "*" not in h:
                p, rest = h.split("-", 1)
                h = f"HLA-{rest[0]}*{rest[1:]}"
            alleles.add(h)

    out = {}
    for a in alleles:
        if a in seq_map:
            out[a] = seq_map[a]
            continue
        if a in fasta:
            out[a] = mature_seq(fasta[a])
            continue
        # last resort: try 4-digit FASTA keys sharing the 2-digit prefix
        hits = [v for k, v in fasta.items()
                if k.startswith(a + ":") or k.startswith(a + "*")]
        if hits:
            out[a] = mature_seq(hits[0])
        else:
            print(f"[warn] no sequence for {a}", flush=True)
    return out


@torch.no_grad()
def encode_perres(encoder, sequences, batch_size=32):
    feats = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        tokens = encoder.tokenizer.tokenize(batch_seqs)
        embs, valid = encoder.forward(tokens, pool=False)
        for j in range(embs.shape[0]):
            feats.append(embs[j][valid[j]].float().cpu())
    return feats


def main():
    parser = argparse.ArgumentParser(description="Precompute full-MHC per-residue ESMC")
    parser.add_argument("--esmc-dir", default=ESMC_DIR)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default=DATA_DIR)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[fullmhc] loading ESMC-6B from {args.esmc_dir} onto {device} (fp16) ...",
          flush=True)
    encoder = ESMCEncoder(args.esmc_dir, device=device, dtype=torch.float16)
    print("[fullmhc] encoder ready.", flush=True)

    if args.smoke:
        seqs = ["SHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDEETGKVKAHSQTDRENLRIALRYYNQSEAGSHTLQMMFGCDVGSDGRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQITKRKWEAAHVAEQQRAYLEGTCVDGLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLR"]
        feats = encode_perres(encoder, seqs, batch_size=args.batch_size)
        for s, f in zip(seqs, feats):
            print(f"[smoke] {s[:12]:<14} -> {tuple(f.shape)} dtype {f.dtype}")
        print("[smoke] OK.")
        return

    iedb = load_iedb_fullseq()
    its = resolve_itsndb_alleles()

    # encode union of unique mature sequences
    uniq = {}
    for a, s in list(iedb.items()) + list(its.items()):
        uniq.setdefault(s, []).append(a)
    seqs = list(uniq.keys())
    print(f"[fullmhc] {len(iedb)} IEDB + {len(its)} ITSNdb alleles, "
          f"{len(seqs)} unique sequences", flush=True)
    lens = [len(s) for s in seqs]
    print(f"[fullmhc] seq lengths: min {min(lens)} max {max(lens)}", flush=True)

    feats = encode_perres(encoder, seqs, batch_size=args.batch_size)
    feat_by_seq = {s: f for s, f in zip(seqs, feats)}

    iedb_out = {a: feat_by_seq[s].clone() for a, s in iedb.items()}
    its_out = {a: feat_by_seq[s].clone() for a, s in its.items()}

    out_iedb = os.path.join(args.out_dir, "esmc_hla_fullmhc_perres.pt")
    out_its = os.path.join(args.out_dir, "esmc_hla_fullmhc_perres_itsndb.pt")
    torch.save(iedb_out, out_iedb)
    torch.save(its_out, out_its)
    print(f"[fullmhc] wrote {out_iedb} ({len(iedb_out)} entries)")
    print(f"[fullmhc] wrote {out_its} ({len(its_out)} entries)")
    # sanity: all keys resolved to a non-trivial length
    bad = [a for a, f in list(iedb_out.items()) + list(its_out.items())
           if f.shape[0] < 200]
    print(f"[fullmhc] entries with <200 residues: {bad or 'none'}")
    print("[fullmhc] DONE.")


if __name__ == "__main__":
    main()
