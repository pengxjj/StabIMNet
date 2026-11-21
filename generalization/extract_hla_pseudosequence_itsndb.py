#!/usr/bin/env python3
"""Extract 34-mer NetMHCpan pseudo-sequences for all ITSNdb HLA alleles.

Builds on extract_hla_pseudosequence.py (same NetMHCpan PSEUDO_POSITIONS).
Covers the 28 alleles with known full sequences and applies fallbacks for the
7 without:
  HLA-A*02:17 -> HLA-A*02:01  (same 2-digit supertype)
  HLA-C*xx    -> HLA-C*04:01  (nearest available full HLA-C seq)

Output: data/HLA_pseudo_sequences_itsndb.csv  {allele, pseudo_sequence}
"""
import os
import re
import ast
import csv

PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
HLA_CSV = os.path.join(PROJ_ROOT, "data", "HLA_allele_sequences.csv")
EVAL_ITSN = os.path.join(PROJ_ROOT, "..", "ImmunoStruct_Phase2_Repro", "eval_itsndb.py")
ITSNB_DIR = os.path.join(PROJ_ROOT, "..", "data", "ITSNdb", "data")
OUT_CSV = os.path.join(PROJ_ROOT, "data", "HLA_pseudo_sequences_itsndb.csv")

PSEUDO_POSITIONS = [
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
    84, 95, 97, 99, 114, 116, 118, 143, 147, 152, 156, 158, 159, 163,
    167, 171, 180,
]
assert len(PSEUDO_POSITIONS) == 34


def extract(full_seq):
    """34-residue pseudo-sequence from a mature-chain HLA full sequence."""
    if full_seq.startswith("GSHSMRY"):
        mature = full_seq
    elif full_seq.startswith("SHSMRY"):
        mature = "G" + full_seq  # restore omitted mature residue 1 (Gly)
    else:
        raise ValueError(f"unexpected sequence start: {full_seq[:12]!r}")
    if len(mature) < max(PSEUDO_POSITIONS):
        raise ValueError(f"sequence too short ({len(mature)})")
    return "".join(mature[p - 1] for p in PSEUDO_POSITIONS)


def normalize_allele(h):
    h = h.strip().strip('"')
    if "HLA-" in h and "*" not in h:
        prefix, rest = h.split("-", 1)
        return f"HLA-{rest[0]}*{rest[1:]}"
    return h


def main():
    # 1. full sequences: 27 CSV + MISSING_HLA_SEQS hardcoded in eval_itsndb.py
    full = {}
    with open(HLA_CSV) as f:
        for r in csv.DictReader(f):
            full[r["allele"].strip()] = r["seqs"].strip()

    src = open(EVAL_ITSN).read()
    m = re.search(r"MISSING_HLA_SEQS = \{(.*?)\n\}", src, re.S)
    if not m:
        raise SystemExit("could not parse MISSING_HLA_SEQS from eval_itsndb.py")
    full.update(ast.literal_eval("{" + m.group(1) + "}"))

    # 2. unique ITSNdb alleles
    itsn = set()
    for fn in ["ITSNdb.csv", "Val_dataset.csv"]:
        with open(os.path.join(ITSNB_DIR, fn)) as f:
            for r in csv.DictReader(f):
                itsn.add(normalize_allele(r["HLA"]))

    # 3. fallbacks for alleles lacking a full sequence
    fallback = {"HLA-A*02:17": "HLA-A*02:01"}
    hla_c_fb = "HLA-C*04:01"
    for a in itsn:
        if a.startswith("HLA-C") and a not in full:
            fallback[a] = hla_c_fb

    # 4. extract + write
    out = []
    for a in sorted(itsn):
        target = fallback.get(a, a)
        if target not in full:
            raise SystemExit(f"no full sequence for {a} (fallback {target})")
        out.append((a, extract(full[target])))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["allele", "pseudo_sequence"])
        w.writerows(out)

    n_fb = sum(1 for a, _ in out if a in fallback)
    print(f"wrote {OUT_CSV}: {len(out)} alleles ({n_fb} fallback)")
    for a, p in out:
        if a in fallback:
            print(f"  fallback {a:16s} -> {fallback[a]}")


if __name__ == "__main__":
    main()
