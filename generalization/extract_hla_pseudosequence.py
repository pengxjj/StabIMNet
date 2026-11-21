#!/usr/bin/env python3
"""Extract the 34-residue NetMHCpan MHC-I pseudo-sequence from full HLA sequences.

The pseudo-sequence is the set of polymorphic binding-groove residues (Nielsen et
al. 2007; Jurtz et al. 2017). Positions are 1-indexed in the mature MHC-I alpha
chain (no signal peptide), which matches the sequences in HLA_allele_sequences.csv
(they start at "SHSMRYF..." = mature position 1).
"""
import os
import csv

PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
HLA_CSV = os.path.join(PROJ_ROOT, "data", "HLA_allele_sequences.csv")
OUT_CSV = os.path.join(PROJ_ROOT, "data", "HLA_pseudo_sequences.csv")

# NetMHCpan MHC-I pseudo-sequence positions (1-indexed, mature chain).
PSEUDO_POSITIONS = [
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
    84, 95, 97, 99, 114, 116, 118, 143, 147, 152, 156, 158, 159, 163,
    167, 171, 180,
]

assert len(PSEUDO_POSITIONS) == 34, f"expected 34 positions, got {len(PSEUDO_POSITIONS)}"


def extract(full_seq):
    # CSV sequences omit the leading Gly (mature residue 1 = 'G'); NetMHCpan
    # pseudo-positions are 1-indexed in the mature chain INCLUDING that Gly, so
    # restore it before indexing (verified against NetMHCpan B0702.fsa reference).
    if full_seq.startswith("GSHSMRY"):
        mature = full_seq
    elif full_seq.startswith("SHSMRY"):
        mature = "G" + full_seq
    else:
        raise ValueError(f"unexpected HLA sequence start: {full_seq[:12]!r}")
    if len(mature) < max(PSEUDO_POSITIONS):
        raise ValueError(f"sequence too short ({len(mature)}) for pseudo-positions")
    return "".join(mature[p - 1] for p in PSEUDO_POSITIONS)


def main():
    with open(HLA_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out_rows = []
    for row in rows:
        allele = row["allele"].strip()
        full = row["seqs"].strip()
        pseudo = extract(full)
        out_rows.append((allele, pseudo))

    # sanity: all 34-aa, polymorphic
    lengths = {len(p) for _, p in out_rows}
    uniq = {p for _, p in out_rows}
    assert lengths == {34}, f"unexpected lengths {lengths}"
    print(f"alleles: {len(out_rows)} | pseudo len: {lengths} | unique pseudo-seq: {len(uniq)}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["allele", "pseudo_sequence"])
        writer.writerows(out_rows)

    print(f"wrote {OUT_CSV}")
    for a, p in out_rows[:5]:
        print(f"  {a:14s} {p}")


if __name__ == "__main__":
    main()
