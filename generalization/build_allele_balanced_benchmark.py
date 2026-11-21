#!/usr/bin/env python3
"""
Build the unified per-allele external immunogenicity benchmark.

Merges the three de-novo-scored external cohorts into one (allele, peptide,
label, source, de-novo features) table:

    TESLA 608   data/TESLA/tesla_608_denovo_features.csv   (hla has HLA- prefix)
    TESLA 310   data/TESLA/tesla_310_denovo_features.csv
    NEPdb       data/NEPdb/nepdb_denovo_features.csv       (allele has no prefix)

All three are de-novo (deployable) feature tables produced by
compute_denovo_features.py / validate_nepdb_denovo.py, so every row carries the
same four sequence-only features:

    binding_score     = -log10(mhcflurry pan-allelic IC50)   (higher = stronger)
    frac_hydrophobic  = fraction of residues with KD > 0
    net_charge        = (#R+#K+#H) - (#D+#E)
    pep_len           = epitope length

This script only NORMALIZES and MERGES; it does not score anything new. It
prints a per-allele census (n / pos) so the HLA-C data ceiling is explicit, and
writes data/allele_balanced_benchmark.csv for eval_allele_balanced.py.

Usage:
    python build_allele_balanced_benchmark.py
"""
import os
import csv
import re
import collections

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, "data")

TESLA_608 = os.path.join(DATA, "TESLA", "tesla_608_denovo_features.csv")
TESLA_310 = os.path.join(DATA, "TESLA", "tesla_310_denovo_features.csv")
NEPDB = os.path.join(DATA, "NEPdb", "nepdb_denovo_features.csv")
OUT_CSV = os.path.join(DATA, "allele_balanced_benchmark.csv")

HLA_RE = re.compile(r"^HLA-([ABC])\*(\d{2}:\d{2})$")
ALLELE_RE = re.compile(r"^([ABC])\*(\d{2}:\d{2})$")

COLS = ["allele", "locus", "peptide", "label", "source",
        "binding_score", "frac_hydrophobic", "net_charge", "pep_len"]


def norm_allele(a):
    """Normalize to 'HLA-A*02:01' style; returns (locus, canonical) or (None,None)."""
    a = a.strip().upper()
    m = HLA_RE.match(a)
    if m:
        return m.group(1), "HLA-" + m.group(1) + "*" + m.group(2)
    m = ALLELE_RE.match(a)
    if m:
        return m.group(1), "HLA-" + m.group(1) + "*" + m.group(2)
    return None, None


def tofloat(v):
    v = (v or "").strip()
    if v == "":
        return ""
    try:
        return float(v)
    except ValueError:
        return ""


def load_tesla(path, source):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            loc, allele = norm_allele(r["hla"])
            if loc is None:
                continue
            rows.append({
                "allele": allele, "locus": loc,
                "peptide": r["peptide"].strip().upper(),
                "label": int(r["label"]), "source": source,
                "binding_score": tofloat(r["binding_score"]),
                "frac_hydrophobic": tofloat(r["frac_hydrophobic"]),
                "net_charge": tofloat(r["net_charge"]),
                "pep_len": tofloat(r["pep_len"]),
            })
    return rows


def load_nepdb(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            loc, allele = norm_allele(r["allele"])
            if loc is None:
                continue
            label = 1 if r["label"].strip().upper() == "P" else 0
            rows.append({
                "allele": allele, "locus": loc,
                "peptide": r["mut_peptide"].strip().upper(),
                "label": label, "source": "NEPdb",
                "binding_score": tofloat(r["binding_score"]),
                "frac_hydrophobic": tofloat(r["frac_hydrophobic"]),
                "net_charge": tofloat(r["net_charge"]),
                "pep_len": tofloat(r["pep_len"]),
            })
    return rows


def main():
    rows = []
    rows += load_tesla(TESLA_608, "TESLA608")
    rows += load_tesla(TESLA_310, "TESLA310")
    rows += load_nepdb(NEPDB)
    print(f"merged rows: {len(rows)}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLS})
    print(f"saved -> {OUT_CSV}")

    # per-allele census
    by_allele = collections.defaultdict(lambda: [0, 0])
    by_locus = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_allele[r["allele"]][0] += 1
        by_allele[r["allele"]][1] += r["label"]
        by_locus[r["locus"]][0] += 1
        by_locus[r["locus"]][1] += r["label"]

    print("\n=== per-locus census ===")
    for loc in "ABC":
        n, p = by_locus[loc]
        print(f"  HLA-{loc}: n={n:5d}  pos={p:4d}  ({100*p/max(1,n):.2f}%)")

    print("\n=== per-allele census (HLA-C first) ===")
    for loc in "CAB":
        print(f"  --- HLA-{loc} ---")
        for a in sorted(k for k in by_allele if k.startswith(f"HLA-{loc}*")):
            n, p = by_allele[a]
            print(f"    {a:14s} n={n:5d}  pos={p:3d}")

    # HLA-C positive ceiling
    c_alleles = [a for a in by_allele if a.startswith("HLA-C*")]
    c_pos = sum(by_allele[a][1] for a in c_alleles)
    c_neg = sum(by_allele[a][0] - by_allele[a][1] for a in c_alleles)
    print("\n=== HLA-C ceiling ===")
    print(f"  HLA-C alleles present: {len(c_alleles)}")
    print(f"  HLA-C total positives : {c_pos}")
    print(f"  HLA-C total negatives : {c_neg}")
    print("  => per-allele balanced (>=20 samples, pos/neg balanced) is data-"
          "impossible for HLA-C;")


if __name__ == "__main__":
    main()
