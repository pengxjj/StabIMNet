#!/usr/bin/env python
"""TCR data inventory: VDJdb (+ McPAS-TCR if reachable) coverage against our
clinical / IEDB HLA alleles and epitopes.

Purpose (feasibility gate for the paper's "TCR diversity" limitation): the paper
excludes TCR information because "the astronomical diversity of the human TCR
repertoire" limits training data. This script measures whether paired pMHC-TCR
data actually exists in enough volume *overlapping our alleles/epitopes* to make
a TCR-augmented model feasible.

It merges VDJdb's chunked TSVs (downloaded via codeload.github.com, which is
reachable through the squid proxy even when raw.githubusercontent.com / the
VDJdb web UI are not), filters to human MHC-I 8-11mer, then computes four levels
of coverage:
  1. VDJdb global scale (records / unique allele / unique epitope / unique CDR3)
  2. per-allele record counts for our 23 clinical + 27 IEDB alleles (4-digit and
     2-digit serotype matching)
  3. epitope overlap: clinical/IEDB peptides that appear as a VDJdb antigen
  4. pair overlap: clinical (allele, mut_pep) pairs that match a VDJdb
     (mhc.a, antigen.epitope) pair

Run:
    python inventory_tcr_pmhc.py --clinical-csv <path> --iedb-csv <path> \
        [--vdjdb-dir /tmp/vdjdb-db-master] [--out tcr_coverage_report.md]
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tarfile

import pandas as pd

HLA_RE = re.compile(r"HLA-([ABC])\*(\d+)(?::(\d+))?")
AMINO = set("ACDEFGHIKLMNPQRSTVWY")


def parse_hla(s):
    """Return (four_digit, two_digit) or (None, None) for a VDJdb mhc.a string."""
    m = HLA_RE.search(str(s))
    if not m:
        return None, None
    locus, field2, field4 = m.group(1), m.group(2), m.group(3)
    two = f"HLA-{locus}*{field2}"
    four = f"HLA-{locus}*{field2}:{field4}" if field4 else two
    return four, two


def is_clean_peptide(seq):
    """True if seq is 8-11 standard amino acids."""
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    return 8 <= len(seq) <= 11 and all(c in AMINO for c in seq)


def download_vdjdb(cache_dir):
    """Download + extract vdjdb-db via codeload (reachable through proxy)."""
    os.makedirs(cache_dir, exist_ok=True)
    tgz = os.path.join(cache_dir, "vdjdb-db.tar.gz")
    url = "https://codeload.github.com/antigenomics/vdjdb-db/tar.gz/refs/heads/master"
    if not os.path.exists(tgz):
        print(f"[download] {url}")
        r = subprocess.run(["curl", "-sL", "--max-time", "300", url, "-o", tgz])
        if r.returncode != 0 or os.path.getsize(tgz) < 1000:
            print("ERROR: VDJdb download failed", file=sys.stderr)
            sys.exit(1)
    outdir = os.path.join(cache_dir, "vdjdb-db-master")
    if not os.path.isdir(outdir):
        with tarfile.open(tgz) as t:
            t.extractall(cache_dir)
    return outdir


def load_vdjdb(vdjdb_dir):
    """Merge chunks/*.tsv into one DataFrame; return empty columns on failure."""
    chunk_dir = os.path.join(vdjdb_dir, "chunks")
    files = sorted(glob.glob(os.path.join(chunk_dir, "*.tsv")))
    if not files:
        print(f"ERROR: no chunks in {chunk_dir}", file=sys.stderr)
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {os.path.basename(f)}: {e}")
    return pd.concat(frames, ignore_index=True)


def try_mcpas(cache_dir):
    """Best-effort McPAS-TCR download; returns DataFrame or None."""
    url = "http://friedmanlab.weizmann.ac.il/McPAS-TCR/export.php"
    out = os.path.join(cache_dir, "McPAS-TCR.csv")
    r = subprocess.run(["curl", "-sL", "--max-time", "60", url, "-o", out])
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 100:
        return None
    try:
        return pd.read_csv(out, sep=",", dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return None


def df_to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def coverage_table(vdjdb, alleles, name):
    """Per-allele record counts (4-digit and 2-digit) for a set of alleles."""
    rows = []
    four_set = set(vdjdb["mhc.a"])
    two_set = set(vdjdb["mhc.two"])
    four_counts = vdjdb["mhc.a"].value_counts()
    two_counts = vdjdb["mhc.two"].value_counts()
    for a in sorted(alleles):
        _, two = parse_hla(a)
        n4 = int(four_counts.get(a, 0))
        n2 = int(two_counts.get(two, 0))
        rows.append({"allele": a, "vdjdb_records_4digit": n4, "vdjdb_records_2digit": n2})
    return pd.DataFrame(rows), four_set, two_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical-csv", default="data/ImmunoStruct_clinical_data.csv")
    ap.add_argument("--iedb-csv", default="data/ImmunoStruct_IEDB_data.csv")
    ap.add_argument("--vdjdb-dir", default=None)
    ap.add_argument("--cache-dir", default="/tmp/vdjdb_cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clinical = pd.read_csv(args.clinical_csv, dtype=str, keep_default_na=False)
    iedb = pd.read_csv(args.iedb_csv, dtype=str, keep_default_na=False)

    clin_alleles = sorted(clinical["allele"].unique())
    iedb_alleles = sorted(iedb["allele"].unique())
    clin_pairs = set(zip(clinical["allele"], clinical["mut_pep"]))
    clin_peps = set(clinical["mut_pep"].unique())
    iedb_peps = set(iedb["peptide"].unique())

    vdjdb_dir = args.vdjdb_dir or download_vdjdb(args.cache_dir)
    vdjdb = load_vdjdb(vdjdb_dir)

    lines = ["# TCR pMHC data coverage report", ""]

    if vdjdb.empty:
        lines.append("**VDJdb: NO DATA LOADED**")
        print("\n".join(lines))
        return

    # ---- filter to human MHC-I ----
    v = vdjdb.copy()
    v = v[v["species"].str.contains("Homo", case=False, na=False)]
    v = v[v["mhc.class"].str.upper().str.contains("MHCI", na=False)]
    v["mhc.a"] = v["mhc.a"].astype(str)
    v[["mhc.four", "mhc.two"]] = v["mhc.a"].apply(
        lambda s: pd.Series(parse_hla(s))
    )
    v = v[v["mhc.four"].notna()]
    v["antigen.epitope"] = v["antigen.epitope"].astype(str).str.upper().str.strip()
    v = v[v["antigen.epitope"].apply(is_clean_peptide)]

    n_rec = len(v)
    n_allele4 = v["mhc.four"].nunique()
    n_allele2 = v["mhc.two"].nunique()
    n_epitope = v["antigen.epitope"].nunique()
    n_pair = v[["mhc.four", "antigen.epitope"]].drop_duplicates().shape[0]
    n_cdr3 = len(set(v["cdr3.alpha"]).union(set(v["cdr3.beta"]))) - (1 if "" in set(v["cdr3.alpha"]).union(set(v["cdr3.beta"])) else 0)

    lines += [
        "## 1. VDJdb global scale (human MHC-I, 8-11mer, after filtering)",
        "",
        f"- records: **{n_rec}**",
        f"- unique 4-digit HLA alleles: **{n_allele4}**",
        f"- unique 2-digit serotypes: **{n_allele2}**",
        f"- unique epitopes: **{n_epitope}**",
        f"- unique (allele, epitope) pairs: **{n_pair}**",
        f"- unique CDR3 (alpha or beta): **{n_cdr3}**",
        "",
    ]

    # ---- per-allele coverage ----
    clin_tbl, clin_f4, clin_f2 = coverage_table(v, clin_alleles, "clinical")
    iedb_tbl, iedb_f4, iedb_f2 = coverage_table(v, iedb_alleles, "IEDB")

    def allele_cov(tbl):
        cov4 = int((tbl["vdjdb_records_4digit"] > 0).sum())
        cov2 = int((tbl["vdjdb_records_2digit"] > 0).sum())
        tot = len(tbl)
        return cov4, cov2, tot

    c4, c2, ct = allele_cov(clin_tbl)
    i4, i2, it = allele_cov(iedb_tbl)

    lines += [
        "## 2. Per-allele coverage",
        "",
        f"- **clinical 23 alleles**: {c4}/23 have >=1 VDJdb record at 4-digit; "
        f"{c2}/23 at 2-digit serotype",
        f"- **IEDB 27 alleles**: {i4}/27 at 4-digit; {i2}/27 at 2-digit",
        "",
        "### clinical alleles (records = VDJdb TCR-pMHC entries naming that allele)",
        "",
        df_to_md(clin_tbl),
        "",
        "### IEDB alleles",
        "",
        df_to_md(iedb_tbl),
        "",
    ]

    # ---- epitope overlap ----
    vdjdb_ep = set(v["antigen.epitope"])
    clin_ep_hit = clin_peps & vdjdb_ep
    iedb_ep_hit = iedb_peps & vdjdb_ep
    lines += [
        "## 3. Epitope overlap (our peptides that are VDJdb antigens)",
        "",
        f"- clinical mut_pep: **{len(clin_ep_hit)}/{len(clin_peps)}** "
        f"({100*len(clin_ep_hit)/max(1,len(clin_peps)):.1f}%) appear as a VDJdb antigen",
        f"- IEDB peptide: **{len(iedb_ep_hit)}/{len(iedb_peps)}** "
        f"({100*len(iedb_ep_hit)/max(1,len(iedb_peps)):.1f}%) appear as a VDJdb antigen",
        "",
    ]

    # ---- pair overlap ----
    vdjdb_pairs = set(zip(v["mhc.four"], v["antigen.epitope"]))
    clin_pair_hit = clin_pairs & vdjdb_pairs
    lines += [
        "## 4. Pair overlap (exact (allele, epitope) match)",
        "",
        f"- clinical (allele, mut_pep): **{len(clin_pair_hit)}/{len(clin_pairs)}** "
        f"({100*len(clin_pair_hit)/max(1,len(clin_pairs)):.2f}%) match a VDJdb pair",
        "",
    ]

    # ---- McPAS-TCR (optional) ----
    mcpas = try_mcpas(args.cache_dir)
    if mcpas is not None:
        lines += [
            "## 5. McPAS-TCR",
            f"- rows: {len(mcpas)}, columns: {list(mcpas.columns)}",
        ]
    else:
        lines += ["## 5. McPAS-TCR", "", "- **unreachable** (proxy 503); skipped."]

    report = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"wrote {args.out}")
    print(report)


if __name__ == "__main__":
    main()
