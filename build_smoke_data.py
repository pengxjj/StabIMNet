#!/usr/bin/env python3
"""Build the small smoke-test IEDB subset for the StabIMNet release package.

Reads the FULL IEDB table + graph dir, selects a stratified ~400-row subset that
preserves the 22% positive rate, peptide-length mix (9/10/11-mer, incl. >=1 11-mer
so the VAE input_dim stays 283*21=5943 and the provided stage-1 weights load), and
>=5 HLA alleles. Copies the matching graph .pt files and writes a reduced HLA table.

This produces ONLY the smoke data (data_smoke/). The full IEDB table/graphs are
NOT copied — see README for how to point the pipeline at the full data.

Usage: python build_smoke_data.py
"""
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)  # bio_models/

SRC_CSV = os.path.join(BASE, "ImmunoStruct-main", "StructIM", "data", "ImmunoStruct_IEDB_data.csv")
SRC_GRAPHS = os.path.join(BASE, "ImmunoStruct-main", "StructIM", "data", "graph_pyg_IEDB_esmfold2")
SRC_HLA = os.path.join(BASE, "ImmunoStruct-main", "StructIM", "data", "HLA_allele_sequences.csv")
DST = os.path.join(HERE, "data_smoke")
N = 400
SEED = 42

def graph_filename(allele, peptide):
    safe = allele.replace("-", "_").replace("*", "_").replace(":", "_")
    return f"{safe}_{peptide}.pt"

def main():
    df = pd.read_csv(SRC_CSV)
    df["L"] = df["peptide"].str.len()
    key = df["immunogenicity"].astype(str) + "_" + df["L"].astype(str)

    # Stratified sample preserving (label x length) mix.
    sub, _ = train_test_split(df, train_size=N, stratify=key, random_state=SEED)
    sub = sub.reset_index(drop=True)

    # Guarantee an 11-mer so max_full_length stays 283 -> VAE input_dim 5943.
    if sub["L"].max() < 11:
        swap = df[df["L"] == 11].sample(1, random_state=SEED)
        sub = pd.concat([sub.iloc[1:], swap], ignore_index=True)

    assert sub["L"].max() == 11, "no 11-mer in smoke subset"
    assert sub["immunogenicity"].nunique() == 2, "subset lacks both classes"
    assert sub["allele"].nunique() >= 5, "subset lacks allele diversity"

    # Copy matching graph files.
    dst_graphs = os.path.join(DST, "graph_pyg_IEDB_esmfold2")
    os.makedirs(dst_graphs, exist_ok=True)
    missing = 0
    for allele, pep in zip(sub["allele"], sub["peptide"]):
        src = os.path.join(SRC_GRAPHS, graph_filename(allele, pep))
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_graphs, os.path.basename(src)))
        else:
            missing += 1
    assert missing == 0, f"{missing} graph files missing"

    # Subset CSVs (drop the helper 'L' column from the data CSV).
    cols = [c for c in df.columns if c != "L"]
    sub[cols].to_csv(os.path.join(DST, "ImmunoStruct_IEDB_data.csv"), index=False)

    hla = pd.read_csv(SRC_HLA)
    hla_sub = hla[hla["allele"].isin(sub["allele"].unique())]
    hla_sub.to_csv(os.path.join(DST, "HLA_allele_sequences.csv"), index=False)

    print(f"smoke subset: n={len(sub)} pos={int(sub['immunogenicity'].sum())} "
          f"neg={int((sub['immunogenicity'] == 0).sum())} alleles={sub['allele'].nunique()}")
    print(f"peptide lengths: {sorted(sub['L'].unique())}")
    print(f"graphs copied: {len(os.listdir(dst_graphs))} -> {dst_graphs}")

if __name__ == "__main__":
    main()
