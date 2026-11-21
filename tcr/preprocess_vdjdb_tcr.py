#!/usr/bin/env python3
"""Build the TCR-pMHC binding dataset from VDJdb.

Filters VDJdb to human MHC-I 8-11mer paired alpha-beta records, generates 1:1
CDR3-shuffle decoys (keeping peptide + allele + V/J genes fixed), and splits by
(peptide, allele) identity (unseen-pMHC test) into 80/10/10 train/val/test.

Reuses helpers from inventory_tcr_pmhc.py (load_vdjdb, parse_hla, is_clean_peptide).

Run:
    python preprocess_vdjdb_tcr.py [--vdjdb-dir /tmp/vdjdb_cache/vdjdb-db-master] \
        [--out data/tcr_pmhc] [--max-records N] [--seed 42]
"""
import argparse
import json
import os
import random

import pandas as pd

from inventory_tcr_pmhc import download_vdjdb, is_clean_peptide, load_vdjdb, parse_hla
from tcr_model import CHAR_VOCAB

AMINO = set("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN = 32  # truncation cap for CDR3; peptides are 8-11


def is_clean_cdr3(seq):
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    return 3 <= len(seq) <= MAX_LEN and all(c in AMINO for c in seq)


def shuffle_cdr3(seq, rng):
    """Shuffle residues; guarantees a different permutation for len>=2."""
    if len(seq) < 2:
        return seq
    lst = list(seq)
    for _ in range(10):
        rng.shuffle(lst)
        out = "".join(lst)
        if out != seq:
            return out
    return "".join(lst)


def build_vocab(values):
    uniq = sorted(set(values))
    return uniq, {v: i for i, v in enumerate(uniq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdjdb-dir", default=None)
    ap.add_argument("--cache-dir", default="/tmp/vdjdb_cache")
    ap.add_argument("--out", default="data/tcr_pmhc")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--negative",
                    choices=["shuffle", "swap", "swap-same-allele"], default="swap",
                    help="shuffle = CDR3-shuffle decoys (literature-comparable but "
                         "trivially separable); swap = real TCR + non-cognate pMHC; "
                         "swap-same-allele = real TCR from the SAME allele, different "
                         "peptide (removes the cross-allele V/J-gene shortcut)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vdjdb_dir = args.vdjdb_dir or download_vdjdb(args.cache_dir)
    print(f"[load] VDJdb from {vdjdb_dir}")
    v = load_vdjdb(vdjdb_dir)

    # ---- filter human MHC-I 8-11mer, paired alpha-beta, clean sequences ----
    v = v[v["species"].str.contains("Homo", case=False, na=False)]
    v = v[v["mhc.class"].str.upper().str.contains("MHCI", na=False)]
    v["mhc.a"] = v["mhc.a"].astype(str)
    v["mhc.four"] = v["mhc.a"].apply(lambda s: parse_hla(s)[0])
    v = v[v["mhc.four"].notna()]
    v["antigen.epitope"] = v["antigen.epitope"].astype(str).str.upper().str.strip()
    v = v[v["antigen.epitope"].apply(is_clean_peptide)]
    v["cdr3.alpha"] = v["cdr3.alpha"].astype(str).str.upper().str.strip()
    v["cdr3.beta"] = v["cdr3.beta"].astype(str).str.upper().str.strip()
    v = v[v["cdr3.alpha"].apply(is_clean_cdr3) & v["cdr3.beta"].apply(is_clean_cdr3)]
    for g in ("v.alpha", "j.alpha", "v.beta", "j.beta"):
        v[g] = v[g].astype(str).str.strip()
        v = v[v[g] != ""]
    print(f"[filter] after human MHC-I paired-alpha-beta + clean: {len(v)} records")

    # ---- dedup on full 8-tuple ----
    keys = ["cdr3.alpha", "cdr3.beta", "v.alpha", "j.alpha", "v.beta", "j.beta",
            "antigen.epitope", "mhc.four"]
    v = v.drop_duplicates(subset=keys).reset_index(drop=True)
    print(f"[dedup] unique records: {len(v)}")

    if args.max_records is not None:
        rng = random.Random(args.seed)
        idx = rng.sample(range(len(v)), min(args.max_records, len(v)))
        v = v.iloc[sorted(idx)].reset_index(drop=True)
        print(f"[smoke] subsampled to {len(v)} positives")

    # ---- vocabularies ----
    va_vocab, _ = build_vocab(v["v.alpha"])
    ja_vocab, _ = build_vocab(v["j.alpha"])
    vb_vocab, _ = build_vocab(v["v.beta"])
    jb_vocab, _ = build_vocab(v["j.beta"])
    allele_vocab, _ = build_vocab(v["mhc.four"])
    vocab = {
        "char": CHAR_VOCAB,
        "v.alpha": va_vocab, "j.alpha": ja_vocab,
        "v.beta": vb_vocab, "j.beta": jb_vocab,
        "allele": allele_vocab,
        "max_len": MAX_LEN,
    }
    print(f"[vocab] char={len(CHAR_VOCAB)} va={len(va_vocab)} ja={len(ja_vocab)} "
          f"vb={len(vb_vocab)} jb={len(jb_vocab)} allele={len(allele_vocab)}")

    # ---- build rows (pos + neg) ----
    rng = random.Random(args.seed)
    cols = ["cdr3.alpha", "cdr3.beta", "v.alpha", "j.alpha", "v.beta", "j.beta",
            "antigen.epitope", "mhc.four"]
    tcr_list = list(zip(*[v[c].tolist() for c in cols]))  # one tuple per record
    n_rec = len(tcr_list)
    # index records by allele and by (allele, peptide) for same-allele sampling
    by_allele = {}
    by_allele_pep = {}
    for idx, t in enumerate(tcr_list):
        by_allele.setdefault(t[7], []).append(idx)
        by_allele_pep.setdefault((t[7], t[6]), []).append(idx)
    rows = []
    for i in range(n_rec):
        ca, cb, va, ja, vb, jb, pep, al = tcr_list[i]
        base = {"cdr3a": ca, "cdr3b": cb, "va": va, "ja": ja,
                "vb": vb, "jb": jb, "peptide": pep, "allele": al}
        rows.append(dict(base, label=1))
        if args.negative == "shuffle":
            nca, ncb = shuffle_cdr3(ca, rng), shuffle_cdr3(cb, rng)
            nva, nja, nvb, njb = va, ja, vb, jb
        elif args.negative == "swap-same-allele":
            cands = by_allele.get(al, ())
            same_pep = by_allele_pep.get((al, pep), ())
            if len(cands) > len(same_pep):
                for _ in range(200):
                    j = rng.choice(cands)
                    if tcr_list[j][6] != pep:
                        break
                else:
                    j = rng.choice([i for i in cands if tcr_list[i][6] != pep])
            else:  # fallback: no other peptide on this allele -> global swap
                for _ in range(100):
                    j = rng.randrange(n_rec)
                    if tcr_list[j][6] != pep or tcr_list[j][7] != al:
                        break
            nca, ncb, nva, nja, nvb, njb, _, _ = tcr_list[j]
        else:  # swap: real TCR from a different (peptide, allele) pMHC
            for _ in range(100):
                j = rng.randrange(n_rec)
                if tcr_list[j][6] != pep or tcr_list[j][7] != al:
                    break
            nca, ncb, nva, nja, nvb, njb, _, _ = tcr_list[j]
        neg = {"cdr3a": nca, "cdr3b": ncb, "va": nva, "ja": nja,
               "vb": nvb, "jb": njb, "peptide": pep, "allele": al, "label": 0}
        rows.append(neg)

    df = pd.DataFrame(rows)
    print(f"[neg] total rows (1:1): {len(df)}  pos={int(df.label.sum())}  "
          f"neg={int((df.label == 0).sum())}")

    # ---- split by (peptide, allele) identity, row-balanced across folds ----
    # VDJdb is dominated by a few immunodominant epitopes (top-3 = 60% of rows;
    # each ~20% of the total). A key-count split (80/10/10 of KEYS per size-bin)
    # let a single mega-epitope land whole in test and swing the row proportions
    # wildly (test = 2%..22% across seeds), making test sets non-comparable. We
    # instead sort pMHC keys by row count descending and greedily assign each
    # whole key to the fold farthest below its row target, landing train/val/test
    # near 80/10/10 of ROWS while keeping each (peptide, allele) intact in exactly
    # one fold (unseen-pMHC test). Deterministic, so all seeds share one honest split.
    size_by_key = df.groupby(["peptide", "allele"]).size().to_dict()
    total_rows = int(sum(size_by_key.values()))

    keys = sorted(size_by_key, key=lambda k: -size_by_key[k])
    targets = {"train": 0.8, "val": 0.1, "test": 0.1}
    fold_keys = {"train": set(), "val": set(), "test": set()}
    fold_rows = {"train": 0, "val": 0, "test": 0}
    for k in keys:
        fold = max(fold_keys, key=lambda f: targets[f] * total_rows - fold_rows[f])
        fold_keys[fold].add(k)
        fold_rows[fold] += size_by_key[k]

    train_keys, val_keys, test_keys = (fold_keys["train"], fold_keys["val"], fold_keys["test"])
    assert not (train_keys & val_keys), "train/val overlap"
    assert not (train_keys & test_keys), "train/test overlap"
    assert not (val_keys & test_keys), "val/test overlap"

    def assign(r):
        k = (r["peptide"], r["allele"])
        return "train" if k in train_keys else ("val" if k in val_keys else "test")

    df["split"] = df.apply(assign, axis=1)

    os.makedirs(args.out, exist_ok=True)
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp].drop(columns=["split"])
        sub.to_csv(os.path.join(args.out, f"{sp}.csv"), index=False)
        print(f"[write] {sp}: {len(sub)} rows  pos={int(sub.label.sum())}")

    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"[write] vocab.json -> {args.out}")


if __name__ == "__main__":
    main()
