#!/usr/bin/env python3
"""Ensemble the 3-seed contrastive binding models and score test.csv.

Uses the per-seed checkpoints that maximize LOEO (leave-one-epitope-out functional
ROC on the 5 trained epitopes), computes the cosine binding score on every test
record, and writes tcr_id,prediction for use in the hybrid submission.
"""
import sys

import numpy as np
import pandas as pd
import torch

HERE = "/public/home/lifex/pengxj/bio_models/ImmunoStruct-main/StructIM-hla"
sys.path.insert(0, HERE)

from predict_tcr_pmhc import build_gene_mapper, encode_all
from tcr_model_contrastive import TCRpMHCContrastiveModel

TEST = "/public/home/lifex/pengxj/bio_models/aIMMREP26/test.csv"
OUT = f"{HERE}/data/tcr_pmhc/test_predictions_contrastive.csv"
CKPTS = [
    f"{HERE}/data/tcr_pmhc/best_model_contrastive_s42.pt.epoch3.pt",
    f"{HERE}/data/tcr_pmhc/best_model_contrastive_s43.pt.epoch9.pt",
    f"{HERE}/data/tcr_pmhc/best_model_contrastive_s44.pt.epoch2.pt",
    f"{HERE}/data/tcr_pmhc/best_model_contrastive_s45.pt.epoch6.pt",
    f"{HERE}/data/tcr_pmhc/best_model_contrastive_s46.pt.epoch6.pt",
    f"{HERE}/data/tcr_pmhc/best_model_contrastive_s47.pt.epoch3.pt",
    f"{HERE}/data/tcr_pmhc/best_model_contrastive_s48.pt.epoch4.pt",
    f"{HERE}/data/tcr_pmhc/best_model_contrastive_s49.pt.epoch5.pt",
]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def build_tensors(vocab, df):
    max_len = vocab["max_len"]
    map_va = build_gene_mapper("v.alpha", vocab)
    map_ja = build_gene_mapper("j.alpha", vocab)
    map_vb = build_gene_mapper("v.beta", vocab)
    map_jb = build_gene_mapper("j.beta", vocab)
    allele_idx = {a: i for i, a in enumerate(vocab["allele"])}
    return {
        "cdr3a": encode_all(df["cdr3_alpha_aa"].tolist(), max_len),
        "cdr3b": encode_all(df["cdr3_beta_aa"].tolist(), max_len),
        "peptide": encode_all(df["epitope_aa"].tolist(), max_len),
        "va": torch.tensor([map_va(g) for g in df["TRAV_IMGT"]], dtype=torch.long),
        "ja": torch.tensor([map_ja(g) for g in df["TRAJ_IMGT"]], dtype=torch.long),
        "vb": torch.tensor([map_vb(g) for g in df["TRBV_IMGT"]], dtype=torch.long),
        "jb": torch.tensor([map_jb(g) for g in df["TRBJ_IMGT"]], dtype=torch.long),
        "allele": torch.tensor([allele_idx.get("HLA-" + str(a).strip(), 0)
                                for a in df["hla_long"]], dtype=torch.long),
    }


def main():
    df = pd.read_csv(TEST)
    vocab = torch.load(CKPTS[0], map_location="cpu", weights_only=False)["vocab"]
    tensors = build_tensors(vocab, df)
    n = len(df)

    scores = np.zeros(n, dtype=np.float32)
    for ck in CKPTS:
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        model = TCRpMHCContrastiveModel(
            d_model=64, n_heads=2, dropout=0.1, max_len=vocab["max_len"],
            n_v_alpha=len(vocab["v.alpha"]), n_j_alpha=len(vocab["j.alpha"]),
            n_v_beta=len(vocab["v.beta"]), n_j_beta=len(vocab["j.beta"]),
            n_allele=len(vocab["allele"]),
        ).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            for s in range(0, n, 512):
                b = slice(s, s + 512)
                batch = {k: tensors[k][b].to(DEVICE) for k in tensors}
                scores[s:s + 512] += model.score(batch).cpu().numpy()
    scores /= len(CKPTS)

    out = df[["tcr_id"]].copy()
    out["prediction"] = scores
    out.to_csv(OUT, index=False)

    df["score"] = scores
    print("=== contrastive binding score per-epitope (test.csv) ===")
    print(df.groupby("epitope_aa").agg(
        n=("tcr_id", "count"), mean=("score", "mean"), std=("score", "std"),
        min=("score", "min"), max=("score", "max"),
    ).round(4).to_string())
    print(f"\n[save] {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
