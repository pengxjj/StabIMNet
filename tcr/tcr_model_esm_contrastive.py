#!/usr/bin/env python3
"""TCRpMHCContrastiveModelESM — contrastive binding model with frozen ESM2 TCR encodings.

Same as TCRpMHCContrastiveModel but the CDR3 alpha/beta char encoders are replaced
by frozen ESM2 (esm2_t30_150M, d=640) mean-pooled embeddings projected to d_model.
The peptide keeps the char-embedding ResidueEncoder (needed for the peptide->HLA
cross-attention). Everything else (V/J genes, allele, pMHC attention, tcr_mlp, and
the contrastive projection heads + InfoNCE objective) is unchanged, so this is a
clean A/B of the TCR sequence encoder under the contrastive objective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tcr_model import ResidueEncoder
from base.models.layers import CrossModalAttentionBlock


class TCRpMHCContrastiveModelESM(nn.Module):
    def __init__(self, esm_dim=640, d_model=64, n_heads=2, dropout=0.1, max_len=32,
                 n_v_alpha=1, n_j_alpha=1, n_v_beta=1, n_j_beta=1, n_allele=1,
                 proj_dim=64):
        super().__init__()
        self.d_model = d_model
        self.esm_proj = nn.Linear(esm_dim, d_model)
        self.res_enc = ResidueEncoder(d_model, n_heads, dropout, max_len)  # peptide only

        self.v_alpha_emb = nn.Embedding(n_v_alpha, d_model)
        self.j_alpha_emb = nn.Embedding(n_j_alpha, d_model)
        self.v_beta_emb = nn.Embedding(n_v_beta, d_model)
        self.j_beta_emb = nn.Embedding(n_j_beta, d_model)
        self.allele_emb = nn.Embedding(n_allele, d_model)

        self.pmhc_attn = CrossModalAttentionBlock(d_model, n_heads, dropout)
        self.tcr_mlp = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.proj_tcr = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim),
        )
        self.proj_pmhc = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())

    def encode(self, batch):
        """batch keys: cdr3a_esm [B,esm_dim], cdr3b_esm [B,esm_dim], peptide [B,L],
        va/ja/vb/jb/allele [B]. Returns (tcr [B,d], pmhc [B,d])."""
        cdr3a = self.esm_proj(batch["cdr3a_esm"])
        cdr3b = self.esm_proj(batch["cdr3b_esm"])
        pep_tokens, _ = self.res_enc(batch["peptide"])

        va = self.v_alpha_emb(batch["va"])
        ja = self.j_alpha_emb(batch["ja"])
        vb = self.v_beta_emb(batch["vb"])
        jb = self.j_beta_emb(batch["jb"])
        allele = self.allele_emb(batch["allele"])

        allele_ctx = allele.unsqueeze(1)
        pep_mask = (batch["peptide"] != 0)
        pmhc_tokens = self.pmhc_attn(pep_tokens, allele_ctx, key_padding_mask=torch.ones(
            batch["peptide"].shape[0], 1, dtype=torch.bool, device=pep_tokens.device))
        pmhc = (pmhc_tokens * pep_mask.unsqueeze(-1).float()).sum(1) / \
               pep_mask.sum(1, keepdim=True).clamp(min=1)

        tcr = self.tcr_mlp(torch.cat([cdr3a, cdr3b, va, ja, vb, jb], dim=-1))
        return tcr, pmhc

    def forward(self, batch):
        tcr, pmhc = self.encode(batch)
        z_tcr = F.normalize(self.proj_tcr(tcr), dim=-1)
        z_pmhc = F.normalize(self.proj_pmhc(pmhc), dim=-1)
        return z_tcr, z_pmhc

    def score(self, batch):
        z_tcr, z_pmhc = self.forward(batch)
        return (z_tcr * z_pmhc).sum(-1)
