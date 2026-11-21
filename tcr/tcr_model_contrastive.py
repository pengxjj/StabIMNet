#!/usr/bin/env python3
"""TCRpMHCContrastiveModel — binding model trained with a contrastive objective.

Identical encoder to TCRpMHCBindingModel (char-embedding ResidueEncoder for
CDR3a/b + peptide, gene/allele embeddings, pMHC cross-attention, tcr_mlp). The
difference is the training objective: instead of a BCE classification head on
swap-decoys, the model projects the TCR and pMHC embeddings into a shared space
and is trained with a symmetric multi-positive InfoNCE (CLIP-style) loss, using
only *positive* (TCR, pMHC) pairs. The binding score is the cosine similarity
between the projected TCR and pMHC embeddings.

This mirrors ConTCR / UpTCR (2025): contrastive objectives are reported to
generalize better to unseen epitopes than BCE classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tcr_model import ResidueEncoder
from base.models.layers import CrossModalAttentionBlock


class TCRpMHCContrastiveModel(nn.Module):
    def __init__(self, d_model=64, n_heads=2, dropout=0.1, max_len=32,
                 n_v_alpha=1, n_j_alpha=1, n_v_beta=1, n_j_beta=1, n_allele=1,
                 proj_dim=64):
        super().__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.res_enc = ResidueEncoder(d_model, n_heads, dropout, max_len)
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
        # CLIP-style 2-layer projection head
        self.proj_tcr = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim),
        )
        self.proj_pmhc = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim),
        )
        # learnable temperature, init ~ CLIP default 1/0.07
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())

    def encode(self, batch):
        """Return (tcr [B,d], pmhc [B,d]) before projection."""
        _, cdr3a = self.res_enc(batch["cdr3a"])
        _, cdr3b = self.res_enc(batch["cdr3b"])
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
        """Return L2-normalized (z_tcr [B,proj], z_pmhc [B,proj])."""
        tcr, pmhc = self.encode(batch)
        z_tcr = F.normalize(self.proj_tcr(tcr), dim=-1)
        z_pmhc = F.normalize(self.proj_pmhc(pmhc), dim=-1)
        return z_tcr, z_pmhc

    def score(self, batch):
        """Cosine similarity in [-1,1] used as the binding score."""
        z_tcr, z_pmhc = self.forward(batch)
        return (z_tcr * z_pmhc).sum(-1)


def multi_positive_infonce(z_tcr, z_pmhc, logit_scale, pmhc_id):
    """Symmetric multi-positive InfoNCE (CLIP-style).

    z_tcr, z_pmhc: [N, d] L2-normalized.
    logit_scale:   scalar nn.Parameter (log of temperature).
    pmhc_id:       [N] long, records with equal id share the same pMHC
                   (peptide+allele) and are mutual positives (incl. self).

    Returns the mean of the tcr->pmhc and pmhc->tcr losses.
    """
    temp = logit_scale.exp().clamp(max=100.0)
    logits = z_tcr @ z_pmhc.T * temp            # [N, N]
    pos_mask = (pmhc_id[:, None] == pmhc_id[None, :]).float()  # symmetric

    def one_way(sim, mask):
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp = sim.exp()
        denom = exp.sum(dim=1, keepdim=True).clamp(min=1e-8)
        num = (exp * mask).sum(dim=1, keepdim=True)
        return -( (num / denom).log() ).mean()

    return 0.5 * (one_way(logits, pos_mask) + one_way(logits.T, pos_mask.T))
