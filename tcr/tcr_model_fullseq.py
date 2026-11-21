#!/usr/bin/env python3
"""TCR-pMHC binding model with full V-region sequence encoding.

Extends the CDR3-only TCRpMHCBindingModel by additionally encoding the assembled
full alpha/beta V regions (framework + CDR1/2/3, ~110-120 aa), which contain the
MHC-contact residues that CDR3 alone misses. The CDR3/peptide encoder, gene
embeddings, allele embedding, pMHC cross-attention and score MLP are shared with
the base model (so a pre-trained checkpoint transfers); the full-sequence encoder
and the TCR fusion MLP (now 8 inputs) are trained fresh.
"""
import torch
import torch.nn as nn

from base.models.layers import CrossModalAttentionBlock
from tcr_model import ResidueEncoder

FULL_MAX_LEN = 140


class TCRpMHCBindingModelFullSeq(nn.Module):
    def __init__(self, d_model=64, n_heads=2, dropout=0.1, max_len=32,
                 full_max_len=FULL_MAX_LEN,
                 n_v_alpha=1, n_j_alpha=1, n_v_beta=1, n_j_beta=1, n_allele=1):
        super().__init__()
        self.d_model = d_model
        self.res_enc = ResidueEncoder(d_model, n_heads, dropout, max_len)       # CDR3 + peptide
        self.full_enc = ResidueEncoder(d_model, n_heads, dropout, full_max_len)  # full V regions
        self.v_alpha_emb = nn.Embedding(n_v_alpha, d_model)
        self.j_alpha_emb = nn.Embedding(n_j_alpha, d_model)
        self.v_beta_emb = nn.Embedding(n_v_beta, d_model)
        self.j_beta_emb = nn.Embedding(n_j_beta, d_model)
        self.allele_emb = nn.Embedding(n_allele, d_model)

        self.pmhc_attn = CrossModalAttentionBlock(d_model, n_heads, dropout)

        self.tcr_mlp = nn.Sequential(
            nn.Linear(d_model * 8, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch):
        _, cdr3a = self.res_enc(batch["cdr3a"])
        _, cdr3b = self.res_enc(batch["cdr3b"])
        pep_tokens, _ = self.res_enc(batch["peptide"])
        _, full_a = self.full_enc(batch["full_a"])
        _, full_b = self.full_enc(batch["full_b"])

        va = self.v_alpha_emb(batch["va"])
        ja = self.j_alpha_emb(batch["ja"])
        vb = self.v_beta_emb(batch["vb"])
        jb = self.j_beta_emb(batch["jb"])
        allele = self.allele_emb(batch["allele"])

        allele_ctx = allele.unsqueeze(1)
        pep_mask = (batch["peptide"] != 0)
        pmhc_tokens = self.pmhc_attn(
            pep_tokens, allele_ctx,
            key_padding_mask=torch.ones(batch["peptide"].shape[0], 1,
                                        dtype=torch.bool, device=pep_tokens.device))
        pmhc = (pmhc_tokens * pep_mask.unsqueeze(-1).float()).sum(1) / \
               pep_mask.sum(1, keepdim=True).clamp(min=1)

        tcr = self.tcr_mlp(torch.cat([cdr3a, cdr3b, full_a, full_b, va, ja, vb, jb], dim=-1))
        score = self.score_mlp(torch.cat([tcr, pmhc], dim=-1))
        return score

    def load_pretrained(self, state_dict):
        """Load shared layers from a CDR3-only checkpoint (strict=False)."""
        own = self.state_dict()
        shared = {k: v for k, v in state_dict.items()
                  if k in own and own[k].shape == v.shape}
        self.load_state_dict(shared, strict=False)
        return shared
