#!/usr/bin/env python3
"""TCR-pMHC binding model (pure PyTorch).

A self-contained baseline that scores whether a TCR (alpha/beta CDR3 + V/J genes)
binds a pMHC (peptide + HLA allele), proving the "TCR recognition" modality is
modelable from sequence alone. Trained on VDJdb paired alpha-beta records with
CDR3-shuffle decoys (see preprocess_vdjdb_tcr.py).

Encoding:
  - character embedding (20 AA + <pad>) + learnable positional embedding
  - 1 transformer encoder layer + masked mean pool for cdr3.alpha/cdr3.beta/peptide
  - learned embedding lookups for v.alpha/j.alpha/v.beta/j.beta and the HLA allele
Fusion:
  - pMHC = CrossModalAttentionBlock(query=peptide residues, context=allele token)
  - TCR  = MLP(concat(cdr3a, cdr3b, V/J genes))
  - score = MLP(concat(tcr, pmhc)) -> scalar logit (BCEWithLogitsLoss)

Reuses CrossModalAttentionBlock from base.models.layers (imports only torch).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.models.layers import CrossModalAttentionBlock

# 20 standard amino acids + <pad>
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
PAD_TOKEN = "<pad>"
CHAR_VOCAB = [PAD_TOKEN] + AA_ORDER  # index 0 = pad
CHAR2IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
UNK_IDX = 0


def encode_seq(seq, max_len):
    """Encode an AA string to a LongTensor of char indices, padded to max_len."""
    ids = [CHAR2IDX.get(c, UNK_IDX) for c in seq[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class ResidueEncoder(nn.Module):
    """Character embedding + positional embedding + 1 transformer encoder layer.

    Returns a [B, d] pooled vector and also exposes the per-residue [B, L, d]
    tokens for downstream cross-attention (peptide -> HLA).
    """

    def __init__(self, d_model=64, n_heads=2, dropout=0.1, max_len=32):
        super().__init__()
        self.d_model = d_model
        self.char_emb = nn.Embedding(len(CHAR_VOCAB), d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x, lengths=None):
        """x: [B, L] LongTensor of char indices.

        Returns (tokens [B, L, d], pooled [B, d]).
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.char_emb(x) + self.pos_emb(pos)  # [B, L, d]
        if lengths is None:
            mask = (x != 0)  # True = valid
        else:
            arange = torch.arange(L, device=x.device).unsqueeze(0)
            mask = arange < lengths.unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~mask)  # mask True = ignored
        pooled = (h * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return h, pooled


class TCRpMHCBindingModel(nn.Module):
    def __init__(self, d_model=64, n_heads=2, dropout=0.1, max_len=32,
                 n_v_alpha=1, n_j_alpha=1, n_v_beta=1, n_j_beta=1, n_allele=1,
                 ablate=None):
        super().__init__()
        self.d_model = d_model
        self.ablate = ablate  # None | "tcr" (pMHC-only) | "pmhc" (TCR-only)
        self.res_enc = ResidueEncoder(d_model, n_heads, dropout, max_len)
        self.v_alpha_emb = nn.Embedding(n_v_alpha, d_model)
        self.j_alpha_emb = nn.Embedding(n_j_alpha, d_model)
        self.v_beta_emb = nn.Embedding(n_v_beta, d_model)
        self.j_beta_emb = nn.Embedding(n_j_beta, d_model)
        self.allele_emb = nn.Embedding(n_allele, d_model)

        # pMHC: peptide residues attend to the HLA allele token
        self.pmhc_attn = CrossModalAttentionBlock(d_model, n_heads, dropout)

        self.tcr_mlp = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch):
        """batch: dict with keys (all LongTensors on device):
            cdr3a, cdr3b, peptide        [B, L] char indices
            va, ja, vb, jb               [B] gene indices
            allele                       [B] allele index
        Returns [B, 1] logits.
        """
        _, cdr3a = self.res_enc(batch["cdr3a"])
        _, cdr3b = self.res_enc(batch["cdr3b"])
        pep_tokens, pep_pooled = self.res_enc(batch["peptide"])

        va = self.v_alpha_emb(batch["va"])
        ja = self.j_alpha_emb(batch["ja"])
        vb = self.v_beta_emb(batch["vb"])
        jb = self.j_beta_emb(batch["jb"])

        allele = self.allele_emb(batch["allele"])  # [B, d]

        # pMHC: peptide query attends to a single HLA allele token
        allele_ctx = allele.unsqueeze(1)  # [B, 1, d]
        pep_mask = (batch["peptide"] != 0)  # [B, L] True=valid
        pmhc_tokens = self.pmhc_attn(pep_tokens, allele_ctx, key_padding_mask=torch.ones(
            batch["peptide"].shape[0], 1, dtype=torch.bool, device=pep_tokens.device))
        pmhc = (pmhc_tokens * pep_mask.unsqueeze(-1).float()).sum(1) / pep_mask.sum(1, keepdim=True).clamp(min=1)

        tcr = self.tcr_mlp(torch.cat([cdr3a, cdr3b, va, ja, vb, jb], dim=-1))  # [B, d]
        if self.ablate == "tcr":
            tcr = torch.zeros_like(tcr)      # keep pMHC only
        elif self.ablate == "pmhc":
            pmhc = torch.zeros_like(pmhc)    # keep TCR only
        score = self.score_mlp(torch.cat([tcr, pmhc], dim=-1))  # [B, 1]
        return score
