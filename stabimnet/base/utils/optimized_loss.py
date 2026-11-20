"""Losses for ImmunoStruct-Opt (standalone, no in-place edits).

Literature grounding
--------------------
* Focal loss — Lin, Goyal et al., "Focal Loss for Dense Object Detection", 2017.
  Down-weights easy examples to sharpen the decision boundary on imbalanced
  immunogenicity labels (~19% positive).
* Free-bits β-VAE — Kingma et al., "Improving Variational Inference with Inverse
  Autoregressive Flow", 2016; Higgins et al., "β-VAE", ICLR 2017. Clamps each
  latent dimension's KL to a floor so the VAE does not posterior-collapse.
* Contrastive — Chen et al., "SimCLR", 2020; Khosla et al., "Supervised
  Contrastive Learning" (SupCon), 2020. Temperature-scaled cosine similarity on
  unit-normalized projected embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OptimizedLosses", "ContrastiveLoss"]


def _focal_bce_with_logits(logits, y, pos_weight, gamma=1.0):
    logits = logits.view(-1)
    y = y.view(-1)
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    if gamma > 0:
        p = torch.sigmoid(logits)
        pt = torch.where(y == 1, p, 1 - p)
        bce = ((1 - pt) ** gamma) * bce
    return bce.mean()


def _kld_free_bits(mu, logvar, free_bits=0.0):
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
    if free_bits > 0:
        kld_per_dim = torch.clamp(kld_per_dim, min=free_bits)
    return kld_per_dim.mean()


def _mmd_rbf(x, y, sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)):
    """Squared MMD between two sample sets with a mixture of RBF kernels.

    Pairwise squared distances are scaled by the latent dim so the kernel is
    scale-invariant to the (high) latent dimensionality; the mixture of
    bandwidths keeps it a characteristic divergence (InfoVAE / MMD-VAE,
    Zhao et al. 2017). Biased estimator (includes diagonals) — stable for
    gradient-based training.
    """
    d = x.shape[1]
    def _sq(a, b):
        aa = (a * a).sum(1, keepdim=True)
        bb = (b * b).sum(1, keepdim=True)
        return (aa + bb.t() - 2.0 * (a @ b.t())) / d

    xx = _sq(x, x)
    yy = _sq(y, y)
    xy = _sq(x, y)
    mmd = 0.0
    for s in sigmas:
        kxx = torch.exp(-xx / (2.0 * s * s))
        kyy = torch.exp(-yy / (2.0 * s * s))
        kxy = torch.exp(-xy / (2.0 * s * s))
        mmd = mmd + kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    return mmd / len(sigmas)


class ContrastiveLoss(nn.Module):
    """Cosine contrastive on cancer/wildtype embeddings (a cosine-embedding /
    supervised-contrastive pull-push objective).

    Non-immunogenic (mutation ~ neutral) pairs are pulled together (target +1);
    immunogenic (mutation matters) pairs are pushed apart (target -1). Similarity
    is the cosine of L2-normalized projected embeddings, bounded in [-1, 1], so
    the loss is naturally stable and scale-free.
    """

    def __init__(self, embed_dim, proj_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim))

    def forward(self, emb_c, emb_w, is_immunogenic):
        if is_immunogenic.unique().numel() < 2:
            return torch.tensor(0.0, device=emb_c.device)
        zc = F.normalize(self.projector(emb_c), dim=-1)
        zw = F.normalize(self.projector(emb_w), dim=-1)
        sim = (zc * zw).sum(-1)                                  # [B], in [-1, 1]
        target = 1.0 - 2.0 * is_immunogenic.float()              # +1 pull / -1 push
        return F.mse_loss(sim, target)


class OptimizedLosses:
    def __init__(self, vae_input_dim, class_weights,
                 bce_weight=2.0, mse_weight=0.5, kld_weight=0.5,
                 focal_gamma=1.0, free_bits=0.0,
                 vae_type="kld", mmd_weight=1.0,
                 contrastive_embed_dim=256, contrastive_proj_dim=128):
        self.vae_input_dim = vae_input_dim
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.kld_weight = kld_weight
        self.focal_gamma = focal_gamma
        self.free_bits = free_bits
        self.vae_type = vae_type
        self.mmd_weight = mmd_weight
        self.pos_weight = torch.tensor(float(class_weights[0]) / float(class_weights[1])).float()
        self.contrastive = ContrastiveLoss(contrastive_embed_dim,
                                           proj_dim=contrastive_proj_dim)

    def kld(self, mu, logvar):
        return _kld_free_bits(mu, logvar, self.free_bits)

    def mmd(self, mu, logvar):
        """MMD between the aggregate posterior q(z) and the prior N(0,I).

        Re-samples z = mu + eps*std inside the loss (an unbiased MC estimate of
        q(z|x)) and compares it against standard-normal samples — the InfoVAE /
        MMD-VAE regularizer (Zhao et al. 2017), which matches distributions
        more softly than the per-sample KL and avoids posterior collapse.
        """
        std = torch.exp(0.5 * logvar)
        z_q = mu + torch.randn_like(std) * std
        z_p = torch.randn_like(z_q)
        return _mmd_rbf(z_q, z_p)

    def recon(self, recon_x, x, mask=None):
        x = x.view(-1, self.vae_input_dim)
        if mask is None:
            return F.mse_loss(recon_x, x, reduction="mean")
        # Masked reconstruction: backprop only the masked residues (mask: [B, L]).
        seq_len = mask.shape[1]
        per_res = self.vae_input_dim // seq_len
        rx = recon_x.view(-1, seq_len, per_res)
        xt = x.view(-1, seq_len, per_res)
        m = mask.unsqueeze(-1).float()
        return (m * (rx - xt) ** 2).sum() / (m.sum() + 1e-8)

    def bce(self, logits, y):
        return _focal_bce_with_logits(logits, y, self.pos_weight, self.focal_gamma)

    def vae_terms(self, recon_x, x, mu, logvar, mask=None):
        """Reconstruction + latent regularizer (KL or MMD, shared across stages)."""
        if self.vae_type == "mmd":
            reg = self.mmd_weight * self.mmd(mu, logvar)
        else:
            reg = self.kld_weight * self.kld(mu, logvar)
        return self.mse_weight * self.recon(recon_x, x, mask=mask) + reg
