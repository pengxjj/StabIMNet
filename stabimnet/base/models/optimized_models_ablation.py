"""Ablation variant of ``HybridModelOpt`` for Phase-2 tower drop-one experiments.

``HybridModelOpt_Ablate`` is identical to ``HybridModelOpt`` (same MLP-VAE,
same geometry-aware EGNN, same property tower, same classifier) except that the
set of tower embeddings concatenated into the classifier input is configurable
via ``fuse``. Everything else -- including the VAE reconstruction + KL loss
terms and the structure tower's forward pass -- is left untouched, so a
drop-one ablation changes *only* which views reach the classifier, keeping the
loss and optimizer schedules byte-for-byte identical to the reference run.

Towers (canonical concat order struct -> seq -> prop):

* ``"struct"``  -> EGNN mean-pool readout, ``gat_hidden_channels`` dims
* ``"seq"``     -> VAE latent ``z``, ``vae_latent_dim`` dims
* ``"prop"``    -> property embedding, ``property_embedding_dim`` dims
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .egnn_opt import EGNNConvOpt
from .egnn_torch import EGNNConvTorch
from .layers import SelfAttention
from .optimized_models import (
    MLPVAE,
    _batch_index,
    DEFAULT_LATENT_DIM,
    DEFAULT_STRUCT_HIDDEN,
    DEFAULT_PROPERTY_EMBEDDING_DIM,
)

__all__ = ["HybridModelOpt_Ablate", "HybridModelOpt_Comparative_Ablate"]

_EGNN_CONV = {"opt": EGNNConvOpt, "torch": EGNNConvTorch}

# Canonical concat order (matches HybridModelOpt's [struct ; z ; prop]).
_TOWER_ORDER = ("struct", "seq", "prop")
_TOWER_DIM = {
    "struct": "gat_hidden_channels",
    "seq": "vae_latent_dim",
    "prop": "property_embedding_dim",
}


class HybridModelOpt_Ablate(nn.Module):
    """Drop-one-tower ablation of the Stage-2 (IEDB) ``HybridModelOpt``.

    ``fuse`` is an iterable of tower names (subset of ``{"struct","seq","prop"}``).
    The classifier input dimension ``cls_in`` is the sum of the selected towers'
    embedding dims. All three towers are still constructed and run (so the VAE
    reconstruction/KL terms and the structure forward pass are unchanged); only
    the ``combined`` vector fed to the classifier differs.
    """

    def __init__(self, vae_input_dim, device,
                 fuse=("struct", "seq", "prop"),
                 gcn_layers=5,
                 vae_hidden_dim=768,
                 vae_latent_dim=DEFAULT_LATENT_DIM,
                 gat_hidden_channels=DEFAULT_STRUCT_HIDDEN,
                 property_embedding_dim=DEFAULT_PROPERTY_EMBEDDING_DIM,
                 dropout=0.1,
                 egnn_type="opt",
                 raw_displacement=False,
                 decoder_layers=3,
                 mask_ratio=0.0,
                 *args, **kwargs):
        super().__init__()
        if not isinstance(fuse, (tuple, list, set)):
            fuse = tuple(fuse.split(","))
        fuse = tuple(fuse)
        unknown = set(fuse) - set(_TOWER_ORDER)
        if unknown:
            raise ValueError(f"unknown towers in fuse: {unknown} (allowed {_TOWER_ORDER})")
        self.fuse = fuse

        self.device = device
        self.vae_latent_dim = vae_latent_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.vae_input_dim = vae_input_dim
        self.egnn_type = egnn_type
        self.raw_displacement = raw_displacement

        conv = _EGNN_CONV[egnn_type]
        conv_kwargs = {"raw_displacement": raw_displacement} if egnn_type == "opt" else {}
        self.GCN_layers = nn.ModuleList(
            [conv(20, gat_hidden_channels, gat_hidden_channels, 1, **conv_kwargs)]
        )
        for _ in range(gcn_layers):
            self.GCN_layers.append(conv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1, **conv_kwargs))
        self.self_attention = SelfAttention(gat_hidden_channels)

        self.vae = MLPVAE(vae_input_dim, vae_hidden_dim=vae_hidden_dim,
                          vae_latent_dim=vae_latent_dim,
                          property_embedding_dim=property_embedding_dim,
                          decoder_layers=decoder_layers,
                          mask_ratio=mask_ratio)

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(32, property_embedding_dim), nn.ReLU(True))

        dims = {"struct": gat_hidden_channels, "seq": vae_latent_dim,
                "prop": property_embedding_dim}
        cls_in = sum(dims[t] for t in _TOWER_ORDER if t in self.fuse)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 32), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(32, 1))

    def _forward_structure(self, graph_data):
        dev = next(self.GCN_layers[0].parameters()).device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)
        batch = _batch_index(graph_data, dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(node_feat.shape[0], -1, self.gat_hidden_channels)
        node_feat, _ = self.self_attention(node_feat)
        node_feat = node_feat.view(-1, self.gat_hidden_channels)
        return global_mean_pool(node_feat, batch)

    def _encode_sequence(self, sequence_data):
        x = sequence_data.view(sequence_data.shape[0], -1)
        mu, logvar = self.vae.encode(x)
        return self.vae.reparameterize(mu, logvar), mu, logvar

    def forward(self, graph_data, sequence_data, peptide_property,
                return_embedding=False, return_attention=False):
        struct_feat = self._forward_structure(graph_data)
        prop_emb = self.property_embedding(peptide_property)
        z, mu, logvar = self._encode_sequence(sequence_data)
        recon_x = self.vae.decode(z, prop_emb)

        parts = []
        if "struct" in self.fuse:
            parts.append(struct_feat)
        if "seq" in self.fuse:
            parts.append(z)
        if "prop" in self.fuse:
            parts.append(prop_emb)
        combined = torch.cat(parts, dim=1)
        final_output = self.classifier(combined)

        if return_embedding:
            return struct_feat, mu, logvar, final_output
        return recon_x, mu, logvar, final_output


class HybridModelOpt_Comparative_Ablate(nn.Module):
    """Delta-head ablation of ``HybridModelOpt_Comparative`` (Stage-3 CEDAR).

    Identical to ``HybridModelOpt_Comparative`` (same shared MLP-VAE + EGNN
    backbone, same ``forward_item``, same ``forward_comparative`` 5-tuple return)
    except that how the cancer/wildtype embeddings are fused into the classifier
    is configurable via ``fusion``. Only the ``combined`` vector differs; the VAE
    reconstruction/KL terms and both structure forward passes are untouched, so a
    drop-delta ablation changes *only* what reaches the classifier.

    ``fusion`` choices (``embed_dim = gat_hidden_channels + vae_latent_dim``):

    * ``"full"``   -> ``[emb_c ; emb_w ; emb_c - emb_w ; ||delta||]``  (cls_in = 3D+1 = 385)
    * ``"concat"`` -> ``[emb_c ; emb_w]``                              (no-delta, 2D = 256)
    * ``"delta"``  -> ``[emb_c - emb_w ; ||delta||]``                  (delta-only, D+1 = 129)
    * ``"add"``    -> ``emb_c + emb_w``                                (D = 128)
    * ``"mul"``    -> ``emb_c * emb_w``                                (D = 128)
    """

    _CLS_IN = None  # set per-instance in __init__

    def __init__(self, vae_input_dim, device,
                 fusion="full",
                 emb_fuse=("struct", "seq"),
                 gcn_layers=5,
                 vae_hidden_dim=512,
                 vae_latent_dim=DEFAULT_LATENT_DIM,
                 gat_hidden_channels=DEFAULT_STRUCT_HIDDEN,
                 property_embedding_dim=DEFAULT_PROPERTY_EMBEDDING_DIM,
                 dropout=0.1,
                 egnn_type="opt",
                 raw_displacement=False,
                 decoder_layers=1,
                 mask_ratio=0.0,
                 *args, **kwargs):
        super().__init__()
        if fusion not in ("full", "concat", "delta", "add", "mul"):
            raise ValueError(f"unknown fusion {fusion!r} (full/concat/delta/add/mul)")
        self.fusion = fusion
        if isinstance(emb_fuse, str):
            emb_fuse = tuple(emb_fuse.split(","))
        emb_fuse = tuple(emb_fuse)
        if not emb_fuse or set(emb_fuse) - {"struct", "seq"}:
            raise ValueError(f"emb_fuse must be a non-empty subset of {{struct,seq}}, got {emb_fuse}")
        self.emb_fuse = emb_fuse

        self.device = device
        self.vae_latent_dim = vae_latent_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.vae_input_dim = vae_input_dim
        self.egnn_type = egnn_type
        self.raw_displacement = raw_displacement
        self.embed_dim = sum(
            {"struct": gat_hidden_channels, "seq": vae_latent_dim}[t] for t in emb_fuse
        )

        conv = _EGNN_CONV[egnn_type]
        conv_kwargs = {"raw_displacement": raw_displacement} if egnn_type == "opt" else {}
        self.GCN_layers = nn.ModuleList(
            [conv(20, gat_hidden_channels, gat_hidden_channels, 1, **conv_kwargs)]
        )
        for _ in range(gcn_layers):
            self.GCN_layers.append(conv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1, **conv_kwargs))
        self.self_attention = SelfAttention(gat_hidden_channels)

        self.vae = MLPVAE(vae_input_dim, vae_hidden_dim=vae_hidden_dim,
                          vae_latent_dim=vae_latent_dim,
                          property_embedding_dim=property_embedding_dim,
                          decoder_layers=decoder_layers,
                          mask_ratio=mask_ratio)

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(32, property_embedding_dim), nn.ReLU(True))

        cls_in = {
            "full": self.embed_dim * 3 + 1,
            "concat": self.embed_dim * 2,
            "delta": self.embed_dim + 1,
            "add": self.embed_dim,
            "mul": self.embed_dim,
        }[fusion]
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 32), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(32, 1))

    def _combine(self, emb_c, emb_w):
        if self.fusion == "full":
            delta = emb_c - emb_w
            magnitude = delta.norm(p=2, dim=1, keepdim=True)
            return torch.cat([emb_c, emb_w, delta, magnitude], dim=1)
        if self.fusion == "concat":
            return torch.cat([emb_c, emb_w], dim=1)
        if self.fusion == "delta":
            delta = emb_c - emb_w
            magnitude = delta.norm(p=2, dim=1, keepdim=True)
            return torch.cat([delta, magnitude], dim=1)
        if self.fusion == "add":
            return emb_c + emb_w
        if self.fusion == "mul":
            return emb_c * emb_w
        raise ValueError(f"unknown fusion {self.fusion}")

    def forward_item(self, graph_data, sequence_data, peptide_property):
        dev = next(self.GCN_layers[0].parameters()).device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)
        batch = _batch_index(graph_data, dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(node_feat.shape[0], -1, self.gat_hidden_channels)
        node_feat, _ = self.self_attention(node_feat)
        node_feat = node_feat.view(-1, self.gat_hidden_channels)
        struct_feat = global_mean_pool(node_feat, batch)

        prop_emb = self.property_embedding(peptide_property)
        x = sequence_data.view(sequence_data.shape[0], -1)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        recon_x = self.vae.decode(z, prop_emb)

        parts = []
        if "struct" in self.emb_fuse:
            parts.append(struct_feat)
        if "seq" in self.emb_fuse:
            parts.append(z)
        emb = torch.cat(parts, dim=1)
        return emb, recon_x, mu, logvar

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):
        gc, gw = graph_data_pair
        sc, sw = sequence_data_pair
        pc, pw = peptide_property_pair

        emb_c, recon_c, mu_c, logvar_c = self.forward_item(gc, sc, pc)
        emb_w, recon_w, mu_w, logvar_w = self.forward_item(gw, sw, pw)

        combined = self._combine(emb_c, emb_w)
        final_output = self.classifier(combined)

        return ([emb_c, emb_w], [recon_c, recon_w], [mu_c, mu_w], [logvar_c, logvar_w], final_output)
