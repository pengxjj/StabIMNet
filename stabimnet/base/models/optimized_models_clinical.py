"""Clinical single-branch variant of dec3's Phase-3 comparative model.

The dec3 Phase-3 model (:class:`optimized_models.HybridModelOpt_Comparative`)
scores a *pair* of mutant vs wildtype via a mutation-delta classifier
``[emb_c ; emb_w ; emb_c - emb_w ; ||emb_c - emb_w||]``, so it cannot score a
single peptide in isolation.  The ImmunoStruct clinical survival validation
(Fig. 5c,d) needs a single-branch immunogenicity head that eats only the cancer
embedding — exactly what the ORIGINAL model does via ``use_wt_for_downstream=False``.

This class reuses the dec3 backbone (geometry-aware ``EGNNConvOpt`` + ``MLPVAE`` +
property tower) and swaps ONLY the classifier: instead of ``embed_dim*3+1`` it
takes ``embed_dim`` (``[struct_feat ; z]``).  ``forward_comparative`` still runs
the two-branch pass so the reconstruction + contrastive objectives keep seeing
the wildtype, but the classifier output now depends on the cancer branch alone.

Standalone file (no in-place edits to ``optimized_models.py``), mirroring the
"Never Modify Models In Place" convention.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .optimized_models import (
    MLPVAE,
    _EGNN_CONV,
    _batch_index,
    DEFAULT_LATENT_DIM,
    DEFAULT_STRUCT_HIDDEN,
    DEFAULT_PROPERTY_EMBEDDING_DIM,
)
from .layers import SelfAttention

__all__ = ["HybridModelOpt_ComparativeSingle"]


class HybridModelOpt_ComparativeSingle(nn.Module):
    """dec3 backbone + single-branch immunogenicity classifier.

    Identical to :class:`optimized_models.HybridModelOpt_Comparative` except the
    classifier head is ``Linear(embed_dim, ...)`` (cancer-only) instead of
    ``Linear(embed_dim*3+1, ...)`` (mutation-delta).  ``forward`` is added so a
    single (wt-free) pMHC graph can be scored directly, matching the stock
    ``HybridModelv2`` / ``HybridModelOpt`` 4-tuple ``(recon_x, mu, logvar,
    final_output)`` signature used by the clinical inference loop.
    """

    def __init__(self, vae_input_dim, device,
                 gcn_layers=5,
                 vae_hidden_dim=512,
                 vae_latent_dim=DEFAULT_LATENT_DIM,
                 gat_hidden_channels=DEFAULT_STRUCT_HIDDEN,
                 property_embedding_dim=DEFAULT_PROPERTY_EMBEDDING_DIM,
                 dropout=0.1,
                 egnn_type="opt",
                 decoder_layers=1,
                 mask_ratio=0.0):
        super().__init__()
        self.device = device
        self.vae_latent_dim = vae_latent_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.vae_input_dim = vae_input_dim
        self.egnn_type = egnn_type
        self.embed_dim = gat_hidden_channels + vae_latent_dim

        conv = _EGNN_CONV[egnn_type]
        self.GCN_layers = nn.ModuleList(
            [conv(20, gat_hidden_channels, gat_hidden_channels, 1)]
        )
        for _ in range(gcn_layers):
            self.GCN_layers.append(conv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))
        self.self_attention = SelfAttention(gat_hidden_channels)

        self.vae = MLPVAE(vae_input_dim, vae_hidden_dim=vae_hidden_dim,
                          vae_latent_dim=vae_latent_dim,
                          property_embedding_dim=property_embedding_dim,
                          decoder_layers=decoder_layers,
                          mask_ratio=mask_ratio)

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(32, property_embedding_dim), nn.ReLU(True))

        # single-branch classifier over [struct_feat ; z] (cancer-only, no wt/delta)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 32), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(32, 1))

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
        struct_feat = global_mean_pool(node_feat, batch)           # [B, D]

        prop_emb = self.property_embedding(peptide_property)       # [B, P]
        x = sequence_data.view(sequence_data.shape[0], -1)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        recon_x = self.vae.decode(z, prop_emb)

        emb = torch.cat([struct_feat, z], dim=1)                   # [B, D+D]
        return emb, recon_x, mu, logvar

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):
        gc, gw = graph_data_pair
        sc, sw = sequence_data_pair
        pc, pw = peptide_property_pair

        emb_c, recon_c, mu_c, logvar_c = self.forward_item(gc, sc, pc)
        emb_w, recon_w, mu_w, logvar_w = self.forward_item(gw, sw, pw)

        final_output = self.classifier(emb_c)                       # cancer-only

        return ([emb_c, emb_w], [recon_c, recon_w], [mu_c, mu_w], [logvar_c, logvar_w], final_output)

    def forward(self, graph_data, sequence_data, peptide_property,
                return_embedding=False, return_attention=False):
        emb, recon_x, mu, logvar = self.forward_item(graph_data, sequence_data, peptide_property)
        final_output = self.classifier(emb)
        if return_embedding:
            return emb, mu, logvar, final_output
        return recon_x, mu, logvar, final_output
