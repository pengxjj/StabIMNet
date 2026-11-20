"""Cancer-Wildtype Contrastive Learning models (ported from ImmunoStruct upstream).

These models pair each cancer (mutant) epitope with its wildtype counterpart and
run both through a *shared* tower (EGNN structure + VAE sequence), then fuse the
two embeddings for the downstream immunogenicity classifier:

    combined = cat([emb_cancer, emb_wt])   # wildtype = natural negative anchor

With ``use_wt_for_downstream=True`` (default) the classifier input dimension is
doubled to ``(vae_latent + property_embedding + gat_hidden) * 2``.

Differences from upstream:
  * ``dgl.nn.EGNNConv`` -> :class:`EGNNConvTorch` (DGL pip wheel is CPU-only).
  * ``encode_vae`` clamps logvar to [-20, 10] (the NaN / mode-collapse fix).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .egnn_torch import EGNNConvTorch
from .layers import MultiHeadAttention

EGNNConv = EGNNConvTorch

__all__ = ["HybridModelv2_Comparative", "HybridModelv2_Comparative_SSL",
           "HybridModelv3_Comparative", "HybridModelv4_Comparative",
           "HybridModelv2_Comparative_Deterministic", "transfer_backbone_from_v2"]


# Default dims match the IEDB-pretrained HybridModelv2 (v2_phase2_best.pt) so
# its encoder backbone can be transferred for CEDAR fine-tuning.
DEFAULT_VAE_HIDDEN_DIM = 768
DEFAULT_VAE_LATENT_DIM = 64
DEFAULT_PROPERTY_EMBEDDING_DIM = 16


def transfer_backbone_from_v2(model, path, map_location=None):
    """Initialize the shared encoder backbone of a comparative model from a
    ``HybridModelv2`` checkpoint (e.g. the IEDB-pretrained ``v2_phase2_best.pt``).

    Only the shared towers are copied (EGNN structure ``GCN_layers``, GAT
    ``self_attention``, VAE sequence ``vae_fc1..4``, ``property_embedding``);
    the comparative fusion (``combined_attention``) and the doubled classifier
    head stay randomly initialized and are fine-tuned on CEDAR.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    for container in ("model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and container in ckpt and isinstance(ckpt[container], dict):
            ckpt = ckpt[container]
            break
    if not isinstance(ckpt, dict):
        raise TypeError(f"expected a state_dict, got {type(ckpt)}")

    prefixes = ("GCN_layers.", "self_attention.", "vae_fc1.", "vae_fc21.",
                "vae_fc22.", "vae_fc3.", "vae_fc4.", "property_embedding.")
    filtered = {k: v for k, v in ckpt.items() if k.startswith(prefixes)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[pretrained] transferred {len(filtered)} backbone tensors from "
          f"{path}; {len(missing)} missing (fusion/head random), "
          f"{len(unexpected)} unexpected")
    return len(filtered)


class HybridModelv2_Comparative(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = DEFAULT_VAE_HIDDEN_DIM,
                 vae_latent_dim: int = DEFAULT_VAE_LATENT_DIM,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim: int = DEFAULT_PROPERTY_EMBEDDING_DIM,
                 self_attention_heads: int = 1,
                 combined_attention_heads: int = 8,
                 use_wt_for_downstream: bool = True):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.use_wt_for_downstream = use_wt_for_downstream

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = MultiHeadAttention(gat_hidden_channels, self_attention_heads)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        self.combined_attention = MultiHeadAttention(32, combined_attention_heads, input_dim=1)

        # Fusion/Classifier layers
        self.classifier = self.get_classifier()

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        if self.use_wt_for_downstream:
            input_dim = (self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels) * 2
        else:
            input_dim = self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def encode_vae(self, x):
        h1 = F.relu(self.vae_fc1(x))
        # Clamp logvar: exp(logvar)>~88 overflows float32 -> NaN loss (mode-collapse death)
        return self.vae_fc21(h1), torch.clamp(self.vae_fc22(h1), min=-20.0, max=10.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_vae(self, z):
        h3 = F.relu(self.vae_fc3(z))
        return self.vae_fc4(h3)  # Sigmoid for reconstruction

    def load_trained(self, path, new_head=False, map_location=None, strict=False):
        self.load_state_dict(torch.load(path, map_location=map_location), strict=strict)
        if new_head:
            self.classifier = self.get_classifier().to(self.device)

    def forward_item(self, graph_data, sequence_data, peptide_property):
        dev = self.device
        x = graph_data
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        # Create batch tensor based on the number of nodes in each graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(x, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gat_hidden_channels)
        attention_output_n, attention_weights_n = self.self_attention(node_feat)
        attention_output_n = attention_output_n.view(-1, self.gat_hidden_channels)

        # Use global_mean_pool with the batch tensor
        x_gat_node = global_mean_pool(attention_output_n, batch_tensor.flatten())

        # peptide property
        peptide_property = self.property_embedding(peptide_property)

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)

        return mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):

        graph_data_cancer, graph_data_wt = graph_data_pair
        sequence_data_cancer, sequence_data_wt = sequence_data_pair
        peptide_property_cancer, peptide_property_wt = peptide_property_pair

        mu_cancer, logvar_cancer, x_gat_node_cancer, z_vae_cancer, attention_weights_n_cancer, recon_x_cancer = \
            self.forward_item(graph_data_cancer, sequence_data_cancer, peptide_property_cancer)
        mu_wt, logvar_wt, x_gat_node_wt, z_vae_wt, attention_weights_n_wt, recon_x_wt = \
            self.forward_item(graph_data_wt, sequence_data_wt, peptide_property_wt)
        combined_gat_only_cancer = torch.cat([x_gat_node_cancer], dim=1)

        # Fusion and final layers
        embeddings_cancer = torch.cat([x_gat_node_cancer, z_vae_cancer], dim=1)
        embeddings_wt = torch.cat([x_gat_node_wt, z_vae_wt], dim=1)
        if self.use_wt_for_downstream:
            combined = torch.cat([embeddings_cancer, embeddings_wt], dim=1)
        else:
            combined = embeddings_cancer

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only_cancer, mu_cancer, logvar_cancer, final_output

        if return_attention:
            return attention_weights_n_cancer, mu_cancer, logvar_cancer, final_output

        return [embeddings_cancer, embeddings_wt], [recon_x_cancer, recon_x_wt], [mu_cancer, mu_wt], [logvar_cancer, logvar_wt], final_output

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        """Single-item forward for pretraining / backward compatibility."""
        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            self.forward_item(graph_data, sequence_data, peptide_property)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        # Fusion and final layers
        if self.use_wt_for_downstream:
            # NOTE: repeating features as a hot fix for classifier dimension.
            combined = torch.cat([x_gat_node, z_vae, x_gat_node, z_vae], dim=1)
        else:
            combined = torch.cat([x_gat_node, z_vae], dim=1)

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output


class HybridModelv2_Comparative_SSL(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = DEFAULT_VAE_HIDDEN_DIM,
                 vae_latent_dim: int = DEFAULT_VAE_LATENT_DIM,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim: int = DEFAULT_PROPERTY_EMBEDDING_DIM,
                 self_attention_heads: int = 1,
                 combined_attention_heads: int = 8,
                 use_wt_for_downstream: bool = True,
                 mlp_features: int = 32):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.property_embedding_dim = property_embedding_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.use_wt_for_downstream = use_wt_for_downstream
        self.mlp_features = mlp_features

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = MultiHeadAttention(gat_hidden_channels, self_attention_heads)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        self.combined_attention = MultiHeadAttention(32, combined_attention_heads, input_dim=1)

        # Fusion/Classifier layers
        self.classifier = self.get_classifier()

        self.classifier_head = nn.Linear(self.mlp_features, 1)
        self.node_predictor_head = nn.Linear(self.mlp_features, 20)  # number of amino acids

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        if self.use_wt_for_downstream:
            input_dim = (self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels) * 2
        else:
            input_dim = self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

    def encode_vae(self, x):
        h1 = F.relu(self.vae_fc1(x))
        # Clamp logvar: exp(logvar)>~88 overflows float32 -> NaN loss (mode-collapse death)
        return self.vae_fc21(h1), torch.clamp(self.vae_fc22(h1), min=-20.0, max=10.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_vae(self, z):
        h3 = F.relu(self.vae_fc3(z))
        return self.vae_fc4(h3)  # Sigmoid for reconstruction

    def load_trained(self, path, new_head=False, map_location=None, strict=False):
        self.load_state_dict(torch.load(path, map_location=map_location), strict=strict)
        if new_head:
            self.classifier_head = nn.Linear(self.mlp_features, 1).to(self.device)

    def forward_item(self, graph_data, sequence_data, peptide_property):
        dev = self.device
        x = graph_data
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(x, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gat_hidden_channels)
        attention_output_n, attention_weights_n = self.self_attention(node_feat)
        attention_output_n = attention_output_n.view(-1, self.gat_hidden_channels)

        x_gat_node = global_mean_pool(attention_output_n, batch_tensor.flatten())

        peptide_property = self.property_embedding(peptide_property)

        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)

        return mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):

        graph_data_cancer, graph_data_wt = graph_data_pair
        sequence_data_cancer, sequence_data_wt = sequence_data_pair
        peptide_property_cancer, peptide_property_wt = peptide_property_pair

        mu_cancer, logvar_cancer, x_gat_node_cancer, z_vae_cancer, attention_weights_n_cancer, recon_x_cancer = \
            self.forward_item(graph_data_cancer, sequence_data_cancer, peptide_property_cancer)
        mu_wt, logvar_wt, x_gat_node_wt, z_vae_wt, attention_weights_n_wt, recon_x_wt = \
            self.forward_item(graph_data_wt, sequence_data_wt, peptide_property_wt)
        combined_gat_only_cancer = torch.cat([x_gat_node_cancer], dim=1)

        embeddings_cancer = torch.cat([x_gat_node_cancer, z_vae_cancer], dim=1)
        embeddings_wt = torch.cat([x_gat_node_wt, z_vae_wt], dim=1)
        if self.use_wt_for_downstream:
            combined = torch.cat([embeddings_cancer, embeddings_wt], dim=1)
        else:
            combined = embeddings_cancer

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        fusion_output = self.classifier(combined)
        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)

        if return_embedding:
            return combined_gat_only_cancer, mu_cancer, logvar_cancer, final_output, node_prediction

        if return_attention:
            return attention_weights_n_cancer, mu_cancer, logvar_cancer, final_output, node_prediction

        return [embeddings_cancer, embeddings_wt], [recon_x_cancer, recon_x_wt], [mu_cancer, mu_wt], [logvar_cancer, logvar_wt], final_output, node_prediction

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        """Single-item forward for pretraining / backward compatibility."""
        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            self.forward_item(graph_data, sequence_data, peptide_property)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        if self.use_wt_for_downstream:
            combined = torch.cat([x_gat_node, z_vae, x_gat_node, z_vae], dim=1)
        else:
            combined = torch.cat([x_gat_node, z_vae], dim=1)

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        fusion_output = self.classifier(combined)
        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output, node_prediction

        if return_attention:
            return attention_weights_n, mu, logvar, final_output, node_prediction

        return recon_x, mu, logvar, final_output, node_prediction


class HybridModelv3_Comparative(HybridModelv2_Comparative):
    """Geometric-decomposition comparative fusion (NeoPrecis-inspired).

    Same shared encoder backbone as ``HybridModelv2_Comparative`` (EGNN
    structure + GAT + VAE sequence + property embedding), so
    ``transfer_backbone_from_v2`` still works unchanged. The ONLY difference
    is the WT/MT fusion: instead of concatenating the two embeddings, we
    decompose the mutation into an explicit geometric vector,

        origin    = emb_wt                       (wildtype context)
        direction = normalize(emb_mt - emb_wt)   (mutation direction)
        magnitude = ||emb_mt - emb_wt||_2         (mutation magnitude)

    and feed ``cat([origin, direction, magnitude])`` to the classifier. This
    injects the "mutation magnitude" (peptide-intrinsic foreignness) as an
    explicit, first-class signal instead of leaving it for the MLP to discover
    from the concatenation.
    """

    def get_classifier(self):
        # +1 for the magnitude scalar.
        input_dim = (self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels) * 2 + 1
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):

        graph_data_cancer, graph_data_wt = graph_data_pair
        sequence_data_cancer, sequence_data_wt = sequence_data_pair
        peptide_property_cancer, peptide_property_wt = peptide_property_pair

        mu_cancer, logvar_cancer, x_gat_node_cancer, z_vae_cancer, attention_weights_n_cancer, recon_x_cancer = \
            self.forward_item(graph_data_cancer, sequence_data_cancer, peptide_property_cancer)
        mu_wt, logvar_wt, x_gat_node_wt, z_vae_wt, attention_weights_n_wt, recon_x_wt = \
            self.forward_item(graph_data_wt, sequence_data_wt, peptide_property_wt)
        combined_gat_only_cancer = torch.cat([x_gat_node_cancer], dim=1)

        embeddings_cancer = torch.cat([x_gat_node_cancer, z_vae_cancer], dim=1)
        embeddings_wt = torch.cat([x_gat_node_wt, z_vae_wt], dim=1)

        # Geometric decomposition of the mutation in embedding space.
        origin = embeddings_wt
        diff = embeddings_cancer - embeddings_wt
        magnitude = diff.norm(p=2, dim=1, keepdim=True)
        direction = diff / (magnitude + 1e-8)

        combined = torch.cat([origin, direction, magnitude], dim=1)

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only_cancer, mu_cancer, logvar_cancer, final_output

        if return_attention:
            return attention_weights_n_cancer, mu_cancer, logvar_cancer, final_output

        return [embeddings_cancer, embeddings_wt], [recon_x_cancer, recon_x_wt], [mu_cancer, mu_wt], [logvar_cancer, logvar_wt], final_output

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        """Single-item forward (comparative pretraining is unused for v3; kept
        dimension-compatible by a degenerate geometric vector with zero diff."""
        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            self.forward_item(graph_data, sequence_data, peptide_property)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        emb = torch.cat([x_gat_node, z_vae], dim=1)
        origin = emb
        direction = torch.zeros_like(emb)
        magnitude = torch.zeros((emb.shape[0], 1), device=emb.device)

        combined = torch.cat([origin, direction, magnitude], dim=1)

        combined = torch.unsqueeze(combined, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output


class HybridModelv4_Comparative(HybridModelv2_Comparative):
    """NeoPrecis-CRD multiplicative fusion (Cross-Reactivity Distance).

    Same shared encoder backbone as ``HybridModelv2_Comparative``, so
    ``transfer_backbone_from_v2`` still works unchanged. The ONLY difference
    is the WT/MT fusion: instead of concatenating the embeddings, we build the
    NeoPrecis ``_geometric_tf`` cross-reactivity vector and modulate its
    *magnitude* multiplicatively,

        origin  = emb_wt
        direction = normalize(emb_mt - emb_wt)
        length  = ||emb_mt - emb_wt||_2
        scaler  = MLP(cat([origin, direction]))          -> scalar in (-inf, inf)
        scaled_length = length * sigmoid(scaler)
        final   = Linear(scaled_length)                   (single logit)

    i.e. the mutation length is gated by a context-dependent scalar, rather
    than fed additively to a classifier over the full concatenation. Unlike
    v3's additive geometric decomposition, the normalize step here does NOT
    discard scale: ``length`` is carried through and only *rescaled*.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        emb_dim = self.gat_hidden_channels + self.vae_latent_dim + self.property_embedding_dim
        self.length_scaler_layer = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, 1),
        )

    def get_classifier(self):
        return nn.Linear(1, 1)

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):

        graph_data_cancer, graph_data_wt = graph_data_pair
        sequence_data_cancer, sequence_data_wt = sequence_data_pair
        peptide_property_cancer, peptide_property_wt = peptide_property_pair

        mu_cancer, logvar_cancer, x_gat_node_cancer, z_vae_cancer, attention_weights_n_cancer, recon_x_cancer = \
            self.forward_item(graph_data_cancer, sequence_data_cancer, peptide_property_cancer)
        mu_wt, logvar_wt, x_gat_node_wt, z_vae_wt, attention_weights_n_wt, recon_x_wt = \
            self.forward_item(graph_data_wt, sequence_data_wt, peptide_property_wt)
        combined_gat_only_cancer = torch.cat([x_gat_node_cancer], dim=1)

        embeddings_cancer = torch.cat([x_gat_node_cancer, z_vae_cancer], dim=1)
        embeddings_wt = torch.cat([x_gat_node_wt, z_vae_wt], dim=1)

        # NeoPrecis multiplicative cross-reactivity distance.
        origin = embeddings_wt
        diff = embeddings_cancer - embeddings_wt
        direction = F.normalize(diff, p=2, dim=1, eps=1e-12)
        length = diff.pow(2).sum(-1, keepdim=True).sqrt()
        scaler = self.length_scaler_layer(torch.cat([origin, direction], dim=1))
        scaled_length = length * torch.sigmoid(scaler)

        final_output = self.classifier(scaled_length)

        if return_embedding:
            return combined_gat_only_cancer, mu_cancer, logvar_cancer, final_output

        if return_attention:
            return attention_weights_n_cancer, mu_cancer, logvar_cancer, final_output

        return [embeddings_cancer, embeddings_wt], [recon_x_cancer, recon_x_wt], [mu_cancer, mu_wt], [logvar_cancer, logvar_wt], final_output

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        """Single-item forward (comparative pretraining is unused for v4; kept
        dimension-compatible by a degenerate zero-diff cross-reactivity vector."""
        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            self.forward_item(graph_data, sequence_data, peptide_property)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        emb = torch.cat([x_gat_node, z_vae], dim=1)
        origin = emb
        direction = torch.zeros_like(emb)
        length = torch.zeros((emb.shape[0], 1), device=emb.device)
        scaler = self.length_scaler_layer(torch.cat([origin, direction], dim=1))
        scaled_length = length * torch.sigmoid(scaler)

        final_output = self.classifier(scaled_length)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output


class HybridModelv2_Comparative_Deterministic(HybridModelv2_Comparative):
    """Deterministic-mu ablation of the v2 concat fusion.

    Identical to ``HybridModelv2_Comparative`` except the downstream WT/MT
    embedding uses the deterministic VAE mean ``mu`` instead of the
    reparameterized sample ``z``. Reconstruction and KLD still use the sampled
    latent (full VAE objective), so the ONLY change is that the immunogenicity
    classifier no longer sees reparameterization sampling noise.

    Hypothesis being tested: a CEDAR mutation is a single-point change in a
    ~281-residue sequence, so ``mu_mt - mu_wt`` is tiny and the sampled ``z``
    adds independent noise that can drown the downstream signal. Feeding the
    deterministic ``mu`` should recover (some of) that signal if the noise was
    the binding constraint.
    """

    def forward_item(self, graph_data, sequence_data, peptide_property):
        mu, logvar, x_gat_node, z_vae, attention_weights_n, recon_x = \
            super().forward_item(graph_data, sequence_data, peptide_property)
        # Rebuild the downstream latent from the deterministic mean (same
        # [mu ; property] shape as the sampled version it replaces).
        prop_emb = self.property_embedding(peptide_property)
        z_det = torch.cat([mu, prop_emb], dim=1)
        return mu, logvar, x_gat_node, z_det, attention_weights_n, recon_x
