"""ImmunoStruct-Opt — literature-grounded architectural upgrades (standalone, no in-place edits).

The stock ``HybridModelv2`` already matches the paper's IEDB AUROC (~0.89); the
optimized variants keep that *proven* backbone (MLP VAE over the flattened
one-hot, EGNN + linear-projection/mean-pool readout, concat fusion) and layer
literature-grounded improvements on top:

1. **Free-bits beta-VAE** (Kingma et al. 2016 "Improving Variational Inference
   with Inverse Autoregressive Flow"; Higgins et al. "beta-VAE", ICLR 2017).
   Each latent dimension's KL is clamped to a floor so the VAE cannot
   posterior-collapse — a strictly more informative latent than the stock
   plain-KL VAE.

2. **Geometry-aware EGNN** (``egnn_opt.EGNNConvOpt``): Gaussian RBF distance
   features + LayerNorm/residual + edge gating, replacing the bare squared
   distance the stock EGNN sees (the stored ``edge_attr`` is a placeholder of
   ones). Grounding: Satorras et al. ICML 2021; Schutt et al. "SchNet" 2017;
   Jing et al. "GVP" ICLR 2021.

3. **Focal BCE** (Lin, Goyal et al. 2017) for the imbalanced immunogenicity
   label (~19% positive).

The comparative (cancer/wildtype) variant additionally fuses an explicit
mutation-delta vector ``[emb_c ; emb_w ; emb_c - emb_w ; ||emb_c - emb_w||]``
(NeoPrecis-style mutation modeling, Bertrand et al. 2024) and is trained with a
cosine pull-push contrastive objective (SimCLR / SupCon, 2020).

A separate ``ConvVAE`` (DeepSequence-style 1D-CNN VAE, Riesselman et al. Nature
Methods 2018) is provided for ablation, but the default backbone uses the MLP
VAE: the dense flatten-then-linear encoder has ~20x the capacity of the CNN and
empirically reaches the paper's IEDB AUROC, whereas the CNN underfits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .egnn_opt import EGNNConvOpt
from .egnn_torch import EGNNConvTorch
from .layers import SelfAttention

__all__ = ["ConvVAE", "MLPVAE", "HybridModelOpt", "HybridModelOpt_Comparative",
           "transfer_shared_from_opt", "transfer_phase1_vae"]

_EGNN_CONV = {"opt": EGNNConvOpt, "torch": EGNNConvTorch}

# Default dims (kept consistent across the three stages so checkpoints transfer).
DEFAULT_LATENT_DIM = 64           # sequence latent z and structure embedding width
DEFAULT_STRUCT_HIDDEN = 64
DEFAULT_PROPERTY_EMBEDDING_DIM = 16
N_AMINO_ACIDS = 21                # 20 AA + padding char


def _batch_index(graph_data, device):
    """Construct a [total_nodes] long batch index from a batched DGL graph."""
    num_nodes = graph_data.batch_num_nodes()
    if not torch.is_tensor(num_nodes):
        num_nodes = torch.tensor(num_nodes, device=device)
    else:
        num_nodes = num_nodes.to(device)
    return torch.repeat_interleave(torch.arange(len(num_nodes), device=device), num_nodes)


class ConvVAE(nn.Module):
    """1D-CNN variational autoencoder over the one-hot pMHC sequence (ablation).

    Encodes ``[B, L, 21]`` into a ``latent_dim`` Gaussian and reconstructs the
    flattened one-hot ``[B, L*21]``. Kept for ablations; the default backbone
    uses :class:`MLPVAE` (see module docstring).
    """

    def __init__(self, vae_input_dim, vae_hidden_dim=256, vae_latent_dim=DEFAULT_LATENT_DIM,
                 property_embedding_dim=DEFAULT_PROPERTY_EMBEDDING_DIM):
        super().__init__()
        self.vae_input_dim = vae_input_dim
        self.vae_latent_dim = vae_latent_dim
        self.seq_len = vae_input_dim // N_AMINO_ACIDS

        self.conv1 = nn.Conv1d(N_AMINO_ACIDS, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.pos_attn = nn.Linear(128, 1)
        self.fc_mu = nn.Linear(128 * 2, vae_latent_dim)
        self.fc_logvar = nn.Linear(128 * 2, vae_latent_dim)

        self.dec_fc1 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.dec_fc2 = nn.Linear(vae_hidden_dim, vae_input_dim)

    def encode(self, x):
        h = x.transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))                      # [B, 128, L]
        a = self.pos_attn(h.transpose(1, 2))           # [B, L, 1]
        a = F.softmax(a, dim=1)                         # [B, L, 1]
        h_attn = (h * a.transpose(1, 2)).sum(dim=2)    # [B, 128]
        h_mean = h.mean(dim=2)                          # [B, 128]
        h = torch.cat([h_attn, h_mean], dim=1)          # [B, 256]
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-20.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, prop_emb):
        zc = torch.cat([z, prop_emb], dim=1)
        h = F.gelu(self.dec_fc1(zc))
        return self.dec_fc2(h)  # [B, L*21]


class MLPVAE(nn.Module):
    """MLP variational autoencoder over the flattened one-hot pMHC sequence.

    Mirrors the stock ``HybridModel`` VAE (a dense ``5943 -> 512 -> latent``
    encoder over the flattened one-hot, whose position-specific dense weights
    preserve where each residue sits) but is trained with a free-bits beta-VAE
    objective so the latent cannot posterior-collapse.
    """

    def __init__(self, vae_input_dim, vae_hidden_dim=512, vae_latent_dim=DEFAULT_LATENT_DIM,
                 property_embedding_dim=DEFAULT_PROPERTY_EMBEDDING_DIM, decoder_layers=1,
                 mask_ratio=0.0):
        super().__init__()
        self.vae_input_dim = vae_input_dim
        self.vae_latent_dim = vae_latent_dim
        self.decoder_layers = decoder_layers
        self.mask_ratio = mask_ratio
        self.mask = None

        self.fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.fc_mu = nn.Linear(vae_hidden_dim, vae_latent_dim)
        self.fc_logvar = nn.Linear(vae_hidden_dim, vae_latent_dim)
        self.fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.dec_hidden = nn.ModuleList(
            [nn.Linear(vae_hidden_dim, vae_hidden_dim) for _ in range(max(0, decoder_layers - 1))]
        )
        self.fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

    def encode(self, x):
        # x: [B, vae_input_dim] (flattened one-hot)
        # Training-only masked reconstruction (MLM/MAE): zero out whole residues and
        # reconstruct only the masked positions, so the latent learns residue
        # co-dependence rather than memorizing the full one-hot. Eval uses full x.
        seq_len = self.vae_input_dim // N_AMINO_ACIDS
        mask = None
        if self.training and self.mask_ratio > 0:
            x2d = x.view(-1, seq_len, N_AMINO_ACIDS)
            mask = torch.rand(x.shape[0], seq_len, device=x.device) < self.mask_ratio
            x2d = x2d * (~mask).unsqueeze(-1).float()
            x = x2d.view(x.shape[0], -1)
        self.mask = mask
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-20.0, max=10.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, prop_emb):
        zc = torch.cat([z, prop_emb], dim=1)
        h = F.relu(self.fc3(zc))
        for layer in self.dec_hidden:
            h = F.relu(layer(h))
        return self.fc4(h)  # [B, vae_input_dim]


class HybridModelOpt(nn.Module):
    """Stage-2 (IEDB) model: MLP VAE sequence + geometry-aware EGNN structure +
    concat fusion -> immunogenicity logit.

    ``forward`` returns ``(recon_x, mu, logvar, final_output)``, matching the
    stock ``HybridModelv2`` signature so existing MC-eval scripts work unchanged.
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
                 mask_ratio=0.0,
                 *args, **kwargs):
        super().__init__()
        self.device = device
        self.vae_latent_dim = vae_latent_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.vae_input_dim = vae_input_dim
        self.egnn_type = egnn_type

        # Structure tower: geometry-aware EGNN + linear-projection/mean-pool readout
        # (the stock model's "self-attention" over a length-1 axis is a linear map).
        conv = _EGNN_CONV[egnn_type]
        self.GCN_layers = nn.ModuleList(
            [conv(20, gat_hidden_channels, gat_hidden_channels, 1)]
        )
        for _ in range(gcn_layers):
            self.GCN_layers.append(conv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))
        self.self_attention = SelfAttention(gat_hidden_channels)

        # Sequence tower: MLP VAE.
        self.vae = MLPVAE(vae_input_dim, vae_hidden_dim=vae_hidden_dim,
                          vae_latent_dim=vae_latent_dim,
                          property_embedding_dim=property_embedding_dim,
                          decoder_layers=decoder_layers,
                          mask_ratio=mask_ratio)

        # Property tower: pI + GRAVY -> embedding.
        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(32, property_embedding_dim), nn.ReLU(True))

        # Classifier over [struct ; z ; prop].
        cls_in = gat_hidden_channels + vae_latent_dim + property_embedding_dim
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
        return global_mean_pool(node_feat, batch)  # [B, struct_dim]

    def _encode_sequence(self, sequence_data):
        x = sequence_data.view(sequence_data.shape[0], -1)
        mu, logvar = self.vae.encode(x)
        return self.vae.reparameterize(mu, logvar), mu, logvar

    def forward(self, graph_data, sequence_data, peptide_property,
                return_embedding=False, return_attention=False):
        struct_feat = self._forward_structure(graph_data)          # [B, D]
        prop_emb = self.property_embedding(peptide_property)        # [B, P]
        z, mu, logvar = self._encode_sequence(sequence_data)        # z [B, D]
        recon_x = self.vae.decode(z, prop_emb)                      # [B, vae_input_dim]

        combined = torch.cat([struct_feat, z, prop_emb], dim=1)
        final_output = self.classifier(combined)

        if return_embedding:
            return struct_feat, mu, logvar, final_output
        return recon_x, mu, logvar, final_output


class HybridModelOpt_Comparative(nn.Module):
    """Stage-3 (CEDAR) comparative model.

    Runs the shared MLP-VAE+EGNN backbone on cancer and wildtype, then fuses an
    explicit mutation-delta vector ``[emb_c ; emb_w ; emb_c - emb_w ; ||delta||]``
    before the immunogenicity classifier.

    ``forward_comparative`` returns
    ``(emb_pair, recon_pair, mu_pair, logvar_pair, final_output)``, matching the
    stock comparative signature so training/eval loops are drop-in compatible.
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

        # classifier over [emb_c ; emb_w ; delta ; magnitude]
        cls_in = self.embed_dim * 3 + 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 32), nn.ReLU(True), nn.Dropout(dropout),
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
        struct_feat = global_mean_pool(node_feat, batch)               # [B, D]

        prop_emb = self.property_embedding(peptide_property)           # [B, P]
        x = sequence_data.view(sequence_data.shape[0], -1)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        recon_x = self.vae.decode(z, prop_emb)

        emb = torch.cat([struct_feat, z], dim=1)                       # [B, D+D]
        return emb, recon_x, mu, logvar

    def forward_comparative(self, graph_data_pair, sequence_data_pair, peptide_property_pair,
                            return_embedding=False, return_attention=False):
        gc, gw = graph_data_pair
        sc, sw = sequence_data_pair
        pc, pw = peptide_property_pair

        emb_c, recon_c, mu_c, logvar_c = self.forward_item(gc, sc, pc)
        emb_w, recon_w, mu_w, logvar_w = self.forward_item(gw, sw, pw)

        delta = emb_c - emb_w
        magnitude = delta.norm(p=2, dim=1, keepdim=True)
        combined = torch.cat([emb_c, emb_w, delta, magnitude], dim=1)
        final_output = self.classifier(combined)

        return ([emb_c, emb_w], [recon_c, recon_w], [mu_c, mu_w], [logvar_c, logvar_w], final_output)


def transfer_shared_from_opt(model, path, map_location=None):
    """Initialize a comparative model's shared backbone from a ``HybridModelOpt``
    (stage-2) checkpoint. Only the shared towers (EGNN ``GCN_layers``, attention
    ``self_attention``, ``vae``, ``property_embedding``) are copied; the
    comparative classifier stays random and is fine-tuned on CEDAR.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    for container in ("model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and container in ckpt and isinstance(ckpt[container], dict):
            ckpt = ckpt[container]
            break
    if not isinstance(ckpt, dict):
        raise TypeError(f"expected a state_dict, got {type(ckpt)}")

    prefixes = ("GCN_layers.", "self_attention.", "vae.", "property_embedding.")
    filtered = {k: v for k, v in ckpt.items() if k.startswith(prefixes)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[pretrained] transferred {len(filtered)} shared tensors from {path}; "
          f"{len(missing)} missing (classifier random), {len(unexpected)} unexpected")
    return len(filtered)


def transfer_phase1_vae(model, path, map_location=None):
    """Warm-start ``MLPVAE`` + ``property_embedding`` from the canonical
    presentation-pretrained VAE checkpoint (``vae_presentation_vae.pt``).

    The pretrained state_dict uses the original HybridModel naming
    (``vae_fc1/vae_fc21/vae_fc22/vae_fc3/vae_fc4``) and a longer flattened one-hot
    (6090 = 290x21); ``MLPVAE`` uses (``fc1/fc_mu/fc_logvar/fc3/fc4``) and the
    IEDB/CEDAR input dim (5943 = 283x21). Keys are renamed and the input-dim axis
    is trimmed to the model's ``vae_input_dim``.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    for container in ("vae", "model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and container in ckpt and isinstance(ckpt[container], dict):
            ckpt = ckpt[container]
            break
    if not isinstance(ckpt, dict):
        raise TypeError(f"expected a state_dict, got {type(ckpt)}")

    target = model.vae.vae_input_dim
    rename = {
        "vae_fc1.": "vae.fc1.",
        "vae_fc21.": "vae.fc_mu.",
        "vae_fc22.": "vae.fc_logvar.",
        "vae_fc3.": "vae.fc3.",
        "vae_fc4.": "vae.fc4.",
    }
    transferred = {}
    for k, v in ckpt.items():
        nk = k
        for old, new in rename.items():
            if k.startswith(old):
                nk = new + k[len(old):]
                break
        # Trim the flattened-one-hot axis (fc1.weight: (H,in), fc4.weight: (in,H),
        # fc4.bias: (in,)) from the pretrained 6090 to the model's input dim.
        if nk == "vae.fc1.weight":
            v = v[:, :target]
        elif nk == "vae.fc4.weight":
            v = v[:target, :]
        elif nk == "vae.fc4.bias":
            v = v[:target]
        transferred[nk] = v
    missing, unexpected = model.load_state_dict(transferred, strict=False)
    print(f"[phase1] transferred {len(transferred)} VAE tensors from {path}; "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    return len(transferred)
