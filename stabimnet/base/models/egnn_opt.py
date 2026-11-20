"""Improved E(n)-equivariant graph layer for ImmunoStruct-Opt (standalone, no in-place edits).

Literature grounding
--------------------
* Satorras, Hoogeboom & Welling, "E(n) Equivariant Graph Neural Networks", ICML 2021
  (base EGNN: node/edge MLP + coordinate displacement update).
* Schutt et al., "SchNet", NeurIPS 2017 — Gaussian radial-basis-function (RBF)
  expansion of interatomic distances as the canonical edge feature for
  geometry-aware GNNs.
* Ba, Kingma et al., "Layer Normalization", 2016 — stabilizes deep stacks.
* Jing, Eismann et al., "Learning from Protein Structure with Geometric Vector
  Perceptrons", ICLR 2021 — geometry-aware protein encoders benefit from richer
  geometric features than a bare squared distance.

Changes vs ``egnn_torch.EGNNConvTorch``
---------------------------------------
1. Richer radial edge features. The stored graphs carry ``edge_attr == ones``
   (a placeholder), so the original EGNN sees only ``dist^2``. Here we compute
   an RBF expansion of the true inter-atomic distance plus the raw distance.
2. LayerNorm + residual on node features (trainable deeper stacks).
3. Edge gating (a learned scalar attention weight per edge) so informative
   contacts dominate the message aggregation.

Crucially the coordinate update keeps the *unit-direction* displacement
normalization (``x_diff / (|x_diff| + eps)``) byte-for-byte compatible with
``dgl.nn.EGNNConv``; using the raw displacement makes updates grow with
``||x_i - x_j||`` and explode super-exponentially across layers (the root cause
of the seeds 42/456 NaN in the v2 audit).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_rbf(dist, num_basis=8, cutoff=20.0):
    """Expand scalar distances into a Gaussian radial-basis feature vector.

    Parameters
    ----------
    dist : torch.Tensor [E, 1]
    num_basis : int
    cutoff : float

    Returns
    -------
    torch.Tensor [E, num_basis]
    """
    centers = torch.linspace(0.0, cutoff, num_basis, device=dist.device)
    width = cutoff / max(num_basis - 1, 1)
    return torch.exp(-((dist - centers) ** 2) / (2.0 * width ** 2))


class EGNNConvOpt(nn.Module):
    """Equivariant graph convolution with RBF edge features, LayerNorm, residual,
    and edge gating. DGL-compatible ``forward(graph, node_feat, coord_feat, edge_feat)``.
    """

    def __init__(self, in_node_dim, hidden_dim, out_node_dim, edge_dim,
                 num_rbf=8, coord_clamp=10.0, raw_displacement=False):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.hidden_dim = hidden_dim
        self.out_node_dim = out_node_dim
        self.edge_dim = edge_dim
        self.num_rbf = num_rbf
        self.coord_clamp = coord_clamp
        self.raw_displacement = raw_displacement

        # Edge MLP: [h_i, h_j, RBF(dist), dist, edge_attr] -> message
        edge_input = in_node_dim * 2 + num_rbf + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Edge gate: scalar attention weight in (0, 1) per edge.
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Coordinate MLP: message -> scalar displacement weight.
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Node update: [h_i, aggregated_messages] -> new h_i (with LayerNorm + residual).
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_node_dim),
        )
        self.node_norm = nn.LayerNorm(out_node_dim)

    def forward(self, graph, node_feat, coord_feat, edge_feat):
        """Returns (new_node_feat, new_coord_feat)."""
        src, dst = graph.edges()
        device = node_feat.device
        src = src.to(device)
        dst = dst.to(device)

        hi = node_feat[src]
        hj = node_feat[dst]

        xi = coord_feat[src]
        xj = coord_feat[dst]
        x_diff = xi - xj                                   # (E, coord_dim)
        dist2 = (x_diff ** 2).sum(dim=-1, keepdim=True)    # (E, 1)
        dist = dist2.sqrt().clamp_min(1e-8)                # (E, 1)
        rbf = gaussian_rbf(dist, self.num_rbf)             # (E, num_rbf)

        edge_input = torch.cat([hi, hj, rbf, dist, edge_feat], dim=-1)
        messages = self.edge_mlp(edge_input)               # (E, hidden)
        gate = self.edge_gate(edge_input)                  # (E, 1)
        messages = messages * gate

        # Coordinate update (unit-direction normalization; NaN-safe).
        if not self.raw_displacement:
            x_diff = x_diff / (dist + 1e-30)
        coord_weight = self.coord_mlp(messages)
        if self.coord_clamp is not None:
            coord_weight = torch.clamp(coord_weight, min=-self.coord_clamp, max=self.coord_clamp)
        disp = x_diff * coord_weight
        agg_disp = torch.zeros_like(coord_feat)
        agg_disp.index_add_(0, dst, disp)
        degree = torch.zeros(node_feat.shape[0], device=device).scatter_add(
            0, dst, torch.ones_like(dst, dtype=torch.float, device=device)
        ).clamp(min=1).unsqueeze(-1)
        new_coord = coord_feat + agg_disp / degree

        # Message aggregation + node update (LayerNorm + residual when dims match).
        agg_messages = torch.zeros(
            node_feat.shape[0], messages.shape[1], device=device, dtype=messages.dtype,
        )
        agg_messages.index_add_(0, dst, messages)
        node_input = torch.cat([node_feat, agg_messages], dim=-1)
        out = self.node_mlp(node_input)
        if out.shape[-1] == node_feat.shape[-1]:
            new_node_feat = node_feat + self.node_norm(out)
        else:
            new_node_feat = self.node_norm(out)

        return new_node_feat, new_coord
