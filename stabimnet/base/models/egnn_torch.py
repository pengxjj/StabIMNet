"""
Pure-PyTorch EGNN (E(n) Equivariant Graph Neural Network) layer.

Replaces ``dgl.nn.EGNNConv`` with identical API — takes a DGL graph to
extract edge indices, but runs all compute in pure PyTorch (no DGL CUDA
backend required).

Reference: Satorras et al. "E(n) Equivariant Graph Neural Networks", ICML 2021.
"""

import torch
import torch.nn as nn


class EGNNConvTorch(nn.Module):
    """Pure-PyTorch EGNNConv with DGL-compatible forward signature.

    Parameters
    ----------
    in_node_dim : int
    hidden_dim : int
    out_node_dim : int
    edge_dim : int
    """

    def __init__(self, in_node_dim, hidden_dim, out_node_dim, edge_dim, coord_clamp=None):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.hidden_dim = hidden_dim
        self.out_node_dim = out_node_dim
        self.edge_dim = edge_dim
        self.coord_clamp = coord_clamp
        # Diagnostic: count of coordinate weights clamped per forward (healthy runs stay 0)
        self.clamp_hits = 0
        # Diagnostic: running max |coord_weight| seen before clamping
        self.max_abs_cw = 0.0

        # Edge MLP: [h_i, h_j, dist^2, a_ij] → message
        edge_input = in_node_dim * 2 + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate MLP: message → scalar weight (with extra output for coord)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Node MLP: [h_i, aggregated_messages] → new h_i
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_node_dim),
        )

    def forward(self, graph, node_feat, coord_feat, edge_feat):
        """Returns (new_node_feat, new_coord_feat)."""
        src, dst = graph.edges()
        device = node_feat.device
        src = src.to(device)
        dst = dst.to(device)

        hi = node_feat[src]   # (E, in_node_dim)
        hj = node_feat[dst]   # (E, in_node_dim)

        # Squared distance (radial feature)
        xi = coord_feat[src]
        xj = coord_feat[dst]
        x_diff = xi - xj                                    # (E, coord_dim)
        dist2 = (x_diff ** 2).sum(dim=-1, keepdim=True)     # (E, 1)

        # Edge messages
        edge_input = torch.cat([hi, hj, dist2, edge_feat], dim=-1)
        messages = self.edge_mlp(edge_input)  # (E, hidden_dim)

        # Coordinate update: unit direction * scalar weight, mean-aggregated to the
        # destination node. This mirrors dgl.nn.EGNNConv exactly (x_diff normalised by
        # its L2 norm); using the raw displacement instead makes the update grow with
        # ||x_i - x_j|| and explode super-exponentially across layers.
        x_diff = x_diff / (dist2.sqrt() + 1e-30)            # unit direction (E, coord_dim)
        coord_weight = self.coord_mlp(messages)  # (E, 1)
        self.max_abs_cw = max(self.max_abs_cw, float(coord_weight.abs().max().item()))
        if self.coord_clamp is not None:
            self.clamp_hits += int((coord_weight.abs() > self.coord_clamp).sum().item())
            coord_weight = torch.clamp(coord_weight, min=-self.coord_clamp, max=self.coord_clamp)
        disp = x_diff * coord_weight                        # (E, coord_dim)
        agg_disp = torch.zeros_like(coord_feat)
        agg_disp.index_add_(0, dst, disp)
        degree = torch.zeros(node_feat.shape[0], device=device).scatter_add(
            0, dst, torch.ones_like(dst, dtype=torch.float, device=device)
        ).clamp(min=1).unsqueeze(-1)
        new_coord = coord_feat + agg_disp / degree

        # Aggregate messages per destination node
        agg_messages = torch.zeros(
            node_feat.shape[0], messages.shape[1],
            device=device, dtype=messages.dtype,
        )
        agg_messages.index_add_(0, dst, messages)

        # Node update (residual only when dims match)
        node_input = torch.cat([node_feat, agg_messages], dim=-1)
        out = self.node_mlp(node_input)
        if out.shape[-1] == node_feat.shape[-1]:
            new_node_feat = node_feat + out
        else:
            new_node_feat = out

        return new_node_feat, new_coord
