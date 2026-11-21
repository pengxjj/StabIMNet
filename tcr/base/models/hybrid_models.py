import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.nn import global_mean_pool
from .egnn_torch import EGNNConvTorch
from .layers import SelfAttention, MultiHeadAttention, CrossModalAttentionBlock

# Use pure-PyTorch EGNN (DGL 2.1 pip wheel is CPU-only)
EGNNConv = EGNNConvTorch

__all__ = ["HybridModel", "HybridModelv2", "HybridModel_SSL", "HybridModelv2_SSL",
           "HybridModelv3", "HybridModelv5", "HybridModelv5NoStruct",
           "HybridModelv6", "HybridModelv7", "HybridModelv8",
           "ESMCEncoder", "ESMCTokenizer", "LoRALinear"]

# ────────────────────────────────────────────────────────────────────
# LoRA: Low-Rank Adaptation for Linear layers
# ────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Adds trainable low-rank matrices A and B alongside a frozen Linear
    layer:  output = W @ x + (alpha / r) * B @ A @ x

    Parameters
    ----------
    linear : nn.Linear
        The frozen base linear layer.
    r : int
        LoRA rank.
    alpha : int
        Scaling factor (applied as alpha / r).
    dropout : float
        Dropout rate on the LoRA path.
    """

    def __init__(self, linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.linear = linear  # frozen base
        in_features = linear.in_features
        out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze base weights
        for p in self.linear.parameters():
            p.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.linear(x)
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        lora = (self.dropout(x) @ lora_A) @ lora_B
        return base + self.scaling * lora

    def merge(self):
        """Merge LoRA weights into the base linear layer for inference."""
        delta = (self.lora_A @ self.lora_B).T * self.scaling
        self.linear.weight.data += delta
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()

    def unmerge(self):
        """Remove merged LoRA weights (revert to original)."""
        delta = (self.lora_A @ self.lora_B).T * self.scaling
        self.linear.weight.data -= delta

    @property
    def weight(self):
        """Expose weight attribute for compatibility (returns effective weight)."""
        delta = (self.lora_A @ self.lora_B).T * self.scaling
        return self.linear.weight + delta

# ────────────────────────────────────────────────────────────────────
# ESMC-6B encoder (pre-computed embeddings, not used in training loop)
# ────────────────────────────────────────────────────────────────────

ESMC_DIM = 2560

ESM2_DIM_MAP = {
    "esm2_t48_15B_UR50D": 5120,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t6_8M_UR50D": 320,
}

# ESMC character-level vocabulary (token id -> AA character)
ESMC_VOCAB = {
    0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>',
    4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E',
    10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K',
    16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H',
    22: 'W', 23: 'C', 24: 'X', 25: 'B', 26: 'U', 27: 'Z',
    28: 'O', 29: '.', 30: '-', 31: '|', 32: '<mask>',
}


class ESMCTokenizer:
    """Character-level tokenizer for ESMC model.

    Maps each amino-acid character to a token ID (4-23 for standard AAs).
    Special tokens: <cls>=0, <pad>=1, <eos>=2, <unk>=3, | =31, <mask>=32.
    """

    def __init__(self):
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 32
        # Build AA -> token_id mapping
        self._aa_to_id = {}
        for tid, aa in ESMC_VOCAB.items():
            if tid >= 4 and tid <= 30 and len(aa) == 1:
                self._aa_to_id[aa] = tid

    def encode(self, sequence, add_special_tokens=True):
        """Convert an AA sequence string to token IDs.

        Parameters
        ----------
        sequence : str
            Raw amino-acid sequence.
        add_special_tokens : bool
            If True, prepend <cls> and append <eos>.

        Returns
        -------
        token_ids : list[int]
        """
        ids = [self._aa_to_id.get(ch, self.unk_token_id) for ch in sequence]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.eos_token_id]
        return ids

    def tokenize(self, sequences):
        """Tokenize a list of AA sequences.

        Parameters
        ----------
        sequences : list[str]

        Returns
        -------
        tokens : torch.LongTensor  [B, L_max]  (padded to max length)
        """
        batch_ids = [self.encode(s) for s in sequences]
        max_len = max(len(ids) for ids in batch_ids)
        padded = torch.full((len(batch_ids), max_len), self.pad_token_id,
                            dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return padded


class ESMCEncoder(nn.Module):
    """ESMC-6B protein language model encoder (standalone, for pre-computing).

    Loads the ESMC-6B model from a local safetensors directory.
    Produces mean-pooled per-residue embeddings [B, 2560].

    This encoder is used ONCE to pre-compute sequence embeddings for all
    samples. It is NOT part of the training loop — HybridModelv3 takes
    pre-computed embeddings directly.

    Parameters
    ----------
    model_dir : str
        Path to ESMC-6B model directory containing safetensors files.
    device : str or torch.device
    dtype : torch.dtype
        Use float16 for GPU memory efficiency (12GB vs 24GB).
    """

    def __init__(self, model_dir, device="cuda", dtype=torch.float16):
        super().__init__()
        self.model_dir = model_dir
        self.embed_dim = ESMC_DIM  # 2560
        self.tokenizer = ESMCTokenizer()

        self._device = device
        self._dtype = dtype

        # Load weights from safetensors
        self._load_weights(model_dir)

        # Freeze
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        # Move to target device / dtype
        self.to(device=device, dtype=dtype)

    def _load_weights(self, model_dir):
        """Load ESMC-6B weights from safetensors shards and build model."""
        from safetensors import safe_open
        import json

        # Read weight map
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            index = json.load(f)

        # Build state dict by loading from shards
        shard_files = set(index["weight_map"].values())
        shard_data = {}
        for shard_file in sorted(shard_files):
            shard_path = os.path.join(model_dir, shard_file)
            with safe_open(shard_path, framework="pt") as sf:
                for key in sf.keys():
                    shard_data[key] = sf.get_tensor(key)

        # ── Build embedding ──
        self.embed = nn.Embedding(64, ESMC_DIM, padding_idx=1)
        self.embed.weight.data.copy_(shard_data["esmc.embed.weight"])

        # ── Build transformer blocks ──
        self.blocks = nn.ModuleList()
        self.num_layers = 80
        for layer_idx in range(self.num_layers):
            prefix = f"esmc.transformer.blocks.{layer_idx}"
            block = _ESMCTransformerBlock(ESMC_DIM, 40, shard_data, prefix)
            self.blocks.append(block)

        # ── Final layer norm ──
        self.norm = nn.LayerNorm(ESMC_DIM, elementwise_affine=True)
        self.norm.weight.data.copy_(shard_data["esmc.transformer.norm.weight"])
        # Bias can be None for LayerNorm without bias

    def forward(self, tokens, pool=True):
        """Encode tokenized sequences.

        Parameters
        ----------
        tokens : torch.LongTensor  [B, L]
        pool : bool
            If True, return mean-pooled embedding [B, 2560].
            If False, return per-residue embeddings [B, L, 2560] + valid mask.

        Returns
        -------
        If pool=True:  embeddings [B, 2560]
        If pool=False: (embeddings [B, L, 2560], valid_mask [B, L])
        """
        device = next(self.parameters()).device
        tokens = tokens.to(device)
        dtype = next(self.parameters()).dtype

        # Attention mask: mask out PAD tokens only
        attn_mask = (tokens != self.tokenizer.pad_token_id)

        x = self.embed(tokens)  # [B, L, D]
        x = x.to(dtype)

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.norm(x)

        # Valid token mask: exclude CLS, EOS, and PAD tokens
        is_pad = (tokens == self.tokenizer.pad_token_id)
        is_cls = (tokens == self.tokenizer.cls_token_id)
        is_eos = (tokens == self.tokenizer.eos_token_id)
        valid = (~is_pad & ~is_cls & ~is_eos)

        if pool:
            valid_f = valid.float().unsqueeze(-1).to(device)
            masked_x = x.float() * valid_f
            summed = masked_x.sum(dim=1)
            counts = valid_f.sum(dim=1).clamp(min=1)
            return summed / counts
        else:
            return x.float(), valid.to(device)

    # ── LoRA management ────────────────────────────────────────────

    def enable_lora(self, r=16, alpha=32, dropout=0.05, num_lora_layers=None):
        """Inject LoRA into transformer blocks and freeze base params.

        After calling this, only LoRA params, Q/K LayerNorms, and the
        final norm are trainable. The embedding, all FFN weights, and
        all original attention projections are frozen.

        Parameters
        ----------
        num_lora_layers : int or None
            Apply LoRA only to the last N layers. None = all layers.
            Default recommendation: 40 (half of 80 layers) for good
            capacity/speed balance.
        """
        if num_lora_layers is None:
            num_lora_layers = self.num_layers

        start_layer = self.num_layers - num_lora_layers
        for i, block in enumerate(self.blocks):
            if i >= start_layer:
                block.enable_lora(r=r, alpha=alpha, dropout=dropout)

        # Unfreeze final norm (small, useful for domain adaptation)
        for p in self.norm.parameters():
            p.requires_grad = True

        # Move LoRA params to the same device as the base model
        self.to(self._device)

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"ESMC LoRA enabled: {trainable:,} / {total:,} params trainable "
              f"({100 * trainable / total:.1f}%) "
              f"[layers {start_layer}-{self.num_layers-1}, "
              f"r={r}, alpha={alpha}]")
        return self

    def lora_parameters(self):
        """Return only parameters that should be optimizer-updated."""
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def save_lora(self, path):
        """Save only LoRA + fine-tuned params (excludes frozen base model)."""
        lora_state = {k: v for k, v in self.state_dict().items()
                      if 'lora_A' in k or 'lora_B' in k
                      or 'q_ln.' in k or 'k_ln.' in k
                      or 'norm.' in k}
        torch.save(lora_state, path)
        print(f"LoRA state saved to {path} ({len(lora_state)} tensors)")

    def load_lora(self, path):
        """Load LoRA checkpoint (must call enable_lora first)."""
        missing, unexpected = self.load_state_dict(
            torch.load(path), strict=False)
        if missing:
            print(f"  LoRA load: {len(missing)} missing keys "
                  f"(expected, base params are separate)")
        return self

    def merge_lora(self):
        """Merge all LoRA weights into base layers (for inference speed)."""
        for block in self.blocks:
            if isinstance(block.qkv_proj, LoRALinear):
                block.qkv_proj.merge()
            if isinstance(block.out_proj, LoRALinear):
                block.out_proj.merge()
            if isinstance(block.ffn_fc1, LoRALinear):
                block.ffn_fc1.merge()
            if isinstance(block.ffn_fc2, LoRALinear):
                block.ffn_fc2.merge()
        print("LoRA weights merged into base layers.")

    @torch.no_grad()
    def encode_batch(self, sequences, batch_size=4):
        """Tokenize and encode a list of AA sequences.

        Parameters
        ----------
        sequences : list[str]
        batch_size : int
            Small batch size due to 6B model memory requirements.

        Returns
        -------
        embeddings : torch.Tensor  [N, 2560]  (float32 on CPU)
        """
        all_emb = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            tokens = self.tokenizer.tokenize(batch_seqs)
            emb = self.forward(tokens)
            all_emb.append(emb.float().cpu())
        return torch.cat(all_emb, dim=0)

    def to(self, *args, **kwargs):
        if 'device' in kwargs:
            self._device = kwargs['device']
        elif len(args) > 0 and isinstance(args[0], (torch.device, str)):
            self._device = args[0]
        return super().to(*args, **kwargs)


class _ESMCTransformerBlock(nn.Module):
    """Single transformer block matching ESMC-6B architecture."""

    def __init__(self, dim, n_heads, state_dict, prefix):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads  # 64

        # QKV layernorm + projection
        self.ln_qkv = nn.LayerNorm(dim, elementwise_affine=True)
        self.ln_qkv.weight.data.copy_(state_dict[f"{prefix}.attn.layernorm_qkv.layer_norm_weight"])
        self.ln_qkv.bias.data.copy_(state_dict[f"{prefix}.attn.layernorm_qkv.layer_norm_bias"])
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.qkv_proj.weight.data.copy_(state_dict[f"{prefix}.attn.layernorm_qkv.weight"])

        # Q/K layer norms (applied before multi-head reshape)
        self.q_ln = nn.LayerNorm(dim, elementwise_affine=True)
        self.q_ln.weight.data.copy_(state_dict[f"{prefix}.attn.q_ln.weight"])
        self.k_ln = nn.LayerNorm(dim, elementwise_affine=True)
        self.k_ln.weight.data.copy_(state_dict[f"{prefix}.attn.k_ln.weight"])

        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj.weight.data.copy_(state_dict[f"{prefix}.attn.out_proj.weight"])

        # FFN (SwiGLU)
        ffn_intermediate = state_dict[f"{prefix}.ffn.fc1_weight"].shape[0]  # 13824
        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn_ln.weight.data.copy_(state_dict[f"{prefix}.ffn.layer_norm_weight"])
        self.ffn_ln.bias.data.copy_(state_dict[f"{prefix}.ffn.layer_norm_bias"])
        self.ffn_fc1 = nn.Linear(dim, ffn_intermediate, bias=False)
        self.ffn_fc1.weight.data.copy_(state_dict[f"{prefix}.ffn.fc1_weight"])
        self.ffn_fc2 = nn.Linear(ffn_intermediate // 2, dim, bias=False)
        self.ffn_fc2.weight.data.copy_(state_dict[f"{prefix}.ffn.fc2_weight"])

    def forward(self, x, mask):
        # ── Attention ──
        residual = x
        x_norm = self.ln_qkv(x)
        qkv = self.qkv_proj(x_norm)  # [B, L, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        # QK LayerNorm (before multi-head reshape)
        q = self.q_ln(q)
        k = self.k_ln(k)

        # Reshape to multi-head
        B, L, D = q.shape
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D_h]
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_weights = (q @ k.transpose(-2, -1)) / scale

        # Apply mask (bool mask → set masked positions to -inf equivalent)
        # Use a large negative safe for float16: -65504.0
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        neg_inf = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~attn_mask, neg_inf)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = attn_weights @ v

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)
        x = residual + attn_out

        # ── FFN (SwiGLU) ──
        residual = x
        x_norm = self.ffn_ln(x)
        fc1_out = self.ffn_fc1(x_norm)
        gate, up = fc1_out.chunk(2, dim=-1)
        ffn_out = self.ffn_fc2(F.silu(gate) * up)
        x = residual + ffn_out

        return x

    def enable_lora(self, r=16, alpha=32, dropout=0.05):
        """Inject LoRA adapters into attention and FFN layers."""
        self.qkv_proj = LoRALinear(self.qkv_proj, r=r, alpha=alpha,
                                   dropout=dropout)
        self.out_proj = LoRALinear(self.out_proj, r=r, alpha=alpha,
                                   dropout=dropout)
        self.ffn_fc1 = LoRALinear(self.ffn_fc1, r=r, alpha=alpha,
                                   dropout=dropout)
        self.ffn_fc2 = LoRALinear(self.ffn_fc2, r=r, alpha=alpha,
                                   dropout=dropout)
        # Unfreeze Q/K LayerNorms (tiny parameter cost, improves adaptation)
        for p in self.q_ln.parameters():
            p.requires_grad = True
        for p in self.k_ln.parameters():
            p.requires_grad = True
        return self


# ────────────────────────────────────────────────────────────────────
# HybridModelv3: ESMC sequence embeddings + EGNN structure encoder
# ────────────────────────────────────────────────────────────────────

class HybridModelv3(nn.Module):
    """ImmunoStruct v3 — ESMC + EGNN two-tower architecture.

    Sequence tower : Pre-computed ESMC embeddings [B, 2560] → Projection → [B, proj_dim]
    Structure tower: EGNN × n_layers → self-attn → mean-pool → [B, gnn_dim]
    Fusion         : Gated cross-modal (seq gates struct, struct gates seq) → classifier

    Two-stage training:
      1. Pretrain (seq-only):   Embeddings → Projection → MLP_head → binding logit
      2. Finetune (seq+struct): Embeddings → Proj + EGNN → Gated Fusion → immunogenicity logit

    The ESMC model is NOT part of this class — embeddings must be
    pre-computed and passed as ``seq_emb`` to forward / forward_pretrain.

    Parameters
    ----------
    seq_emb_dim : int
        Pre-computed sequence embedding dimension (2560 for ESMC-6B).
    proj_dim : int
        Projection output dimension (default 256).
    gnn_dim : int
        EGNN hidden dimension (default 64).
    gnn_layers : int
        Number of EGNN layers (default 6).
    dropout : float
        Classifier dropout rate.
    """

    def __init__(self,
                 seq_emb_dim=2560,
                 proj_dim=256,
                 gnn_dim=64,
                 gnn_layers=6,
                 dropout=0.1,
                 device="cuda",
                 property_embedding_dim=32,
                 combined_attention_heads=8,
                 **kwargs):
        super().__init__()
        self.device_name = device
        self.proj_dim = proj_dim
        self.gnn_dim = gnn_dim
        self.seq_emb_dim = seq_emb_dim
        self.property_embedding_dim = property_embedding_dim

        # ── Sequence tower (projection only; ESMC run separately) ─────
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_emb_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # ── Per-residue projection + position attention ──────────────
        self.residue_proj = nn.Sequential(
            nn.Linear(seq_emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.pos_attn = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 4),
            nn.Tanh(),
            nn.Linear(proj_dim // 4, 1),
        )

        # ── Structure tower (EGNN) ──────────────────────────────────
        self.GCN_layers = nn.ModuleList([
            EGNNConv(20, gnn_dim, gnn_dim, 1)
        ])
        for _ in range(gnn_layers - 1):
            self.GCN_layers.append(EGNNConv(gnn_dim, gnn_dim, gnn_dim, 1))

        self.self_attention = MultiHeadAttention(gnn_dim, 1)

        # ── Biochemical property embedding (matching paper v2) ──────
        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, property_embedding_dim),
            nn.ReLU(True),
        )

        # ── Cross-modal gated fusion ────────────────────────────────
        self.struct_proj = nn.Sequential(
            nn.Linear(gnn_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.seq_gate = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.Sigmoid())
        self.struct_gate = nn.Sequential(nn.Linear(gnn_dim, proj_dim), nn.Sigmoid())

        # ── Post-fusion multi-head attention (matching paper v2) ────
        self.combined_attention = MultiHeadAttention(
            16, combined_attention_heads, input_dim=1)

        # ── Classifier ──────────────────────────────────────────────
        cls_input_dim = proj_dim * 3 + property_embedding_dim
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(cls_input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # ── Stage 1a: seq-only pretraining head (MHC-I binding BCE) ─
        self.seq_pretrain_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # ── Stage 1b: multimodal foreignness regression head (MSE) ──
        self.pretrain_head = nn.Sequential(
            nn.Linear(cls_input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # ── Helpers ─────────────────────────────────────────────────────
    def _forward_sequence(self, seq_emb):
        """Project pre-computed ESMC embeddings (two-layer MLP)."""
        return self.seq_proj(seq_emb)  # [B, proj_dim]

    def _forward_structure(self, graph_data):
        """EGNN structure encoding."""
        dev = next(self.GCN_layers[0].parameters()).device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        batch_tensor = torch.cat(
            [torch.full((1, n), i)
             for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat,
                                          coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gnn_dim)
        attn_out, _ = self.self_attention(node_feat)
        attn_out = attn_out.view(-1, self.gnn_dim)
        pooled = global_mean_pool(attn_out, batch_tensor.flatten())
        return pooled  # [B, gnn_dim]

    def _forward_multimodal(self, graph_data, seq_emb, peptide_property=None):
        """Shared multimodal encoder: seq + struct + property → combined feat.

        Used identically by both pretraining and finetuning stages
        (matching the paper's design where only the head differs).
        """
        seq_feat = self._forward_sequence(seq_emb)          # [B, proj_dim]
        struct_feat = self._forward_structure(graph_data)   # [B, gnn_dim]

        struct_up = self.struct_proj(struct_feat)           # [B, proj_dim]
        g_seq = self.seq_gate(seq_feat)                     # [B, proj_dim]
        g_struct = self.struct_gate(struct_feat)            # [B, proj_dim]

        fused = g_seq * seq_feat + g_struct * struct_up    # [B, proj_dim]

        if peptide_property is not None:
            prop_emb = self.property_embedding(
                peptide_property.to(seq_feat.device))            # [B, prop_dim]
        else:
            prop_emb = torch.zeros(
                seq_feat.shape[0], self.property_embedding_dim,
                device=seq_feat.device)

        combined = torch.cat([fused, seq_feat, struct_up, prop_emb], dim=1)

        # Post-fusion multi-head attention (matching paper v2 pattern)
        combined = torch.unsqueeze(combined, 2)          # [B, D, 1]
        combined, _ = self.combined_attention(combined)  # [B, D, 16]
        combined = torch.mean(combined, dim=2)           # [B, D]

        return combined

    # ── Forward modes ───────────────────────────────────────────────

    def forward_pretrain(self, seq_emb, valid_mask=None):
        """Stage 1a: seq-only → MHC-I binding BCE.

        Supports two modes:
          - Per-residue [B, L, D]: position attention → weighted pool → head
          - Mean-pooled [B, D]: projection → head (legacy)

        External MHC-I eluted ligand data has no structure or biochemical
        properties, so only the sequence projection + seq_pretrain_head
        are trained here (alongside ESMC LoRA).
        """
        if seq_emb.dim() == 3:
            # Per-residue mode: learn which positions matter for MHC-I binding
            x = self.residue_proj(seq_emb)               # [B, L, proj_dim]
            attn = self.pos_attn(x)                       # [B, L, 1]
            if valid_mask is not None:
                attn = attn.masked_fill(
                    ~valid_mask.unsqueeze(-1), -1e9)
            attn_w = torch.softmax(attn, dim=1)           # [B, L, 1]
            seq_feat = (attn_w * x).sum(dim=1)            # [B, proj_dim]
        else:
            # Legacy mean-pooled mode
            seq_feat = self.seq_proj(seq_emb)
        return self.seq_pretrain_head(seq_feat)

    def forward_foreignness(self, graph_data, seq_emb, peptide_property=None):
        """Stage 1b: all 3 modalities → foreignness regression (MSE).

        Same multimodal encoder as finetuning; only the output head
        differs (pretrain_head vs classifier), matching the paper's
        transfer-learning design.
        """
        combined = self._forward_multimodal(graph_data, seq_emb,
                                            peptide_property)
        return self.pretrain_head(combined)

    def forward(self, graph_data, seq_emb, peptide_property=None):
        """Stage 2: all 3 modalities → immunogenicity classification (BCE)."""
        combined = self._forward_multimodal(graph_data, seq_emb,
                                           peptide_property)
        return self.classifier(combined)

    # ── Checkpoint IO ───────────────────────────────────────────────
    def load_pretrained_projection(self, path, map_location=None):
        """Load encoder weights from a prior stage, excluding all output heads.

        Stage 1a → 1b: loads seq_proj (trained), EGNN/gates/MHA stay random.
        Stage 1b → 2:  loads full multimodal encoder, classifier reinitialized.

        Matches the paper's transfer-learning pattern: warm-start encoder
        modules from the previous stage, reinitialize only the final head.
        """
        ckpt = torch.load(path, map_location=map_location)

        # Exclude all output heads — each stage uses its own
        encoder_state = {k: v for k, v in ckpt.items()
                        if not k.startswith('pretrain_head.')
                        and not k.startswith('seq_pretrain_head.')
                        and not k.startswith('classifier.')}

        missing, unexpected = self.load_state_dict(encoder_state, strict=False)

        loaded = len(encoder_state) - len(unexpected)
        if loaded > 0:
            print(f"  Loaded {loaded} encoder params from checkpoint "
                  f"(output heads excluded; reinitialize for current stage).")
        else:
            print("  [WARN] No compatible encoder weights found. "
                  "Model will start from scratch.")

    def load_trained(self, path, new_head=False, map_location=None):
        """Load full checkpoint."""
        self.load_state_dict(torch.load(path, map_location=map_location), strict=False)
        if new_head:
            dev = next(self.parameters()).device
            self.classifier = self.get_classifier().to(dev)

    def get_classifier(self):
        cls_input_dim = self.proj_dim * 3 + self.property_embedding_dim
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(cls_input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

class HybridModelv5(nn.Module):
    """ImmunoStruct v5 — HLA-conditioned three-tower pan-allelic model.

    Designed to fix the cross-locus generalization failure of v2 (which
    entangled HLA + peptide in a shared VAE latent and a mixed pMHC graph).
    Here the HLA allele is an explicit, separately-encoded conditioning signal
    (its 34-residue NetMHCpan pseudo-sequence), and it modulates the peptide
    and structure features via FiLM.

    Towers
    ------
    Peptide tower : precomputed ESMC embedding of the peptide → proj_dim
    HLA tower     : precomputed ESMC embedding of the 34-aa pseudo-seq → proj_dim
    Structure tower: EGNN over the pMHC graph → gnn_dim → proj_dim
    Fusion        : FiLM (HLA modulates peptide + structure) → concat → classifier

    The ESMC model is NOT part of this class — embeddings must be precomputed
    and passed as ``pep_emb`` / ``hla_emb`` (see precompute_esmc_embeddings.py).

    Parameters
    ----------
    seq_emb_dim : int
        Precomputed ESMC embedding dimension (2560).
    proj_dim : int
        Projection output dimension for peptide/HLA/structure (default 256).
    gnn_dim : int
        EGNN hidden dimension (default 64).
    gnn_layers : int
        Number of EGNN layers (default 6).
    dropout : float
        Classifier dropout rate.
    property_embedding_dim : int
        Biochemical property embedding dimension (default 32).
    """

    def __init__(self,
                 seq_emb_dim=2560,
                 proj_dim=256,
                 gnn_dim=64,
                 gnn_layers=6,
                 dropout=0.1,
                 device="cuda",
                 property_embedding_dim=32,
                 **kwargs):
        super().__init__()
        self.device_name = device
        self.proj_dim = proj_dim
        self.gnn_dim = gnn_dim
        self.seq_emb_dim = seq_emb_dim
        self.property_embedding_dim = property_embedding_dim

        # ── Peptide tower (ESMC → proj_dim) ─────────────────────────
        self.pep_proj = nn.Sequential(
            nn.Linear(seq_emb_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # ── HLA tower (ESMC pseudo-seq → proj_dim) ──────────────────
        self.hla_proj = nn.Sequential(
            nn.Linear(seq_emb_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # ── Structure tower (EGNN) ──────────────────────────────────
        self.GCN_layers = nn.ModuleList([
            EGNNConv(20, gnn_dim, gnn_dim, 1)
        ])
        for _ in range(gnn_layers - 1):
            self.GCN_layers.append(EGNNConv(gnn_dim, gnn_dim, gnn_dim, 1))

        self.self_attention = MultiHeadAttention(gnn_dim, 1)
        self.struct_proj = nn.Sequential(
            nn.Linear(gnn_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # ── FiLM: HLA conditions peptide + structure ────────────────
        self.film_pep_gamma = nn.Linear(proj_dim, proj_dim)
        self.film_pep_beta = nn.Linear(proj_dim, proj_dim)
        self.film_struct_gamma = nn.Linear(proj_dim, proj_dim)
        self.film_struct_beta = nn.Linear(proj_dim, proj_dim)
        self._zero_init_film()

        # ── Biochemical property embedding ──────────────────────────
        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, property_embedding_dim),
            nn.ReLU(True),
        )

        # ── Classifier ──────────────────────────────────────────────
        cls_input_dim = proj_dim * 3 + property_embedding_dim
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(cls_input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _zero_init_film(self):
        """Zero-init FiLM so the model starts as plain peptide+structure fusion."""
        for m in (self.film_pep_gamma, self.film_pep_beta,
                  self.film_struct_gamma, self.film_struct_beta):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _forward_structure(self, graph_data):
        """EGNN structure encoding → pooled [B, gnn_dim]."""
        dev = next(self.GCN_layers[0].parameters()).device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        batch_tensor = torch.cat(
            [torch.full((1, n), i)
             for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat,
                                          coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gnn_dim)
        attn_out, _ = self.self_attention(node_feat)
        attn_out = attn_out.view(-1, self.gnn_dim)
        pooled = global_mean_pool(attn_out, batch_tensor.flatten())
        return pooled  # [B, gnn_dim]

    def forward(self, graph_data, pep_emb, hla_emb, peptide_property=None):
        """Immunogenicity classification logits [B, 1].

        Parameters
        ----------
        graph_data : batched DGL graph
        pep_emb : torch.Tensor [B, seq_emb_dim]  (precomputed ESMC peptide emb)
        hla_emb : torch.Tensor [B, seq_emb_dim]  (precomputed ESMC pseudo-seq emb)
        peptide_property : torch.Tensor [B, 2] or None
        """
        struct_feat = self._forward_structure(graph_data)   # [B, gnn_dim]
        struct_up = self.struct_proj(struct_feat)           # [B, proj_dim]

        pep_proj = self.pep_proj(pep_emb.to(struct_up.device))   # [B, proj_dim]
        hla_proj = self.hla_proj(hla_emb.to(struct_up.device))   # [B, proj_dim]

        # FiLM conditioning (identity at init → stable start)
        gamma_p = self.film_pep_gamma(hla_proj)
        beta_p = self.film_pep_beta(hla_proj)
        pep_film = (1.0 + gamma_p) * pep_proj + beta_p

        gamma_s = self.film_struct_gamma(hla_proj)
        beta_s = self.film_struct_beta(hla_proj)
        struct_film = (1.0 + gamma_s) * struct_up + beta_s

        if peptide_property is not None:
            prop_emb = self.property_embedding(
                peptide_property.to(struct_up.device))
        else:
            prop_emb = torch.zeros(
                struct_up.shape[0], self.property_embedding_dim,
                device=struct_up.device)

        combined = torch.cat([pep_film, struct_film, hla_proj, prop_emb], dim=1)
        return self.classifier(combined)

    def load_trained(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location),
                             strict=False)


class HybridModelv5NoStruct(nn.Module):
    """Ablation variant of v5: peptide + HLA pseudo-sequence only, no EGNN tower.

    Used to test whether the allele-specific structure tower (which encodes the
    pMHC complex, and therefore does not transfer to unseen HLA alleles) helps or
    hurts cross-locus generalization. Identical to HybridModelv5 except:
      - no EGNN / struct_proj / struct-FiLM,
      - classifier input = proj_dim*2 + property_embedding_dim
        (pep_film + hla_proj + prop_emb).
    ``graph_data`` is accepted in forward() for interface compatibility but unused.
    """

    def __init__(self,
                 seq_emb_dim=2560,
                 proj_dim=256,
                 gnn_dim=64,
                 gnn_layers=6,
                 dropout=0.1,
                 device="cuda",
                 property_embedding_dim=32,
                 **kwargs):
        super().__init__()
        self.proj_dim = proj_dim
        self.seq_emb_dim = seq_emb_dim
        self.property_embedding_dim = property_embedding_dim

        self.pep_proj = nn.Sequential(
            nn.Linear(seq_emb_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        self.hla_proj = nn.Sequential(
            nn.Linear(seq_emb_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # FiLM: HLA conditions the peptide tower only (no structure to modulate).
        self.film_pep_gamma = nn.Linear(proj_dim, proj_dim)
        self.film_pep_beta = nn.Linear(proj_dim, proj_dim)
        for m in (self.film_pep_gamma, self.film_pep_beta):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, property_embedding_dim),
            nn.ReLU(True),
        )

        cls_input_dim = proj_dim * 2 + property_embedding_dim
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(cls_input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, graph_data, pep_emb, hla_emb, peptide_property=None):
        dev = pep_emb.device
        pep_proj = self.pep_proj(pep_emb.to(dev))
        hla_proj = self.hla_proj(hla_emb.to(dev))

        gamma_p = self.film_pep_gamma(hla_proj)
        beta_p = self.film_pep_beta(hla_proj)
        pep_film = (1.0 + gamma_p) * pep_proj + beta_p

        if peptide_property is not None:
            prop_emb = self.property_embedding(peptide_property.to(dev))
        else:
            prop_emb = torch.zeros(
                pep_film.shape[0], self.property_embedding_dim, device=dev)

        combined = torch.cat([pep_film, hla_proj, prop_emb], dim=1)
        return self.classifier(combined)

    def load_trained(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location),
                             strict=False)


class HybridModelv6(nn.Module):
    """ImmunoStruct v6 — sequence-primary cross-attention model.

    v5 conditions pooled peptide/structure with a *global* HLA FiLM scale-shift;
    it never lets peptide residues interact locally with HLA pseudo-seq residues.
    v6 replaces FiLM with per-residue cross-attention, the NetMHCpan / mhcSFM
    inductive bias that transfers to unseen HLA alleles:

      peptide tokens (9-11 residues) cross-attend to the 34 HLA pseudo-seq
      residues → mean-pooled peptide representation.
      structure (EGNN) is kept as a secondary pooled auxiliary, not the
      HLA-conditioned centerpiece.

    Inputs
    ------
    pep_emb : torch.Tensor [B, Lp, seq_emb_dim]  per-residue ESMC peptide
    hla_emb : torch.Tensor [B, 34, seq_emb_dim]  per-residue ESMC pseudo-seq
    pep_mask : torch.BoolTensor [B, Lp]  True = valid peptide residue
    hla_mask : torch.BoolTensor [B, 34]  True = valid pseudo-seq residue
    """

    def __init__(self,
                 seq_emb_dim=2560,
                 proj_dim=256,
                 gnn_dim=64,
                 gnn_layers=6,
                 n_heads=4,
                 dropout=0.1,
                 device="cuda",
                 property_embedding_dim=32,
                 **kwargs):
        super().__init__()
        self.proj_dim = proj_dim
        self.seq_emb_dim = seq_emb_dim
        self.gnn_dim = gnn_dim
        self.property_embedding_dim = property_embedding_dim

        # Per-residue projections (peptide + HLA pseudo-seq share dim)
        self.pep_proj = nn.Linear(seq_emb_dim, proj_dim)
        self.hla_proj = nn.Linear(seq_emb_dim, proj_dim)

        # Cross-attention: peptide residues ↔ HLA pseudo-seq residues
        self.pep_hla_attn = CrossModalAttentionBlock(proj_dim, n_heads, dropout)

        # Structure tower (EGNN → self-attn → pool), identical to v5
        self.GCN_layers = nn.ModuleList([EGNNConv(20, gnn_dim, gnn_dim, 1)])
        for _ in range(gnn_layers - 1):
            self.GCN_layers.append(EGNNConv(gnn_dim, gnn_dim, gnn_dim, 1))

        self.self_attention = MultiHeadAttention(gnn_dim, 1)
        self.struct_proj = nn.Linear(gnn_dim, proj_dim)

        # Biochemical property embedding
        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, property_embedding_dim),
            nn.ReLU(True),
        )

        cls_input_dim = proj_dim * 3 + property_embedding_dim
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(cls_input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _forward_structure(self, graph_data):
        """EGNN structure encoding → pooled [B, gnn_dim] (identical to v5)."""
        dev = next(self.GCN_layers[0].parameters()).device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        batch_tensor = torch.cat(
            [torch.full((1, n), i)
             for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat,
                                          coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gnn_dim)
        attn_out, _ = self.self_attention(node_feat)
        attn_out = attn_out.view(-1, self.gnn_dim)
        pooled = global_mean_pool(attn_out, batch_tensor.flatten())
        return pooled  # [B, gnn_dim]

    @staticmethod
    def _masked_mean(x, mask):
        """Mean-pool sequence dim with a [B, L] bool mask (True = valid)."""
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)

    def forward(self, graph_data, pep_emb, hla_emb, pep_mask=None, hla_mask=None,
                peptide_property=None):
        """Immunogenicity classification logits [B, 1]."""
        dev = pep_emb.device
        B = pep_emb.shape[0]

        pep = self.pep_proj(pep_emb)  # [B, Lp, d]
        hla = self.hla_proj(hla_emb)  # [B, 34, d]

        if pep_mask is None:
            pep_mask = torch.ones(B, pep.shape[1], dtype=torch.bool, device=dev)
        if hla_mask is None:
            hla_mask = torch.ones(B, hla.shape[1], dtype=torch.bool, device=dev)

        # Peptide residues attend locally to HLA pseudo-seq residues
        pep_hla = self.pep_hla_attn(pep, hla, key_padding_mask=hla_mask)  # [B, Lp, d]
        pep_vec = self._masked_mean(pep_hla, pep_mask)                    # [B, d]
        hla_vec = self._masked_mean(hla, hla_mask)                        # [B, d]

        # Structure (secondary auxiliary)
        struct_vec = self.struct_proj(self._forward_structure(graph_data))  # [B, d]

        if peptide_property is not None:
            prop_emb = self.property_embedding(peptide_property.to(dev))
        else:
            prop_emb = torch.zeros(B, self.property_embedding_dim, device=dev)

        combined = torch.cat([pep_vec, hla_vec, struct_vec, prop_emb], dim=1)
        return self.classifier(combined)

    def load_trained(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location),
                             strict=False)


class HybridModelv7(HybridModelv6):
    """ImmunoStruct v7 — full-MHC-sequence allele representation.

    Identical architecture to v6 (peptide residues cross-attend to allele
    residues, structure demoted to a pooled EGNN auxiliary), but the HLA key
    is the FULL MHC class-I extracellular sequence (~276 aa, α1+α2+α3)
    per-residue ESMC embedding instead of the 34-residue NetMHCpan
    pseudo-sequence.

    Rationale: NetMHCpan's 34 pocket positions are a hand-crafted prior that
    discards flanking groove + domain context; protein-language models
    (MHCRoBERTa / SoloPepHLAPan) condition on the full MHC sequence and
    transfer better to unseen alleles. v7 lets peptide tokens attend over the
    entire groove + domain instead of 34 fixed positions.

    Inputs
    ------
    pep_emb : torch.Tensor [B, Lp, seq_emb_dim]  per-residue ESMC peptide
    hla_emb : torch.Tensor [B, Lm, seq_emb_dim]  per-residue ESMC full MHC (~276)
    pep_mask : torch.BoolTensor [B, Lp]  True = valid peptide residue
    hla_mask : torch.BoolTensor [B, Lm]  True = valid MHC residue
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HybridModelv8(HybridModelv6):
    """ImmunoStruct v8 — structure as a cross-attention target.

    v6 lets peptide residues cross-attend ONLY to the 34 HLA pseudo-seq
    residues; structure is demoted to a pooled EGNN vector. v8 adds a SECOND
    parallel cross-attention path where the peptide tokens also attend to
    per-residue STRUCTURE node tokens (EGNN output before pooling), so the
    sequence tower reads structural context locally instead of only through a
    pooled summary. The pooled EGNN vector is kept unchanged (v6-identical) as
    an auxiliary, making this a clean single-variable ablation.

    Classifier input grows from 3·proj_dim (v6) to 4·proj_dim:
      [pep_hla_vec, pep_struct_vec, hla_vec, struct_vec, prop_emb]

    Padded graph nodes (all-zero 20-dim AA one-hot) are masked out of the
    structure cross-attention via a valid-node mask derived from raw features.

    Inputs (identical to v6): pep_emb [B,Lp,D], hla_emb [B,34,D],
    pep_mask [B,Lp], hla_mask [B,34].
    """

    def __init__(self, seq_emb_dim=2560, proj_dim=256, gnn_dim=64, gnn_layers=6,
                 n_heads=4, dropout=0.1, device="cuda", property_embedding_dim=32,
                 **kwargs):
        super().__init__(seq_emb_dim=seq_emb_dim, proj_dim=proj_dim, gnn_dim=gnn_dim,
                         gnn_layers=gnn_layers, n_heads=n_heads, dropout=dropout,
                         device=device, property_embedding_dim=property_embedding_dim,
                         **kwargs)
        self.n_heads = n_heads
        self.dropout = dropout

        # New structure cross-attention path (peptide -> structure node tokens)
        self.struct_node_proj = nn.Linear(gnn_dim, proj_dim)
        self.pep_struct_attn = CrossModalAttentionBlock(proj_dim, n_heads, dropout)

        # Grow the classifier to absorb the extra structure-attended peptide vector.
        cls_input_dim = proj_dim * 4 + property_embedding_dim
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(cls_input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _forward_structure_tokens(self, graph_data):
        """EGNN → per-node structure tokens + valid mask + pooled auxiliary.

        Returns
        -------
        attn_out : torch.Tensor [B, N, gnn_dim]  per-node structure tokens
        struct_mask : torch.BoolTensor [B, N]  True = real (non-padded) node
        pooled : torch.Tensor [B, gnn_dim]  global mean pool (v6-identical)
        """
        dev = next(self.GCN_layers[0].parameters()).device
        raw_x = graph_data.ndata['x']
        node_feat = raw_x[:, :20].to(dev)
        coord_feat = raw_x[:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        batch_tensor = torch.cat(
            [torch.full((1, n), i)
             for i, n in enumerate(graph_data.batch_num_nodes())], dim=0).to(dev)
        B = batch_tensor.shape[0]
        N = node_feat.shape[0] // B

        # Valid-node mask from RAW features (padded nodes are all-zero one-hot).
        valid_flat = node_feat.abs().sum(dim=-1) > 0.5      # [B*N]
        struct_mask = valid_flat.view(B, N)                  # [B, N]

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat,
                                          coord_feat, edge_feat)

        node_feat = node_feat.view(B, N, self.gnn_dim)
        attn_out, _ = self.self_attention(node_feat)          # [B, N, gnn_dim]
        attn_out_flat = attn_out.reshape(-1, self.gnn_dim)
        pooled = global_mean_pool(attn_out_flat, batch_tensor.reshape(-1))
        return attn_out, struct_mask, pooled                   # pooled [B, gnn_dim]

    def forward(self, graph_data, pep_emb, hla_emb, pep_mask=None, hla_mask=None,
                peptide_property=None):
        """Immunogenicity classification logits [B, 1]."""
        dev = pep_emb.device
        B = pep_emb.shape[0]

        pep = self.pep_proj(pep_emb)  # [B, Lp, d]
        hla = self.hla_proj(hla_emb)  # [B, 34, d]

        if pep_mask is None:
            pep_mask = torch.ones(B, pep.shape[1], dtype=torch.bool, device=dev)
        if hla_mask is None:
            hla_mask = torch.ones(B, hla.shape[1], dtype=torch.bool, device=dev)

        # Path 1: peptide residues attend to HLA pseudo-seq residues (v6)
        pep_hla = self.pep_hla_attn(pep, hla, key_padding_mask=hla_mask)  # [B, Lp, d]
        pep_hla_vec = self._masked_mean(pep_hla, pep_mask)                 # [B, d]
        hla_vec = self._masked_mean(hla, hla_mask)                         # [B, d]

        # Path 2 (new): peptide residues attend to structure node tokens
        struct_tokens, struct_mask, struct_pooled = self._forward_structure_tokens(graph_data)
        struct_tokens = self.struct_node_proj(struct_tokens)               # [B, N, d]
        pep_struct = self.pep_struct_attn(pep, struct_tokens,
                                          key_padding_mask=struct_mask)    # [B, Lp, d]
        pep_struct_vec = self._masked_mean(pep_struct, pep_mask)           # [B, d]
        struct_vec = self.struct_proj(struct_pooled)                       # [B, d] (v6 aux)

        if peptide_property is not None:
            prop_emb = self.property_embedding(peptide_property.to(dev))
        else:
            prop_emb = torch.zeros(B, self.property_embedding_dim, device=dev)

        combined = torch.cat(
            [pep_hla_vec, pep_struct_vec, hla_vec, struct_vec, prop_emb], dim=1)
        return self.classifier(combined)


class HybridModel(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim : int = 8,
                 *args, **kwargs):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = SelfAttention(gat_hidden_channels)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels, 32),
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

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier = self.get_classifier().to(self.device)

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        dev = self.device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat, coord_feat, edge_feat)

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

        # Fusion and final layers
        combined = torch.cat([x_gat_node, z_vae], dim=1)
        combined_gat_only = torch.cat([x_gat_node], dim=1)
        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output

class HybridModel_SSL(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim : int = 8,
                 mlp_features = 32,
                 *args, **kwargs):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
        self.mlp_features = mlp_features

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = SelfAttention(gat_hidden_channels)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.classifier_head = nn.Linear(self.mlp_features, 1)
        self.node_predictor_head = nn.Linear(self.mlp_features, 20) # number of amino acids

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels, 32),
            nn.ReLU(True),
            nn.Dropout(0.1)
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

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier_head = nn.Linear(self.mlp_features, 1).to(self.device)

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        dev = self.device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat, coord_feat, edge_feat)

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

        # Fusion and final layers
        combined = torch.cat([x_gat_node, z_vae], dim=1)
        combined_gat_only = torch.cat([x_gat_node], dim=1)

        fusion_output = self.classifier(combined)

        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output, node_prediction

        if return_attention:
            return attention_weights_n, mu, logvar, final_output, node_prediction

        return recon_x, mu, logvar, final_output, node_prediction

class HybridModelv2(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 self_attention_heads: int = 1,
                 property_embedding_dim: int = 8,
                 combined_attention_heads: int = 8,
                 *args, **kwargs):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim

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

        self.combined_attention = MultiHeadAttention(16, combined_attention_heads, input_dim=1)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels, 32),
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

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier = self.get_classifier().to(self.device)

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        dev = self.device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat, coord_feat, edge_feat)

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


        # Fusion and final layers
        combined_x = torch.cat([x_gat_node, z_vae], dim=1)
        combined = torch.unsqueeze(combined_x, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        combined_gat_only = torch.cat([x_gat_node], dim=1)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output

class HybridModelv2_SSL(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 property_embedding_dim: int = 8,
                 self_attention_heads: int = 1,
                 combined_attention_heads: int = 8,
                 mlp_features: int = 32,
                 *args, **kwargs):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim
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

         # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.classifier_head = nn.Linear(self.mlp_features, 1)
        self.node_predictor_head = nn.Linear(self.mlp_features, 20) # number of amino acids

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels, 32),
            nn.ReLU(True),
            nn.Dropout(0.1)
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

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier_head = nn.Linear(self.mlp_features, 1).to(self.device)

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        dev = self.device
        node_feat = graph_data.ndata['x'][:, :20].to(dev)
        coord_feat = graph_data.ndata['x'][:, 20:].to(dev)
        edge_feat = graph_data.edata['edge_attr'].to(dev)

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(dev)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(graph_data, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gat_hidden_channels)
        attention_output_n, attention_weights_n = self.self_attention(node_feat)
        attention_output_n = attention_output_n.view(-1, self.gat_hidden_channels)

        # Use global_mean_pool with the batch tensor
        x_gat_node_mean = global_mean_pool(attention_output_n, batch_tensor.flatten())

        # peptide property
        peptide_property = self.property_embedding(peptide_property)

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)

        # Fusion and final layers
        combined_x = torch.cat([x_gat_node_mean, z_vae], dim=1)
        combined = torch.unsqueeze(combined_x, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        combined_gat_only = torch.cat([x_gat_node_mean], dim=1)

        fusion_output = self.classifier(combined)

        final_output = self.classifier_head(fusion_output)
        node_prediction = self.node_predictor_head(fusion_output)


        if return_embedding:
            return combined_gat_only, mu, logvar, final_output, node_prediction

        if return_attention:
            return attention_weights_n, mu, logvar, final_output, node_prediction

        return recon_x, mu, logvar, final_output, node_prediction
