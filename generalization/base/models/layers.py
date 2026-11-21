import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights
    
# https://github.com/hyunwoongko/transformer/blob/master/models/layers/scale_dot_product_attention.py
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = F.softmax(score, dim=-1)

        # 4. multiply with Value
        v = score @ v

        return v, score

# https://github.com/hyunwoongko/transformer/blob/master/models/layers/multi_head_attention.py
class MultiHeadAttention(nn.Module):

    def __init__(self, feature_dim, n_head, input_dim=None):
        super(MultiHeadAttention, self).__init__()

        assert feature_dim % n_head == 0, "Embedding dimension must be 0 modulo number of heads."

        if not input_dim:
            input_dim = feature_dim
        self.n_head = n_head
        self.w_q = nn.Linear(input_dim, feature_dim)
        self.w_k = nn.Linear(input_dim, feature_dim)
        self.w_v = nn.Linear(input_dim, feature_dim)
        self.w_concat = nn.Linear(feature_dim, feature_dim)
        self.attention = ScaleDotProductAttention()

    def forward(self, x, mask=None):
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        q, k, v = self.split(q), self.split(k), self.split(v)

        # attention here is multi head attention
        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class CrossModalAttentionBlock(nn.Module):
    """Cross-attention: query tokens attend to a separate key/value context.

    Used by HybridModelv6 so peptide residues can attend locally to HLA
    pseudo-sequence residues (the NetMHCpan / mhcSFM position-pocket bias)
    and to per-residue structure tokens. Both ``query`` and ``context`` must
    already be projected to ``d_model``.

    Parameters
    ----------
    d_model : int
        Hidden dimension (query and context share it).
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate for the residual FFN.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, context, key_padding_mask=None):
        """Cross-attention forward.

        Parameters
        ----------
        query : torch.Tensor [B, Lq, d_model]
        context : torch.Tensor [B, Lk, d_model]
        key_padding_mask : torch.BoolTensor [B, Lk] or None
            True = valid key position, False = padding (masked out of softmax).

        Returns
        -------
        torch.Tensor [B, Lq, d_model]
        """
        B, Lq, _ = query.shape
        Lk = context.shape[1]

        q = self.w_q(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(context).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(context).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        attn = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # [B, H, Lq, Lk]
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                ~key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard all-masked rows

        out = attn @ v  # [B, H, Lq, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(out)

        out = self.norm1(query + out)
        out = self.norm2(out + self.ffn(out))
        return out
