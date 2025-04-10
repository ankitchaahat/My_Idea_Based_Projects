import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Optimized Layer Normalization (Pre-LN)
# Define LayerNorm (if not already defined)
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim, dtype=torch.float32))  # Learnable scale
        self.beta = nn.Parameter(torch.zeros(embed_dim, dtype=torch.float32))  # Learnable shift
        self.eps = eps  # Small value for numerical stability

    def forward(self, x):
        # Apply Layer Normalization
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta, self.eps)

# Define ResidualConnection
class ResidualConnection(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(embed_dim)  # Pre-LayerNorm
        self.dropout = nn.Dropout(dropout)  # Regularization

    def forward(self, x, sublayer):
        # Apply Pre-LayerNorm, sublayer, and residual connection
        return x + self.dropout(sublayer(self.norm(x)))

# Define MultiHeadSelfAttention (if not already defined)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads  # Ensure num_heads is set as an attribute
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Fused QKV Projection (Single Linear Layer for Efficiency)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, dtype=torch.float32)
        self.out_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.float32)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V in a single pass
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        Q, K, V = qkv.unbind(dim=2)  # Split into separate tensors

        # Reshape for multi-head attention
        Q = Q.transpose(1, 2)  # Shape: (batch, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Flash Attention (Optimized Scaled Dot-Product Attention)
        if mask is not None:
            mask = mask.to(dtype=torch.float16, device=x.device)  # Ensure mask is on the correct device and dtype
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

        # Reshape back to original shape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Apply output projection
        return self.out_proj(output)
