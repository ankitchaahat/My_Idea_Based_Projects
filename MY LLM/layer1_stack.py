import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Layer 1: Sublayers
class Layer1(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.1):
        super(Layer1, self).__init__()
        # Sublayers
        self.pre_layer_norm = LayerNorm(embed_dim)  # Pre-LayerNorm
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)  # Self-Attention
        self.residual1 = ResidualConnection(embed_dim, dropout)  # Residual Connection 1
        self.layer_norm1 = LayerNorm(embed_dim)  # LayerNorm after Self-Attention
        self.ffn = FeedforwardNetwork(embed_dim, hidden_dim, dropout)  # Feedforward Network
        self.residual2 = ResidualConnection(embed_dim, dropout)  # Residual Connection 2
        self.layer_norm2 = LayerNorm(embed_dim)  # LayerNorm after FFN

    def forward(self, x):
        # Sublayer 1: Self-Attention with Residual Connection
        x = self.residual1(x, self.self_attention)  # Self-Attention + Residual
        x = self.layer_norm1(x)  # LayerNorm after Self-Attention

        # Sublayer 2: Feedforward Network with Residual Connection
        x = self.residual2(x, self.ffn)  # FFN + Residual
        x = self.layer_norm2(x)  # LayerNorm after FFN

        return x
