import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Optimized Layer Normalization (Pre-LN)
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim, dtype=torch.float32))  # Learnable scale
        self.beta = nn.Parameter(torch.zeros(embed_dim, dtype=torch.float32))  # Learnable shift
        self.eps = eps  # Small value for numerical stability

    def forward(self, x):
        # Apply Layer Normalization
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta, self.eps)

# ✅ Optimized Feedforward Network (FFN)
class FeedforwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        """
        Feedforward Network with GELU Activation and Dropout.
        """
        super(FeedforwardNetwork, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim, dtype=torch.float32)  # Expansion
        self.linear2 = nn.Linear(hidden_dim, embed_dim, dtype=torch.float32)  # Compression
        self.dropout = nn.Dropout(dropout)  # Regularization
        self.activation = nn.GELU()  # Activation function

    def forward(self, x):
        x = self.linear1(x)  # Expand dimensions
        x = self.activation(x)  # Apply GELU
        x = self.dropout(x)  # Apply dropout
        x = self.linear2(x)  # Compress dimensions
        return x

# ✅ Optimized Residual Connection with Pre-Norm
class ResidualConnection(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        """
        Residual Connection with Pre-Norm (Better Stability).
        """
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(embed_dim)  # Pre-LayerNorm
        self.dropout = nn.Dropout(dropout)  # Regularization

    def forward(self, x, sublayer):
        # Apply Pre-LayerNorm, sublayer, and residual connection
        return x + self.dropout(sublayer(self.norm(x)))
