import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        """
        Rotary Positional Encoding (RoPE) for transformers.

        Args:
            embed_dim (int): Dimension of token embeddings.
        """
        super(RotaryPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        # Compute inverse frequency terms for RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim))
        self.register_buffer("inv_freq", inv_freq)  # Store as buffer

    def rotate_half(self, x):
        """
        Rotates the last dimension by 90 degrees.

        Args:
            x (torch.Tensor): Input tensor of shape (..., embed_dim).

        Returns:
            torch.Tensor: Rotated tensor of same shape.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        """
        Forward pass for RoPE.

        Args:
            x (torch.Tensor): Token embeddings of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Rotated embeddings with positional information.
        """
        batch_size, seq_len, embed_dim = x.shape

        # Generate position indices
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)

        # Compute rotation frequencies
        freqs = torch.matmul(positions, self.inv_freq.unsqueeze(0))  # Shape: [seq_len, embed_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # Shape: [seq_len, embed_dim]

        # Compute cos and sin embeddings
        cos_emb, sin_emb = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)  # Shape: [1, seq_len, embed_dim]

        # Apply RoPE transformation
        x_rotated = (x * cos_emb) + (self.rotate_half(x) * sin_emb)

        return x_rotated


# ✅ Hyperparameters
batch_size = 8
seq_len = 128
embed_dim = 512

# ✅ Initialize RoPE
rotary_pe = RotaryPositionalEncoding(embed_dim).to(device)

# ✅ Example Input (Random Token Embeddings)
input_embeddings = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32, device=device)

# ✅ Apply RoPE
output_embeddings = rotary_pe(input_embeddings)

# ✅ Debugging Info
print("✅ Input Embeddings Shape:", input_embeddings.shape)  # Expected: (8, 128, 512)
print("✅ RoPE Output Shape:", output_embeddings.shape)  # Expected: (8, 128, 512)
print("✅ RoPE Output dtype:", output_embeddings.dtype)  # Expected: float32
