import torch
import torch.nn as nn
from torch.amp import autocast

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=0, dropout=0.1):
        """
        Token Embedding Layer using PyTorch nn.Embedding.

        Args:
            vocab_size (int): Number of unique tokens in vocabulary.
            embed_dim (int): Dimension of each token embedding.
            padding_idx (int, optional): Index of padding token. Default: 0.
            dropout (float, optional): Dropout probability. Default: 0.1.
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx  # Helps handle padding tokens efficiently
        )
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        nn.init.xavier_uniform_(self.embedding.weight)  # Better initialization

    def forward(self, input_tokens):
        """
        Forward pass to convert token IDs to embeddings.

        Args:
            input_tokens (torch.Tensor): Tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Token embeddings of shape (batch_size, seq_len, embed_dim).
        """
        # Ensure input is on the correct device
        input_tokens = input_tokens.to(self.embedding.weight.device)

        # Apply embedding and dropout

        embeddings = self.embedding(input_tokens)
        embeddings = self.dropout(embeddings)  # Apply dropout

        return embeddings


# ✅ Hyperparameters
vocab_size = 10000  # Size of vocabulary
embed_dim = 512  # Embedding dimension per token
batch_size = 8  # Number of sequences processed in parallel
seq_len = 128  # Max sequence length

# ✅ Initialize Token Embedding Layer
# ✅ Initialize Token Embedding Layer with correct vocab size
token_embedding = TokenEmbedding(
    vocab_size=tokenizer_layer.vocab_size,  # Use actual vocab size
    embed_dim=512
).to(device)

# ✅ Example Input (Random Token IDs)
input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

# ✅ Apply Token Embedding
output_embeddings = token_embedding(input_tokens)

# ✅ Debugging Info
print("\n✅ Input Tokens Shape:", input_tokens.shape)  # Expected: (8, 128)
print("✅ Token Embedding Output Shape:", output_embeddings.shape)  # Expected: (8, 128, 512)
print("✅ Token Embedding Output dtype:", output_embeddings.dtype)  # Should be float16 if AMP enabled
