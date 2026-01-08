"""
Token and Positional Embeddings

PURPOSE:
This file converts token IDs (integers) into continuous vector representations
that the Transformer can process. It also adds positional information so the
model knows the order of tokens (since attention has no inherent notion of position).

WHAT THIS STEP DOES:
1. Token Embedding: Maps each token ID to a learned vector of size d_model
   - Input: [batch_size, seq_len] of token IDs
   - Output: [batch_size, seq_len, d_model] of vectors

2. Positional Embedding: Adds position information to each token
   - Two approaches: learned embeddings or sinusoidal encoding
   - Same shape as token embeddings: [batch_size, seq_len, d_model]

3. Combines both: token_emb + pos_emb
   - Final output: [batch_size, seq_len, d_model]

PACKAGES USED:
- torch: PyTorch tensors and neural network modules
- torch.nn: Embedding layer, Dropout

FILES FROM THIS PROJECT:
- None (this is a base component)

TENSOR SHAPES EXPLAINED:
- batch_size: Number of sequences processed together
- seq_len: Length of each sequence (512 in our case)
- vocab_size: Size of tokenizer vocabulary (8000)
- d_model: Embedding dimension (256-512)

COMMON FAILURE MODES TO AVOID:
- Forgetting dropout → overfitting
- Wrong positional encoding dimension → shape mismatch
- Not scaling embeddings → training instability
- Using fixed positions > max_seq_len → index out of bounds
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Convert token IDs to embeddings.

    Tensor shape flow:
        Input:  [batch_size, seq_len] (token IDs)
        Output: [batch_size, seq_len, d_model] (embeddings)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, d_model]

        Note: We scale by sqrt(d_model) following the original Transformer paper.
        This prevents embeddings from being too small relative to positional encodings.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.

    Alternative to sinusoidal encoding - the model learns optimal position representations.
    GPT uses learned embeddings, so we follow that.

    Tensor shape flow:
        Input:  [batch_size, seq_len]
        Output: [batch_size, seq_len, d_model]
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs [batch_size, seq_len] (we only use seq_len from this)

        Returns:
            Position embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.size()

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]

        # Expand to batch: [batch_size, seq_len]
        positions = positions.expand(batch_size, seq_len)

        return self.embedding(positions)


class TokenPositionalEmbedding(nn.Module):
    """
    Combined token + positional embeddings.

    This is what the transformer actually uses.

    Tensor shape flow:
        Input:  [batch_size, seq_len] (token IDs)
        Output: [batch_size, seq_len, d_model] (combined embeddings)
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs [batch_size, seq_len]

        Returns:
            Combined embeddings [batch_size, seq_len, d_model]
        """
        # Get token embeddings: [batch_size, seq_len, d_model]
        token_emb = self.token_embedding(x)

        # Get positional embeddings: [batch_size, seq_len, d_model]
        pos_emb = self.positional_embedding(x)

        # Combine by addition
        # Why addition? Because we want the model to learn relationships
        # between both "what" (token) and "where" (position)
        embeddings = token_emb + pos_emb

        # Apply dropout for regularization
        return self.dropout(embeddings)
