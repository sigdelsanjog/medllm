"""
Transformer Decoder Block

PURPOSE:
Combines attention, feed-forward network, residual connections, and layer
normalization into a single reusable block. Multiple blocks are stacked
to form the full transformer.

WHAT THIS STEP DOES:
1. Multi-head causal self-attention
   - With residual connection: x + attention(x)
   - With layer normalization

2. Feed-forward network
   - With residual connection: x + ffn(x)
   - With layer normalization

Architecture pattern (Pre-LN vs Post-LN):
We use Pre-LN (normalize before sublayer) because it's more stable:
  x = x + attention(LayerNorm(x))
  x = x + ffn(LayerNorm(x))

PACKAGES USED:
- torch: PyTorch tensors
- torch.nn: Module, LayerNorm, Dropout

FILES FROM THIS PROJECT:
- architecture/attention.py: Multi-head attention module
- architecture/feedforward.py: FFN module

TENSOR SHAPES:
- Input: [batch_size, seq_len, d_model]
- Output: [batch_size, seq_len, d_model] (unchanged)

COMMON FAILURE MODES TO AVOID:
- Post-LN instead of Pre-LN → training instability
- Forgetting residual connections → vanishing gradients
- Wrong LayerNorm dimension → incorrect normalization
- Dropout too high → underfitting
- Dropout too low → overfitting
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerDecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.

    This is one "layer" of the transformer. We stack multiple of these.

    Architecture (Pre-LN):
        1. LayerNorm → Multi-Head Attention → Residual
        2. LayerNorm → Feed-Forward → Residual

    Tensor shape flow:
        Input:  [batch_size, seq_len, d_model]
        Output: [batch_size, seq_len, d_model] (same shape)

    Why Pre-LN instead of Post-LN?
    - Pre-LN: Normalize BEFORE sublayer → more stable gradients
    - Post-LN: Normalize AFTER sublayer → can have training instability
    - GPT-2 and modern transformers use Pre-LN
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization (one for each sublayer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Causal mask [seq_len, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, d_model]

        Step-by-step explanation:
        1. Normalize input
        2. Apply attention
        3. Add residual (skip connection)
        4. Normalize result
        5. Apply feed-forward
        6. Add residual (skip connection)
        """
        # Sublayer 1: Self-Attention with residual
        # Pre-LN: normalize first
        normed = self.norm1(x)

        # Apply attention
        attention_output = self.attention(normed, mask)

        # Residual connection: x + attention(norm(x))
        x = x + self.dropout(attention_output)

        # Sublayer 2: Feed-Forward with residual
        # Pre-LN: normalize first
        normed = self.norm2(x)

        # Apply feed-forward
        ff_output = self.feed_forward(normed)

        # Residual connection: x + ffn(norm(x))
        x = x + self.dropout(ff_output)

        return x
