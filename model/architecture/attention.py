"""
Multi-Head Causal Self-Attention

PURPOSE:
This is the core mechanism that allows the model to attend to different parts
of the input sequence. "Causal" means the model can only look at previous tokens,
not future ones (essential for next-token prediction).

WHAT THIS STEP DOES:
1. Linear projections: Create Query, Key, Value matrices
   - Input: [batch_size, seq_len, d_model]
   - Q, K, V each: [batch_size, seq_len, d_model]

2. Split into multiple heads
   - Reshape to: [batch_size, n_heads, seq_len, d_head]
   - where d_head = d_model / n_heads

3. Scaled dot-product attention
   - Compute attention scores: Q @ K^T / sqrt(d_head)
   - Apply causal mask (CRITICAL: prevents looking at future)
   - Softmax to get attention weights
   - Apply to values: attention_weights @ V

4. Concatenate heads and project back
   - Output: [batch_size, seq_len, d_model]

PACKAGES USED:
- torch: PyTorch tensors and operations
- torch.nn: Linear layers, Dropout
- torch.nn.functional: Softmax
- math: sqrt for scaling

FILES FROM THIS PROJECT:
- None (this is a base component)

TENSOR SHAPES EXPLAINED:
- n_heads: Number of attention heads (4-8)
- d_head: Dimension per head (d_model / n_heads)
- Causal mask: Lower triangular matrix [seq_len, seq_len]

COMMON FAILURE MODES TO AVOID:
- Missing causal mask → model cheats by seeing future tokens
- Wrong mask shape → silent failures or crashes
- Not scaling attention scores → vanishing/exploding gradients
- Forgetting dropout → overfitting
- Wrong tensor transpose/reshape → incorrect attention patterns
- Not masking padding tokens → attending to meaningless tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    This is the CORE of the transformer. Understanding this is critical.

    Key concept: Attention lets each token "look at" other tokens to gather context.
    "Causal" means token at position i can ONLY look at positions <= i (not future).

    Tensor shape flow:
        Input:  [batch_size, seq_len, d_model]
        Q,K,V:  [batch_size, seq_len, d_model]
        Split:  [batch_size, n_heads, seq_len, d_head]
        Attention: [batch_size, n_heads, seq_len, seq_len]
        Output: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # Dimension per head

        # Linear projections for Q, K, V
        # We use separate projections for each, not combined
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Causal mask [seq_len, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()

        # 1. Linear projections
        # Each: [batch_size, seq_len, d_model]
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # 2. Split into multiple heads
        # Reshape: [batch_size, seq_len, n_heads, d_head]
        # Then transpose: [batch_size, n_heads, seq_len, d_head]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Scaled dot-product attention
        # Q @ K^T: [batch_size, n_heads, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 4. Apply causal mask (CRITICAL!)
        # This prevents position i from attending to positions > i
        if mask is not None:
            # mask should be [seq_len, seq_len] or [batch_size, 1, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        # 5. Softmax to get attention weights
        # [batch_size, n_heads, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights (standard practice)
        attention_weights = self.dropout(attention_weights)

        # 6. Apply attention to values
        # [batch_size, n_heads, seq_len, d_head]
        attended_values = torch.matmul(attention_weights, V)

        # 7. Concatenate heads
        # Transpose back: [batch_size, seq_len, n_heads, d_head]
        # Then reshape: [batch_size, seq_len, d_model]
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, self.d_model)

        # 8. Final linear projection
        output = self.out_linear(attended_values)

        return output


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal mask for autoregressive generation.

    The mask is a lower triangular matrix:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    This ensures position i can only attend to positions <= i.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Causal mask [seq_len, seq_len]

    Why this works:
    - Position 0 can see position 0 only
    - Position 1 can see positions 0, 1
    - Position 2 can see positions 0, 1, 2
    - etc.

    This is the ESSENCE of causal language modeling!
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask  # [seq_len, seq_len]
