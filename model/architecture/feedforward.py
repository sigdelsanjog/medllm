"""
Feed-Forward Network (FFN)

PURPOSE:
Applies position-wise transformations to each token independently.
This adds non-linear processing power to the model beyond what attention provides.

WHAT THIS STEP DOES:
1. First linear projection: Expand dimension
   - Input: [batch_size, seq_len, d_model]
   - Output: [batch_size, seq_len, d_ff]
   - Typically d_ff = 4 * d_model (expansion)

2. Non-linear activation (GELU or ReLU)
   - Introduces non-linearity
   - GELU is smoother than ReLU, often better for transformers

3. Second linear projection: Project back
   - Input: [batch_size, seq_len, d_ff]
   - Output: [batch_size, seq_len, d_model]

4. Dropout for regularization

PACKAGES USED:
- torch: PyTorch tensors
- torch.nn: Linear, Dropout, GELU/ReLU

FILES FROM THIS PROJECT:
- None (this is a base component)

TENSOR SHAPES EXPLAINED:
- d_ff: Hidden dimension in FFN (usually 4 * d_model)
- For d_model=256, d_ff=1024
- For d_model=512, d_ff=2048

COMMON FAILURE MODES TO AVOID:
- d_ff too small → insufficient expressiveness
- d_ff too large → OOM on GPU, overfitting
- Forgetting activation → just linear transformation (useless)
- Using ReLU with high learning rate → dead neurons
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applies the same transformation to each position independently.
    This is why it's called "position-wise" - no interaction between positions.

    Architecture:
        Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Dropout

    Tensor shape flow:
        Input:  [batch_size, seq_len, d_model]
        Hidden: [batch_size, seq_len, d_ff]
        Output: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (embedding size)
            d_ff: Feed-forward hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()

        # First linear projection: expand
        self.linear1 = nn.Linear(d_model, d_ff)

        # Non-linear activation
        # GELU (Gaussian Error Linear Unit) - smoother than ReLU
        # Used in GPT-2, BERT, and most modern transformers
        self.activation = nn.GELU()

        # Second linear projection: compress back
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]

        Note: Each position is processed identically and independently.
        No attention or interaction between different positions here.
        """
        # Expand: [batch_size, seq_len, d_model] → [batch_size, seq_len, d_ff]
        x = self.linear1(x)

        # Non-linearity
        x = self.activation(x)

        # Compress back: [batch_size, seq_len, d_ff] → [batch_size, seq_len, d_model]
        x = self.linear2(x)

        # Dropout
        x = self.dropout(x)

        return x
