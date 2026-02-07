"""
Transformer Decoder Block (From Scratch)

Combines multi-head attention and feed-forward network with residual connections
and layer normalization.
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForwardNetwork


class TransformerDecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block
    
    Architecture:
    1. Multi-Head Self-Attention (causal)
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network
    4. Add & Norm (residual connection + layer normalization)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension (default: 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Layer normalization for attention
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization for FFN
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional mask for padding tokens
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Multi-head attention with residual connection and normalization
        attention_output = self.attention(x, attention_mask)
        attention_output = self.dropout(attention_output)
        x = self.norm1(x + attention_output)  # Add & Norm
        
        # Feed-forward network with residual connection and normalization
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)  # Add & Norm
        
        return x
