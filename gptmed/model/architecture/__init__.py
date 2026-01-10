"""
Model Architecture Components - Package Initializer

PURPOSE:
Makes it easy to import transformer components from a single location.

WHAT THIS FILE DOES:
Exposes main classes for import:
- from model.architecture import GPTTransformer
- from model.architecture import TransformerDecoderBlock
- from model.architecture import MultiHeadAttention

PACKAGES USED:
- None (just Python imports)

FILES FROM THIS PROJECT:
- All components in this architecture/ directory
"""

from .embeddings import TokenEmbedding, PositionalEmbedding, TokenPositionalEmbedding
from .attention import MultiHeadAttention, create_causal_mask
from .feedforward import FeedForward
from .decoder_block import TransformerDecoderBlock
from .transformer import GPTTransformer

__all__ = [
    "TokenEmbedding",
    "PositionalEmbedding",
    "TokenPositionalEmbedding",
    "MultiHeadAttention",
    "create_causal_mask",
    "FeedForward",
    "TransformerDecoderBlock",
    "GPTTransformer",
]
