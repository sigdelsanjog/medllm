"""
Full GPT-Style Transformer Model

PURPOSE:
Assembles all components (embeddings, decoder blocks, output projection)
into a complete causal language model for next-token prediction.

WHAT THIS STEP DOES:
1. Token + Positional Embeddings
   - Convert input IDs to vectors: [batch_size, seq_len, d_model]

2. Stack of N Decoder Blocks
   - Each block applies self-attention + FFN
   - Typically N = 4-6 layers for our hardware

3. Final Layer Normalization
   - Stabilize outputs before projection

4. Output Projection (Language Modeling Head)
   - Project to vocabulary: [batch_size, seq_len, vocab_size]
   - No activation (raw logits for CrossEntropyLoss)

5. Forward Pass
   - Input: token IDs [batch_size, seq_len]
   - Output: logits [batch_size, seq_len, vocab_size]

PACKAGES USED:
- torch: PyTorch tensors
- torch.nn: Module, Linear, LayerNorm, ModuleList

FILES FROM THIS PROJECT:
- architecture/embeddings.py: TokenPositionalEmbedding
- architecture/decoder_block.py: TransformerDecoderBlock
- configs/model_config.py: Hyperparameters (d_model, n_layers, etc.)

TENSOR SHAPES:
- Input IDs: [batch_size, seq_len] (integers)
- Embeddings: [batch_size, seq_len, d_model]
- After blocks: [batch_size, seq_len, d_model]
- Logits: [batch_size, seq_len, vocab_size]

HYPERPARAMETERS (for GTX 1080):
- vocab_size: 8000 (from tokenizer)
- d_model: 256-512 (embedding dimension)
- n_layers: 4-6 (number of transformer blocks)
- n_heads: 4-8 (attention heads)
- d_ff: 4 * d_model (FFN hidden size)
- dropout: 0.1-0.2
- max_seq_len: 512

COMMON FAILURE MODES TO AVOID:
- Not tying embeddings and output weights → slower convergence
- Too many layers → OOM or slow training
- d_model not divisible by n_heads → shape mismatch
- Missing final LayerNorm → unstable outputs
- Forgetting to handle padding mask → attending to padding
"""

import torch
import torch.nn as nn

from .embeddings import TokenPositionalEmbedding
from .decoder_block import TransformerDecoderBlock
from .attention import create_causal_mask


class GPTTransformer(nn.Module):
    """
    GPT-style Causal Language Model.

    This is the COMPLETE model. Everything comes together here.

    Architecture:
        1. Token + Positional Embeddings
        2. N x Decoder Blocks
        3. Final LayerNorm
        4. LM Head (projects to vocab)

    Training objective: Next-token prediction
        - Given tokens [t0, t1, t2, ..., tn]
        - Predict [t1, t2, t3, ..., tn+1]

    Tensor flow:
        Input:  [batch_size, seq_len] token IDs
        Embed:  [batch_size, seq_len, d_model]
        Blocks: [batch_size, seq_len, d_model]
        Logits: [batch_size, seq_len, vocab_size]
    """

    def __init__(self, config):
        """
        Args:
            config: ModelConfig object with hyperparameters
        """
        super().__init__()

        self.config = config

        # 1. Embeddings (token + positional)
        self.embeddings = TokenPositionalEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # 2. Stack of decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # 3. Final layer normalization
        self.final_norm = nn.LayerNorm(config.d_model)

        # 4. Language modeling head (output projection)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: Share weights between embeddings and output projection
        # This is standard practice in language models
        # Why? Reduces parameters and improves generalization
        self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights using Xavier/Glorot initialization.

        This is CRITICAL for training stability.
        Poor initialization → vanishing/exploding gradients
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            mask: Optional causal mask [seq_len, seq_len]

        Returns:
            Logits [batch_size, seq_len, vocab_size]

        Step-by-step:
        1. Convert token IDs to embeddings
        2. Pass through each decoder block
        3. Apply final normalization
        4. Project to vocabulary
        """
        batch_size, seq_len = input_ids.size()

        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(seq_len, device=input_ids.device)

        # 1. Embeddings: [batch_size, seq_len] → [batch_size, seq_len, d_model]
        x = self.embeddings(input_ids)

        # 2. Pass through decoder blocks
        for block in self.decoder_blocks:
            x = block(x, mask)

        # 3. Final layer norm
        x = self.final_norm(x)

        # 4. Project to vocabulary: [batch_size, seq_len, d_model] → [batch_size, seq_len, vocab_size]
        logits = self.lm_head(x)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get parameter count.

        Args:
            non_embedding: If True, exclude embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.embedding.weight.numel()
            n_params -= self.embeddings.positional_embedding.embedding.weight.numel()
        return n_params
