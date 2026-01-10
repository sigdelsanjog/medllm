"""
Model Configuration

PURPOSE:
Central place to store all model hyperparameters. Makes it easy to experiment
with different model sizes without changing code.

WHAT THIS FILE CONTAINS:
1. ModelConfig dataclass with all hyperparameters:
   - vocab_size: From tokenizer (8000)
   - d_model: Embedding/hidden dimension
   - n_layers: Number of transformer blocks
   - n_heads: Number of attention heads
   - d_ff: Feed-forward hidden dimension
   - dropout: Dropout probability
   - max_seq_len: Maximum sequence length

2. Predefined configurations:
   - Tiny: For quick testing (d_model=128, n_layers=2)
   - Small: For GTX 1080 training (d_model=256, n_layers=4)
   - Medium: Larger if memory allows (d_model=512, n_layers=6)

PACKAGES USED:
- dataclasses: For clean config structure
- json: For saving/loading configs

FILES FROM THIS PROJECT:
- None (this defines configs for other files to use)

DESIGN DECISIONS:
- d_model must be divisible by n_heads
- d_ff typically 4 * d_model (expansion ratio)
- dropout 0.1-0.2 (too high → underfitting, too low → overfitting)
- max_seq_len matches tokenization (512)

MEMORY ESTIMATION (approximate):
- Model parameters ≈ 12 * n_layers * d_model^2
- Small config: ~10M parameters (~40MB)
- Medium config: ~40M parameters (~160MB)
- Fits comfortably in 8GB VRAM with batch_size=16-32
"""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Transformer model configuration.

    All hyperparameters in one place for easy experimentation.
    """

    # Vocabulary
    vocab_size: int = 8000  # From SentencePiece tokenizer

    # Architecture
    d_model: int = 256  # Embedding/hidden dimension
    n_layers: int = 4  # Number of transformer blocks
    n_heads: int = 4  # Number of attention heads
    d_ff: int = 1024  # FFN hidden dimension (4 * d_model)

    # Regularization
    dropout: float = 0.1  # Dropout probability

    # Sequence
    max_seq_len: int = 512  # Maximum sequence length

    # Special tokens (from tokenizer)
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    def __post_init__(self):
        """Validate configuration."""
        assert (
            self.d_model % self.n_heads == 0
        ), f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Load from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, path: Path) -> "ModelConfig":
        """Load from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different use cases


def get_tiny_config() -> ModelConfig:
    """
    Tiny model for quick testing and debugging.
    ~2M parameters, very fast training.
    """
    return ModelConfig(
        vocab_size=8000, d_model=128, n_layers=2, n_heads=4, d_ff=512, dropout=0.1, max_seq_len=512
    )


def get_small_config() -> ModelConfig:
    """
    Small model for GTX 1080 (8GB VRAM).
    ~10M parameters, good balance of speed and capacity.
    """
    return ModelConfig(
        vocab_size=8000, d_model=256, n_layers=4, n_heads=4, d_ff=1024, dropout=0.1, max_seq_len=512
    )


def get_medium_config() -> ModelConfig:
    """
    Medium model for GPUs with more memory.
    ~40M parameters, better quality but slower.
    """
    return ModelConfig(
        vocab_size=8000, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1, max_seq_len=512
    )
