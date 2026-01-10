"""
Generation Configuration

PURPOSE:
Hyperparameters for text generation (inference).
Controls how creative vs conservative the model's outputs are.

WHAT THIS FILE CONTAINS:
- Temperature: Randomness control
- Top-k, top-p: Sampling constraints
- Repetition penalty: Prevent repetitive text
- Max length: Stop generation

PACKAGES USED:
- dataclasses: Clean config structure

FILES FROM THIS PROJECT:
- None (base config)

KEY PARAMETERS EXPLAINED:
- temperature: 0.0 = greedy, 0.7 = balanced, 1.5 = very creative
- top_k: Only sample from top k tokens (50-100 typical)
- top_p: Sample from tokens with cumulative prob p (0.9-0.95 typical)
- repetition_penalty: >1.0 discourages repetition (1.2 is good)
"""

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    # Sampling strategy
    temperature: float = 0.8  # Higher = more random (0.0 = greedy)
    top_k: int = 50  # Only sample from top k tokens (0 = disabled)
    top_p: float = 0.95  # Nucleus sampling threshold (1.0 = disabled)

    # Repetition control
    repetition_penalty: float = 1.2  # >1.0 penalizes repetition
    no_repeat_ngram_size: int = 3  # Block repeating n-grams (0 = disabled)

    # Length control
    max_length: int = 200  # Maximum tokens to generate
    min_length: int = 10  # Minimum tokens to generate

    # Stopping criteria
    stop_tokens: list = None  # Token IDs that stop generation

    # Special tokens
    bos_token_id: int = 2  # Beginning of sequence
    eos_token_id: int = 3  # End of sequence
    pad_token_id: int = 0  # Padding

    def __post_init__(self):
        """Validate config."""
        if self.stop_tokens is None:
            self.stop_tokens = [self.eos_token_id]

        assert self.temperature >= 0.0, "temperature must be >= 0"
        assert self.top_k >= 0, "top_k must be >= 0"
        assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0, 1]"
        assert self.repetition_penalty >= 1.0, "repetition_penalty must be >= 1.0"


def get_greedy_config() -> GenerationConfig:
    """Greedy decoding (deterministic, picks highest prob)."""
    return GenerationConfig(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)


def get_balanced_config() -> GenerationConfig:
    """Balanced sampling (good default)."""
    return GenerationConfig(temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.2)


def get_creative_config() -> GenerationConfig:
    """Creative sampling (more diverse, less coherent)."""
    return GenerationConfig(temperature=1.2, top_k=100, top_p=0.95, repetition_penalty=1.3)


def get_conservative_config() -> GenerationConfig:
    """Conservative sampling (safe, coherent, less diverse)."""
    return GenerationConfig(temperature=0.5, top_k=30, top_p=0.9, repetition_penalty=1.1)
