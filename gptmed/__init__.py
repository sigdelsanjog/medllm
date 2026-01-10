"""
llm-med: A lightweight medical question-answering language model

This package provides a GPT-based transformer architecture trained on the MedQuAD dataset
for medical domain question answering.

Main Components:
- model: GPT transformer architecture
- inference: Text generation and sampling
- training: Training loop and utilities
- tokenizer: SentencePiece tokenizer
- configs: Configuration management
- utils: Utility functions

Example:
    >>> from llm_med.model.architecture import GPTTransformer
    >>> from llm_med.model.configs.model_config import get_small_config
    >>> from llm_med.inference.generator import TextGenerator
    >>> 
    >>> config = get_small_config()
    >>> model = GPTTransformer(config)
"""

__version__ = "0.2.0"
__author__ = "Sanjog Sigdel"
__email__ = "sigdelsanjog@gmail.com"

# Expose main components at package level for convenience
from llm_med.model.architecture import GPTTransformer
from llm_med.model.configs.model_config import ModelConfig, get_small_config, get_tiny_config

__all__ = [
    "GPTTransformer",
    "ModelConfig",
    "get_small_config",
    "get_tiny_config",
]
