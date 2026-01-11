"""
GptMed: A lightweight GPT-based language model framework

A domain-agnostic framework for training custom question-answering models.
Train your own GPT model on any Q&A dataset - medical, technical support,
education, or any other domain.

Quick Start:
    >>> import gptmed
    >>> 
    >>> # 1. Create a config file
    >>> gptmed.create_config('my_config.yaml')
    >>> 
    >>> # 2. Edit my_config.yaml with your settings
    >>> 
    >>> # 3. Train your model
    >>> results = gptmed.train_from_config('my_config.yaml')
    >>> 
    >>> # 4. Generate answers
    >>> answer = gptmed.generate(
    ...     checkpoint=results['best_checkpoint'],
    ...     tokenizer='tokenizer/my_tokenizer.model',
    ...     prompt='Your question here?'
    ... )

Advanced Usage:
    >>> from gptmed.model.architecture import GPTTransformer
    >>> from gptmed.model.configs.model_config import get_small_config
    >>> from gptmed.inference.generator import TextGenerator
    >>> 
    >>> config = get_small_config()
    >>> model = GPTTransformer(config)
"""

__version__ = "0.3.3"
__author__ = "Sanjog Sigdel"
__email__ = "sigdelsanjog@gmail.com"

# High-level API - Main user interface
from gptmed.api import (
    create_config,
    train_from_config,
    generate,
)

# Expose main components at package level for convenience
from gptmed.model.architecture import GPTTransformer
from gptmed.model.configs.model_config import ModelConfig, get_small_config, get_tiny_config

__all__ = [
    # Simple API
    "create_config",
    "train_from_config", 
    "generate",
    # Advanced API
    "GPTTransformer",
    "ModelConfig",
    "get_small_config",
    "get_tiny_config",
]
