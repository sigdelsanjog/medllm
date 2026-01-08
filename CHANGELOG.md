# Changelog

All notable changes to llm-med will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-08

### Added

- Initial release of llm-med package
- GPT-based transformer architecture for medical QA
- SentencePiece tokenizer trained on medical text
- Training pipeline with configurable hyperparameters
- Inference engine with multiple sampling strategies
- Command-line tools for training and generation
- Support for model checkpointing and resumption
- Comprehensive documentation and examples

### Features

- Small (~10M), Medium (~50M) model configurations
- Cosine learning rate scheduling with warmup
- Gradient clipping for training stability
- Mixed precision training support
- Validation monitoring and early stopping
- Perplexity metrics for evaluation

### Models

- Pre-trained on MedQuAD dataset
- Supports medical domain question answering
- Context length: 512 tokens
- Custom vocabulary: ~8000 tokens

## [0.0.1] - 2026-01-01

### Added

- Project initialization
- Basic model architecture
- Training scripts
- Data preprocessing pipeline
