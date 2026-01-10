# Changelog

All notable changes to gptmed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-01-10

### Changed

- **BREAKING CHANGE**: Complete package rename from gptgpt to gptmed
  - Package renamed to reflect medical AI vision and purpose
  - All imports changed from `gptgpt` to `gptmed`
  - CLI commands renamed: `gptmed-generate`, `gptmed-train`
  - Model files renamed: `gptmed_model.pt`, `gptmed_tokenizer.model`
  - Repository and documentation updated
  - Version reset to 0.0.1 to mark new beginning

### Vision

- Establishing foundation for end-to-end medical assistance platform
- Future roadmap includes:
  - Casual question-answering (current capability)
  - Conversational models
  - Logic and reasoning capabilities
  - Medical prescription analysis
  - Comprehensive healthcare domain support for hospitals, clinics, pharmacies, doctors, and patients

## [0.2.0] - 2026-01-09

### Fixed

- **BREAKING CHANGE**: Proper package namespace structure
  - All modules now under `llm_med` namespace
  - Imports changed from `from model.architecture` to `from llm_med.model.architecture`
  - This fixes the ModuleNotFoundError with v0.1.0

### Changed

- Restructured package with proper namespace: `llm_med/`
- Updated all internal imports to use `llm_med.` prefix
- Updated entry points to use proper namespace paths

### Added

- Package-level `__init__.py` with convenience imports
- `setup.py` for editable install compatibility

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
