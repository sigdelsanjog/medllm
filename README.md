# llm-med 🏥

A lightweight medical question-answering language model trained on the MedQuAD dataset. This package provides a transformer-based GPT architecture optimized for medical domain question answering.

[![PyPI version](https://badge.fury.io/py/llm-med.svg)](https://badge.fury.io/py/llm-med)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🧠 **Custom GPT Architecture**: Lightweight transformer model designed for medical QA
- 💊 **Medical Domain**: Trained on MedQuAD dataset with medical terminology
- ⚡ **Fast Inference**: Optimized for quick medical question answering
- 🔧 **Flexible**: Easy to fine-tune on your own medical datasets
- 📦 **Lightweight**: Small model size suitable for edge deployment

## Installation

### From PyPI (Recommended)

```bash
pip install llm-med
```

### From Source

```bash
git clone https://github.com/yourusername/medllm.git
cd medllm
pip install -e .
```

### With Optional Dependencies

```bash
# For development
pip install llm-med[dev]

# For training
pip install llm-med[training]

# All dependencies
pip install llm-med[dev,training]
```

## Quick Start

### Inference (Generate Medical Answers)

```python
from inference.generator import MedicalQAGenerator
from model.architecture import GPTTransformer
from model.configs.model_config import get_small_config

# Load model
config = get_small_config()
model = GPTTransformer(config)

# Load your trained checkpoint
# model.load_state_dict(torch.load('path/to/checkpoint.pt'))

# Create generator
generator = MedicalQAGenerator(
    model=model,
    tokenizer_path='path/to/tokenizer.model'
)

# Generate answer
question = "What are the symptoms of diabetes?"
answer = generator.generate(
    prompt=question,
    max_length=100,
    temperature=0.7
)

print(f"Q: {question}")
print(f"A: {answer}")
```

### Using Command Line

```bash
# Generate answers
medllm-generate --prompt "What causes hypertension?" --max-length 100

# Train model
medllm-train --model-size small --num-epochs 10 --batch-size 16
```

### Training Your Own Model

```python
from training.train import main
from configs.train_config import get_default_config
from model.configs.model_config import get_small_config

# Configure training
train_config = get_default_config()
train_config.batch_size = 16
train_config.num_epochs = 10
train_config.learning_rate = 3e-4

# Start training
main()
```

## Model Architecture

The model uses a custom GPT-based transformer architecture:

- **Embedding**: Token + positional embeddings
- **Transformer Blocks**: Multi-head self-attention + feed-forward networks
- **Parameters**: ~10M (small), ~50M (medium)
- **Context Length**: 512 tokens
- **Vocabulary**: Custom SentencePiece tokenizer trained on medical text

## Configuration

### Model Sizes

```python
from model.configs.model_config import (
    get_tiny_config,   # ~2M parameters - for testing
    get_small_config,  # ~10M parameters - recommended
    get_medium_config  # ~50M parameters - higher quality
)
```

### Training Configuration

```python
from configs.train_config import TrainingConfig

config = TrainingConfig(
    batch_size=16,
    learning_rate=3e-4,
    num_epochs=10,
    warmup_steps=100,
    grad_clip=1.0
)
```

## Project Structure

```
llm-med/
├── model/
│   ├── architecture/      # GPT transformer implementation
│   └── configs/           # Model configurations
├── inference/
│   ├── generator.py       # Text generation
│   └── sampling.py        # Sampling strategies
├── training/
│   ├── train.py          # Training script
│   ├── trainer.py        # Training loop
│   └── dataset.py        # Data loading
├── tokenizer/
│   └── train_tokenizer.py # SentencePiece tokenizer
├── configs/
│   └── train_config.py   # Training configurations
└── utils/
    ├── checkpoints.py    # Model checkpointing
    └── logging.py        # Training logging
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- sentencepiece >= 0.1.99
- numpy >= 1.24.0
- tqdm >= 4.65.0

## Documentation

For detailed documentation, visit [GitHub Repository](https://github.com/yourusername/medllm).

### Key Guides

- [Training Guide](docs/training.md)
- [Inference Guide](docs/inference.md)
- [Model Architecture](docs/architecture.md)
- [API Reference](docs/api.md)

## Performance

| Model Size | Parameters | Training Time | Inference Speed |
| ---------- | ---------- | ------------- | --------------- |
| Tiny       | ~2M        | 2 hours       | ~100 tokens/sec |
| Small      | ~10M       | 8 hours       | ~80 tokens/sec  |
| Medium     | ~50M       | 24 hours      | ~50 tokens/sec  |

_Tested on GTX 1080 8GB_

## Examples

### Medical Question Answering

```python
# Example 1: Symptoms inquiry
question = "What are the early signs of Alzheimer's disease?"
answer = generator.generate(question, temperature=0.7)

# Example 2: Treatment information
question = "How is Type 2 diabetes treated?"
answer = generator.generate(question, temperature=0.6)

# Example 3: Medical definitions
question = "What is hypertension?"
answer = generator.generate(question, temperature=0.5)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this model in your research, please cite:

```bibtex
@software{llm_med_2026,
  author = {Your Name},
  title = {llm-med: Medical Question-Answering Language Model},
  year = {2026},
  url = {https://github.com/yourusername/medllm}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MedQuAD dataset creators
- PyTorch team
- Hugging Face for inspiration

## Disclaimer

⚠️ **Medical Disclaimer**: This model is for research and educational purposes only. It should NOT be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## Support

- 📫 Issues: [GitHub Issues](https://github.com/yourusername/medllm/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/medllm/discussions)
- 📧 Email: your.email@example.com

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

Made with ❤️ for the medical AI community
