# Architecture Extension Guide

## Adding New Model Architectures to GptMed

This guide shows how to extend the gptmed package to support multiple model types (Q&A, Conversational, Summarization, etc.)

## Current Structure (Single Architecture)

```
gptmed/model/architecture/
├── transformer.py        # GPTTransformer (Q&A focused)
├── attention.py
├── embeddings.py
├── feedforward.py
└── decoder_block.py
```

## Proposed Structure (Multi-Architecture)

```
gptmed/model/
├── architecture/
│   ├── __init__.py
│   ├── base_transformer.py           # NEW: Base class
│   ├── qa_transformer.py             # Current GPTTransformer renamed
│   ├── conversational_transformer.py # NEW: Chat model
│   ├── summarization_transformer.py  # NEW: Summarization (optional)
│   │
│   └── shared/                       # Reusable components
│       ├── __init__.py
│       ├── attention.py              # Multi-head attention
│       ├── embeddings.py             # Token + positional
│       ├── feedforward.py            # FFN layer
│       └── decoder_block.py          # Transformer block
│
└── configs/
    ├── base_config.py                # Common config
    ├── qa_config.py                  # Q&A specific
    └── conversational_config.py      # Chat specific
```

---

## Step 1: Create Base Transformer Class

**File: `gptmed/model/architecture/base_transformer.py`**

```python
"""
Base Transformer class that all architectures inherit from.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseTransformer(nn.Module, ABC):
    """
    Abstract base class for all transformer architectures.

    All model types (Q&A, Conversational, etc.) inherit from this.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

    @abstractmethod
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass - must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate(self, prompt_ids, max_length):
        """
        Text generation - must be implemented by subclasses.
        """
        pass

    def save_pretrained(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__
        }, path)

    @classmethod
    def from_pretrained(cls, path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        # Reconstruct config and model
        # Implementation here...
```

---

## Step 2: Refactor Current Q&A Model

**File: `gptmed/model/architecture/qa_transformer.py`**

```python
"""
Q&A Transformer - optimized for question answering.

Focused on:
- Single-turn Q&A
- Concise answers
- Domain-specific (medical, tech, etc.)
"""

from .base_transformer import BaseTransformer
from .shared.embeddings import TokenPositionalEmbedding
from .shared.decoder_block import TransformerDecoderBlock

class QATransformer(BaseTransformer):
    """
    GPT-style transformer optimized for Q&A.

    Differences from conversational:
    - Shorter context window (512 tokens)
    - Higher temperature default (more precise)
    - No special conversation tokens
    """

    def __init__(self, config):
        super().__init__(config)

        # Components
        self.embeddings = TokenPositionalEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len
        )

        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(config)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights (standard practice)
        self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(self, input_ids, attention_mask=None):
        """Standard GPT forward pass."""
        x = self.embeddings(input_ids)

        for block in self.decoder_blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
```

---

## Step 3: Add Conversational Transformer

**File: `gptmed/model/architecture/conversational_transformer.py`**

```python
"""
Conversational Transformer - optimized for multi-turn dialogue.

Key features:
- Longer context window (1024-2048 tokens)
- Special tokens for user/assistant
- Turn tracking
- Context management
"""

import torch
import torch.nn as nn
from .base_transformer import BaseTransformer
from .shared.embeddings import TokenPositionalEmbedding
from .shared.decoder_block import TransformerDecoderBlock

class ConversationalTransformer(BaseTransformer):
    """
    GPT-style transformer optimized for conversations.

    Differences from Q&A:
    - Longer context (2048 tokens vs 512)
    - Special conversation tokens: <user>, <assistant>, <turn>
    - Role embeddings (user vs assistant)
    - Better handling of multi-turn context
    """

    def __init__(self, config):
        super().__init__(config)

        # Token embeddings
        self.embeddings = TokenPositionalEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len  # 2048 for conversations
        )

        # NEW: Role embeddings (user vs assistant)
        self.role_embeddings = nn.Embedding(
            num_embeddings=2,  # 0=user, 1=assistant
            embedding_dim=config.d_model
        )

        # Transformer blocks (same as Q&A)
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(config)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, role_ids=None, attention_mask=None):
        """
        Forward pass with role awareness.

        Args:
            input_ids: [batch, seq_len] - token IDs
            role_ids: [batch, seq_len] - 0=user, 1=assistant
            attention_mask: [batch, seq_len] - attention mask
        """
        # Token + positional embeddings
        x = self.embeddings(input_ids)

        # Add role embeddings (NEW for conversations)
        if role_ids is not None:
            role_emb = self.role_embeddings(role_ids)
            x = x + role_emb

        # Apply transformer blocks
        for block in self.decoder_blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate_response(self, conversation_history, max_length=150):
        """
        Generate assistant response given conversation history.

        conversation_history format:
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        """
        # Format conversation into tokens
        # Add special tokens for turns
        # Generate response
        pass
```

---

## Step 4: Configuration for Each Architecture

**File: `gptmed/model/configs/conversational_config.py`**

```python
"""
Configuration for conversational models.
"""

from dataclasses import dataclass
from .base_config import BaseModelConfig

@dataclass
class ConversationalConfig(BaseModelConfig):
    """
    Config for conversational transformer.

    Differences from Q&A:
    - Longer context (2048 vs 512)
    - Different generation defaults
    - Special tokens for conversation
    """

    # Model architecture (same as Q&A)
    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # Context (LONGER for conversations)
    max_seq_len: int = 2048  # vs 512 for Q&A

    # Special tokens
    user_token_id: int = 8001
    assistant_token_id: int = 8002
    turn_separator_id: int = 8003

    # Generation defaults (more creative for chat)
    default_temperature: float = 0.8  # vs 0.6 for Q&A
    default_top_p: float = 0.95
    default_max_length: int = 200

    # Conversation-specific
    max_turns: int = 10
    role_embedding: bool = True


def get_conversational_small():
    """Get small conversational model config."""
    return ConversationalConfig(
        vocab_size=8000,
        d_model=512,
        n_layers=6,
        max_seq_len=2048
    )


def get_conversational_large():
    """Get large conversational model config."""
    return ConversationalConfig(
        vocab_size=8000,
        d_model=768,
        n_layers=12,
        max_seq_len=4096  # Even longer for complex conversations
    )
```

---

## Step 5: Update Package Exports

**File: `gptmed/model/__init__.py`**

```python
"""
Model architectures for different tasks.
"""

# Import all architectures
from .architecture.qa_transformer import QATransformer
from .architecture.conversational_transformer import ConversationalTransformer

# Import configs
from .configs.qa_config import get_qa_small, get_qa_medium
from .configs.conversational_config import get_conversational_small, get_conversational_large

__all__ = [
    # Q&A models
    "QATransformer",
    "get_qa_small",
    "get_qa_medium",

    # Conversational models
    "ConversationalTransformer",
    "get_conversational_small",
    "get_conversational_large",
]
```

---

## Usage Examples

### **Q&A Model (Current)**

```python
from gptmed.model import QATransformer, get_qa_small
from gptmed.inference.generator import TextGenerator

# Create Q&A model
config = get_qa_small()
model = QATransformer(config)

# Generate answers
generator = TextGenerator(model, tokenizer)
answer = generator.generate("What causes diabetes?")
```

### **Conversational Model (New)**

```python
from gptmed.model import ConversationalTransformer, get_conversational_small
from gptmed.inference.conversation_generator import ConversationGenerator

# Create chat model
config = get_conversational_small()
model = ConversationalTransformer(config)

# Multi-turn conversation
chat = ConversationGenerator(model, tokenizer)

# Turn 1
response1 = chat.add_user_message("Hello!")
print(response1)  # "Hi! How can I help you today?"

# Turn 2
response2 = chat.add_user_message("Tell me about yourself")
print(response2)  # Uses full conversation history
```

---

## Training Different Architectures

**Same training code, different models:**

```python
from gptmed.training.train import train_model
from gptmed.model import QATransformer, ConversationalTransformer

# Train Q&A model
qa_model = QATransformer(qa_config)
train_model(qa_model, qa_dataset, output_dir="models/qa")

# Train conversational model
chat_model = ConversationalTransformer(chat_config)
train_model(chat_model, conversation_dataset, output_dir="models/chat")
```

---

## Backend Integration

**Update your backend to support multiple model types:**

```python
# backend/services/model_factory.py

from gptmed.model import QATransformer, ConversationalTransformer

class ModelFactory:

    @staticmethod
    def load_model(model_type: str, checkpoint_path: str):
        if model_type == "qa":
            return QATransformer.from_pretrained(checkpoint_path)
        elif model_type == "conversational":
            return ConversationalTransformer.from_pretrained(checkpoint_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# backend/config.py
MODEL_CONFIGS = {
    "GptMed-QA": {
        "type": "qa",
        "model_path": "models/gptmed-qa",
        "display_name": "GptMed Q&A"
    },
    "GptMed-Chat": {
        "type": "conversational",
        "model_path": "models/gptmed-chat",
        "display_name": "GptMed Conversational"
    }
}
```

---

## Summary

### **Key Design Principles:**

1. **✅ Shared Components** - Reuse attention, embeddings, FFN
2. **✅ Base Class** - Common interface for all architectures
3. **✅ Specific Configs** - Each architecture has optimized settings
4. **✅ Same Training** - Use same training pipeline
5. **✅ Backend Flexibility** - Load different models dynamically

### **Architecture Comparison:**

| Feature         | Q&A Transformer | Conversational Transformer  |
| --------------- | --------------- | --------------------------- |
| Context Length  | 512 tokens      | 2048 tokens                 |
| Special Tokens  | None            | <user>, <assistant>, <turn> |
| Role Embeddings | No              | Yes                         |
| Temperature     | 0.6 (precise)   | 0.8 (creative)              |
| Use Case        | Single Q&A      | Multi-turn chat             |

This design keeps the package **modular and extensible** while sharing most code!
