"""
Training Configuration

PURPOSE:
Central place for all training hyperparameters. Separating training config
from model config follows separation of concerns - you can train the same
model architecture with different training strategies.

WHAT THIS FILE CONTAINS:
- Batch size, learning rate, epochs
- Optimizer settings (weight decay, betas)
- Learning rate schedule parameters
- Gradient clipping threshold
- Checkpoint and logging intervals

PACKAGES USED:
- dataclasses: Clean config structure
- json: Save/load configs

FILES FROM THIS PROJECT:
- None (base config)

DESIGN DECISIONS:
- Small batch size (16-32) for 8GB VRAM
- Learning rate ~1e-4 to 3e-4 (typical for small transformers)
- Weight decay 0.01 (L2 regularization)
- Gradient clipping at 1.0 (prevents exploding gradients)
- Warmup steps to stabilize early training

COMMON FAILURE MODES:
- LR too high → loss explodes, NaN
- LR too low → very slow convergence
- No warmup → unstable early training
- Batch size too large → OOM
- No gradient clipping → gradient explosion
"""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Data
    train_data_path: str = "./data/tokenized/train.npy"
    val_data_path: str = "./data/tokenized/val.npy"

    # Batch size (adjust based on VRAM)
    batch_size: int = 16  # GTX 1080: 16-32 works well

    # Training duration
    num_epochs: int = 10  # Total training epochs
    max_steps: int = -1  # -1 means train for num_epochs

    # Optimization
    learning_rate: float = 3e-4  # Peak learning rate
    weight_decay: float = 0.01  # L2 regularization
    betas: tuple = (0.9, 0.999)  # Adam beta1, beta2
    eps: float = 1e-8  # Adam epsilon

    # Learning rate schedule
    warmup_steps: int = 100  # Linear warmup steps
    lr_decay: str = "cosine"  # 'cosine' or 'linear' or 'constant'
    min_lr: float = 1e-5  # Minimum LR for decay

    # Gradient clipping (CRITICAL for stability)
    grad_clip: float = 1.0  # Clip gradient norm to this value

    # Evaluation
    eval_interval: int = 500  # Evaluate every N steps
    eval_iters: int = 100  # Number of eval batches

    # Checkpointing
    checkpoint_dir: str = "./model/checkpoints"
    save_interval: int = 1000  # Save checkpoint every N steps
    keep_last_n: int = 3  # Keep only last N checkpoints

    # Logging
    log_interval: int = 10  # Log every N steps
    log_dir: str = "./logs"

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'

    # Mixed precision (optional, for faster training)
    use_amp: bool = False  # Automatic Mixed Precision

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "warmup_steps": self.warmup_steps,
            "lr_decay": self.lr_decay,
            "min_lr": self.min_lr,
            "grad_clip": self.grad_clip,
            "eval_interval": self.eval_interval,
            "eval_iters": self.eval_iters,
            "checkpoint_dir": self.checkpoint_dir,
            "save_interval": self.save_interval,
            "keep_last_n": self.keep_last_n,
            "log_interval": self.log_interval,
            "log_dir": self.log_dir,
            "device": self.device,
            "use_amp": self.use_amp,
            "seed": self.seed,
        }

    def save(self, path: Path):
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Load from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, path: Path):
        """Load from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config() -> TrainingConfig:
    """Default training config for GTX 1080."""
    return TrainingConfig()


def get_quick_test_config() -> TrainingConfig:
    """Quick config for testing (small batch, few steps)."""
    return TrainingConfig(
        batch_size=4,
        num_epochs=1,
        max_steps=100,
        eval_interval=50,
        save_interval=50,
        log_interval=5,
    )
