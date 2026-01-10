"""
Training Utilities

PURPOSE:
Helper functions for training loop:
- Gradient clipping
- Learning rate scheduling
- Training state management

WHAT THIS FILE DOES:
1. Clip gradients to prevent explosion
2. Calculate learning rate with warmup + decay
3. Compute gradient norms for monitoring

WHY THESE ARE CRITICAL:
- Gradient clipping: Prevents training collapse from exploding gradients
- LR warmup: Stabilizes early training (large steps can destabilize)
- LR decay: Helps model converge to better minima

PACKAGES USED:
- torch: Gradient operations
- math: Cosine calculations

FILES FROM THIS PROJECT:
- None (utility functions)

COMMON FAILURE MODES:
- No gradient clipping → NaN loss
- No warmup → unstable first epochs
- Constant LR → suboptimal convergence
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def clip_grad_norm(model: nn.Module, max_norm: float) -> float:
    """
    Clip gradient norms to prevent explosion.

    Args:
        model: Model with gradients
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm (before clipping)

    How it works:
    - Compute total gradient norm across all parameters
    - If norm > max_norm, scale all gradients down
    - This prevents single large gradients from destroying training

    Typical values:
    - max_norm=1.0: Standard for transformers
    - max_norm=5.0: More lenient
    - max_norm=0.5: Very conservative
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def get_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
    max_steps: int,
    decay_type: str = "cosine",
) -> float:
    """
    Calculate learning rate with warmup and decay.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_lr: Peak learning rate
        min_lr: Minimum learning rate
        max_steps: Total training steps
        decay_type: 'cosine', 'linear', or 'constant'

    Returns:
        Learning rate for this step

    Schedule:
    1. Warmup (0 to warmup_steps): Linear increase from 0 to max_lr
    2. Decay (warmup_steps to max_steps): Cosine/linear decay to min_lr

    Why warmup?
    - Random initialization → large gradients early on
    - Large LR + large gradients = explosion
    - Warmup gives model time to stabilize
    """
    # Warmup phase
    if step < warmup_steps:
        # Linear warmup from 0 to max_lr
        return max_lr * (step / warmup_steps)

    # After warmup
    if decay_type == "constant":
        return max_lr

    # Progress through decay phase
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(progress, 1.0)  # Cap at 1.0

    if decay_type == "cosine":
        # Cosine decay: smooth curve to min_lr
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    elif decay_type == "linear":
        # Linear decay: straight line to min_lr
        lr = max_lr - (max_lr - min_lr) * progress
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    return lr


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """
    Set learning rate for optimizer.

    Args:
        optimizer: PyTorch optimizer
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_loss_single_batch(model: nn.Module, batch: tuple, device: str) -> float:
    """
    Compute loss on a single batch (for evaluation).

    Args:
        model: Model to evaluate
        batch: (input_ids, target_ids) batch
        device: Device to run on

    Returns:
        Loss value
    """
    input_ids, target_ids = batch
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    # Forward pass
    logits = model(input_ids)

    # Compute loss
    # Reshape for CrossEntropyLoss:
    # logits: [batch_size * seq_len, vocab_size]
    # targets: [batch_size * seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(batch_size * seq_len, vocab_size)
    targets = target_ids.view(batch_size * seq_len)

    loss = nn.functional.cross_entropy(logits, targets)

    return loss.item()


def estimate_loss_dataloader(
    model: nn.Module, dataloader, device: str, max_batches: Optional[int] = None
) -> float:
    """
    Estimate average loss over a dataloader.

    Args:
        model: Model to evaluate
        dataloader: DataLoader to evaluate on
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (None = all)

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            loss = estimate_loss_single_batch(model, batch, device)
            total_loss += loss
            num_batches += 1

    model.train()

    return total_loss / num_batches if num_batches > 0 else 0.0


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.

    Perplexity = exp(loss)

    Lower is better. Perplexity measures how "surprised" the model is.
    - Perplexity of 10 = model is choosing between ~10 likely tokens
    - Perplexity of 100 = model is very uncertain
    """
    return math.exp(loss)
