"""
Sampling Strategies for Text Generation

PURPOSE:
Different methods to select the next token during generation.
Each strategy has trade-offs between quality, diversity, and speed.

WHAT THIS FILE DOES:
1. Greedy sampling: Always pick highest probability (deterministic)
2. Temperature sampling: Control randomness
3. Top-k sampling: Sample from top k tokens only
4. Top-p (nucleus) sampling: Sample from cumulative probability p

WHY DIFFERENT STRATEGIES:
- Greedy: Fast, deterministic, but boring and repetitive
- Temperature: Simple randomness control
- Top-k: Prevents sampling very unlikely tokens
- Top-p: More adaptive than top-k (adjusts to probability distribution)

PACKAGES USED:
- torch: PyTorch tensors and operations

FILES FROM THIS PROJECT:
- None (utility functions)

COMMON ISSUES:
- Temperature too low → boring, repetitive
- Temperature too high → incoherent
- Top-k too small → limited diversity
- Top-p too low → truncates good options
"""

import torch
import torch.nn.functional as F


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling: Always pick the highest probability token.

    Args:
        logits: Logits from model [batch_size, vocab_size]

    Returns:
        Next token IDs [batch_size]

    Pros: Fast, deterministic, reproducible
    Cons: Boring, repetitive, gets stuck in loops

    Use when: You want deterministic outputs or testing
    """
    return torch.argmax(logits, dim=-1)


def temperature_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Temperature sampling: Scale logits before softmax.

    Args:
        logits: Logits from model [batch_size, vocab_size]
        temperature: Temperature parameter (>0)

    Returns:
        Sampled token IDs [batch_size]

    How it works:
    - temperature = 1.0: No change (normal sampling)
    - temperature < 1.0: More conservative (peaks sharper)
    - temperature > 1.0: More random (distribution flatter)

    Example:
    - temp=0.1: Almost greedy
    - temp=0.7: Balanced (recommended)
    - temp=1.5: Very creative

    Pros: Simple, interpretable
    Cons: No control over tail probabilities
    """
    if temperature == 0.0:
        return greedy_sample(logits)

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sample from distribution
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token


def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Top-k sampling: Only sample from top k tokens.

    Args:
        logits: Logits from model [batch_size, vocab_size]
        k: Number of top tokens to keep
        temperature: Temperature scaling

    Returns:
        Sampled token IDs [batch_size]

    How it works:
    - Keep only top k highest probability tokens
    - Set all other tokens to -inf (zero probability)
    - Sample from remaining tokens

    Why it helps:
    - Prevents sampling very unlikely tokens (noise)
    - Reduces incoherent outputs

    Typical values:
    - k=1: Greedy
    - k=10: Very conservative
    - k=50: Balanced
    - k=100: More diverse

    Limitation:
    - Fixed k doesn't adapt to probability distribution
    - If top-1 has 99% probability, k=50 wastes options
    """
    if k == 0 or k >= logits.size(-1):
        # No filtering
        return temperature_sample(logits, temperature)

    # Get top k values and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Create filtered logits (set non-top-k to -inf)
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(-1, top_k_indices, top_k_logits)

    # Sample with temperature
    return temperature_sample(filtered_logits, temperature)


def top_p_sample(logits: torch.Tensor, p: float, temperature: float = 1.0) -> torch.Tensor:
    """
    Top-p (nucleus) sampling: Sample from smallest set with cumulative prob >= p.

    Args:
        logits: Logits from model [batch_size, vocab_size]
        p: Cumulative probability threshold (0 < p <= 1)
        temperature: Temperature scaling

    Returns:
        Sampled token IDs [batch_size]

    How it works:
    1. Sort tokens by probability (descending)
    2. Find smallest set where cumulative probability >= p
    3. Sample only from this set

    Why better than top-k:
    - Adapts to probability distribution
    - When model is confident (one token has 90%), nucleus is small
    - When uncertain, nucleus is larger (more options)

    Typical values:
    - p=0.9: Conservative (90% probability mass)
    - p=0.95: Balanced (recommended)
    - p=0.99: More diverse

    Used in: GPT-3, ChatGPT, most modern LLMs
    """
    if p >= 1.0:
        # No filtering
        return temperature_sample(logits, temperature)

    # Scale by temperature first
    scaled_logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: first position where cumsum > p
    # Shift right by 1 to keep at least one token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create mask in original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )

    # Set removed indices to -inf
    filtered_logits = scaled_logits.clone()
    filtered_logits[indices_to_remove] = float("-inf")

    # Sample from filtered distribution
    probs = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0
) -> torch.Tensor:
    """
    Unified sampling function combining temperature, top-k, and top-p.

    Args:
        logits: Logits from model [batch_size, vocab_size]
        temperature: Temperature parameter
        top_k: Top-k filtering (0 = disabled)
        top_p: Top-p filtering (1.0 = disabled)

    Returns:
        Sampled token IDs [batch_size]

    Order of operations:
    1. Temperature scaling
    2. Top-k filtering (if enabled)
    3. Top-p filtering (if enabled)
    4. Sample from remaining distribution
    """
    # Greedy if temperature is 0
    if temperature == 0.0:
        return greedy_sample(logits)

    # Apply temperature
    scaled_logits = logits / temperature

    # Apply top-k if enabled
    if top_k > 0 and top_k < logits.size(-1):
        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
        filtered_logits = torch.full_like(scaled_logits, float("-inf"))
        filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
        scaled_logits = filtered_logits

    # Apply top-p if enabled
    if top_p < 1.0:
        probs = F.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        scaled_logits[indices_to_remove] = float("-inf")

    # Sample
    probs = F.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token
