"""
Decoding Utilities

PURPOSE:
Helper functions for text generation:
- Repetition penalty
- N-gram blocking
- Stopping criteria

WHAT THIS FILE DOES:
1. Apply repetition penalty to discourage repeated tokens
2. Block n-gram repetition (prevents "the the the")
3. Check stopping conditions

WHY THESE ARE NEEDED:
- Models often get stuck in repetition loops
- "The patient has has has has..."
- N-gram blocking prevents this
- Repetition penalty makes it less likely

PACKAGES USED:
- torch: PyTorch tensors

FILES FROM THIS PROJECT:
- None (utility functions)
"""

import torch
from typing import List, Set


def apply_repetition_penalty(
    logits: torch.Tensor, generated_tokens: torch.Tensor, penalty: float = 1.0
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repeated tokens.

    Args:
        logits: Current logits [batch_size, vocab_size]
        generated_tokens: Previously generated tokens [batch_size, seq_len]
        penalty: Penalty factor (>1.0 penalizes, <1.0 encourages)

    Returns:
        Modified logits

    How it works:
    - For each token that appeared before:
      - If its logit is positive: divide by penalty
      - If its logit is negative: multiply by penalty
    - This makes repeated tokens less likely

    Typical values:
    - penalty=1.0: No penalty
    - penalty=1.1: Mild penalty
    - penalty=1.2: Moderate (recommended)
    - penalty=1.5: Strong penalty

    Warning: Too high penalty can make model avoid common words!
    """
    if penalty == 1.0:
        return logits

    batch_size = logits.size(0)

    for batch_idx in range(batch_size):
        # Get unique tokens in this sequence
        unique_tokens = generated_tokens[batch_idx].unique()

        for token_id in unique_tokens:
            token_id = token_id.item()

            # Apply penalty
            if logits[batch_idx, token_id] > 0:
                logits[batch_idx, token_id] /= penalty
            else:
                logits[batch_idx, token_id] *= penalty

    return logits


def get_ngrams(tokens: List[int], n: int) -> Set[tuple]:
    """
    Extract all n-grams from token list.

    Args:
        tokens: List of token IDs
        n: N-gram size

    Returns:
        Set of n-grams (tuples)

    Example:
        tokens = [1, 2, 3, 4]
        get_ngrams(tokens, 2) = {(1,2), (2,3), (3,4)}
    """
    if len(tokens) < n:
        return set()

    ngrams = set()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams.add(ngram)

    return ngrams


def block_ngram_repeats(
    logits: torch.Tensor, generated_tokens: List[int], ngram_size: int = 3
) -> torch.Tensor:
    """
    Block tokens that would create repeated n-grams.

    Args:
        logits: Current logits [vocab_size]
        generated_tokens: Previously generated tokens (list)
        ngram_size: N-gram size to block

    Returns:
        Modified logits with blocked tokens set to -inf

    How it works:
    - Look at last (n-1) tokens
    - Find all tokens that appeared after this (n-1)-gram before
    - Set their logits to -inf (can't be sampled)

    Example with n=3:
    - Generated: "The patient has the patient"
    - Last 2 tokens: "the patient"
    - Previously after "the patient": "has"
    - Block "has" from being generated again

    This prevents: "The patient has the patient has the patient has..."
    """
    if ngram_size == 0 or len(generated_tokens) < ngram_size:
        return logits

    # Get context (last n-1 tokens)
    context = tuple(generated_tokens[-(ngram_size - 1) :])

    # Find all tokens that appeared after this context
    blocked_tokens = set()

    for i in range(len(generated_tokens) - ngram_size + 1):
        # Check if this position matches our context
        if tuple(generated_tokens[i : i + ngram_size - 1]) == context:
            # The next token creates a repeated n-gram
            next_token = generated_tokens[i + ngram_size - 1]
            blocked_tokens.add(next_token)

    # Block these tokens
    for token_id in blocked_tokens:
        logits[token_id] = float("-inf")

    return logits


def should_stop_generation(
    generated_tokens: List[int], stop_tokens: List[int], max_length: int, min_length: int
) -> bool:
    """
    Check if generation should stop.

    Args:
        generated_tokens: Generated token IDs
        stop_tokens: Token IDs that trigger stopping (e.g., EOS)
        max_length: Maximum allowed length
        min_length: Minimum required length

    Returns:
        True if should stop, False otherwise

    Stopping criteria:
    1. Reached max_length
    2. Generated a stop token (and past min_length)
    """
    current_length = len(generated_tokens)

    # Must generate at least min_length tokens
    if current_length < min_length:
        return False

    # Stop if reached max length
    if current_length >= max_length:
        return True

    # Stop if generated a stop token
    if generated_tokens[-1] in stop_tokens:
        return True

    return False
