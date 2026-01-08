"""
Dataset Tokenization

Applies trained tokenizer to processed text and creates train/validation splits.

Design decisions:
- Sequence length: 512 tokens (fits GTX 1080, captures most Q&A pairs)
- Train/val split: 90/10 (enough validation data to detect overfitting)
- Padding: Left-pad or truncate (causal LM sees left-to-right)
- No data augmentation: Keep it simple for Phase 1

Common failure modes:
- Truncating answers → model never learns to generate long responses
- Too short sequences → can't learn context
- Too long sequences → OOM on GPU
- Wrong padding → breaks attention masks
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import random

import sentencepiece as spm
import numpy as np


def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    """Load trained SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def tokenize_text(text: str, tokenizer: spm.SentencePieceProcessor) -> List[int]:
    """
    Tokenize text to IDs.

    Args:
        text: Input text
        tokenizer: SentencePiece processor

    Returns:
        List of token IDs
    """
    return tokenizer.encode_as_ids(text)


def create_sequences(token_ids: List[int], max_length: int, stride: int = None) -> List[List[int]]:
    """
    Split token IDs into fixed-length sequences.

    Args:
        token_ids: Full token ID sequence
        max_length: Maximum sequence length
        stride: Stride for overlapping windows (None = no overlap)

    Returns:
        List of sequences

    Design note: For causal LM, we create non-overlapping chunks.
    Each chunk is a separate training example for next-token prediction.
    """
    if stride is None:
        stride = max_length  # No overlap

    sequences = []
    for i in range(0, len(token_ids) - max_length + 1, stride):
        seq = token_ids[i : i + max_length]
        sequences.append(seq)

    # Handle remaining tokens (last incomplete sequence)
    remainder = len(token_ids) % stride
    if remainder > 0 and len(sequences) > 0:
        # Include last incomplete sequence if it's at least half the max_length
        last_start = len(token_ids) - remainder
        if remainder >= max_length // 2:
            last_seq = token_ids[last_start:]
            # Pad to max_length
            last_seq = last_seq + [0] * (max_length - len(last_seq))
            sequences.append(last_seq)

    return sequences


def analyze_lengths(text_file: Path, tokenizer: spm.SentencePieceProcessor) -> dict:
    """
    Analyze token length distribution to choose optimal sequence length.

    This helps avoid:
    - Truncating too much data (information loss)
    - Wasting memory on padding (inefficiency)
    """
    lengths = []

    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newline (our document separator)
    documents = content.split("\n\n")

    for doc in documents:
        if doc.strip():
            tokens = tokenizer.encode_as_ids(doc.strip())
            lengths.append(len(tokens))

    lengths = np.array(lengths)

    stats = {
        "num_documents": len(lengths),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "percentile_50": float(np.percentile(lengths, 50)),
        "percentile_75": float(np.percentile(lengths, 75)),
        "percentile_90": float(np.percentile(lengths, 90)),
        "percentile_95": float(np.percentile(lengths, 95)),
        "percentile_99": float(np.percentile(lengths, 99)),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Tokenize MedQuAD dataset with trained tokenizer")
    parser.add_argument(
        "--input-file",
        type=str,
        default="./data/processed/medquad_simple.txt",
        help="Input text file",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="./tokenizer/medquad_tokenizer.model",
        help="Path to trained tokenizer model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/tokenized",
        help="Output directory for tokenized data",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512 for GTX 1080)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9, help="Train/val split ratio (default: 0.9)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Dataset Tokenization")
    print("=" * 60)
    print(f"Input: {args.input_file}")
    print(f"Tokenizer: {args.tokenizer_model}")
    print(f"Max length: {args.max_length}")
    print(f"Train ratio: {args.train_ratio}")
    print()

    # Load tokenizer
    input_file = Path(args.input_file)
    tokenizer_model = Path(args.tokenizer_model)

    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return

    if not tokenizer_model.exists():
        print(f"❌ Error: Tokenizer not found: {tokenizer_model}")
        print("\nPlease train tokenizer first:")
        print("  python tokenizer/train_tokenizer.py")
        return

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_model)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size()})")

    # Analyze sequence lengths
    print("\nAnalyzing sequence lengths...")
    stats = analyze_lengths(input_file, tokenizer)

    print("\nSequence Length Statistics:")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Mean: {stats['mean']:.1f} tokens")
    print(f"  Median: {stats['median']:.1f} tokens")
    print(f"  Std dev: {stats['std']:.1f} tokens")
    print(f"  Min: {stats['min']} tokens")
    print(f"  Max: {stats['max']} tokens")
    print(f"\nPercentiles:")
    print(f"  50th: {stats['percentile_50']:.1f} tokens")
    print(f"  75th: {stats['percentile_75']:.1f} tokens")
    print(f"  90th: {stats['percentile_90']:.1f} tokens")
    print(f"  95th: {stats['percentile_95']:.1f} tokens")
    print(f"  99th: {stats['percentile_99']:.1f} tokens")

    # Warning about truncation
    truncated_pct = (
        np.array([l for l in stats.values() if isinstance(l, (int, float))]) > args.max_length
    ).sum()
    if stats["percentile_95"] > args.max_length:
        print(
            f"\n⚠️  WARNING: max_length={args.max_length} will truncate ~{100-95:.0f}% of sequences"
        )
        print(f"   Consider increasing to {int(stats['percentile_95'])} to capture 95% of data")

    # Tokenize full dataset
    print("\nTokenizing dataset...")
    with open(input_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    token_ids = tokenizer.encode_as_ids(full_text)
    print(f"Total tokens: {len(token_ids):,}")

    # Create sequences
    print(f"\nCreating sequences (max_length={args.max_length})...")
    sequences = create_sequences(token_ids, max_length=args.max_length)
    print(f"Total sequences: {len(sequences):,}")

    # Train/val split
    random.shuffle(sequences)
    split_idx = int(len(sequences) * args.train_ratio)

    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_sequences):,} sequences")
    print(f"  Val: {len(val_sequences):,} sequences")

    # Save tokenized data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving tokenized data...")

    # Save as numpy arrays (efficient for PyTorch DataLoader)
    train_array = np.array(train_sequences, dtype=np.int32)
    val_array = np.array(val_sequences, dtype=np.int32)

    np.save(output_dir / "train.npy", train_array)
    np.save(output_dir / "val.npy", val_array)

    print(f"✅ Train data saved: {output_dir / 'train.npy'}")
    print(f"✅ Val data saved: {output_dir / 'val.npy'}")

    # Save metadata
    metadata = {
        "vocab_size": tokenizer.vocab_size(),
        "max_length": args.max_length,
        "num_train_sequences": len(train_sequences),
        "num_val_sequences": len(val_sequences),
        "total_tokens": len(token_ids),
        "length_stats": stats,
        "tokenizer_model": str(tokenizer_model),
        "seed": args.seed,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved: {output_dir / 'metadata.json'}")

    print("\n" + "=" * 60)
    print("✅ Tokenization complete!")
    print("=" * 60)
    print("\nNext steps (Phase 2):")
    print("1. Implement Transformer architecture")
    print("2. Create PyTorch Dataset and DataLoader")
    print("3. Define training loop")


if __name__ == "__main__":
    main()
