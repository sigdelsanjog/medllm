"""
SentencePiece Tokenizer Training

Trains a BPE (Byte-Pair Encoding) tokenizer on the processed MedQuAD text.

Design decisions explained:
- BPE over WordPiece: Better for medical terminology (handles rare words via subwords)
- Vocab size: Trade-off between model size and expressiveness
- Character coverage: 0.9995 captures medical unicode (Greek letters, symbols)
- No pre-tokenization: Let BPE learn from raw text

Common failure modes to avoid:
- Too small vocab → repetitive, generic text
- Too large vocab → overfitting, slow training
- Missing special tokens → can't mark boundaries
- Wrong normalization → "COVID-19" vs "covid-19" treated differently
"""

import argparse
from pathlib import Path
import sentencepiece as spm


def train_sentencepiece_tokenizer(
    input_file: Path,
    output_prefix: Path,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    add_special_tokens: bool = True,
):
    """
    Train a SentencePiece tokenizer.

    Args:
        input_file: Path to training text file
        output_prefix: Output path prefix (will create .model and .vocab files)
        vocab_size: Vocabulary size (default 8000 for small dataset)
        model_type: 'bpe' or 'unigram'
        character_coverage: Character coverage for unicode (0.9995 = medical terms)
        add_special_tokens: Whether to add custom special tokens

    Design notes:
    - vocab_size=8000: Good for 40K examples on GTX 1080
      * Too small (2K): Poor compression, repetitive outputs
      * Too large (32K): Overfits, wastes memory on rare terms
    - BPE: Deterministic, easier to debug than unigram
    - character_coverage=0.9995: Captures medical unicode (μ, α, β, etc.)
    """

    # Build training command
    train_args = [
        f"--input={input_file}",
        f"--model_prefix={output_prefix}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        "--pad_id=0",  # <pad> token at ID 0
        "--unk_id=1",  # <unk> token at ID 1
        "--bos_id=2",  # <bos> (beginning of sequence) at ID 2
        "--eos_id=3",  # <eos> (end of sequence) at ID 3
        "--pad_piece=[PAD]",
        "--unk_piece=[UNK]",
        "--bos_piece=[BOS]",
        "--eos_piece=[EOS]",
    ]

    # Add user-defined special tokens if requested
    if add_special_tokens:
        # These match our text formatting (Q:, A:)
        user_defined = "[Q],[A]"
        train_args.append(f"--user_defined_symbols={user_defined}")

    # Normalization rules (CAREFUL: medical text is case-sensitive)
    # We use minimal normalization to preserve:
    # - "COVID-19" vs "covid-19"
    # - Drug names (proper capitalization matters)
    # - Abbreviations (BP vs bp)
    train_args.extend(
        [
            "--normalization_rule_name=identity",  # No normalization
            "--remove_extra_whitespaces=true",  # Clean whitespace
            "--split_by_unicode_script=true",  # Split CJK if present
            "--split_by_whitespace=true",
            "--split_by_number=true",  # Split numbers
            "--max_sentence_length=4096",  # Long medical answers
        ]
    )

    print("=" * 60)
    print("Training SentencePiece Tokenizer")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model type: {model_type}")
    print(f"Character coverage: {character_coverage}")
    print()

    # Train the tokenizer
    spm.SentencePieceTrainer.Train(" ".join(train_args))

    print("\n✅ Tokenizer training complete!")
    print(f"Model saved: {output_prefix}.model")
    print(f"Vocab saved: {output_prefix}.vocab")

    # Load and inspect the tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f"{output_prefix}.model")

    print("\n" + "=" * 60)
    print("Tokenizer Statistics")
    print("=" * 60)
    print(f"Vocabulary size: {sp.vocab_size()}")
    print(f"PAD token: {sp.id_to_piece(0)} (ID: 0)")
    print(f"UNK token: {sp.id_to_piece(1)} (ID: 1)")
    print(f"BOS token: {sp.id_to_piece(2)} (ID: 2)")
    print(f"EOS token: {sp.id_to_piece(3)} (ID: 3)")

    # Test tokenization on medical example
    print("\n" + "=" * 60)
    print("Sample Tokenization")
    print("=" * 60)

    test_texts = [
        "What is diabetes?",
        "COVID-19 vaccination side effects",
        "Hypertension treatment guidelines",
    ]

    for text in test_texts:
        tokens = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Token count: {len(tokens)}")

    # Vocabulary inspection
    print("\n" + "=" * 60)
    print("Sample Vocabulary (first 20 tokens)")
    print("=" * 60)
    for i in range(min(20, sp.vocab_size())):
        piece = sp.id_to_piece(i)
        print(f"ID {i:4d}: {piece}")

    return sp


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for MedQuAD")
    parser.add_argument(
        "--input-file",
        type=str,
        default="./data/processed/medquad_simple.txt",
        help="Input text file for training",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./tokenizer", help="Output directory for tokenizer files"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size (default: 8000 for small dataset)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bpe", "unigram"],
        default="bpe",
        help="Tokenizer algorithm",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage (0.9995 = captures medical unicode)",
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print("\nPlease run preprocessing first:")
        print("  python preprocess.py")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output prefix for .model and .vocab files
    output_prefix = output_dir / "medquad_tokenizer"

    # Train tokenizer
    tokenizer = train_sentencepiece_tokenizer(
        input_file=input_file,
        output_prefix=output_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        add_special_tokens=True,
    )

    print("\n" + "=" * 60)
    print("✅ Tokenizer training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Inspect vocabulary for medical terms")
    print("2. Test on sample medical texts")
    print("3. Tokenize full dataset: python tokenizer/tokenize_data.py")


if __name__ == "__main__":
    main()
