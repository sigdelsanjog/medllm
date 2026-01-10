"""
Main Training Script

PURPOSE:
Entry point for training the GPT model.
Ties everything together and starts training.

WHAT THIS FILE DOES:
1. Load configuration (model + training)
2. Create model
3. Load tokenized data
4. Create optimizer
5. Start training
6. Handle command-line arguments

USAGE:
    python training/train.py                    # Use default config
    python training/train.py --batch-size 32    # Override batch size
    python training/train.py --resume          # Resume from checkpoint

PACKAGES USED:
- torch: PyTorch
- argparse: Command-line arguments
- pathlib: Path handling

FILES FROM THIS PROJECT:
- All model, training, and utility modules

EXECUTION ORDER:
1. Parse arguments
2. Set random seeds (reproducibility)
3. Create model
4. Load data
5. Create optimizer
6. Initialize trainer
7. Start training
"""

import torch
import argparse
from pathlib import Path
import random
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_med.model.architecture import GPTTransformer
from llm_med.model.configs.model_config import get_small_config, get_tiny_config
from llm_med.configs.train_config import get_default_config, get_quick_test_config
from llm_med.training.dataset import create_dataloaders
from llm_med.training.trainer import Trainer


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed

    Why this matters:
    - Makes training reproducible
    - Critical for debugging (can recreate issues)
    - Scientific experiments need reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make cudnn deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Train GPT model on MedQuAD")

    # Model config
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium"],
        help="Model size (tiny/small/medium)",
    )

    # Training config
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Use quick test config (small batches, few steps)"
    )

    # Paths
    parser.add_argument(
        "--train-data", type=str, default="./data/tokenized/train.npy", help="Path to training data"
    )
    parser.add_argument(
        "--val-data", type=str, default="./data/tokenized/val.npy", help="Path to validation data"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./model/checkpoints", help="Checkpoint directory"
    )

    # Resume training
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Resume from specific checkpoint"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to train on"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("GPT Training - MedQuAD")
    print("=" * 60)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"

    # Set random seed
    print(f"\nSetting random seed: {args.seed}")
    set_seed(args.seed)

    # Load configurations
    print(f"\nLoading configurations...")

    # Model config
    if args.model_size == "tiny":
        model_config = get_tiny_config()
    elif args.model_size == "small":
        model_config = get_small_config()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    print(f"Model config: {args.model_size}")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  n_heads: {model_config.n_heads}")

    # Training config
    if args.quick_test:
        train_config = get_quick_test_config()
        print("Using quick test config (fast debugging)")
    else:
        train_config = get_default_config()

    # Override with command-line args
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        train_config.num_epochs = args.num_epochs
    if args.train_data:
        train_config.train_data_path = args.train_data
    if args.val_data:
        train_config.val_data_path = args.val_data
    if args.checkpoint_dir:
        train_config.checkpoint_dir = args.checkpoint_dir

    train_config.device = args.device
    train_config.seed = args.seed

    print(f"\nTraining config:")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Num epochs: {train_config.num_epochs}")
    print(f"  Device: {train_config.device}")

    # Create model
    print(f"\nCreating model...")
    model = GPTTransformer(model_config)
    total_params = count_parameters(model)
    print(f"Model created with {total_params:,} parameters")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")

    # Load data
    print(f"\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        train_path=Path(train_config.train_data_path),
        val_path=Path(train_config.val_data_path),
        batch_size=train_config.batch_size,
        num_workers=0,
    )

    # Create optimizer
    print(f"\nCreating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=train_config.betas,
        eps=train_config.eps,
        weight_decay=train_config.weight_decay,
    )
    print(f"Optimizer: AdamW")
    print(f"  LR: {train_config.learning_rate}")
    print(f"  Weight decay: {train_config.weight_decay}")

    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=train_config,
        device=args.device,
    )

    # Resume if requested
    if args.resume or args.resume_from:
        print(f"\nResuming training...")
        checkpoint_path = Path(args.resume_from) if args.resume_from else None
        trainer.resume_from_checkpoint(checkpoint_path)

    # Start training
    print(f"\n{'='*60}")
    print("Ready to train!")
    print(f"{'='*60}\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            val_loss=trainer.best_val_loss,
            model_config=model_config.to_dict(),
            train_config=train_config.to_dict(),
        )
        print("Checkpoint saved. You can resume with --resume")

    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)
    print(f"\nBest model saved in: {train_config.checkpoint_dir}/best_model.pt")
    print(f"Logs saved in: {train_config.log_dir}")


if __name__ == "__main__":
    main()
