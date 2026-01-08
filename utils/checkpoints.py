"""
Model Checkpointing Utilities

PURPOSE:
Save and load model checkpoints during training. This is essential for:
- Resuming interrupted training
- Saving best model based on validation loss
- Preventing loss of work from crashes

WHAT THIS FILE DOES:
1. Save model state_dict + optimizer + training state
2. Load checkpoints to resume training
3. Manage checkpoint files (keep only best/recent)
4. Save configuration alongside weights

PACKAGES USED:
- torch: Save/load model state
- pathlib: File management
- json: Save metadata

FILES FROM THIS PROJECT:
- model/checkpoints/ (checkpoint directory)

CHECKPOINT CONTENTS:
- model_state_dict: Model weights
- optimizer_state_dict: Optimizer state (for resuming)
- step: Current training step
- epoch: Current epoch
- best_val_loss: Best validation loss so far
- config: Model and training config

COMMON ISSUES:
- Not saving optimizer → can't resume training properly
- Not saving RNG state → non-reproducible results
- Checkpoints too large → disk space issues
- Not testing loading → corrupt checkpoints discovered too late
"""

import torch
from pathlib import Path
import json
from typing import Dict, Optional
import shutil


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Handles saving, loading, and cleanup of checkpoint files.
    """

    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 3):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

        # Track best validation loss
        self.best_val_loss = float("inf")

        print(f"Checkpoint directory: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        val_loss: float,
        model_config: dict,
        train_config: dict,
        is_best: bool = False,
    ):
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            epoch: Current epoch
            val_loss: Validation loss
            model_config: Model configuration
            train_config: Training configuration
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "model_config": model_config,
            "train_config": train_config,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save as best if applicable
        if is_best or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path} (val_loss: {val_loss:.4f})")

        # Save as latest (for easy resuming)
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        # Get all checkpoint files (except best and latest)
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        # Remove old ones
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[: -self.keep_last_n]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt.name}")

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> Dict:
        """
        Load a checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Specific checkpoint to load (or None for latest)
            device: Device to load to

        Returns:
            Checkpoint dictionary with metadata
        """
        # Use latest if no path specified
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint from step {checkpoint['step']}, epoch {checkpoint['epoch']}")
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")

        return checkpoint

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists."""
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        return latest_path.exists()


def save_model_for_inference(
    model: torch.nn.Module, tokenizer_path: Path, save_path: Path, model_config: dict
):
    """
    Save model for inference (weights only, no optimizer).

    Args:
        model: Trained model
        tokenizer_path: Path to tokenizer model
        save_path: Where to save
        model_config: Model configuration
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "tokenizer_path": str(tokenizer_path),
    }

    torch.save(save_dict, save_path)
    print(f"Model saved for inference: {save_path}")


def load_model_for_inference(model: torch.nn.Module, checkpoint_path: Path, device: str = "cuda"):
    """
    Load model for inference.

    Args:
        model: Model architecture (will be populated with weights)
        checkpoint_path: Path to saved model
        device: Device to load to
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded for inference from: {checkpoint_path}")

    return checkpoint.get("model_config")
