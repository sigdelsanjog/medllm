"""
Trainer Class

PURPOSE:
Core training loop logic. This is the heart of Phase 3.
Handles forward pass, backward pass, optimization, and evaluation.

WHAT THIS FILE DOES:
1. Training loop: iterate over batches, compute loss, backprop
2. Evaluation: compute validation loss
3. Checkpointing: save model periodically
4. Logging: track metrics

TRAINING ALGORITHM:
For each batch:
  1. Forward pass: model(input) → logits
  2. Compute loss: CrossEntropyLoss(logits, targets)
  3. Backward pass: loss.backward()
  4. Clip gradients: prevent explosion
  5. Optimizer step: update weights
  6. Update learning rate: warmup + decay

PACKAGES USED:
- torch: PyTorch training
- time: Track speed

FILES FROM THIS PROJECT:
- model/architecture/transformer.py: Model
- training/dataset.py: DataLoader
- training/utils.py: Helper functions
- utils/logging.py: Metrics logging
- utils/checkpoints.py: Save/load

TENSOR SHAPES:
- Input: [batch_size, seq_len]
- Logits: [batch_size, seq_len, vocab_size]
- Targets: [batch_size, seq_len]
- Loss: scalar

COMMON TRAINING ISSUES:
- Loss = NaN → gradient explosion (reduce LR, check grad clipping)
- Loss stuck → LR too low, bad initialization
- Slow convergence → LR too low, increase it
- Overfitting → add dropout, weight decay
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from pathlib import Path
from typing import Optional

from llm_med.model.architecture import GPTTransformer
from llm_med.training.utils import (
    clip_grad_norm,
    get_lr_with_warmup,
    set_learning_rate,
    estimate_loss_dataloader,
    compute_perplexity,
)
from llm_med.utils.logging import MetricsLogger, log_training_step, log_validation
from llm_med.utils.checkpoints import CheckpointManager


class Trainer:
    """
    Training orchestrator for GPT model.

    Handles the full training loop including:
    - Forward/backward passes
    - Optimization
    - Evaluation
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: GPTTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config,  # TrainingConfig
        device: str = "cuda",
    ):
        """
        Args:
            model: GPT model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (e.g., AdamW)
            config: TrainingConfig object
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # Initialize utilities
        self.logger = MetricsLogger(log_dir=config.log_dir, experiment_name="gpt_training")

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir, keep_last_n=config.keep_last_n
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Calculate total steps
        steps_per_epoch = len(train_loader)
        if config.max_steps > 0:
            self.total_steps = config.max_steps
        else:
            self.total_steps = steps_per_epoch * config.num_epochs

        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Num epochs: {config.num_epochs}")

    def train_step(self, batch: tuple) -> dict:
        """
        Single training step.

        Args:
            batch: (input_ids, target_ids) tuple

        Returns:
            Dictionary with step metrics
        """
        # Move batch to device
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass
        logits = self.model(input_ids)

        # Compute loss
        # CrossEntropyLoss expects:
        #   - Input: [N, C] where N = batch_size * seq_len, C = vocab_size
        #   - Target: [N] with class indices
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        targets_flat = target_ids.view(batch_size * seq_len)

        loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients (CRITICAL for stability)
        grad_norm = clip_grad_norm(self.model, self.config.grad_clip)

        # Optimizer step
        self.optimizer.step()

        # Return metrics
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    def evaluate(self) -> dict:
        """
        Evaluate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        print("\nRunning validation...")

        val_loss = estimate_loss_dataloader(
            self.model, self.val_loader, self.device, max_batches=self.config.eval_iters
        )

        val_perplexity = compute_perplexity(val_loss)

        log_validation(self.global_step, val_loss, val_perplexity)

        return {"val_loss": val_loss, "val_perplexity": val_perplexity}

    def train(self):
        """
        Main training loop.

        This is where everything comes together.
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        self.model.train()

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")

            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                step_start_time = time.time()

                # Update learning rate (warmup + decay)
                lr = get_lr_with_warmup(
                    step=self.global_step,
                    warmup_steps=self.config.warmup_steps,
                    max_lr=self.config.learning_rate,
                    min_lr=self.config.min_lr,
                    max_steps=self.total_steps,
                    decay_type=self.config.lr_decay,
                )
                set_learning_rate(self.optimizer, lr)

                # Training step
                metrics = self.train_step(batch)

                # Calculate tokens per second
                step_time = time.time() - step_start_time
                tokens_per_sec = (metrics["batch_size"] * metrics["seq_len"]) / step_time

                # Log to console
                if self.global_step % self.config.log_interval == 0:
                    log_training_step(
                        step=self.global_step,
                        loss=metrics["loss"],
                        lr=lr,
                        grad_norm=metrics["grad_norm"],
                        tokens_per_sec=tokens_per_sec,
                    )

                # Log metrics
                self.logger.log(
                    self.global_step,
                    {
                        "train_loss": metrics["loss"],
                        "learning_rate": lr,
                        "grad_norm": metrics["grad_norm"],
                        "tokens_per_sec": tokens_per_sec,
                    },
                )

                # Evaluate
                if self.global_step % self.config.eval_interval == 0 and self.global_step > 0:
                    val_metrics = self.evaluate()

                    # Log validation metrics
                    self.logger.log(self.global_step, val_metrics)

                    # Check if best model
                    if val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        is_best = True
                    else:
                        is_best = False

                    # Save checkpoint
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        step=self.global_step,
                        epoch=epoch,
                        val_loss=val_metrics["val_loss"],
                        model_config=self.model.config.to_dict(),
                        train_config=self.config.to_dict(),
                        is_best=is_best,
                    )

                    self.model.train()  # Back to training mode

                # Save checkpoint periodically
                if self.global_step % self.config.save_interval == 0 and self.global_step > 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        step=self.global_step,
                        epoch=epoch,
                        val_loss=self.best_val_loss,
                        model_config=self.model.config.to_dict(),
                        train_config=self.config.to_dict(),
                    )

                self.global_step += 1

                # Check if reached max steps
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    print(f"\nReached max_steps ({self.config.max_steps}). Stopping training.")
                    return

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def resume_from_checkpoint(self, checkpoint_path: Optional[Path] = None):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (or None for latest)
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

        self.global_step = checkpoint["step"]
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
