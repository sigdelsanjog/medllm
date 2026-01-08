"""
Training Logging Utilities

PURPOSE:
Track and log training metrics (loss, learning rate, gradient norms).
This is CRITICAL for debugging training issues.

WHAT THIS FILE DOES:
1. Log training metrics to console
2. Track loss history
3. Compute moving averages
4. Save metrics to file

WHY LOGGING IS CRITICAL:
- Detect gradient explosions (loss → NaN)
- Spot overfitting (train loss ↓, val loss ↑)
- Monitor learning rate schedule
- Debug slow convergence

PACKAGES USED:
- json: Save metrics
- time: Track training speed
- statistics: Compute averages

FILES FROM THIS PROJECT:
- None (utility module)

COMMON ISSUES TO DETECT:
- Loss = NaN → exploding gradients, LR too high
- Loss stuck → LR too low, bad initialization
- Loss oscillating → LR too high, reduce it
- Val loss increasing → overfitting, add regularization
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class MetricsLogger:
    """
    Simple metrics logger for training.

    Tracks loss, learning rate, and other metrics over time.
    """

    def __init__(self, log_dir: Path, experiment_name: str = "training"):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name for this training run
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}_metrics.jsonl"

        # Metrics storage
        self.metrics = defaultdict(list)

        # Timing
        self.start_time = time.time()
        self.step_times = []

        print(f"Logging to: {self.log_file}")

    def log(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics for a training step.

        Args:
            step: Training step number
            metrics: Dictionary of metric_name -> value
        """
        # Add timestamp and step
        log_entry = {"step": step, "timestamp": time.time() - self.start_time, **metrics}

        # Store in memory
        for key, value in metrics.items():
            self.metrics[key].append(value)

        # Append to file (JSONL format)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """Log epoch-level metrics."""
        print(f"\nEpoch {epoch}:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")

        self.log(epoch, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    def get_last(self, metric_name: str, n: int = 1) -> Optional[float]:
        """Get last n values of a metric."""
        if metric_name not in self.metrics:
            return None
        values = self.metrics[metric_name]
        if len(values) < n:
            return None
        return sum(values[-n:]) / n

    def get_average(self, metric_name: str, window: int = 100) -> Optional[float]:
        """Get moving average of a metric."""
        if metric_name not in self.metrics:
            return None
        values = self.metrics[metric_name]
        if len(values) == 0:
            return None
        window = min(window, len(values))
        return sum(values[-window:]) / window


def log_training_step(
    step: int,
    loss: float,
    lr: float,
    grad_norm: float,
    tokens_per_sec: float,
    print_output: bool = True,
):
    """
    Log a single training step to console.

    Args:
        step: Training step
        loss: Current loss
        lr: Current learning rate
        grad_norm: Gradient norm (before clipping)
        tokens_per_sec: Processing speed
        print_output: Whether to print to console
    """
    if not print_output:
        return

    # Format output
    msg = f"Step {step:5d} | "
    msg += f"Loss: {loss:.4f} | "
    msg += f"LR: {lr:.2e} | "
    msg += f"Grad: {grad_norm:.3f} | "
    msg += f"Tok/s: {tokens_per_sec:.0f}"

    # Check for issues
    if loss != loss:  # NaN check
        msg += " [WARNING: NaN loss!]"
    elif loss > 100:
        msg += " [WARNING: High loss!]"

    if grad_norm > 10:
        msg += " [WARNING: Large gradients!]"

    print(msg)


def log_validation(step: int, val_loss: float, val_perplexity: float):
    """
    Log validation metrics.

    Args:
        step: Training step
        val_loss: Validation loss
        val_perplexity: Validation perplexity (exp(loss))
    """
    print(f"\n{'='*60}")
    print(f"Validation at step {step}")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Perplexity: {val_perplexity:.2f}")
    print(f"{'='*60}\n")


def save_training_summary(log_dir: Path, config: dict, final_metrics: dict):
    """
    Save final training summary.

    Args:
        log_dir: Directory to save summary
        config: Training configuration
        final_metrics: Final metrics (best val loss, etc.)
    """
    summary = {"config": config, "final_metrics": final_metrics, "timestamp": time.time()}

    summary_file = log_dir / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary saved to: {summary_file}")
