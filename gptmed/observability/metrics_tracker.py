"""
Metrics Tracker for Training Observability

PURPOSE:
Enhanced metrics tracking with loss curves, moving averages,
gradient statistics, and export capabilities.

FEATURES:
- Loss curve history (train & validation)
- Moving averages for smoothed visualization
- Perplexity tracking
- Learning rate schedule visualization
- Gradient norm monitoring
- Export to JSON, CSV, and plots

WHAT TO LOOK FOR:
- Train loss ↓, Val loss ↓ → Healthy learning
- Train loss ↓, Val loss ↑ → Overfitting
- Loss plateau → Stuck (increase LR or check data)
- Loss spikes → Instability (reduce LR)
- Loss = NaN → Exploding gradients
"""


import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque

from gptmed.observability.base import (
    TrainingObserver,
    StepMetrics,
    ValidationMetrics,
    GradientMetrics,
)

# Import the interface but not the Redis implementation directly (for loose coupling)
from gptmed.observability.redis_metrics_storage import MetricsStorageInterface


@dataclass
class LossCurvePoint:
    """Single point on the loss curve."""
    step: int
    loss: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "step": self.step,
            "loss": self.loss,
            "timestamp": self.timestamp,
        }


class MetricsTracker(TrainingObserver):
    """
    Comprehensive metrics tracking for training observability.
    
    Tracks:
    - Training loss curve
    - Validation loss curve
    - Learning rate schedule
    - Gradient norms
    - Perplexity
    - Moving averages
    
    Example:
        >>> tracker = MetricsTracker(log_dir='logs/experiment1')
        >>> trainer.add_observer(tracker)
        >>> # After training:
        >>> tracker.plot_loss_curves()
        >>> tracker.export_to_csv('metrics.csv')
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "training",
        moving_avg_window: int = 100,
        log_interval: int = 10,
        verbose: bool = True,
        storage_backend: Optional[MetricsStorageInterface] = None,
    ):
        """
        Initialize MetricsTracker.
        
        Args:
            log_dir: Directory to save logs and exports
            experiment_name: Name for this experiment
            moving_avg_window: Window size for moving average
            log_interval: How often to log to file (every N steps)
            verbose: Whether to print progress
        """
        super().__init__(name="MetricsTracker")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.moving_avg_window = moving_avg_window
        self.log_interval = log_interval
        self.verbose = verbose

        # Optional metrics storage backend (e.g., Redis)
        self.storage_backend = storage_backend

        # Initialize storage
        self._reset_storage()

        # File paths
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        self.summary_file = self.log_dir / f"{experiment_name}_summary.json"

        if self.verbose:
            print(f"📊 MetricsTracker initialized")
            print(f"   Log directory: {self.log_dir}")
            print(f"   Moving average window: {moving_avg_window}")
            if self.storage_backend:
                print(f"   Using external metrics storage: {type(self.storage_backend).__name__}")
    
    def _reset_storage(self) -> None:
        """Reset all metric storage."""
        # Loss curves
        self.train_losses: List[LossCurvePoint] = []
        self.val_losses: List[LossCurvePoint] = []
        
        # Moving average buffer
        self._loss_buffer: deque = deque(maxlen=self.moving_avg_window)
        
        # Learning rate history
        self.learning_rates: List[Tuple[int, float]] = []
        
        # Gradient norms
        self.gradient_norms: List[Tuple[int, float]] = []
        
        # Perplexity
        self.train_perplexities: List[Tuple[int, float]] = []
        self.val_perplexities: List[Tuple[int, float]] = []
        
        # Timing
        self.start_time: Optional[float] = None
        self.step_times: List[float] = []
        
        # Training config
        self.config: Dict[str, Any] = {}
        
        # Best metrics
        self.best_val_loss: float = float('inf')
        self.best_val_step: int = 0
    
    # === TrainingObserver Implementation ===
    
    def on_train_start(self, config: Dict[str, Any]) -> None:
        """Called when training begins."""
        self._reset_storage()
        self.start_time = time.time()
        self.config = config.copy()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 Training started - MetricsTracker active")
            print(f"{'='*60}")
        
        # Log config
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def on_step(self, metrics: StepMetrics) -> None:
        """Called after each training step."""
        timestamp = time.time() - self.start_time if self.start_time else 0

        # Store loss
        self.train_losses.append(LossCurvePoint(
            step=metrics.step,
            loss=metrics.loss,
            timestamp=timestamp,
        ))

        # Update moving average buffer
        self._loss_buffer.append(metrics.loss)

        # Store learning rate
        self.learning_rates.append((metrics.step, metrics.learning_rate))

        # Store gradient norm
        self.gradient_norms.append((metrics.step, metrics.grad_norm))

        # Store perplexity
        self.train_perplexities.append((metrics.step, metrics.perplexity))

        # Log to file periodically
        if metrics.step % self.log_interval == 0:
            self._log_step(metrics, timestamp)
            # Also log to external storage if available
            if self.storage_backend:
                self.storage_backend.save_step_metrics({
                    "type": "step",
                    "timestamp": timestamp,
                    "moving_avg_loss": self.get_moving_average(),
                    **metrics.to_dict(),
                })
    
    def on_validation(self, metrics: ValidationMetrics) -> None:
        """Called after validation."""
        timestamp = time.time() - self.start_time if self.start_time else 0

        # Store validation loss
        self.val_losses.append(LossCurvePoint(
            step=metrics.step,
            loss=metrics.val_loss,
            timestamp=timestamp,
        ))

        # Store validation perplexity
        self.val_perplexities.append((metrics.step, metrics.val_perplexity))

        # Track best
        if metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
            self.best_val_step = metrics.step
            if self.verbose:
                print(f"   ⭐ New best val_loss: {metrics.val_loss:.4f}")

        # Log to file
        self._log_validation(metrics, timestamp)
        # Also log to external storage if available
        if self.storage_backend:
            self.storage_backend.save_validation_metrics({
                "type": "validation",
                "timestamp": timestamp,
                "is_best": metrics.val_loss <= self.best_val_loss,
                **metrics.to_dict(),
            })
    
    def on_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training completes."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Create summary
        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": len(self.train_losses),
            "total_time_seconds": total_time,
            "final_train_loss": self.train_losses[-1].loss if self.train_losses else None,
            "final_val_loss": self.val_losses[-1].loss if self.val_losses else None,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
            "final_perplexity": self.train_perplexities[-1][1] if self.train_perplexities else None,
            "config": self.config,
            **final_metrics,
        }
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 Training completed - MetricsTracker summary")
            print(f"{'='*60}")
            print(f"   Total steps: {len(self.train_losses)}")
            print(f"   Total time: {total_time/60:.2f} minutes")
            print(f"   Final train loss: {summary['final_train_loss']:.4f}" if summary['final_train_loss'] else "")
            print(f"   Best val loss: {self.best_val_loss:.4f} (step {self.best_val_step})")
            print(f"   Summary saved: {self.summary_file}")
    
    def on_gradient_computed(self, metrics: GradientMetrics) -> None:
        """Called after gradients are computed."""
        # Additional gradient tracking if needed
        pass
    
    # === Metrics Access Methods ===
    
    def get_train_loss_curve(self) -> List[Tuple[int, float]]:
        """Get training loss curve as (step, loss) pairs."""
        return [(p.step, p.loss) for p in self.train_losses]
    
    def get_val_loss_curve(self) -> List[Tuple[int, float]]:
        """Get validation loss curve as (step, loss) pairs."""
        return [(p.step, p.loss) for p in self.val_losses]
    
    def get_moving_average(self) -> float:
        """Get current moving average of training loss."""
        if not self._loss_buffer:
            return 0.0
        return sum(self._loss_buffer) / len(self._loss_buffer)
    
    def get_smoothed_loss_curve(self, window: int = None) -> List[Tuple[int, float]]:
        """
        Get smoothed training loss curve using moving average.
        
        Args:
            window: Smoothing window size (default: self.moving_avg_window)
            
        Returns:
            List of (step, smoothed_loss) tuples
        """
        window = window or self.moving_avg_window
        if len(self.train_losses) < window:
            return self.get_train_loss_curve()
        
        smoothed = []
        losses = [p.loss for p in self.train_losses]
        steps = [p.step for p in self.train_losses]
        
        for i in range(window - 1, len(losses)):
            avg = sum(losses[i - window + 1:i + 1]) / window
            smoothed.append((steps[i], avg))
        
        return smoothed
    
    def get_loss_at_step(self, step: int) -> Optional[float]:
        """Get training loss at specific step."""
        for point in self.train_losses:
            if point.step == step:
                return point.loss
        return None
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient norm statistics."""
        if not self.gradient_norms:
            return {}
        
        norms = [n for _, n in self.gradient_norms]
        return {
            "mean": sum(norms) / len(norms),
            "max": max(norms),
            "min": min(norms),
            "last": norms[-1],
        }
    
    def detect_issues(self) -> List[str]:
        """
        Detect potential training issues from metrics.
        
        Returns:
            List of warning messages
        """
        issues = []
        
        if not self.train_losses:
            return ["No training data recorded yet"]
        
        # Check for NaN
        if any(math.isnan(p.loss) for p in self.train_losses):
            issues.append("⚠️ NaN loss detected - likely exploding gradients")
        
        # Check for loss explosion
        recent_losses = [p.loss for p in self.train_losses[-100:]]
        if recent_losses and max(recent_losses) > 100:
            issues.append("⚠️ Very high loss (>100) - check learning rate")
        
        # Check for gradient explosion
        if self.gradient_norms:
            recent_grads = [n for _, n in self.gradient_norms[-100:]]
            if max(recent_grads) > 100:
                issues.append("⚠️ Large gradient norms (>100) - consider gradient clipping")
        
        # Check for overfitting
        if len(self.val_losses) >= 3:
            recent_val = [p.loss for p in self.val_losses[-3:]]
            if all(recent_val[i] > recent_val[i-1] for i in range(1, len(recent_val))):
                issues.append("⚠️ Validation loss increasing - possible overfitting")
        
        # Check for stalled training
        if len(self.train_losses) >= 1000:
            early_avg = sum(p.loss for p in self.train_losses[:100]) / 100
            recent_avg = sum(p.loss for p in self.train_losses[-100:]) / 100
            if abs(early_avg - recent_avg) < 0.01:
                issues.append("⚠️ Loss not improving - training may be stuck")
        
        return issues if issues else ["✓ No issues detected"]
    
    # === Export Methods ===
    
    def export_to_csv(self, filepath: str = None) -> str:
        """
        Export metrics to CSV file.
        
        Args:
            filepath: Output path (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        filepath = filepath or str(self.log_dir / f"{self.experiment_name}_metrics.csv")
        
        with open(filepath, 'w') as f:
            # Header
            f.write("step,train_loss,val_loss,learning_rate,grad_norm,perplexity,timestamp\n")
            
            # Create lookup dicts
            val_lookup = {p.step: p.loss for p in self.val_losses}
            lr_lookup = dict(self.learning_rates)
            grad_lookup = dict(self.gradient_norms)
            ppl_lookup = dict(self.train_perplexities)
            
            for point in self.train_losses:
                val_loss = val_lookup.get(point.step, "")
                lr = lr_lookup.get(point.step, "")
                grad = grad_lookup.get(point.step, "")
                ppl = ppl_lookup.get(point.step, "")
                f.write(f"{point.step},{point.loss},{val_loss},{lr},{grad},{ppl},{point.timestamp}\n")
        
        if self.verbose:
            print(f"📁 Exported to CSV: {filepath}")
        return filepath
    
    def export_to_json(self, filepath: str = None) -> str:
        """
        Export all metrics to JSON file.
        
        Args:
            filepath: Output path (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        filepath = filepath or str(self.log_dir / f"{self.experiment_name}_full_metrics.json")
        
        data = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "train_losses": [p.to_dict() for p in self.train_losses],
            "val_losses": [p.to_dict() for p in self.val_losses],
            "learning_rates": self.learning_rates,
            "gradient_norms": self.gradient_norms,
            "train_perplexities": self.train_perplexities,
            "val_perplexities": self.val_perplexities,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"📁 Exported to JSON: {filepath}")
        return filepath
    
    def plot_loss_curves(
        self,
        filepath: str = None,
        show_smoothed: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Optional[str]:
        """
        Plot training and validation loss curves.
        
        Args:
            filepath: Output path (default: auto-generated)
            show_smoothed: Whether to show smoothed curve
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved figure, or None if matplotlib not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not installed. Run: pip install matplotlib")
            return None
        
        filepath = filepath or str(self.log_dir / f"{self.experiment_name}_loss_curves.png")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # === Plot 1: Loss Curves ===
        ax1 = axes[0, 0]
        
        # Training loss
        train_steps = [p.step for p in self.train_losses]
        train_loss = [p.loss for p in self.train_losses]
        ax1.plot(train_steps, train_loss, alpha=0.3, label='Train Loss (raw)', color='blue')
        
        # Smoothed training loss
        if show_smoothed and len(self.train_losses) > self.moving_avg_window:
            smoothed = self.get_smoothed_loss_curve()
            smooth_steps, smooth_loss = zip(*smoothed)
            ax1.plot(smooth_steps, smooth_loss, label=f'Train Loss (MA-{self.moving_avg_window})', color='blue')
        
        # Validation loss
        if self.val_losses:
            val_steps = [p.step for p in self.val_losses]
            val_loss = [p.loss for p in self.val_losses]
            ax1.plot(val_steps, val_loss, 'o-', label='Val Loss', color='orange', markersize=4)
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === Plot 2: Learning Rate ===
        ax2 = axes[0, 1]
        if self.learning_rates:
            lr_steps, lr_values = zip(*self.learning_rates)
            ax2.plot(lr_steps, lr_values, color='green')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        # === Plot 3: Gradient Norms ===
        ax3 = axes[1, 0]
        if self.gradient_norms:
            grad_steps, grad_values = zip(*self.gradient_norms)
            ax3.plot(grad_steps, grad_values, alpha=0.5, color='red')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Norms')
        ax3.grid(True, alpha=0.3)
        
        # === Plot 4: Perplexity ===
        ax4 = axes[1, 1]
        if self.train_perplexities:
            ppl_steps, ppl_values = zip(*self.train_perplexities)
            # Cap perplexity for visualization
            ppl_values = [min(p, 1000) for p in ppl_values]
            ax4.plot(ppl_steps, ppl_values, alpha=0.5, label='Train Perplexity', color='purple')
        if self.val_perplexities:
            val_ppl_steps, val_ppl_values = zip(*self.val_perplexities)
            val_ppl_values = [min(p, 1000) for p in val_ppl_values]
            ax4.plot(val_ppl_steps, val_ppl_values, 'o-', label='Val Perplexity', color='magenta', markersize=4)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Perplexity')
        ax4.set_title('Perplexity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Metrics: {self.experiment_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        if self.verbose:
            print(f"📊 Loss curves saved: {filepath}")
        
        return filepath
    
    # === Private Methods ===
    
    def _log_step(self, metrics: StepMetrics, timestamp: float) -> None:
        """Log step metrics to file."""
        log_entry = {
            "type": "step",
            "timestamp": timestamp,
            "moving_avg_loss": self.get_moving_average(),
            **metrics.to_dict(),
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_validation(self, metrics: ValidationMetrics, timestamp: float) -> None:
        """Log validation metrics to file."""
        log_entry = {
            "type": "validation",
            "timestamp": timestamp,
            "is_best": metrics.val_loss <= self.best_val_loss,
            **metrics.to_dict(),
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
