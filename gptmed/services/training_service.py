"""
Training Service

PURPOSE:
Encapsulates training logic following Service Layer Pattern.
Provides a high-level interface for model training with device flexibility.

DESIGN PATTERNS:
- Service Layer Pattern: Business logic separated from API layer
- Dependency Injection: DeviceManager injected for flexibility
- Single Responsibility: Only handles training orchestration
- Open/Closed Principle: Extensible without modification

WHAT THIS FILE DOES:
1. Orchestrates the training process
2. Manages device configuration via DeviceManager
3. Coordinates model, data, optimizer, and trainer
4. Provides clean interface for training operations

PACKAGES USED:
- torch: PyTorch training
- pathlib: Path handling
"""

import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from gptmed.services.device_manager import DeviceManager
from gptmed.model.architecture import GPTTransformer
from gptmed.model.configs.model_config import get_tiny_config, get_small_config, get_medium_config
from gptmed.configs.train_config import TrainingConfig
from gptmed.training.dataset import create_dataloaders
from gptmed.training.trainer import Trainer

# Observability imports
from gptmed.observability import (
    TrainingObserver,
    MetricsTracker,
    ConsoleCallback,
)


class TrainingService:
    """
    High-level service for model training.
    
    Implements Service Layer Pattern to encapsulate training logic.
    Uses Dependency Injection for DeviceManager.
    
    Example:
        >>> device_manager = DeviceManager(preferred_device='cpu')
        >>> service = TrainingService(device_manager=device_manager)
        >>> results = service.train_from_config('config.yaml', verbose=True)
    """
    
    def __init__(
        self, 
        device_manager: Optional[DeviceManager] = None,
        verbose: bool = True
    ):
        """
        Initialize TrainingService.
        
        Args:
            device_manager: DeviceManager instance (if None, creates default)
            verbose: Whether to print training information
        """
        self.device_manager = device_manager or DeviceManager(preferred_device='cuda')
        self.verbose = verbose
    
    def set_seed(self, seed: int) -> None:
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def create_model(self, model_size: str) -> GPTTransformer:
        """
        Create model based on size specification.
        
        Args:
            model_size: Model size ('tiny', 'small', or 'medium')
            
        Returns:
            GPTTransformer model instance
            
        Raises:
            ValueError: If model_size is invalid
        """
        if model_size == 'tiny':
            model_config = get_tiny_config()
        elif model_size == 'small':
            model_config = get_small_config()
        elif model_size == 'medium':
            model_config = get_medium_config()
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        return GPTTransformer(model_config)
    
    def prepare_training(
        self,
        model: GPTTransformer,
        train_config: TrainingConfig,
        device: str
    ) -> tuple:
        """
        Prepare components for training.
        
        Args:
            model: Model to train
            train_config: Training configuration
            device: Device to use
            
        Returns:
            Tuple of (train_loader, val_loader, optimizer)
        """
        # Load data
        if self.verbose:
            print(f"\n📊 Loading data...")
            print(f"  Train: {train_config.train_data_path}")
            print(f"  Val: {train_config.val_data_path}")
        
        train_loader, val_loader = create_dataloaders(
            train_path=Path(train_config.train_data_path),
            val_path=Path(train_config.val_data_path),
            batch_size=train_config.batch_size,
            num_workers=0,
        )
        
        if self.verbose:
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
        
        # Create optimizer
        if self.verbose:
            print(f"\n⚙️  Setting up optimizer...")
            print(f"  Learning rate: {train_config.learning_rate}")
            print(f"  Weight decay: {train_config.weight_decay}")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=train_config.betas,
            eps=train_config.eps,
            weight_decay=train_config.weight_decay,
        )
        
        return train_loader, val_loader, optimizer
    
    def execute_training(
        self,
        model: GPTTransformer,
        train_loader,
        val_loader,
        optimizer,
        train_config: TrainingConfig,
        device: str,
        model_config_dict: dict,
        observers: Optional[List[TrainingObserver]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the training process.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            train_config: Training configuration
            device: Device to use
            model_config_dict: Model configuration as dictionary
            observers: Optional list of TrainingObserver instances.
                       If None, default observers (MetricsTracker) will be used.
            
        Returns:
            Dictionary with training results
        """
        # Set up default observers if none provided
        if observers is None:
            observers = self._create_default_observers(train_config)
        
        # Create trainer with observers
        if self.verbose:
            print(f"\n🎯 Initializing trainer...")
            print(f"   Observers: {len(observers)} ({', '.join(o.name for o in observers)})")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=train_config,
            device=device,
            observers=observers,
        )
        
        # Resume if requested
        if hasattr(train_config, 'resume_from') and train_config.resume_from is not None:
            if self.verbose:
                print(f"\n📥 Resuming from checkpoint: {train_config.resume_from}")
            trainer.resume_from_checkpoint(Path(train_config.resume_from))
        elif train_config.checkpoint_dir and hasattr(train_config, 'checkpoint_dir'):
            # Check if there's a resume_from in the checkpoint dir
            resume_path = Path(train_config.checkpoint_dir) / "resume_from.pt"
            if resume_path.exists() and self.verbose:
                print(f"\n📥 Found checkpoint to resume: {resume_path}")
        
        # Start training
        if self.verbose:
            print(f"\n{'='*60}")
            print("🚀 Starting Training!")
            print(f"{'='*60}\n")
        
        interrupted = False
        try:
            trainer.train()
        except KeyboardInterrupt:
            interrupted = True
            if self.verbose:
                print("\n\n⏸️  Training interrupted by user")
                print("💾 Saving checkpoint...")
            trainer.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=trainer.global_step,
                epoch=trainer.current_epoch,
                val_loss=trainer.best_val_loss,
                model_config=model_config_dict,
                train_config=train_config.to_dict(),
            )
            if self.verbose:
                print("✓ Checkpoint saved. Resume with resume_from in config.")
        
        # Generate observability reports (on BOTH normal and abnormal exit)
        self._generate_observability_reports(
            observers=observers,
            train_config=train_config,
            trainer=trainer,
            interrupted=interrupted,
        )
        
        # Return results
        final_checkpoint = Path(train_config.checkpoint_dir) / "final_model.pt"
        
        results = {
            'final_checkpoint': str(final_checkpoint),
            'best_checkpoint': str(final_checkpoint),  # Alias for backward compatibility
            'final_val_loss': trainer.best_val_loss,
            'total_epochs': trainer.current_epoch,
            'total_steps': trainer.global_step,
            'checkpoint_dir': train_config.checkpoint_dir,
            'log_dir': train_config.log_dir,
            'interrupted': interrupted,
        }
        
        # Get training issues from metrics tracker
        metrics_tracker = self._get_metrics_tracker(observers)
        if metrics_tracker:
            results['training_issues'] = metrics_tracker.detect_issues()
        
        if self.verbose:
            status = "⏸️  Training Interrupted" if interrupted else "✅ Training Complete!"
            print(f"\n{'='*60}")
            print(status)
            print(f"{'='*60}")
            print(f"\n📁 Results:")
            print(f"  Final checkpoint: {results['final_checkpoint']}")
            print(f"  Best val loss: {results['final_val_loss']:.4f}")
            print(f"  Total steps: {results['total_steps']}")
            print(f"  Total epochs: {results['total_epochs']}")
            print(f"  Logs: {results['log_dir']}")
        
        return results
    
    def _generate_observability_reports(
        self,
        observers: List[TrainingObserver],
        train_config: TrainingConfig,
        trainer,
        interrupted: bool = False,
    ) -> None:
        """
        Generate observability reports from metrics tracker.
        
        Called on both normal completion and abnormal exit (Ctrl+C).
        
        Args:
            observers: List of training observers
            train_config: Training configuration
            trainer: Trainer instance
            interrupted: Whether training was interrupted
        """
        metrics_tracker = self._get_metrics_tracker(observers)
        
        if not metrics_tracker:
            if self.verbose:
                print("\n⚠️  No MetricsTracker found - skipping observability reports")
            return
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("📊 Generating Observability Reports")
            print(f"{'='*60}")
        
        try:
            # Export metrics to CSV
            csv_path = metrics_tracker.export_to_csv()
            if self.verbose:
                print(f"   ✓ CSV exported: {csv_path}")
        except Exception as e:
            if self.verbose:
                print(f"   ✗ CSV export failed: {e}")
        
        try:
            # Export metrics to JSON
            json_path = metrics_tracker.export_to_json()
            if self.verbose:
                print(f"   ✓ JSON exported: {json_path}")
        except Exception as e:
            if self.verbose:
                print(f"   ✗ JSON export failed: {e}")
        
        try:
            # Generate loss curve plots
            plot_path = metrics_tracker.plot_loss_curves()
            if plot_path and self.verbose:
                print(f"   ✓ Loss curves plotted: {plot_path}")
        except ImportError:
            if self.verbose:
                print(f"   ⚠️  Plotting skipped (matplotlib not installed)")
        except Exception as e:
            if self.verbose:
                print(f"   ✗ Plotting failed: {e}")
        
        # Training health check
        issues = metrics_tracker.detect_issues()
        if self.verbose:
            print(f"\n📋 Training Health Check:")
            for issue in issues:
                print(f"   {issue}")
        
        # Add interrupted notice if applicable
        if interrupted and self.verbose:
            print(f"\n⚠️  Note: Training was interrupted at step {trainer.global_step}")
            print(f"   Reports reflect partial training data only.")
    
    def _create_default_observers(self, train_config: TrainingConfig) -> List[TrainingObserver]:
        """
        Create default observers for training.
        
        Args:
            train_config: Training configuration
            
        Returns:
            List of default TrainingObserver instances
        """
        observers = []
        
        # MetricsTracker - comprehensive metrics logging
        metrics_tracker = MetricsTracker(
            log_dir=train_config.log_dir,
            experiment_name="gptmed_training",
            moving_avg_window=100,
            log_interval=train_config.log_interval,
            verbose=self.verbose,
        )
        observers.append(metrics_tracker)
        
        # Note: ConsoleCallback is optional since Trainer already has console output
        # Uncomment if you want additional formatted console output:
        # console_callback = ConsoleCallback(log_interval=train_config.log_interval)
        # observers.append(console_callback)
        
        return observers
    
    def _get_metrics_tracker(self, observers: List[TrainingObserver]) -> Optional[MetricsTracker]:
        """
        Get MetricsTracker from observers list if present.
        
        Args:
            observers: List of observers
            
        Returns:
            MetricsTracker instance or None
        """
        for obs in observers:
            if isinstance(obs, MetricsTracker):
                return obs
        return None
        
        return results
    
    def train(
        self,
        model_size: str,
        train_data_path: str,
        val_data_path: str,
        batch_size: int = 16,
        learning_rate: float = 3e-4,
        num_epochs: int = 10,
        checkpoint_dir: str = "./model/checkpoints",
        log_dir: str = "./logs",
        seed: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        High-level training interface.
        
        Args:
            model_size: Model size ('tiny', 'small', 'medium')
            train_data_path: Path to training data
            val_data_path: Path to validation data
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
            seed: Random seed
            **kwargs: Additional training config parameters
            
        Returns:
            Dictionary with training results
        """
        # Set seed
        if self.verbose:
            print(f"\n🎲 Setting random seed: {seed}")
        self.set_seed(seed)
        
        # Get device
        device = self.device_manager.get_device()
        self.device_manager.print_device_info(verbose=self.verbose)
        
        # Create model
        if self.verbose:
            print(f"\n🧠 Creating model: {model_size}")
        
        model = self.create_model(model_size)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if self.verbose:
            print(f"  Model size: {model_size}")
            print(f"  Parameters: {total_params:,}")
            print(f"  Memory: ~{total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Create training config
        train_config = TrainingConfig(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            device=device,
            seed=seed,
            **{k: v for k, v in kwargs.items() if hasattr(TrainingConfig, k)}
        )
        
        # Prepare training components
        train_loader, val_loader, optimizer = self.prepare_training(
            model, train_config, device
        )
        
        # Execute training
        results = self.execute_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            train_config=train_config,
            device=device,
            model_config_dict=model.config.to_dict()
        )
        
        return results
