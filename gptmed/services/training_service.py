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
from typing import Dict, Any, Optional

from gptmed.services.device_manager import DeviceManager
from gptmed.model.architecture import GPTTransformer
from gptmed.model.configs.model_config import get_tiny_config, get_small_config, get_medium_config
from gptmed.configs.train_config import TrainingConfig
from gptmed.training.dataset import create_dataloaders
from gptmed.training.trainer import Trainer


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
        model_config_dict: dict
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
            
        Returns:
            Dictionary with training results
        """
        # Create trainer
        if self.verbose:
            print(f"\n🎯 Initializing trainer...")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=train_config,
            device=device,
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
        
        try:
            trainer.train()
        except KeyboardInterrupt:
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
        
        # Return results
        best_checkpoint = Path(train_config.checkpoint_dir) / "best_model.pt"
        
        results = {
            'best_checkpoint': str(best_checkpoint),
            'final_val_loss': trainer.best_val_loss,
            'total_epochs': trainer.current_epoch,
            'checkpoint_dir': train_config.checkpoint_dir,
            'log_dir': train_config.log_dir,
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("✅ Training Complete!")
            print(f"{'='*60}")
            print(f"\n📁 Results:")
            print(f"  Best checkpoint: {results['best_checkpoint']}")
            print(f"  Best val loss: {results['final_val_loss']:.4f}")
            print(f"  Total epochs: {results['total_epochs']}")
            print(f"  Logs: {results['log_dir']}")
        
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
