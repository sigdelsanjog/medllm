"""
High-Level API for GptMed

Simple, user-friendly functions to train and use GPT models.
This is the main interface users should use.

Example:
    >>> import gptmed
    >>> 
    >>> # Create a config file
    >>> gptmed.create_config('my_config.yaml')
    >>> 
    >>> # Edit my_config.yaml with your settings
    >>> 
    >>> # Train the model
    >>> gptmed.train_from_config('my_config.yaml')
    >>> 
    >>> # Generate text
    >>> answer = gptmed.generate(
    ...     checkpoint='model/checkpoints/best_model.pt',
    ...     prompt='Your question?',
    ...     tokenizer='tokenizer/my_tokenizer.model'
    ... )
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any

from gptmed.configs.config_loader import (
    load_yaml_config,
    validate_config,
    config_to_args,
    create_default_config_file
)
from gptmed.model.architecture import GPTTransformer
from gptmed.model.configs.model_config import get_tiny_config, get_small_config, get_medium_config
from gptmed.configs.train_config import TrainingConfig
from gptmed.training.dataset import create_dataloaders
from gptmed.training.trainer import Trainer
from gptmed.inference.generator import TextGenerator


def create_config(output_path: str = 'training_config.yaml') -> None:
    """
    Create a default training configuration file.
    
    This creates a YAML file that you can edit with your training settings.
    
    Args:
        output_path: Where to save the config file (default: 'training_config.yaml')
    
    Example:
        >>> import gptmed
        >>> gptmed.create_config('my_training_config.yaml')
        >>> # Now edit my_training_config.yaml with your settings
    """
    create_default_config_file(output_path)


def train_from_config(config_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Train a GPT model using a YAML configuration file.
    
    This is the simplest way to train a model. Just create a config file
    with create_config(), edit it with your settings, and pass it here.
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to print training progress (default: True)
    
    Returns:
        Dictionary with training results:
            - best_checkpoint: Path to best model checkpoint
            - final_val_loss: Final validation loss
            - total_epochs: Number of epochs trained
    
    Example:
        >>> import gptmed
        >>> 
        >>> # Create and edit config file
        >>> gptmed.create_config('config.yaml')
        >>> # ... edit config.yaml ...
        >>> 
        >>> # Train the model
        >>> results = gptmed.train_from_config('config.yaml')
        >>> print(f"Best model: {results['best_checkpoint']}")
    
    Raises:
        FileNotFoundError: If config file or data files don't exist
        ValueError: If configuration is invalid
    """
    if verbose:
        print("=" * 60)
        print("GptMed Training from Configuration File")
        print("=" * 60)
    
    # Load and validate config
    if verbose:
        print(f"\n📄 Loading configuration from: {config_path}")
    config = load_yaml_config(config_path)
    
    if verbose:
        print("✓ Configuration loaded")
        print("\n🔍 Validating configuration...")
    validate_config(config)
    
    if verbose:
        print("✓ Configuration valid")
    
    # Convert to arguments
    args = config_to_args(config)
    
    # Import here to avoid circular imports
    import random
    import numpy as np
    
    # Set random seed
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if verbose:
        print(f"\n🎲 Setting random seed: {args['seed']}")
    set_seed(args['seed'])
    
    # Check device
    device = args['device']
    if device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model config
    if verbose:
        print(f"\n🧠 Creating model: {args['model_size']}")
    
    if args['model_size'] == 'tiny':
        model_config = get_tiny_config()
    elif args['model_size'] == 'small':
        model_config = get_small_config()
    elif args['model_size'] == 'medium':
        model_config = get_medium_config()
    else:
        raise ValueError(f"Unknown model size: {args['model_size']}")
    
    # Create model
    model = GPTTransformer(model_config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"  Model size: {args['model_size']}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Memory: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Load data
    if verbose:
        print(f"\n📊 Loading data...")
        print(f"  Train: {args['train_data']}")
        print(f"  Val: {args['val_data']}")
    
    train_loader, val_loader = create_dataloaders(
        train_path=Path(args['train_data']),
        val_path=Path(args['val_data']),
        batch_size=args['batch_size'],
        num_workers=0,
    )
    
    if verbose:
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    
    # Create training config
    train_config = TrainingConfig(
        batch_size=args['batch_size'],
        learning_rate=args['learning_rate'],
        num_epochs=args['num_epochs'],
        warmup_steps=args['warmup_steps'],
        grad_clip=args['grad_clip'],
        weight_decay=args['weight_decay'],
        betas=args['betas'],
        eps=args['eps'],
        max_steps=args['max_steps'],
        save_interval=args['save_interval'],
        eval_interval=args['eval_interval'],
        log_interval=args['log_interval'],
        keep_last_n=args['keep_last_n'],
        train_data_path=args['train_data'],
        val_data_path=args['val_data'],
        checkpoint_dir=args['checkpoint_dir'],
        log_dir=args['log_dir'],
        device=device,
        seed=args['seed'],
    )
    
    # Create optimizer
    if verbose:
        print(f"\n⚙️  Setting up optimizer...")
        print(f"  Learning rate: {args['learning_rate']}")
        print(f"  Weight decay: {args['weight_decay']}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args['learning_rate'],
        betas=args['betas'],
        eps=args['eps'],
        weight_decay=args['weight_decay'],
    )
    
    # Create trainer
    if verbose:
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
    if args['resume_from'] is not None:
        if verbose:
            print(f"\n📥 Resuming from checkpoint: {args['resume_from']}")
        trainer.resume_from_checkpoint(Path(args['resume_from']))
    
    # Start training
    if verbose:
        print(f"\n{'='*60}")
        print("🚀 Starting Training!")
        print(f"{'='*60}\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        if verbose:
            print("\n\n⏸️  Training interrupted by user")
            print("💾 Saving checkpoint...")
        trainer.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            val_loss=trainer.best_val_loss,
            model_config=model_config.to_dict(),
            train_config=train_config.to_dict(),
        )
        if verbose:
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
    
    if verbose:
        print(f"\n{'='*60}")
        print("✅ Training Complete!")
        print(f"{'='*60}")
        print(f"\n📁 Results:")
        print(f"  Best checkpoint: {results['best_checkpoint']}")
        print(f"  Best val loss: {results['final_val_loss']:.4f}")
        print(f"  Total epochs: {results['total_epochs']}")
        print(f"  Logs: {results['log_dir']}")
    
    return results


def generate(
    checkpoint: str,
    tokenizer: str,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda"
) -> str:
    """
    Generate text using a trained model.
    
    Args:
        checkpoint: Path to model checkpoint (.pt file)
        tokenizer: Path to tokenizer model (.model file)
        prompt: Input text/question
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Generated text
    
    Example:
        >>> import gptmed
        >>> 
        >>> answer = gptmed.generate(
        ...     checkpoint='model/checkpoints/best_model.pt',
        ...     tokenizer='tokenizer/my_tokenizer.model',
        ...     prompt='What is machine learning?',
        ...     max_length=150,
        ...     temperature=0.7
        ... )
        >>> print(answer)
    """
    # Load checkpoint
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    
    # Load model config
    from gptmed.model.configs.model_config import ModelConfig
    model_config = ModelConfig.from_dict(checkpoint_data['model_config'])
    
    # Create and load model
    model = GPTTransformer(model_config)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    
    # Load tokenizer
    import sentencepiece as spm
    from gptmed.inference.generation_config import GenerationConfig
    
    tokenizer_sp = spm.SentencePieceProcessor()
    tokenizer_sp.Load(tokenizer)
    
    # Create generator
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer_sp,
        device=device
    )
    
    # Create generation config
    gen_config = GenerationConfig(
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    # Generate
    output = generator.generate(
        prompt=prompt,
        gen_config=gen_config
    )
    
    return output
