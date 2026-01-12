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
from gptmed.services.device_manager import DeviceManager
from gptmed.services.training_service import TrainingService


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


def train_from_config(
    config_path: str, 
    verbose: bool = True, 
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a GPT model using a YAML configuration file.
    
    This is the simplest way to train a model. Just create a config file
    with create_config(), edit it with your settings, and pass it here.
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to print training progress (default: True)
        device: Device to use ('cuda', 'cpu', or 'auto'). If None, uses config value.
                'auto' will select best available device.
    
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
        >>> # Train the model on CPU
        >>> results = gptmed.train_from_config('config.yaml', device='cpu')
        >>> print(f"Best model: {results['best_checkpoint']}")
        >>> 
        >>> # Train with auto device selection
        >>> results = gptmed.train_from_config('config.yaml', device='auto')
    
    Raises:
        FileNotFoundError: If config file or data files don't exist
        ValueError: If configuration is invalid or device is invalid
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
    
    # Override device if provided as parameter
    if device is not None:
        # Validate and normalize device
        device = DeviceManager.validate_device(device)
        if verbose:
            print(f"\n⚙️  Device override: {device} (from parameter)")
        args['device'] = device
    
    # Create DeviceManager with the selected device
    device_manager = DeviceManager(
        preferred_device=args['device'],
        allow_fallback=True
    )
    
    # Print device information
    device_manager.print_device_info(verbose=verbose)
    
    # Create TrainingService with DeviceManager
    training_service = TrainingService(
        device_manager=device_manager,
        verbose=verbose
    )
    
    # Set random seed
    if verbose:
        print(f"\n🎲 Setting random seed: {args['seed']}")
    training_service.set_seed(args['seed'])
    
    # Get actual device to use
    actual_device = device_manager.get_device()
    
    # Load model config
    if verbose:
        print(f"\n🧠 Creating model: {args['model_size']}")
    
    model = training_service.create_model(args['model_size'])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"  Model size: {args['model_size']}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Memory: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Load data using TrainingService
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
    
    # Create training config with actual device
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
        device=actual_device,  # Use actual device from DeviceManager
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
    
    # Execute training using TrainingService
    results = training_service.execute_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        train_config=train_config,
        device=actual_device,
        model_config_dict=model.config.to_dict()
    )
    
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
