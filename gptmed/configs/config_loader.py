"""
Configuration File Loader

Load training configuration from YAML file for easy user customization.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['model', 'data', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate model size
    valid_sizes = ['tiny', 'small', 'medium']
    if config['model']['size'] not in valid_sizes:
        raise ValueError(f"Invalid model size: {config['model']['size']}. "
                        f"Must be one of {valid_sizes}")
    
    # Validate data paths
    train_path = Path(config['data']['train_data'])
    val_path = Path(config['data']['val_data'])
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    # Validate training parameters
    if config['training']['num_epochs'] <= 0:
        raise ValueError("num_epochs must be positive")
    if config['training']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    if config['training']['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")
    
    # Validate device
    valid_devices = ['cuda', 'cpu', 'auto']
    device_value = config.get('device', {}).get('device', 'cuda').lower()
    if device_value not in valid_devices:
        raise ValueError(
            f"Invalid device: {device_value}. "
            f"Must be one of {valid_devices}"
        )


def config_to_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert YAML config to training arguments.
    
    Args:
        config: Configuration dictionary from YAML
        
    Returns:
        Flattened dictionary suitable for training
    """
    args = {
        # Model
        'model_size': config['model']['size'],
        
        # Data
        'train_data': config['data']['train_data'],
        'val_data': config['data']['val_data'],
        
        # Training
        'num_epochs': config['training']['num_epochs'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'grad_clip': config['training']['grad_clip'],
        'warmup_steps': config['training']['warmup_steps'],
        
        # Optimizer
        'betas': tuple(config['optimizer']['betas']),
        'eps': config['optimizer']['eps'],
        
        # Checkpointing
        'checkpoint_dir': config['checkpointing']['checkpoint_dir'],
        'save_interval': config['checkpointing'].get('save_interval', config['checkpointing'].get('save_every', 1)),
        'keep_last_n': config['checkpointing']['keep_last_n'],
        
        # Logging
        'log_dir': config['logging']['log_dir'],
        'eval_interval': config['logging'].get('eval_interval', config['logging'].get('eval_every', 100)),
        'log_interval': config['logging'].get('log_interval', config['logging'].get('log_every', 10)),
        
        # Device
        'device': config['device']['device'],
        'seed': config['device']['seed'],
        
        # Advanced
        'max_steps': config.get('advanced', {}).get('max_steps', -1),
        'resume_from': config.get('advanced', {}).get('resume_from'),
        'quick_test': config.get('advanced', {}).get('quick_test', False),
    }
    
    return args


def create_default_config_file(output_path: str = 'training_config.yaml') -> None:
    """
    Create a default configuration file template.
    
    Args:
        output_path: Path where to save the config file
    """
    default_config = {
        'model': {
            'size': 'small'
        },
        'data': {
            'train_data': './data/tokenized/train.npy',
            'val_data': './data/tokenized/val.npy'
        },
        'training': {
            'num_epochs': 10,
            'batch_size': 16,
            'learning_rate': 0.0003,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'warmup_steps': 100
        },
        'optimizer': {
            'betas': [0.9, 0.95],
            'eps': 1.0e-8
        },
        'checkpointing': {
            'checkpoint_dir': './model/checkpoints',
            'save_interval': 1,
            'keep_last_n': 3
        },
        'logging': {
            'log_dir': './logs',
            'eval_interval': 100,
            'log_interval': 10
        },
        'device': {
            'device': 'cuda',  # Options: 'cuda', 'cpu', or 'auto'
            'seed': 42
        },
        'advanced': {
            'max_steps': -1,
            'resume_from': None,
            'quick_test': False
        }
    }
    
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created default configuration file: {output_path}")
    print(f"  Edit this file and then run: gptmed.train_from_config('{output_path}')")
