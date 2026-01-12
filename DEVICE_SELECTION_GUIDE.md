# Device Selection Guide

This guide explains how to configure CPU or GPU training in the GPTMed package.

## Overview

The GPTMed package now supports flexible device selection through:
- **DeviceManager Service**: Strategy pattern implementation for device management
- **TrainingService**: Service layer that uses DeviceManager for training orchestration
- **API Enhancement**: `train_from_config()` now accepts a `device` parameter

## Design Patterns Used

### 1. Strategy Pattern (DeviceManager)
The DeviceManager implements the Strategy pattern to handle different device types:
- **CUDAStrategy**: Handles GPU/CUDA devices
- **CPUStrategy**: Handles CPU devices

### 2. Service Layer Pattern (TrainingService)
TrainingService encapsulates training logic and uses Dependency Injection to accept a DeviceManager instance.

### 3. Single Responsibility Principle
- DeviceManager: Only handles device selection and validation
- TrainingService: Only orchestrates training
- API Layer: Provides user-friendly interface

## Usage

### Method 1: Using API Parameter (Recommended)

Override device directly in the API call:

```python
import gptmed

# Train on CPU
results = gptmed.train_from_config(
    config_path='config.yaml',
    device='cpu'
)

# Train on GPU
results = gptmed.train_from_config(
    config_path='config.yaml',
    device='cuda'
)

# Auto-select best available device
results = gptmed.train_from_config(
    config_path='config.yaml',
    device='auto'  # Uses CUDA if available, else CPU
)
```

### Method 2: Using Configuration File

Set device in your YAML config file:

```yaml
device:
  device: cpu  # Options: 'cuda', 'cpu', or 'auto'
  seed: 42
```

Then train normally:

```python
import gptmed

results = gptmed.train_from_config('config.yaml')
```

### Method 3: Using DeviceManager Directly

For advanced use cases:

```python
from gptmed.services.device_manager import DeviceManager
from gptmed.services.training_service import TrainingService

# Create DeviceManager with CPU preference
device_manager = DeviceManager(
    preferred_device='cpu',
    allow_fallback=True
)

# Create TrainingService with DeviceManager
training_service = TrainingService(
    device_manager=device_manager,
    verbose=True
)

# Use the service for training
results = training_service.train(
    model_size='small',
    train_data_path='./data/train.npy',
    val_data_path='./data/val.npy',
    num_epochs=10,
    batch_size=16
)
```

## Device Options

| Device Value | Behavior |
|-------------|----------|
| `'cuda'` | Use GPU if available, fallback to CPU if not |
| `'cpu'` | Always use CPU |
| `'auto'` | Automatically select best available (CUDA if available, else CPU) |

## API Reference

### DeviceManager

```python
DeviceManager(preferred_device='cuda', allow_fallback=True)
```

**Methods:**
- `get_device()`: Returns the actual device to use
- `get_device_info()`: Returns detailed device information
- `print_device_info(verbose=True)`: Prints device information
- `validate_device(device)`: Static method to validate device string
- `get_optimal_device()`: Static method to get optimal device

**Example:**
```python
from gptmed.services.device_manager import DeviceManager

# Create manager
mgr = DeviceManager(preferred_device='cpu')

# Get device
device = mgr.get_device()  # Returns 'cpu'

# Get info
info = mgr.get_device_info()
```

### TrainingService

```python
TrainingService(device_manager=None, verbose=True)
```

**Methods:**
- `set_seed(seed)`: Set random seeds
- `create_model(model_size)`: Create model
- `prepare_training(...)`: Prepare training components
- `execute_training(...)`: Execute training
- `train(...)`: High-level training interface

### train_from_config (Updated)

```python
gptmed.train_from_config(config_path, verbose=True, device=None)
```

**Parameters:**
- `config_path` (str): Path to YAML config file
- `verbose` (bool): Print training progress (default: True)
- `device` (str, optional): Device to use ('cuda', 'cpu', 'auto'). Overrides config file.

**Returns:**
- Dictionary with training results

## Examples

### Example 1: CPU Training

```python
import gptmed

# Force CPU training
results = gptmed.train_from_config(
    'config.yaml',
    device='cpu'
)
```

### Example 2: GPU Training with Fallback

```python
import gptmed

# Try GPU, fallback to CPU if unavailable
results = gptmed.train_from_config(
    'config.yaml',
    device='cuda'
)
```

### Example 3: Automatic Selection

```python
import gptmed

# Let the system decide
results = gptmed.train_from_config(
    'config.yaml',
    device='auto'
)
```

## Configuration File Examples

### CPU Training Config

```yaml
model:
  size: small

data:
  train_data: ./data/tokenized/train.npy
  val_data: ./data/tokenized/val.npy

training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 0.0003
  weight_decay: 0.01
  grad_clip: 1.0
  warmup_steps: 100

optimizer:
  betas: [0.9, 0.95]
  eps: 1.0e-8

checkpointing:
  checkpoint_dir: ./model/checkpoints
  save_interval: 1
  keep_last_n: 3

logging:
  log_dir: ./logs
  eval_interval: 100
  log_interval: 10

device:
  device: cpu  # Force CPU
  seed: 42

advanced:
  max_steps: -1
  resume_from: null
  quick_test: false
```

### GPU Training Config

```yaml
# ... same as above ...

device:
  device: cuda  # Use GPU
  seed: 42
```

### Auto Device Config

```yaml
# ... same as above ...

device:
  device: auto  # Auto-select
  seed: 42
```

## Testing

### Test CPU Training

```bash
python gptmed-api/test_cpu_training.py
```

### Test Auto Device Selection

```bash
python gptmed-api/test_auto_device.py
```

### Test DeviceManager

```bash
python gptmed/test_device_manager.py
```

## Architecture

```
API Layer (api.py)
    ↓
TrainingService (services/training_service.py)
    ↓
DeviceManager (services/device_manager.py)
    ↓
DeviceStrategy (CPUStrategy / CUDAStrategy)
    ↓
PyTorch Device
```

## Benefits of This Implementation

1. **Separation of Concerns**: Device management is isolated from training logic
2. **Flexibility**: Easy to switch between CPU and GPU
3. **Extensibility**: Easy to add new device types (e.g., MPS for Apple Silicon)
4. **Testability**: DeviceManager and TrainingService can be tested independently
5. **User-Friendly**: Multiple ways to specify device preference
6. **Fallback Support**: Graceful degradation from GPU to CPU

## Error Handling

The system provides clear error messages:

```python
# Invalid device
results = gptmed.train_from_config('config.yaml', device='gpu')
# ValueError: Invalid device: 'gpu'. Must be 'cuda', 'cpu', or 'auto'

# CUDA not available with fallback disabled
device_mgr = DeviceManager('cuda', allow_fallback=False)
device = device_mgr.get_device()
# RuntimeError: Device 'cuda' is not available and fallback is disabled
```

## Backward Compatibility

The implementation is fully backward compatible:
- Existing code without `device` parameter continues to work
- Config files without device specification use default ('cuda')
- Original behavior is preserved when no device override is provided
