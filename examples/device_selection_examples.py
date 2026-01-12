"""
Quick Example: Device Selection

This demonstrates the three ways to select training device.
"""

# Example 1: Via API Parameter (Recommended)
# ============================================
import gptmed

# Option A: Force CPU training
print("Example 1a: Force CPU training")
results = gptmed.train_from_config(
    config_path='config.yaml',
    device='cpu',  # Override to CPU
    verbose=True
)

# Option B: Force GPU training (with automatic fallback to CPU if unavailable)
print("\nExample 1b: GPU training with fallback")
results = gptmed.train_from_config(
    config_path='config.yaml',
    device='cuda',  # Try GPU, fallback to CPU
    verbose=True
)

# Option C: Automatic device selection
print("\nExample 1c: Auto device selection")
results = gptmed.train_from_config(
    config_path='config.yaml',
    device='auto',  # Let system choose best
    verbose=True
)


# Example 2: Via Config File
# ===========================
# Edit your config.yaml:
"""
device:
  device: cpu  # Options: 'cuda', 'cpu', 'auto'
  seed: 42
"""

# Then train normally:
print("\nExample 2: Using config file setting")
results = gptmed.train_from_config('config.yaml')


# Example 3: Using Services Directly (Advanced)
# ===============================================
from gptmed.services.device_manager import DeviceManager
from gptmed.services.training_service import TrainingService

print("\nExample 3: Direct service usage")

# Create device manager for CPU
device_manager = DeviceManager(
    preferred_device='cpu',
    allow_fallback=True
)

# Create training service
training_service = TrainingService(
    device_manager=device_manager,
    verbose=True
)

# Train using the service
results = training_service.train(
    model_size='small',
    train_data_path='./data/tokenized/train.npy',
    val_data_path='./data/tokenized/val.npy',
    num_epochs=10,
    batch_size=16,
    learning_rate=3e-4,
    checkpoint_dir='./model/checkpoints',
    log_dir='./logs',
    seed=42
)

print(f"\nTraining complete! Best model: {results['best_checkpoint']}")
