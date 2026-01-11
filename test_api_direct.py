"""
Test GPTMed API Directly

This script tests the gptmed API functions directly in the gptmed folder.
"""

import sys
from pathlib import Path

# Add gptmed to path to use local version
sys.path.insert(0, str(Path(__file__).parent))

import gptmed

print("="*60)
print("Testing GPTMed API Directly")
print("="*60)
print(f"\nGPTMed version: {gptmed.__version__}")
print(f"Available functions: {gptmed.__all__}\n")

# Step 1: Create a config file
print("Step 1: Creating configuration file...")
config_file = "test_api_config.yaml"
gptmed.create_config(config_file)
print(f"✅ Created: {config_file}\n")

# Step 2: Train the model
print("Step 2: Training model...")
print("Note: Using default config which points to ./data/tokenized/train.npy and val.npy")
print("Starting training...\n")

try:
    results = gptmed.train_from_config(
        config_path=config_file,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best checkpoint: {results['best_checkpoint']}")
    print(f"Best val loss: {results['final_val_loss']:.4f}")
    print(f"Total epochs: {results['total_epochs']}")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
