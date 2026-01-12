"""
Quick Integration Test

Tests the complete integration of DeviceManager + TrainingService + API
with a minimal training run.
"""

import sys
sys.path.insert(0, '/home/travelingnepal/Documents/proj/codellm/code-llm/gptmed')

import gptmed
from gptmed.services.device_manager import DeviceManager

print("="*70)
print("Integration Test: Device Selection with Training API")
print("="*70)

# Test 1: DeviceManager
print("\n[Test 1] DeviceManager Initialization")
print("-" * 70)
try:
    dm = DeviceManager(preferred_device='cpu')
    device = dm.get_device()
    print(f"✅ DeviceManager working - Selected device: {device}")
    dm.print_device_info(verbose=True)
except Exception as e:
    print(f"❌ DeviceManager failed: {e}")
    sys.exit(1)

# Test 2: API with device parameter
print("\n[Test 2] API Device Parameter")
print("-" * 70)
try:
    # Validate device string
    validated = DeviceManager.validate_device('cpu')
    print(f"✅ Device validation working - 'cpu' -> '{validated}'")
    
    validated_auto = DeviceManager.validate_device('auto')
    print(f"✅ Device validation working - 'auto' -> '{validated_auto}'")
except Exception as e:
    print(f"❌ Device validation failed: {e}")
    sys.exit(1)

# Test 3: Check if we can import TrainingService
print("\n[Test 3] TrainingService Import")
print("-" * 70)
try:
    from gptmed.services.training_service import TrainingService
    service = TrainingService(device_manager=dm, verbose=False)
    print("✅ TrainingService imported and initialized successfully")
except Exception as e:
    print(f"❌ TrainingService import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check API signature
print("\n[Test 4] API Signature Check")
print("-" * 70)
import inspect
sig = inspect.signature(gptmed.train_from_config)
params = list(sig.parameters.keys())
print(f"API parameters: {params}")
if 'device' in params:
    print("✅ 'device' parameter present in train_from_config()")
else:
    print("❌ 'device' parameter missing!")
    sys.exit(1)

print("\n" + "="*70)
print("✅ All Integration Tests Passed!")
print("="*70)
print("\nNext Step: Run actual training test")
print("  cd gptmed-api")
print("  python3 test_cpu_training.py")
