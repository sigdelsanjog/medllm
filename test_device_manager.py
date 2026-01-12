"""
Device Manager Demo

Demonstrates the DeviceManager service for device selection and management.
"""

from gptmed.services.device_manager import DeviceManager

print("="*60)
print("Device Manager Service Demo")
print("="*60)

# Example 1: Create DeviceManager with CUDA preference
print("\n1. DeviceManager with CUDA preference:")
print("-" * 60)
device_mgr_cuda = DeviceManager(preferred_device='cuda', allow_fallback=True)
device_mgr_cuda.print_device_info(verbose=True)
print(f"Selected device: {device_mgr_cuda.get_device()}")

# Example 2: Create DeviceManager with CPU preference
print("\n2. DeviceManager with CPU preference:")
print("-" * 60)
device_mgr_cpu = DeviceManager(preferred_device='cpu')
device_mgr_cpu.print_device_info(verbose=True)
print(f"Selected device: {device_mgr_cpu.get_device()}")

# Example 3: Get optimal device automatically
print("\n3. Optimal Device Selection:")
print("-" * 60)
optimal = DeviceManager.get_optimal_device()
print(f"Optimal device for this system: {optimal}")

# Example 4: Validate device strings
print("\n4. Device Validation:")
print("-" * 60)
test_devices = ['cuda', 'cpu', 'auto', 'CUDA', 'CPU']
for dev in test_devices:
    try:
        validated = DeviceManager.validate_device(dev)
        print(f"  '{dev}' -> '{validated}' ✓")
    except ValueError as e:
        print(f"  '{dev}' -> ERROR: {e}")
