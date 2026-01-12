"""
Device Manager Service

PURPOSE:
Manages device selection and configuration for model training and inference.
Implements Strategy Pattern for flexible device handling.

DESIGN PATTERNS:
- Strategy Pattern: Different strategies for CPU vs GPU
- Dependency Injection: DeviceManager can be injected into services
- Single Responsibility: Only handles device-related concerns

WHAT THIS FILE DOES:
1. Validates device availability (CUDA check)
2. Provides device selection logic with fallback
3. Manages device-specific configurations
4. Ensures consistent device handling across the codebase

PACKAGES USED:
- torch: Device detection and management
- abc: Abstract base classes for strategy pattern
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch


class DeviceStrategy(ABC):
    """
    Abstract base class for device strategies.
    Implements Strategy Pattern for different device types.
    """
    
    @abstractmethod
    def get_device(self) -> str:
        """
        Get the device string for PyTorch.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the device is available.
        
        Returns:
            True if device is available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_device_info(self) -> dict:
        """
        Get information about the device.
        
        Returns:
            Dictionary with device information
        """
        pass


class CUDAStrategy(DeviceStrategy):
    """Strategy for CUDA/GPU devices."""
    
    def get_device(self) -> str:
        """Get CUDA device if available."""
        return 'cuda' if self.is_available() else 'cpu'
    
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    def get_device_info(self) -> dict:
        """Get CUDA device information."""
        if not self.is_available():
            return {
                'device': 'cuda',
                'available': False,
                'message': 'CUDA not available'
            }
        
        return {
            'device': 'cuda',
            'available': True,
            'device_name': torch.cuda.get_device_name(0),
            'device_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda if torch.version.cuda else 'N/A',
        }


class CPUStrategy(DeviceStrategy):
    """Strategy for CPU devices."""
    
    def get_device(self) -> str:
        """Always return CPU."""
        return 'cpu'
    
    def is_available(self) -> bool:
        """CPU is always available."""
        return True
    
    def get_device_info(self) -> dict:
        """Get CPU device information."""
        return {
            'device': 'cpu',
            'available': True,
            'num_threads': torch.get_num_threads(),
        }


class DeviceManager:
    """
    Manages device selection and configuration.
    
    Follows Single Responsibility Principle - only handles device concerns.
    Uses Strategy Pattern for different device types.
    
    Example:
        >>> device_manager = DeviceManager(preferred_device='cuda')
        >>> device = device_manager.get_device()
        >>> print(f"Using device: {device}")
    """
    
    def __init__(self, preferred_device: str = 'cuda', allow_fallback: bool = True):
        """
        Initialize DeviceManager.
        
        Args:
            preferred_device: Preferred device ('cuda' or 'cpu')
            allow_fallback: If True, fallback to CPU if CUDA unavailable
        """
        self.preferred_device = preferred_device.lower()
        self.allow_fallback = allow_fallback
        
        # Validate device input
        if self.preferred_device not in ['cuda', 'cpu']:
            raise ValueError(
                f"Invalid device: {preferred_device}. Must be 'cuda' or 'cpu'"
            )
        
        # Select strategy based on preferred device
        if self.preferred_device == 'cuda':
            self.strategy = CUDAStrategy()
        else:
            self.strategy = CPUStrategy()
    
    def get_device(self) -> str:
        """
        Get the actual device to use.
        
        Returns fallback device if preferred is unavailable and fallback is allowed.
        
        Returns:
            Device string ('cuda' or 'cpu')
            
        Raises:
            RuntimeError: If preferred device unavailable and fallback disabled
        """
        if self.strategy.is_available():
            return self.strategy.get_device()
        
        # Handle unavailable device
        if self.allow_fallback and self.preferred_device == 'cuda':
            # Fallback to CPU
            return 'cpu'
        else:
            raise RuntimeError(
                f"Device '{self.preferred_device}' is not available and "
                f"fallback is {'disabled' if not self.allow_fallback else 'not applicable'}"
            )
    
    def get_device_info(self) -> dict:
        """
        Get information about the current device.
        
        Returns:
            Dictionary with device information
        """
        info = self.strategy.get_device_info()
        info['preferred_device'] = self.preferred_device
        info['actual_device'] = self.get_device()
        info['allow_fallback'] = self.allow_fallback
        return info
    
    def print_device_info(self, verbose: bool = True) -> None:
        """
        Print device information.
        
        Args:
            verbose: If True, print detailed information
        """
        if not verbose:
            return
        
        info = self.get_device_info()
        actual = info['actual_device']
        preferred = info['preferred_device']
        
        print(f"\n💻 Device Configuration:")
        print(f"  Preferred: {preferred}")
        print(f"  Using: {actual}")
        
        if preferred != actual:
            print(f"  ⚠️  Fallback to CPU (CUDA not available)")
        
        if actual == 'cuda' and info.get('available'):
            print(f"  GPU: {info.get('device_name', 'Unknown')}")
            print(f"  CUDA Version: {info.get('cuda_version', 'N/A')}")
            print(f"  GPU Count: {info.get('device_count', 0)}")
        elif actual == 'cpu':
            print(f"  CPU Threads: {info.get('num_threads', 'N/A')}")
    
    @staticmethod
    def validate_device(device: str) -> str:
        """
        Validate and normalize device string.
        
        Args:
            device: Device string to validate
            
        Returns:
            Normalized device string
            
        Raises:
            ValueError: If device is invalid
        """
        device = device.lower().strip()
        
        if device not in ['cuda', 'cpu', 'auto']:
            raise ValueError(
                f"Invalid device: '{device}'. Must be 'cuda', 'cpu', or 'auto'"
            )
        
        # Auto-select best available device
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return device
    
    @staticmethod
    def get_optimal_device() -> str:
        """
        Get the optimal device for the current environment.
        
        Returns:
            'cuda' if available, otherwise 'cpu'
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'
