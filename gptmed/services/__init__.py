"""
Services Layer

Business logic services following SOLID principles.
This layer implements the service pattern to encapsulate complex operations.
"""

from gptmed.services.device_manager import DeviceManager, DeviceStrategy
from gptmed.services.training_service import TrainingService

__all__ = [
    'DeviceManager',
    'DeviceStrategy',
    'TrainingService',
]
