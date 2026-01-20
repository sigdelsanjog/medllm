"""
Redis client for real-time metrics storage.
Follows SOLID principles: single responsibility, dependency inversion, and interface segregation.
"""

import redis
from typing import Any, Dict
from gptmed.configs.configs import REDIS_CONFIG

class MetricsStorageInterface:
    """Interface for metrics storage backends."""
    def save_step_metrics(self, metrics: Dict[str, Any]):
        raise NotImplementedError
    def save_validation_metrics(self, metrics: Dict[str, Any]):
        raise NotImplementedError

class RedisMetricsStorage(MetricsStorageInterface):
    """Redis implementation for metrics storage."""
    def __init__(self):
        self.client = redis.Redis(**REDIS_CONFIG)
    def save_step_metrics(self, metrics: Dict[str, Any]):
        # Use a Redis list for steps
        self.client.rpush("training:steps", str(metrics))
    def save_validation_metrics(self, metrics: Dict[str, Any]):
        # Use a Redis list for validation
        self.client.rpush("training:validation", str(metrics))
