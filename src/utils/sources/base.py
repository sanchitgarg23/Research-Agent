"""Base interface and common utilities for multi-source research providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging
import time

from src.state import SearchResult, SourceType
from src.utils.web_utils import CircuitBreaker, HTTPClientManager
from src.exceptions import SearchError, RateLimitError, CircuitOpenError

logger = logging.getLogger(__name__)


class BaseSourceProvider(ABC):
    """Abstract base class for all research source providers."""
    
    def __init__(self, name: str, source_type: SourceType, max_results: int = 5, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self._name = name
        self._source_type = source_type
        self.max_results = max_results
        self.circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout
        )
        self.last_query_time: float = 0.0
        self.min_delay_seconds: float = 0.5

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @abstractmethod
    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Search the external source and return standardized SearchResult objects."""
        pass

    async def _enforce_rate_limit(self):
        """Ensure minimum delay between consecutive calls to the provider."""
        import asyncio
        elapsed = time.time() - self.last_query_time
        if elapsed < self.min_delay_seconds:
            await asyncio.sleep(self.min_delay_seconds - elapsed)
        self.last_query_time = time.time()

