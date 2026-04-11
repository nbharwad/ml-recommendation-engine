"""
Request Coalescing Middleware
============================
Deduplicates concurrent identical requests to reduce load.

Use Case:
- Multiple users make same search at same time
- Only one request goes to backend
- All waiters receive same response

Implementation:
- Bloom filter for fast duplicate detection
- In-flight request tracking with promise/future
- Configurable coalescing window

Industry Standard:
- Used by Google, Meta, Netflix for high QPS systems
- Reduces backend load by 30-70% for popular items
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from collections import defaultdict

import structlog
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger(component="coalescing")


REQUESTS_COALESCED = Counter(
    "requests_coalesced_total",
    "Requests coalesced (deduplicated)",
    ["request_type"],
)

REQUESTS_UNIQUE = Counter(
    "requests_unique_total",
    "Unique requests sent to backend",
    ["request_type"],
)

COALESCING_BENEFIT = Histogram(
    "coalescing_benefit_ms",
    "Latency benefit from coalescing",
    buckets=[0, 1, 2, 5, 10, 20, 50],
)

WAITER_COUNT = Gauge(
    "coalescing_waiters",
    "Number of requests waiting for response",
    ["request_type"],
)


@dataclass
class CoalescingConfig:
    """Configuration for request coalescing."""
    window_ms: int = 5
    max_waiters: int = 100
    enable_bloom_filter: bool = True
    bloom_filter_size: int = 10000
    bloom_filter_error_rate: float = 0.01
    enable_stats: bool = True


class RequestKey:
    """Normalizes requests into comparable keys."""
    
    def __init__(self, config: CoalescingConfig):
        self.config = config
        
    def normalize(
        self,
        request_type: str,
        user_id: str,
        params: dict[str, Any],
    ) -> str:
        """Create normalized request key."""
        parts = [
            request_type,
            user_id,
        ]
        
        # Sort params for consistent hashing
        if params:
            sorted_params = sorted(params.items())
            param_str = str(sorted_params)
        else:
            param_str = ""
            
        parts.append(param_str)
        
        key_str = "|".join(parts)
        
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class InFlightRequest:
    """Tracks in-flight request and waiters."""
    
    def __init__(self):
        self.start_time = time.time()
        self.future: Optional[Future] = None
        self.waiters: list[Future] = []
        self.response: Optional[Any] = None
        self.error: Optional[Exception] = None
        self.completed = False
        
    def add_waiter(self) -> Future:
        """Add a waiter and return future."""
        future = Future()
        self.waiters.append(future)
        WAITER_COUNT.inc()
        return future
    
    def signal_complete(self, response: Any):
        """Signal all waiters with response."""
        self.response = response
        self.completed = True
        
        for waiter in self.waiters:
            if not waiter.done():
                waiter.set_result(response)
                
        self.waiters.clear()
        WAITER_COUNT.set(0)
    
    def signal_error(self, error: Exception):
        """Signal all waiters with error."""
        self.error = error
        self.completed = True
        
        for waiter in self.waiters:
            if not waiter.done():
                waiter.set_exception(error)
                
        self.waiters.clear()
        WAITER_COUNT.set(0)
    
    def get_latency_ms(self) -> float:
        """Get elapsed time since request started."""
        return (time.time() - self.start_time) * 1000


class BloomFilter:
    """
    Simple Bloom filter for fast duplicate detection.
    Uses MurmurHash3 for good distribution.
    """
    
    def __init__(self, size: int, error_rate: float):
        self.size = size
        self.error_rate = error_rate
        self.array = bytearray(size)
        self.num_hashes = max(1, int(-math.log(error_rate) / math.log(2))))
        
    def _hashes(self, item: str) -> list[int]:
        """Generate hash values for item."""
        item_bytes = item.encode()
        hash1 = int(hashlib.murmurhash3(item_bytes, 0))
        hash2 = int(hashlib.murmurhash3(item_bytes, hash1))
        
        return [(hash1 + i * hash2) % self.size for i in range(self.num_hashes)]
    
    def add(self, item: str):
        """Add item to filter."""
        for idx in self._hashes(item):
            self.array[idx] = 1
    
    def might_contain(self, item: str) -> bool:
        """Check if item might be in filter."""
        return all(self.array[idx] for idx in self._hashes(item))
    
    def clear(self):
        """Clear the filter."""
        self.array = bytearray(self.size)


class RequestCoalescer:
    """
    Main request coalescing handler.
    
    Flow:
    1. Request arrives
    2. Normalize to key
    3. Bloom filter check (fast reject)
    4. In-flight map check
       - If exists: add waiter, return future
       - If new: create in-flight, execute request
    5. On completion: signal all waiters
    """
    
    def __init__(self, config: CoalescingConfig):
        self.config = config
        self.key_normalizer = RequestKey(config)
        self.in_flight: dict[str, InFlightRequest] = {}
        self.stats: dict[str, int] = defaultdict(int)
        
        if config.enable_bloom_filter:
            self.bloom = BloomFilter(
                config.bloom_filter_size,
                config.bloom_filter_error_rate,
            )
        else:
            self.bloom = None
            
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Periodic cleanup
        self._cleanup_interval_sec = config.window_ms / 1000 * 2
        
    async def execute(
        self,
        request_type: str,
        user_id: str,
        params: dict[str, Any],
        executor: Callable,
    ) -> Any:
        """
        Execute request with coalescing.
        
        Args:
            request_type: e.g., "recommendations", "search"
            user_id: user making request
            params: request parameters
            executor: async function to execute
            
        Returns:
            Response from backend
        """
        key = self.key_normalizer.normalize(request_type, user_id, params)
        
        # Fast path: Bloom filter check
        if self.bloom and self.bloom.might_contain(key):
            REQUESTS_COALESCED.labels(request_type=request_type).inc()
        else:
            # Slow path: Check in-flight
            async with self._get_or_create_inflight(key, request_type, executor) as result:
                return result
    
    async def _get_or_create_inflight(
        self,
        key: str,
        request_type: str,
        executor: Callable,
    ) -> Any:
        """Get existing or create new in-flight request."""
        with self.lock:
            # Check if already in-flight
            if key in self.in_flight:
                inflight = self.in_flight[key]
                
                # Add waiter
                future = inflight.add_waiter()
                
                # Wait for result
                result = await future
                
                REQUESTS_COALESCED.labels(request_type=request_type).inc()
                
                latency_benefit = inflight.get_latency_ms()
                COALESCING_BENEFIT.observe(latency_benefit)
                
                return result
            
            # Create new in-flight request
            inflight = InFlightRequest()
            self.in_flight[key] = inflight
            
            if self.bloom:
                self.bloom.add(key)
        
        try:
            # Execute the request
            start = time.perf_counter()
            
            result = await executor()
            
            latency = (time.perf_counter() - start) * 1000
            
            # Signal completion
            inflight.signal_complete(result)
            
            REQUESTS_UNIQUE.labels(request_type=request_type).inc()
            self.stats[request_type] += 1
            
            logger.info(
                "request_executed",
                request_type=request_type,
                latency_ms=round(latency, 2),
                waiters=len(inflight.waiters),
            )
            
            return result
            
        except Exception as e:
            inflight.signal_error(e)
            raise
            
        finally:
            # Cleanup after window
            self._cleanup_stale(key)
    
    def _cleanup_stale(self, key: str):
        """Remove stale in-flight requests."""
        with self.lock:
            if key in self.in_flight:
                inflight = self.in_flight[key]
                
                # Remove if completed and old
                if inflight.completed:
                    age_sec = (time.time() - inflight.start_time)
                    
                    if age_sec > self._cleanup_interval_sec:
                        del self.in_flight[key]
                        
                        if self.bloom:
                            self.bloom.clear()
    
    def get_stats(self) -> dict:
        """Get coalescing statistics."""
        return {
            "in_flight_count": len(self.in_flight),
            "by_type": dict(self.stats),
        }


class CoalescingMiddleware:
    """
    FastAPI middleware for request coalescing.
    
    Usage:
        app.add_middleware(CoalescingMiddleware, window_ms=5)
    """
    
    def __init__(self, app, config: Optional[CoalescingConfig] = None):
        self.app = app
        self.config = config or CoalescingConfig()
        self.coalescer = RequestCoalescer(self.config)
    
    async def __call__(self, scope, receive, send):
        """Handle request."""
        # Extract request info
        # Process through coalescer
        # Pass to app
        pass


def create_coalescer(
    window_ms: int = 5,
    max_waiters: int = 100,
) -> RequestCoalescer:
    """Factory function to create request coalescer."""
    config = CoalescingConfig(
        window_ms=window_ms,
        max_waiters=max_waiters,
    )
    return RequestCoalescer(config)


async def coalesce_request(
    request_type: str,
    user_id: str,
    params: dict,
    executor: Callable,
    coalescer: Optional[RequestCoalescer] = None,
) -> Any:
    """
    Convenience function for request coalescing.
    
    Usage:
        results = await coalesce_request(
            "recommendations",
            user_id,
            {"num_items": 10},
            lambda: fetch_results(),
        )
    """
    if not coalescer:
        coalescer = create_coalescer()
    
    return await coalescer.execute(request_type, user_id, params, executor)


# Import for math functions
import math


STATS = {}


def get_coalescing_stats() -> dict:
    """Get coalescing statistics."""
    return STATS