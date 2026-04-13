"""Base gRPC client with resilience patterns."""

import asyncio
import logging
import time
from typing import Any, Callable, Optional

import grpc

logger = logging.getLogger(__name__)

RETRYABLE_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
}


def is_retryable(e: Exception) -> bool:
    if isinstance(e, grpc.aio.AioRpcError):
        return e.code() in RETRYABLE_CODES
    return False


class BaseGRPCClient:
    """Base client with connection pooling and retry."""

    def __init__(
        self,
        service_name: str,
        host: str,
        port: int,
        timeout: float = 2.0,
        max_retries: int = 3,
    ):
        self.service_name = service_name
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[Any] = None

    def set_stub_class(self, stub_class):
        """Set the gRPC stub class."""
        self._stub_class = stub_class

    async def connect(self):
        if self._stub:
            return
        # Lazy import to avoid circular dependency
        from services.serving.main import CircuitBreaker, CircuitBreakerOpenError, CircuitState

        # Create circuit breaker for this service
        if not hasattr(self, "_circuit_breaker"):
            self._circuit_breaker = CircuitBreaker(
                service_name=self.service_name,
                failure_threshold=5,
                recovery_timeout=30.0,
                half_open_max_calls=3,
            )
        # Check circuit state before connecting
        if self._circuit_breaker.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit open for {self.service_name}")
        # Use stub class if set, otherwise skip
        if not hasattr(self, "_stub_class") or self._stub_class is None:
            return
        try:
            self._channel = grpc.aio.insecure_channel(
                f"{self.host}:{self.port}",
                options=[
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.max_send_message_length", 10 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 10 * 1024 * 1024),
                ],
            )
            self._stub = self._stub_class(self._channel)
            logger.info(f"Connected to {self.service_name}")
            self._circuit_breaker.record_success()
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise

    async def close(self):
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def call(self, method_name: str, request: Any, **kwargs):
        if not self._stub:
            await self.connect()
        method = getattr(self._stub, method_name)
        return await method(request, timeout=self.timeout)
