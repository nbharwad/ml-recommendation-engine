"""Redis caching layer with 3-tier cache system."""

import hashlib
import json
from enum import Enum
from typing import Any, Optional

import redis.asyncio as redis
from pydantic import BaseModel

from app.config import settings
from app.observability.logger import get_logger


class CacheType(str, Enum):
    """Cache type enumeration with prefixes and TTLs."""

    QUERY = "query"
    RETRIEVAL = "retrieval"
    LLM = "llm"

    @property
    def prefix(self) -> str:
        """Get cache key prefix."""
        return f"{self.value}cache"

    @property
    def ttl(self) -> int:
        """Get TTL in seconds from settings."""
        return {
            CacheType.QUERY: settings.cache_ttl_query,
            CacheType.RETRIEVAL: settings.cache_ttl_retrieval,
            CacheType.LLM: settings.cache_ttl_llm,
        }[self]


class CacheConfig(BaseModel):
    """Cache configuration model."""

    cache_type: CacheType
    key_data: str
    value: dict[str, Any]


class RedisCache:
    """Three-tier Redis cache with graceful fallback."""

    def __init__(self, redis_url: Optional[str] = None, password: Optional[str] = None):
        """Initialize Redis cache."""
        self._redis_url = redis_url or settings.redis_url
        self._password = password or settings.redis_password
        self._client: Optional[redis.Redis] = None
        self._logger = get_logger(__name__)

    async def connect(self) -> bool:
        """Connect to Redis and verify connectivity."""
        try:
            self._client = redis.from_url(
                self._redis_url,
                password=self._password,
                decode_responses=True,
            )
            await self._client.ping()
            self._logger.info("Redis cache connected", extra={"url": self._redis_url})
            return True
        except Exception as e:
            self._logger.warning(
                "Redis connection failed, cache will operate in passthrough mode",
                extra={"error": str(e)},
            )
            self._client = None
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None

    def _normalize_key(self, key_data: str) -> str:
        """Normalize key data and compute SHA-256 hash."""
        normalized = key_data.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _build_key(self, cache_type: CacheType, key_data: str) -> str:
        """Build full cache key with prefix and hash."""
        key_hash = self._normalize_key(key_data)
        return f"{cache_type.prefix}:{key_hash}"

    async def get(self, cache_type: CacheType, key_data: str) -> Optional[dict[str, Any]]:
        """
        Get cached value by type and key data.

        Args:
            cache_type: Type of cache (QUERY, RETRIEVAL, LLM)
            key_data: Raw key data to be hashed

        Returns:
            Deserialized JSON value or None if not found / Redis unavailable
        """
        if not self._client:
            return None

        try:
            full_key = self._build_key(cache_type, key_data)
            value = await self._client.get(full_key)

            if value is None:
                return None

            return json.loads(value)

        except redis.RedisError as e:
            self._logger.warning(
                "Redis get failed, returning cache miss",
                extra={"cache_type": cache_type.value, "error": str(e)},
            )
            return None
        except json.JSONDecodeError as e:
            self._logger.warning(
                "Cache value corrupted, skipping",
                extra={"cache_type": cache_type.value, "error": str(e)},
            )
            return None
        except Exception as e:
            self._logger.warning(
                "Unexpected error during cache get",
                extra={"error": str(e)},
            )
            return None

    async def set(
        self,
        cache_type: CacheType,
        key_data: str,
        value: dict[str, Any],
    ) -> bool:
        """
        Set cached value by type and key data.

        Args:
            cache_type: Type of cache (QUERY, RETRIEVAL, LLM)
            key_data: Raw key data to be hashed
            value: Value to cache (will be JSON serialized)

        Returns:
            True on success, False on failure
        """
        if not self._client:
            return False

        try:
            full_key = self._build_key(cache_type, key_data)
            serialized = json.dumps(value)

            await self._client.setex(
                full_key,
                cache_type.ttl,
                serialized,
            )

            return True

        except redis.RedisError as e:
            self._logger.warning(
                "Redis set failed",
                extra={"cache_type": cache_type.value, "error": str(e)},
            )
            return False
        except (TypeError, ValueError) as e:
            self._logger.warning(
                "Value serialization failed",
                extra={"cache_type": cache_type.value, "error": str(e)},
            )
            return False
        except Exception as e:
            self._logger.warning(
                "Unexpected error during cache set",
                extra={"error": str(e)},
            )
            return False

    async def invalidate(self, cache_type: CacheType, key_data: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            cache_type: Type of cache
            key_data: Raw key data

        Returns:
            True if deleted, False otherwise
        """
        if not self._client:
            return False

        try:
            full_key = self._build_key(cache_type, key_data)
            result = await self._client.delete(full_key)
            return result > 0

        except redis.RedisError as e:
            self._logger.warning(
                "Redis invalidate failed",
                extra={"cache_type": cache_type.value, "error": str(e)},
            )
            return False
        except Exception as e:
            self._logger.warning(
                "Unexpected error during cache invalidate",
                extra={"error": str(e)},
            )
            return False

    async def invalidate_by_prefix(self, cache_type: CacheType) -> int:
        """
        Invalidate all cache entries matching prefix.

        Args:
            cache_type: Type of cache to clear

        Returns:
            Number of keys deleted
        """
        if not self._client:
            return 0

        try:
            pattern = f"{cache_type.prefix}:*"
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self._client.delete(*keys)
                self._logger.info(
                    "Cache invalidated by prefix",
                    extra={"cache_type": cache_type.value, "deleted": deleted},
                )
                return deleted

            return 0

        except redis.RedisError as e:
            self._logger.warning(
                "Redis invalidate by prefix failed",
                extra={"cache_type": cache_type.value, "error": str(e)},
            )
            return 0
        except Exception as e:
            self._logger.warning(
                "Unexpected error during cache invalidate by prefix",
                extra={"error": str(e)},
            )
            return 0

    async def health_check(self) -> bool:
        """Check Redis connectivity."""
        if not self._client:
            return False

        try:
            await self._client.ping()
            return True
        except Exception:
            return False


async def create_cache() -> RedisCache:
    """Factory function to create cache from settings."""
    cache = RedisCache()
    await cache.connect()
    return cache
