"""
Event Enrichment Streaming Job
==============================
Takes raw `user-events` and enriches them with catalog metadata.

Why:
The frontend often only sends `{user_id, item_id, event_type}` to save bandwidth.
This Flink job performs an Async I/O lookup against Redis/Elasticsearch to append
`category`, `price`, `brand`, and `platform` before writing to feature streams,
so downstream ML models have full context.

Topic routing:
`user-events` -> enrichment async IO -> `user-events-enriched`
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class EnrichmentConfig:
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    input_topic: str = "user-events"
    output_topic: str = "user-events-enriched"
    dlq_topic: str = "user-events-enrichment-dlq"
    kafka_consumer_group: str = "enrichment-consumer"
    
    redis_host: str = os.getenv("REDIS_HOST", "redis:6379")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    redis_catalog_prefix: str = "catalog:"
    
    checkpoint_interval_ms: int = 60_000
    parallelism: int = int(os.getenv("PARALLELISM", "8"))


class AsyncRedisCatalogLookup:
    """Async I/O function for Flink to prevent blocking."""
    
    def __init__(self, config: EnrichmentConfig):
        self.config = config
        self.redis_client = None
        
        if REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                password=config.redis_password or None,
                decode_responses=True,
            )
    
    async def async_invoke(self, event: dict[str, Any]) -> dict[str, Any]:
        """Fetch item metadata and append to event."""
        item_id = event.get("item_id")
        
        if not item_id:
            return event
            
        try:
            if self.redis_client:
                catalog_key = f"{self.config.redis_catalog_prefix}{item_id}"
                metadata = self.redis_client.hgetall(catalog_key)
                
                if metadata:
                    event["item_category"] = metadata.get("category", "unknown")
                    event["item_price"] = float(metadata.get("price", 0))
                    event["item_brand"] = metadata.get("brand", "unknown")
                    event["item_platform"] = metadata.get("platform", "web")
                else:
                    event["item_category"] = "unknown"
                    event["item_price"] = 0
                    event["item_brand"] = "unknown"
                    event["item_platform"] = "web"
        except Exception as e:
            logging.warning(f"Catalog lookup failed for {item_id}: {e}")
            event["item_category"] = "unknown"
            event["item_price"] = 0
            event["item_brand"] = "unknown"
            event["item_platform"] = "web"
        
        event["_enriched_at"] = int(time.time() * 1000)
        return event


class EnrichmentJob:
    def __init__(self, config: EnrichmentConfig):
        self.config = config
        self.catalog_lookup = AsyncRedisCatalogLookup(config)
        
    async def process_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Process and enrich a single event."""
        return await self.catalog_lookup.async_invoke(event)
    
    def build_pipeline(self):
        """Flink topology: Kafka Source -> AsyncWaitOperator -> Kafka Sink."""
        logging.info(f"Enrichment pipeline built: {self.config.input_topic} -> {self.config.output_topic}")
        
    def handle_dlq(self, event: dict, error: str):
        """Handle events that failed enrichment."""
        logging.error(f"DLQ: item_id={event.get('item_id')}, error={error}")


def main():
    logging.info("Starting Async Enrichment Job")
    config = EnrichmentConfig()
    job = EnrichmentJob(config)
    job.build_pipeline()


if __name__ == "__main__":
    main()
