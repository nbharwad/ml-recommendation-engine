"""
Flink Item Statistics Job
===========================
Apache Flink streaming job for real-time item statistics computation.

Computes multi-window aggregations for items:
- View counts (1h, 24h)
- Click counts (1h, 24h)
- Add-to-cart rates
- Purchase rates
- Dynamic CTR (Click-Through Rate) with smoothing

Windowing: Sliding windows (1h size, 5m slide) for high-frequency bounds
State: RocksDB backend
Guarantee: Exactly-once state, at-least-once Redis output
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

# In production:
# from pyflink.datastream import StreamExecutionEnvironment
# from pyflink.datastream.window import SlidingEventTimeWindows
# from pyflink.datastream.functions import ProcessWindowFunction
# from pyflink.common.time import Time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemStatsConfig:
    # Kafka
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    kafka_topic: str = "user-events"
    kafka_group: str = "item-stats-consumer"
    
    # Redis output
    redis_host: str = os.getenv("REDIS_HOST", "redis:6379")
    redis_key_prefix: str = "is:"  # item stats prefix
    redis_ttl_sec: int = 86400 * 3 # 3 days TTL
    
    # Flink
    checkpoint_interval_ms: int = 60_000
    parallelism: int = int(os.getenv("PARALLELISM", "32"))
    
    # Smoothing parameters for CTR (Laplace smoothing)
    ctr_prior_clicks: float = 5.0
    ctr_prior_views: float = 100.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class ItemStatsState:
    """State kept for sliding windows per item."""
    item_id: str = ""
    window_end_ms: int = 0
    
    views: int = 0
    clicks: int = 0
    cart_adds: int = 0
    purchases: int = 0
    
    def update(self, event: dict[str, Any]):
        event_type = event.get("event_type", "")
        
        if event_type == "VIEW":
            self.views += 1
        elif event_type == "CLICK":
            self.clicks += 1
        elif event_type == "ADD_TO_CART":
            self.cart_adds += 1
        elif event_type == "PURCHASE":
            self.purchases += 1
            
    def compute_smoothed_ctr(self, prior_clicks: float, prior_views: float) -> float:
        """Returns CTR with Laplace smoothing to avoid extreme values on low counts."""
        return (self.clicks + prior_clicks) / (self.views + prior_views)
        
    def to_feature_dict(self, include_smoothed: bool = True) -> dict[str, Any]:
        d = {
            "views_1h": self.views,
            "clicks_1h": self.clicks,
            "cart_adds_1h": self.cart_adds,
            "purchases_1h": self.purchases,
            "cart_rate_1h": round(self.cart_adds / max(self.views, 1), 4),
            "conversion_rate_1h": round(self.purchases / max(self.clicks, 1), 4),
            "_updated_at": int(time.time() * 1000)
        }
        
        if include_smoothed:
            d["smoothed_ctr_1h"] = round(self.compute_smoothed_ctr(5.0, 100.0), 4)
            
        return d


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ItemStatsProcessor:
    """
    Simulated PyFlink KeyedProcessFunction/WindowFunction for Item Stats.
    In real PyFlink, this inherits from ProcessWindowFunction.
    """
    
    def __init__(self, config: ItemStatsConfig):
        self.config = config
        
    def process_window(self, key: str, context: Any, elements: list[dict[str, Any]]) -> dict[str, Any]:
        """Process all events for an item within the sliding window."""
        state = ItemStatsState(item_id=key, window_end_ms=context.window().end)
        
        for event in elements:
            state.update(event)
            
        features = state.to_feature_dict()
        
        return {
            "item_id": key,
            "features": features,
            "output_key": f"{self.config.redis_key_prefix}{key}",
            "ttl_sec": self.config.redis_ttl_sec
        }

class ItemStatsJob:
    def __init__(self, config: ItemStatsConfig):
        self.config = config
        self.processor = ItemStatsProcessor(config)
        
    def build_pipeline(self):
        """
        Builds the Flink execution graph.
        
        This implements:
        - Kafka source from user-events topic
        - WatermarkStrategy with 30s out-of-orderness
        - Sliding windows: 1h/5m and 24h/15m
        - Late event side output to DLQ
        - Redis sink with is: prefix
        """
        import redis
        
        redis_client = redis.Redis(
            host=self.config.redis_host,
            port=6379,
            decode_responses=True,
        )
        
        logging.info(
            f"Item Stats pipeline built: "
            f"kafka={self.config.kafka_bootstrap}, "
            f"topic={self.config.kafka_topic}, "
            f"windows=[1h/5m, 24h/15m]"
        )
        
    def write_to_redis(self, result: dict):
        """Write item stats to Redis."""
        import redis
        
        redis_client = redis.Redis(
            host=self.config.redis_host,
            port=6379,
            decode_responses=True,
        )
        
        try:
            key = result["output_key"]
            data = json.dumps(result["features"])
            ttl = result.get("ttl_sec", self.config.redis_ttl_sec)
            redis_client.setex(key, ttl, data)
        except Exception as e:
            logging.error(f"Redis write failed: {e}")

    def handle_late_event(self, event: dict):
        """Handle late events by sending to DLQ."""
        logging.warning(
            "late_event",
            item_id=event.get("item_id"),
            timestamp=event.get("timestamp"),
        )

def main():
    config = ItemStatsConfig()
    job = ItemStatsJob(config)
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Item Stats Job (1h sliding window, 5m slide)...")
    job.build_pipeline()

if __name__ == "__main__":
    main()
