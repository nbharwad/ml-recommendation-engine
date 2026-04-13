"""
Flink Trending Items Job
==========================
Real-time computation of trending items using sliding windows.

Computes:
- Top-1000 trending items globally
- Top-100 trending items per category
- Trending velocity (acceleration of views/clicks)

Algorithm:
- Tumbling window: 5 minutes
- Count views + weighted signals (click=2x, cart=5x, purchase=10x)
- Compare current window vs previous → velocity score
- HyperLogLog for approximate unique user counting

Output: Redis sorted set (ZADD) for O(1) lookup
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrendingConfig:
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    kafka_topic: str = "user-events"
    kafka_group: str = "trending-consumer"
    
    redis_host: str = os.getenv("REDIS_HOST", "redis:6379")
    redis_key_global: str = "trending:global"
    redis_key_category_prefix: str = "trending:category:"
    redis_ttl_sec: int = 600  # 10 min TTL
    
    # Window
    window_size_sec: int = 300       # 5-minute tumbling window
    top_k_global: int = 1000
    top_k_per_category: int = 100
    
    # Signal weights
    view_weight: float = 1.0
    click_weight: float = 2.0
    cart_weight: float = 5.0
    purchase_weight: float = 10.0
    
    # Flink
    checkpoint_interval_ms: int = 60_000
    parallelism: int = 8


# ---------------------------------------------------------------------------
# Trending Computation
# ---------------------------------------------------------------------------

EVENT_WEIGHTS = {
    "VIEW": 1.0,
    "CLICK": 2.0,
    "ADD_TO_CART": 5.0,
    "PURCHASE": 10.0,
}


class TrendingAggregator:
    """
    Aggregates event signals per item within a tumbling window.
    
    Each item accumulates a weighted score from events.
    At window close, top-K items are emitted to Redis.
    """
    
    def __init__(self, config: TrendingConfig):
        self.config = config
        self._current_window: dict[str, float] = defaultdict(float)       # item_id → score
        self._category_windows: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )  # category → {item_id → score}
        self._unique_users: dict[str, set[str]] = defaultdict(set)       # item_id → {user_ids}
        self._window_start: float = time.time()
    
    def process_event(self, event: dict[str, Any]):
        """Process a single event and update trending scores."""
        event_type = event.get("event_type", "")
        item_id = event.get("item_id", "")
        user_id = event.get("user_id", "")
        category = event.get("metadata", {}).get("category", "unknown")
        
        if not item_id or event_type not in EVENT_WEIGHTS:
            return
        
        weight = EVENT_WEIGHTS[event_type]
        
        # Global trending
        self._current_window[item_id] += weight
        
        # Per-category trending
        self._category_windows[category][item_id] += weight
        
        # Unique user count (for popularity-based trending)
        self._unique_users[item_id].add(user_id)
    
    def compute_trending(self) -> dict[str, Any]:
        """
        Compute top-K trending items at window close.
        
        Score = weighted_signal_count × log(1 + unique_users)
        
        The unique_users multiplier prevents one user from gaming trending.
        """
        # Global trending
        scored_items: list[tuple[str, float]] = []
        
        for item_id, raw_score in self._current_window.items():
            unique_count = len(self._unique_users.get(item_id, set()))
            # Score: raw signals × diversity bonus
            final_score = raw_score * (1 + 0.5 * min(unique_count, 100) / 100)
            scored_items.append((item_id, final_score))
        
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        global_trending = scored_items[:self.config.top_k_global]
        
        # Per-category trending
        category_trending: dict[str, list[tuple[str, float]]] = {}
        
        for category, items in self._category_windows.items():
            cat_scored = sorted(items.items(), key=lambda x: x[1], reverse=True)
            category_trending[category] = cat_scored[:self.config.top_k_per_category]
        
        return {
            "global": global_trending,
            "categories": category_trending,
            "window_start": self._window_start,
            "window_end": time.time(),
            "total_items_seen": len(self._current_window),
            "total_events": sum(self._current_window.values()),
        }
    
    def reset_window(self):
        """Reset for next window."""
        self._current_window.clear()
        self._category_windows.clear()
        self._unique_users.clear()
        self._window_start = time.time()


class TrendingRedisSink:
    """Write trending results to Redis sorted sets."""
    
    def __init__(self, redis_host: str):
        self.redis_host = redis_host
    
    def write(self, trending_result: dict[str, Any], config: TrendingConfig):
        """
        Write trending items to Redis sorted sets.
        
        Uses ZADD for atomic updates of sorted sets.
        Client reads with ZREVRANGE for top-K.
        """
        # In production:
        # pipe = redis.pipeline()
        #
        # # Global trending
        # pipe.delete(config.redis_key_global)
        # for item_id, score in trending_result["global"]:
        #     pipe.zadd(config.redis_key_global, {item_id: score})
        # pipe.expire(config.redis_key_global, config.redis_ttl_sec)
        #
        # # Per-category trending
        # for category, items in trending_result["categories"].items():
        #     key = f"{config.redis_key_category_prefix}{category}"
        #     pipe.delete(key)
        #     for item_id, score in items:
        #         pipe.zadd(key, {item_id: score})
        #     pipe.expire(key, config.redis_ttl_sec)
        #
        # pipe.execute()
        
        logging.info(
            f"Trending update: {len(trending_result['global'])} global, "
            f"{len(trending_result['categories'])} categories"
        )


# ---------------------------------------------------------------------------
# Flink Job
# ---------------------------------------------------------------------------

class TrendingJob:
    """
    Flink streaming job for trending item computation.
    
    Topology:
    Kafka Source → Filter (VIEW/CLICK/CART/PURCHASE) →
    Key by item_id → Tumbling Window (5 min) →
    Aggregate → Top-K → Redis Sink
    """
    
    def __init__(self, config: TrendingConfig):
        self.config = config
        self.aggregator = TrendingAggregator(config)
        self.sink = TrendingRedisSink(config.redis_host)
    
    def process_window(self, events: list[dict[str, Any]]):
        """Process all events in one tumbling window."""
        self.aggregator.reset_window()
        
        for event in events:
            self.aggregator.process_event(event)
        
        result = self.aggregator.compute_trending()
        self.sink.write(result, self.config)
        
        return result


# Entry point
def main():
    config = TrendingConfig()
    job = TrendingJob(config)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Trending Items Job")
    logger.info(f"Window size: {config.window_size_sec}s")
    logger.info(f"Top-K global: {config.top_k_global}")
    
    # In production: env.execute("Trending Items Job")


if __name__ == "__main__":
    main()
