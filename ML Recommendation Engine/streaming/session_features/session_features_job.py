"""
Flink Session Features Job
============================
Apache Flink streaming job for real-time session feature computation.

Consumes user events from Kafka, maintains session state,
and writes session-level features to Redis.

Features computed:
- Last N viewed items (session-scoped)
- Session click count
- Session dwell time
- Session search queries
- Category browse distribution

Windowing: Session window with 30-minute gap
State: RocksDB backend with S3 checkpointing
Guarantee: At-least-once (Redis writes are idempotent SET operations)
"""

# Note: This is a Flink Python (PyFlink) implementation.
# In production, many teams prefer Java/Scala Flink for performance.
# This Python version works for moderate scale; switch to Java at >100K events/sec.

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# In production:
# from pyflink.datastream import StreamExecutionEnvironment
# from pyflink.datastream.window import SessionWindows
# from pyflink.datastream.functions import ProcessWindowFunction, KeyedProcessFunction
# from pyflink.common.serialization import SimpleStringSchema
# from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
# from pyflink.common.watermark_strategy import WatermarkStrategy
# import redis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionFeatureConfig:
    # Kafka
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    kafka_topic: str = "user-events"
    kafka_group: str = "session-feature-consumer"
    
    # Redis output
    redis_host: str = os.getenv("REDIS_HOST", "redis:6379")
    redis_key_prefix: str = "sf:"  # session feature prefix
    redis_ttl_sec: int = 86400     # 24h TTL for session features
    
    # Session window
    session_gap_minutes: int = 30
    max_session_duration_hours: int = 4
    
    # Feature parameters
    max_viewed_items: int = 50
    max_search_queries: int = 20
    
    # Flink
    checkpoint_interval_ms: int = 60_000  # 60s
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "s3://rec-system/checkpoints/session-features")
    parallelism: int = int(os.getenv("PARALLELISM", "16"))
    
    # Watermark
    max_out_of_orderness_sec: int = 120  # 2-minute watermark delay


# ---------------------------------------------------------------------------
# Session State (Flink Keyed State)
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """
    Per-user session state maintained in Flink's keyed state.
    Backed by RocksDB for large state sizes.
    """
    user_id: str = ""
    session_id: str = ""
    session_start_ms: int = 0
    last_event_ms: int = 0
    
    # Behavioral features
    viewed_items: list[str] = field(default_factory=list)
    clicked_items: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    cart_items: list[str] = field(default_factory=list)
    
    # Aggregates
    view_count: int = 0
    click_count: int = 0
    cart_add_count: int = 0
    purchase_count: int = 0
    total_dwell_time_sec: float = 0.0
    
    # Category distribution
    category_views: dict[str, int] = field(default_factory=dict)
    
    def update(self, event: dict[str, Any]):
        """Update session state with new event."""
        event_type = event.get("event_type", "")
        item_id = event.get("item_id", "")
        timestamp_ms = event.get("timestamp_ms", int(time.time() * 1000))
        
        # Update timing
        if self.session_start_ms == 0:
            self.session_start_ms = timestamp_ms
        
        if self.last_event_ms > 0:
            gap_sec = (timestamp_ms - self.last_event_ms) / 1000
            self.total_dwell_time_sec += min(gap_sec, 300)  # cap at 5 min per gap
        
        self.last_event_ms = timestamp_ms
        self.user_id = event.get("user_id", self.user_id)
        
        # Update by event type
        if event_type == "VIEW":
            self.view_count += 1
            if item_id and item_id not in self.viewed_items[-10:]:  # dedup recent
                self.viewed_items.append(item_id)
                # Trim to max
                if len(self.viewed_items) > 50:
                    self.viewed_items = self.viewed_items[-50:]
            
            # Category tracking
            category = event.get("metadata", {}).get("category", "unknown")
            self.category_views[category] = self.category_views.get(category, 0) + 1
        
        elif event_type == "CLICK":
            self.click_count += 1
            if item_id:
                self.clicked_items.append(item_id)
        
        elif event_type == "ADD_TO_CART":
            self.cart_add_count += 1
            if item_id:
                self.cart_items.append(item_id)
        
        elif event_type == "PURCHASE":
            self.purchase_count += 1
        
        elif event_type == "SEARCH":
            query = event.get("metadata", {}).get("query", "")
            if query:
                self.search_queries.append(query)
                if len(self.search_queries) > 20:
                    self.search_queries = self.search_queries[-20:]
    
    def to_feature_dict(self) -> dict[str, Any]:
        """Convert session state to feature dictionary for Redis."""
        session_duration_sec = (
            (self.last_event_ms - self.session_start_ms) / 1000
            if self.session_start_ms > 0 and self.last_event_ms > 0
            else 0
        )
        
        return {
            "session_view_count": self.view_count,
            "session_click_count": self.click_count,
            "session_cart_add_count": self.cart_add_count,
            "session_purchase_count": self.purchase_count,
            "session_duration_sec": round(session_duration_sec, 1),
            "session_dwell_time_sec": round(self.total_dwell_time_sec, 1),
            "session_ctr": round(
                self.click_count / max(self.view_count, 1), 4
            ),
            "last_viewed_items": self.viewed_items[-10:],  # last 10
            "last_clicked_items": self.clicked_items[-5:],  # last 5
            "cart_items": self.cart_items[-10:],
            "search_queries": self.search_queries[-5:],
            "top_categories": dict(
                sorted(self.category_views.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "_timestamp_ms": self.last_event_ms,
            "_session_start_ms": self.session_start_ms,
        }


# ---------------------------------------------------------------------------
# Flink Job (Pseudo-code with PyFlink structure)
# ---------------------------------------------------------------------------

class SessionFeatureJob:
    """
    Flink streaming job for session feature computation.
    
    Topology:
    Kafka Source → Deserialize → Key by user_id → Session Window → 
    → Process (update state, emit features) → Redis Sink
    
    Also emits incremental updates per-event (not just on window close)
    for real-time freshness.
    """
    
    def __init__(self, config: SessionFeatureConfig):
        self.config = config
    
    def build_pipeline(self):
        """
        Build the Flink execution graph.
        
        In production (Java/Scala Flink):
        ```java
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.enableCheckpointing(60000);
        env.getCheckpointConfig().setCheckpointStorage(checkpointDir);
        
        KafkaSource<String> source = KafkaSource.<String>builder()
            .setBootstrapServers(kafkaBootstrap)
            .setTopics(topic)
            .setGroupId(groupId)
            .setStartingOffsets(OffsetsInitializer.committedOffsets())
            .setDeserializer(new SimpleStringSchema())
            .build();
        
        DataStream<UserEvent> events = env
            .fromSource(source, WatermarkStrategy
                .forBoundedOutOfOrderness(Duration.ofSeconds(120))
                .withTimestampAssigner((event, ts) -> event.getTimestampMs()),
                "Kafka Source")
            .map(this::parseEvent)
            .keyBy(UserEvent::getUserId)
            .process(new SessionFeatureProcessor())
        
        events.addSink(new RedisSink(redisConfig));
        ```
        """
        pass

    def process_event(self, event: dict[str, Any], state: SessionState) -> dict[str, Any]:
        """
        Process a single event and update session state.
        Called for each incoming event.
        
        Emits updated features to Redis after each event
        for real-time freshness (not waiting for window close).
        """
        state.update(event)
        features = state.to_feature_dict()
        
        return {
            "user_id": event.get("user_id"),
            "features": features,
            "output_key": f"{self.config.redis_key_prefix}{event.get('user_id')}",
            "ttl_sec": self.config.redis_ttl_sec,
        }
    
    def handle_late_event(self, event: dict[str, Any]):
        """
        Handle events that arrive after the watermark.
        
        Strategy:
        - Events within 2 minutes: processed normally by Flink watermark
        - Events 2-60 minutes late: routed to side output for batch processing
        - Events >60 minutes late: dropped with monitoring counter
        """
        lateness_sec = (time.time() * 1000 - event.get("timestamp_ms", 0)) / 1000
        
        if lateness_sec > 3600:
            # Extremely late — drop
            logging.warning(f"Dropping extremely late event: {lateness_sec:.0f}s late")
            return None
        else:
            # Moderately late — route to late-event topic
            return {
                "topic": "user-events-late",
                "event": event,
                "lateness_sec": lateness_sec,
            }


# ---------------------------------------------------------------------------
# Redis Sink
# ---------------------------------------------------------------------------

class RedisSink:
    """
    Flink sink that writes session features to Redis.
    
    Write strategy:
    - Idempotent SET operation (at-least-once is fine)
    - Pipeline writes for efficiency
    - Features stored as JSON blob at key: sf:{user_id}
    """
    
    def __init__(self, redis_host: str, redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        # self.client = redis.Redis(host=redis_host, port=redis_port)
    
    def write(self, user_id: str, features: dict[str, Any], ttl_sec: int = 86400):
        """Write session features to Redis."""
        key = f"sf:{user_id}"
        value = json.dumps(features)
        
        # In production:
        # self.client.setex(key, ttl_sec, value)
        
        logging.debug(f"Redis SET {key} (TTL: {ttl_sec}s)")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """Start the Flink session features job."""
    config = SessionFeatureConfig()
    job = SessionFeatureJob(config)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Session Feature Job")
    logger.info(f"Kafka: {config.kafka_bootstrap} / {config.kafka_topic}")
    logger.info(f"Redis: {config.redis_host}")
    logger.info(f"Session gap: {config.session_gap_minutes} minutes")
    logger.info(f"Checkpoint interval: {config.checkpoint_interval_ms}ms")
    logger.info(f"Parallelism: {config.parallelism}")
    
    job.build_pipeline()
    
    # In production: env.execute("Session Feature Job")


if __name__ == "__main__":
    main()
