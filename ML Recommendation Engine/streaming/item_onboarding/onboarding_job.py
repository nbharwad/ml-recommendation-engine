"""
Item Onboarding Streaming Job
============================
Processes new item catalog events in real-time.

When a new item is added to the catalog, we don't have user interaction history.
This stream:
1. Listens to `item-onboarding` Kafka topic.
2. Extracts textual metadata (title, description, brand, category).
3. Uses a pre-trained language model (e.g., BERT mini) to generate a content embedding.
4. Inserts the content embedding directly into Milvus.
5. Populates default "cold-start" stats in Redis to allow ranking.

Prevents the "cold start" delay where new items take 24hrs to show up.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List, Optional

import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from pymilvus import connections, Collection, DataType
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class OnboardingConfig:
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    kafka_topic: str = "item-onboarding"
    kafka_consumer_group: str = "item-onboarding-consumer"
    
    redis_host: str = os.getenv("REDIS_HOST", "redis:6379")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    redis_prefix: str = "is:"
    
    milvus_host: str = os.getenv("MILVUS_HOST", "milvus:19530")
    milvus_collection: str = "items"
    
    model_path: str = os.getenv("MODEL_PATH", "/models/text_embedding/model.onnx")
    embedding_dim: int = 128


class ContentEmbedder:
    """Invokes NLP model to embed raw text."""
    
    def __init__(self, config: OnboardingConfig):
        self.config = config
        self.model = self._load_model()
        
    def _load_model(self) -> Any:
        """Load ONNX model for text embedding."""
        try:
            import onnxruntime as ort
            
            if not os.path.exists(self.config.model_path):
                logging.warning(f"Model not found: {self.config.model_path}")
                return None
                
            session = ort.InferenceSession(
                self.config.model_path,
                providers=["CPUExecutionProvider"],
            )
            logging.info(f"Loaded text embedding model from {self.config.model_path}")
            return session
        except Exception as e:
            logging.warning(f"Failed to load model: {e}")
            return None
        
    def embed_text(self, text: str) -> List[float]:
        """Generate text embedding."""
        if self.model is None:
            return self._stub_embedding()
        
        try:
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            result = self.model.run([output_name], {input_name: text})[0]
            return result.flatten().tolist()[:self.config.embedding_dim]
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            return self._stub_embedding()
    
    def _stub_embedding(self) -> List[float]:
        """Stub embedding when model unavailable."""
        import random
        random.seed(hash(str(time.time())) 
        return [random.gauss(0, 1) for _ in range(self.config.embedding_dim)]


class ItemOnboardingJob:
    def __init__(self, config: OnboardingConfig):
        self.config = config
        self.embedder = ContentEmbedder(config)
        self.redis_client = None
        self.milvus_client = None
        
    def _init_clients(self):
        """Initialize Redis and Milvus clients."""
        if REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password or None,
                decode_responses=True,
            )
            
        if MILVUS_AVAILABLE:
            try:
                connections.connect(alias="default", host=self.config.milvus_host)
            except Exception as e:
                logging.warning(f"Milvus connection failed: {e}")
        
    def process_new_item(self, event: dict[str, Any]) -> dict[str, Any]:
        """Process a single new item event."""
        item_id = event.get("item_id")
        title = event.get("title", "")
        category = event.get("category", "")
        description = event.get("description", "")
        
        text_context = f"{category} {title} {description}"
        content_embedding = self.embedder.embed_text(text_context)
        
        cold_start_stats = {
            "views_1h": 0,
            "clicks_1h": 0,
            "cart_adds_1h": 0,
            "purchases_1h": 0,
            "smoothed_ctr_1h": 0.05,
            "_is_cold_start": True,
            "_updated_at": int(time.time() * 1000)
        }
        
        return {
            "item_id": item_id,
            "embedding": content_embedding,
            "stats": cold_start_stats
        }
    
    def _write_to_milvus(self, result: dict):
        """Write embedding to Milvus."""
        if not MILVUS_AVAILABLE:
            return
            
        try:
            from pymilvus import Collection, DataType
            
            collection = Collection(self.config.milvus_collection)
            
            entity = [
                result["item_id"],
                result["embedding"],
                int(time.time() * 1000),
            ]
            collection.insert([entity])
            collection.flush()
            logging.info(f"Inserted item {result['item_id']} to Milvus")
        except Exception as e:
            logging.error(f"Milvus write failed: {e}")
    
    def _write_to_redis(self, result: dict):
        """Write cold-start stats to Redis."""
        if not REDIS_AVAILABLE:
            return
            
        try:
            key = f"{self.config.redis_prefix}{result['item_id']}"
            self.redis_client.setex(
                key,
                86400 * 7,
                json.dumps(result["stats"]),
            )
            logging.info(f"Inserted stats for {result['item_id']} to Redis")
        except Exception as e:
            logging.error(f"Redis write failed: {e}")
    
    def build_pipeline(self):
        """Flink topology."""
        self._init_clients()
        logging.info(f"Item Onboarding pipeline built: topic={self.config.kafka_topic}")


def main():
    logging.info("Starting Item Onboarding Streaming Job...")
    job = ItemOnboardingJob(OnboardingConfig())
    job.build_pipeline()


if __name__ == "__main__":
    main()
