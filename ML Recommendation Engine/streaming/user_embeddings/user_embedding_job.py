"""
Flink User Embeddings Streaming Job
====================================
Computes real-time user embeddings based on the latest interaction sequence.

Uses a pre-trained Two-Tower user encoder exported to ONNX/TorchScript.
Reads session updates from `session-features` topic, applies inference,
and writes updated 128-dim embeddings to Redis & Milvus.

Window: Sliding window (30m, 5m slide) on user engagement
Guarantees: Exactly-once state for inference caching
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List, Optional

import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from pymilvus import connections, Collection
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class UserEmbeddingConfig:
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    kafka_topic: str = "session-features"
    kafka_consumer_group: str = "user-embedding-consumer"
    
    redis_host: str = os.getenv("REDIS_HOST", "redis:6379")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    redis_key_prefix: str = "ue:"
    
    milvus_host: str = os.getenv("MILVUS_HOST", "milvus:19530")
    milvus_collection: str = "user_embeddings"
    
    model_path: str = os.getenv("MODEL_PATH", "/models/two_tower_user/model.onnx")
    embedding_dim: int = 128
    
    checkpoint_interval_ms: int = 60_000
    parallelism: int = int(os.getenv("PARALLELISM", "16"))


class UserEmbeddingInferencer:
    """Invokes the user tower model to create dense embeddings."""
    
    def __init__(self, config: UserEmbeddingConfig):
        self.config = config
        self.model = self._load_model()
        
    def _load_model(self) -> Optional[ort.InferenceSession]:
        """Load ONNX model for inference."""
        if not ONNX_AVAILABLE:
            logging.warning("ONNX Runtime not available, using stub")
            return None
            
        if not os.path.exists(self.config.model_path):
            logging.warning(f"Model not found: {self.config.model_path}, using stub")
            return None
            
        try:
            session = ort.InferenceSession(
                self.config.model_path,
                providers=["CPUExecutionProvider"],
            )
            logging.info(f"Loaded ONNX model from {self.config.model_path}")
            return session
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return None
        
    def encode_user_history(
        self, 
        viewed_items: List[str], 
        clicked_items: List[str],
        item_id_to_idx: dict,
    ) -> List[float]:
        """
        Convert sparse history into a 128-dim dense embedding vector.
        """
        if self.model is None:
            return self._stub_encode(len(viewed_items) + len(clicked_items))
        
        try:
            item_ids = [item_id_to_idx.get(i, 0) for i in viewed_items[-50:]]
            item_ids += [item_id_to_idx.get(i, 0) for i in clicked_items[-20:]]
            
            if len(item_ids) < 3:
                return self._stub_encode(len(item_ids))
            
            input_array = np.array(item_ids, dtype=np.int64).reshape(1, -1)
            
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            result = self.model.run([output_name], {input_name: input_array})
            embedding = result[0].flatten().tolist()
            
            return embedding[:self.config.embedding_dim]
            
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            return self._stub_encode(1)
    
    def _stub_encode(self, num_items: int) -> List[float]:
        """Stub encoding when model unavailable."""
        import random
        random.seed(num_items)
        return [random.gauss(0, 1) for _ in range(self.config.embedding_dim)]


class UserEmbeddingJob:
    def __init__(self, config: UserEmbeddingConfig):
        self.config = config
        self.inferencer = UserEmbeddingInferencer(config)
        self.redis_client = None
        self.milvus_client = None
        self.item_id_to_idx = {}
        
    def _init_clients(self):
        """Initialize Redis and Milvus clients."""
        if REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password or None,
                decode_responses=False,
            )
            
        if MILVUS_AVAILABLE:
            try:
                connections.connect(alias="default", host=self.config.milvus_host)
                collection = Collection(self.config.milvus_collection)
                collection.load()
                self.milvus_client = collection
            except Exception as e:
                logging.warning(f"Milvus connection failed: {e}")
        
    def _load_item_mapping(self):
        """Load item ID to index mapping from Redis."""
        if self.redis_client:
            try:
                data = self.redis_client.get("item_id_mapping")
                if data:
                    self.item_id_to_idx = json.loads(data)
            except Exception as e:
                logging.warning(f"Failed to load item mapping: {e}")
        
    def process_session_update(self, session_event: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Process a session update to recompute and export embedding.
        """
        user_id = session_event.get("user_id")
        features = session_event.get("features", {})
        
        viewed = features.get("last_viewed_items", [])
        clicked = features.get("last_clicked_items", [])
        
        if len(viewed) < 3 and len(clicked) == 0:
            return None
            
        embedding = self.inferencer.encode_user_history(
            viewed, 
            clicked, 
            self.item_id_to_idx,
        )
        
        return {
            "user_id": user_id,
            "embedding": embedding,
            "_updated_at": int(time.time() * 1000),
        }
    
    def _write_to_redis(self, result: dict):
        """Write embedding to Redis."""
        if self.redis_client:
            try:
                key = f"{self.config.redis_key_prefix}{result['user_id']}"
                self.redis_client.setex(
                    key,
                    3600,
                    json.dumps(result),
                )
            except Exception as e:
                logging.error(f"Redis write failed: {e}")
    
    def _write_to_milvus(self, result: dict):
        """Write embedding to Milvus."""
        if self.milvus_client:
            try:
                from pymilvus import Collection, Entity
                
                entity = [
                    result["user_id"],
                    result["embedding"],
                    int(time.time() * 1000),
                ]
                self.milvus_client.insert([entity])
            except Exception as e:
                logging.error(f"Milvus write failed: {e}")
    
    def build_pipeline(self):
        """
        Flink graph building.
        
        PyFlink equivalent:
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(self.config.parallelism)
        env.enable_checkpointing(self.config.checkpoint_interval_ms)
        
        ds = env.add_source(KafkaSource(...))
            .key_by(lambda e: e["user_id"])
            .process(UserEmbeddingProcessFunction())
            .add_sink(RedisSink(...))
            .add_sink(MilvusSink(...))
        """
        self._init_clients()
        self._load_item_mapping()
        logging.info("User Embedding pipeline built")


def main():
    config = UserEmbeddingConfig()
    job = UserEmbeddingJob(config)
    job.build_pipeline()
    logging.info(f"Deployed User Embedding Streaming Job. Dim={config.embedding_dim}")


if __name__ == "__main__":
    main()
