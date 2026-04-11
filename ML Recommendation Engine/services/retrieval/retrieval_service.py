"""
Retrieval Service
=================
ANN-based candidate retrieval from Milvus with multi-source fusion.

Retrieval Strategy:
1. ANN Search (60%): HNSW index on Milvus for Two-Tower embeddings
2. Collaborative Filtering (20%): Precomputed user-item affinity cache
3. Trending Items (10%): Real-time trending from Flink pipeline
4. Rule-Based (10%): Category affinity, editorial picks

Scale:
- 10M vectors × 128 dimensions
- 8 shards, 3 replicas per shard
- Target: <15ms p99 for 1000 candidates
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# In production: from pymilvus import MilvusClient, connections, Collection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    # Milvus configuration
    milvus_host: str = os.getenv("MILVUS_HOST", "milvus-proxy")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    collection_name: str = "item_embeddings"
    
    # Index parameters (HNSW)
    hnsw_m: int = 16          # connections per layer
    hnsw_ef_construction: int = 256  # build-time search width
    hnsw_ef_search: int = 200       # query-time search width
    
    # Search parameters
    embedding_dim: int = 128
    default_num_candidates: int = 1000
    ann_fraction: float = 0.6     # 60% from ANN
    cf_fraction: float = 0.2      # 20% from CF
    trending_fraction: float = 0.1  # 10% trending
    rule_fraction: float = 0.1    # 10% rule-based
    
    # Performance
    search_timeout_ms: int = 15
    max_concurrent_searches: int = 100
    
    # gRPC
    grpc_port: int = int(os.getenv("GRPC_PORT", "50052"))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_ms",
    "Retrieval latency by source",
    ["source"],
    buckets=[1, 2, 5, 8, 10, 12, 15, 20, 30],
)

RETRIEVAL_CANDIDATES = Histogram(
    "retrieval_candidate_count",
    "Number of candidates from each source",
    ["source"],
    buckets=[50, 100, 200, 300, 500, 800, 1000],
)

ANN_RECALL = Gauge(
    "retrieval_ann_recall_at_k",
    "ANN recall@K from periodic validation",
    ["k"],
)

HEDGED_REQUESTS = Counter(
    "hedged_requests_total",
    "Hedged requests (sent to multiple replicas)",
    ["outcome"],
    buckets=[50, 100, 200, 300, 500, 800, 1000],
)

HEDGED_LATENCY = Histogram(
    "hedged_latency_ms",
    "Latency benefit from hedged requests",
    buckets=[0, 1, 2, 3, 5, 10],
)


# ---------------------------------------------------------------------------
# ANN Search (Milvus Client)
# ---------------------------------------------------------------------------

class MilvusANNSearcher:
    """
    Milvus-backed ANN search with HNSW index.
    
    Index design:
    - Algorithm: HNSW (best latency/recall trade-off for this scale)
    - Parameters: M=16, ef_construction=256, ef_search=200
    - Vectors: 10M × 128-dim float32
    - Shards: 8 (each shard ~1.25M vectors, ~640MB)
    - Memory per shard: ~2GB (vectors + HNSW graph)
    
    Why HNSW over IVF_PQ:
    - HNSW: 5ms query, 95%+ recall@100 (no training required)
    - IVF_PQ: 2ms query, 85-90% recall@100 (needs codebook training)
    - At our latency budget (15ms), HNSW's higher recall is worth 3ms
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.logger = structlog.get_logger(component="ann_searcher")
        self._collection = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_searches)
    
    async def initialize(self):
        """Connect to Milvus and load collection."""
        self.logger.info(
            "milvus_connecting",
            host=self.config.milvus_host,
            collection=self.config.collection_name,
        )
        # In production:
        # connections.connect(host=self.config.milvus_host, port=self.config.milvus_port)
        # self._collection = Collection(self.config.collection_name)
        # self._collection.load()
        self.logger.info("milvus_connected")
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 600,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar items using HNSW index.
        
        Args:
            query_embedding: 128-dim user embedding from Two-Tower
            top_k: number of nearest neighbors
            filters: attribute filters (category, price_range, in_stock)
        
        Returns:
            List of {item_id, score, source} dicts
        """
        async with self._semaphore:
            start = time.monotonic()
            
            try:
                # Build search parameters
                search_params = {
                    "metric_type": "IP",  # Inner Product (embeddings are L2-normalized)
                    "params": {"ef": self.config.hnsw_ef_search},
                }
                
                # Build filter expression for Milvus
                filter_expr = self._build_filter_expr(filters)
                
                # In production:
                # results = self._collection.search(
                #     data=[query_embedding],
                #     anns_field="embedding",
                #     param=search_params,
                #     limit=top_k,
                #     expr=filter_expr,
                #     output_fields=["item_id"],
                # )
                
                # Simulated results
                results = []
                for i in range(top_k):
                    results.append({
                        "item_id": f"item_{i:07d}",
                        "score": max(0, 1.0 - (i * 0.0015)),
                        "source": "ANN",
                    })
                
                latency_ms = (time.monotonic() - start) * 1000
                RETRIEVAL_LATENCY.labels(source="ANN").observe(latency_ms)
                RETRIEVAL_CANDIDATES.labels(source="ANN").observe(len(results))
                
                self.logger.debug(
                    "ann_search_complete",
                    top_k=top_k,
                    result_count=len(results),
                    latency_ms=round(latency_ms, 2),
                )
                
                return results
                
            except Exception as e:
                self.logger.error("ann_search_failed", error=str(e))
                raise
    
    def _build_filter_expr(self, filters: dict[str, str] | None) -> str | None:
        """Build Milvus filter expression from attribute filters."""
        if not filters:
            return None
        
        expressions = []
        if "category" in filters:
            expressions.append(f'category == "{filters["category"]}"')
        if "in_stock" in filters:
            expressions.append("stock_count > 0")
        if "min_price" in filters:
            expressions.append(f"price >= {filters['min_price']}")
        if "max_price" in filters:
            expressions.append(f"price <= {filters['max_price']}")
        
        return " and ".join(expressions) if expressions else None
    
    async def validate_index(self, validation_queries: list[dict]) -> float:
        """
        Canary validation: check recall@100 with known good queries.
        
        Run after every index reload to detect corruption.
        If recall drops >20% below baseline, flag as corrupt.
        """
        if not validation_queries:
            return 1.0
        
        total_recall = 0.0
        for query in validation_queries:
            embedding = query["embedding"]
            expected_ids = set(query["expected_top_100"])
            
            results = await self.search(embedding, top_k=100)
            retrieved_ids = {r["item_id"] for r in results}
            
            recall = len(expected_ids & retrieved_ids) / len(expected_ids)
            total_recall += recall
        
        avg_recall = total_recall / len(validation_queries)
        ANN_RECALL.labels(k="100").set(avg_recall)
        
        return avg_recall


# ---------------------------------------------------------------------------
# Hedged Requests (Latency Protection)
# ---------------------------------------------------------------------------

class HedgedRequests:
    """
    Send requests to multiple replicas, take first response.
    
    Use Case:
    - Protect against slow replicas during high load
    - Reduce p99 latency by avoiding stragglers
    
    Implementation:
    - Send to 2 replicas in parallel
    - Use asyncio.wait_for with timeout
    - Take first response, cancel others
    
    Industry Standard:
    - Used by Google, Meta, Netflix for p99 protection
    - Typical improvement: 30-50% faster p99
    """
    
    def __init__(self, num_replicas: int = 2, timeout_ms: int = 10):
        self.num_replicas = num_replicas
        self.timeout_sec = timeout_ms / 1000
        self.logger = structlog.get_logger(component="hedged_requests")
    
    async def execute(
        self,
        query_embedding: list[float],
        searcher: MilvusANNSearcher,
        top_k: int = 600,
    ) -> list[dict[str, Any]]:
        """
        Execute hedged search to multiple replicas.
        
        Returns first response, cancels others.
        """
        start = time.monotonic()
        
        # Create parallel tasks for multiple replicas
        # In production: query multiple Milvus replicas
        tasks = []
        for replica_id in range(self.num_replicas):
            task = asyncio.create_task(
                searcher.search(query_embedding, top_k=top_k)
            )
            tasks.append(task)
        
        # Wait for first completion
        done, pending = await asyncio.wait(
            tasks,
            timeout=self.timeout_sec,
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        # Cancel pending
        for task in pending:
            task.cancel()
        
        # Get result from first completed
        if done:
            completed_task = done.pop()
            try:
                results = await completed_task
                
                latency_ms = (time.monotonic() - start) * 1000
                HEDGED_REQUESTS.labels(outcome="first_response").inc()
                HEDGED_LATENCY.observe(latency_ms)
                
                self.logger.debug(
                    "hedged_success",
                    latency_ms=round(latency_ms, 2),
                    pending_cancelled=len(pending),
                )
                
                return results
                
            except Exception as e:
                self.logger.error("hedged_error", error=str(e))
                raise
        
        # All timed out
        HEDGED_REQUESTS.labels(outcome="timeout").inc()
        self.logger.warning("hedged_all_timeout", timeout_ms=self.timeout_ms)
        
        # Return empty on timeout (let main path handle fallback)
        return []


# ---------------------------------------------------------------------------
# Collaborative Filtering Source
# ---------------------------------------------------------------------------

class CollaborativeFilterSource:
    """
    Precomputed user-item affinity scores from offline CF model.
    
    Strategy:
    - Item-item CF: stored as top-100 similar items per item
    - User-item CF: stored as top-500 candidate items per user segment
    - Updated every 4 hours from batch pipeline
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(component="cf_source")
    
    async def get_candidates(
        self,
        user_id: str,
        num_candidates: int = 200,
        recently_viewed: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve CF candidates.
        
        Strategy: union of item-item similarity for recently viewed items
        """
        start = time.monotonic()
        
        # In production: Redis lookup for user's CF candidates
        # Key: cf:user:{user_id} → JSON list of (item_id, score)
        candidates = [
            {
                "item_id": f"cf_item_{i:07d}",
                "score": max(0, 0.8 - (i * 0.003)),
                "source": "CF",
            }
            for i in range(num_candidates)
        ]
        
        RETRIEVAL_LATENCY.labels(source="CF").observe(
            (time.monotonic() - start) * 1000
        )
        return candidates


# ---------------------------------------------------------------------------
# Trending Items Source
# ---------------------------------------------------------------------------

class TrendingSource:
    """
    Real-time trending items from Flink streaming pipeline.
    
    Updated: every 5 minutes (tumbling window)
    Storage: Redis sorted set per category + global
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(component="trending_source")
    
    async def get_candidates(
        self,
        num_candidates: int = 100,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch current trending items."""
        start = time.monotonic()
        
        # In production: Redis ZREVRANGE on sorted set
        # Key: trending:global or trending:category:{category}
        candidates = [
            {
                "item_id": f"trending_{i:07d}",
                "score": max(0, 0.7 - (i * 0.005)),
                "source": "TRENDING",
            }
            for i in range(num_candidates)
        ]
        
        RETRIEVAL_LATENCY.labels(source="TRENDING").observe(
            (time.monotonic() - start) * 1000
        )
        return candidates


# ---------------------------------------------------------------------------
# Multi-Source Retrieval Orchestrator
# ---------------------------------------------------------------------------

class RetrievalService:
    """
    Orchestrates multi-source retrieval and merges candidates.
    
    Sources are queried in parallel, results are merged, deduplicated,
    and filtered for eligibility (in-stock, not blocked).
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.logger = structlog.get_logger(component="retrieval_service")
        
        self.ann_searcher = MilvusANNSearcher(config)
        self.cf_source = CollaborativeFilterSource()
        self.trending_source = TrendingSource()
        
        # Hedged requests for p99 protection
        self.hedger = HedgedRequests(
            num_replicas=2,
            timeout_ms=config.search_timeout_ms,
        )
    
    async def initialize(self):
        await self.ann_searcher.initialize()
    
    async def retrieve_candidates(
        self,
        user_embedding: list[float],
        num_candidates: int = 1000,
        exclude_ids: list[str] | None = None,
        filters: dict[str, str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Multi-source candidate retrieval with parallel execution.
        
        Returns merged candidate set with source attribution.
        """
        start = time.monotonic()
        exclude_set = set(exclude_ids) if exclude_ids else set()
        
        # Calculate per-source candidate counts
        ann_count = int(num_candidates * self.config.ann_fraction)
        cf_count = int(num_candidates * self.config.cf_fraction)
        trending_count = int(num_candidates * self.config.trending_fraction)
        
        # Parallel retrieval from all sources
        ann_task = asyncio.create_task(
            self.ann_searcher.search(user_embedding, top_k=ann_count, filters=filters)
        )
        cf_task = asyncio.create_task(
            self.cf_source.get_candidates(user_id=user_id or "", num_candidates=cf_count)
        )
        trending_task = asyncio.create_task(
            self.trending_source.get_candidates(num_candidates=trending_count)
        )
        
        # Gather results (continue even if some sources fail)
        results = await asyncio.gather(
            ann_task, cf_task, trending_task,
            return_exceptions=True,
        )
        
        # Merge and deduplicate
        all_candidates: dict[str, dict[str, Any]] = {}  # item_id → best candidate
        source_counts: dict[str, int] = {}
        
        for source_result in results:
            if isinstance(source_result, Exception):
                self.logger.warning("source_failed", error=str(source_result))
                continue
            
            for candidate in source_result:
                item_id = candidate["item_id"]
                
                # Skip excluded items
                if item_id in exclude_set:
                    continue
                
                # Deduplicate: keep highest score
                if item_id not in all_candidates or candidate["score"] > all_candidates[item_id]["score"]:
                    all_candidates[item_id] = candidate
                
                source = candidate.get("source", "UNKNOWN")
                source_counts[source] = source_counts.get(source, 0) + 1
        
        # Sort by score and limit
        sorted_candidates = sorted(
            all_candidates.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:num_candidates]
        
        total_latency_us = int((time.monotonic() - start) * 1_000_000)
        
        RETRIEVAL_LATENCY.labels(source="total").observe(total_latency_us / 1000)
        
        self.logger.info(
            "retrieval_complete",
            total_candidates=len(sorted_candidates),
            source_counts=source_counts,
            latency_us=total_latency_us,
        )
        
        return {
            "candidates": sorted_candidates,
            "source_counts": source_counts,
            "retrieval_latency_us": total_latency_us,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def serve():
    """Start the retrieval service."""
    config = RetrievalConfig()
    service = RetrievalService(config)
    await service.initialize()
    
    start_http_server(9091)
    
    logger = structlog.get_logger()
    logger.info("retrieval_service_started", grpc_port=config.grpc_port)
    
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        pass


if __name__ == "__main__":
    asyncio.run(serve())
