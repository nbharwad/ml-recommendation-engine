"""
Recommendation System Serving Layer
====================================
FastAPI-based orchestrator that coordinates the full recommendation pipeline:
Feature Fetch → Retrieval → Ranking → Re-Ranking

Production-grade with:
- Circuit breakers per downstream service
- Graceful degradation chain
- Structured logging with trace propagation
- Latency budget management
- A/B experiment assignment
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Generated gRPC stubs
from recommendation.v1 import recommendation_pb2_grpc

from services.serving.clients import (
    FeatureStoreClient,
    RetrievalClient,
    RankingClient,
    ReRankingClient,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceConfig:
    """Immutable configuration for downstream service connections."""

    host: str
    port: int
    timeout_ms: int
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_sec: int = 30
    max_retries: int = 1


@dataclass(frozen=True)
class ServingConfig:
    """Main serving layer configuration."""

    # Downstream services
    feature_service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(
            host=os.getenv("FEATURE_SERVICE_HOST", "feature-service"),
            port=int(os.getenv("FEATURE_SERVICE_PORT", "50051")),
            timeout_ms=10,
        )
    )
    retrieval_service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(
            host=os.getenv("RETRIEVAL_SERVICE_HOST", "retrieval-service"),
            port=int(os.getenv("RETRIEVAL_SERVICE_PORT", "50052")),
            timeout_ms=20,
        )
    )
    ranking_service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(
            host=os.getenv("RANKING_SERVICE_HOST", "ranking-service"),
            port=int(os.getenv("RANKING_SERVICE_PORT", "50053")),
            timeout_ms=30,
        )
    )
    reranking_service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(
            host=os.getenv("RERANKING_SERVICE_HOST", "reranking-service"),
            port=int(os.getenv("RERANKING_SERVICE_PORT", "50054")),
            timeout_ms=8,
        )
    )
    experiment_service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(
            host=os.getenv("EXPERIMENT_SERVICE_HOST", "experiment-service"),
            port=int(os.getenv("EXPERIMENT_SERVICE_PORT", "50055")),
            timeout_ms=5,
        )
    )

    # Serving parameters
    total_latency_budget_ms: int = 75
    default_num_items: int = 20
    max_num_items: int = 50
    num_candidates: int = 1000  # candidates to retrieve from ANN
    cache_ttl_home_sec: int = 30
    cache_ttl_pdp_sec: int = 0  # no cache for PDP

    # Rate limiting
    global_rate_limit_qps: int = 50_000
    per_user_rate_limit_qps: int = 100


# ---------------------------------------------------------------------------
# Request / Response Models (REST API)
# ---------------------------------------------------------------------------


class PageContextEnum(str, Enum):
    HOME = "home"
    PDP = "pdp"
    CART = "cart"
    SEARCH = "search"
    CATEGORY = "category"


class RecommendationRequest(BaseModel):
    """External API request schema."""

    user_id: str = Field(..., description="User identifier", min_length=1, max_length=64)
    session_id: Optional[str] = Field(None, description="Session identifier")
    page_context: PageContextEnum = Field(PageContextEnum.HOME, description="Page type for context")
    num_items: int = Field(20, ge=1, le=50, description="Number of items to return")
    client_context: dict[str, str] = Field(default_factory=dict, description="Device, locale, etc.")


class RecommendedItemResponse(BaseModel):
    """Single recommended item in the response."""

    item_id: str
    position: int
    score: float
    explanation: Optional[str] = None
    tracking: dict[str, str] = Field(default_factory=dict)


class FallbackInfoResponse(BaseModel):
    """Indicates whether a fallback was used and why."""

    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    fallback_type: Optional[str] = None


class RecommendationResponse(BaseModel):
    """External API response schema."""

    items: list[RecommendedItemResponse]
    request_id: str
    experiment_id: Optional[str] = None
    total_latency_ms: int
    fallback: FallbackInfoResponse = Field(default_factory=FallbackInfoResponse)


class SimilarItemsRequest(BaseModel):
    """Request for similar items."""

    item_id: str = Field(..., min_length=1)
    num_items: int = Field(20, ge=1, le=50)
    user_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    dependencies: dict[str, str]


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Production circuit breaker with three states:
    - CLOSED: normal operation, requests pass through
    - OPEN: failures exceeded threshold, requests fail fast
    - HALF_OPEN: recovery period, allow one test request
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_sec: int = 30,
        success_threshold: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._lock = asyncio.Lock()

        self.logger = structlog.get_logger(component="circuit_breaker", service=name)

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time > self.recovery_timeout_sec:
                    self._state = CircuitState.HALF_OPEN
                    self.logger.info("circuit_breaker_half_open", service=self.name)
                else:
                    CIRCUIT_BREAKER_STATE.labels(service=self.name).set(2)  # OPEN
                    raise CircuitBreakerOpenError(f"Circuit breaker OPEN for {self.name}")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    CIRCUIT_BREAKER_STATE.labels(service=self.name).set(0)  # CLOSED
                    self.logger.info("circuit_breaker_closed", service=self.name)
            else:
                self._failure_count = 0

    async def _on_failure(self):
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                CIRCUIT_BREAKER_STATE.labels(service=self.name).set(2)  # OPEN
                self.logger.warning(
                    "circuit_breaker_opened",
                    service=self.name,
                    failure_count=self._failure_count,
                )

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                CIRCUIT_BREAKER_STATE.labels(service=self.name).set(2)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# ---------------------------------------------------------------------------
# Latency Budget Manager
# ---------------------------------------------------------------------------


class LatencyBudget:
    """
    Tracks remaining latency budget across the request pipeline.
    Allows downstream services to adapt (e.g., reduce candidate count)
    when budget is running low.
    """

    def __init__(self, total_budget_ms: int):
        self.total_budget_ms = total_budget_ms
        self._start_time = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start_time) * 1000

    @property
    def remaining_ms(self) -> float:
        return max(0, self.total_budget_ms - self.elapsed_ms)

    @property
    def is_expired(self) -> bool:
        return self.remaining_ms <= 0

    def can_afford(self, operation_budget_ms: float) -> bool:
        """Check if we have enough budget for an operation."""
        return self.remaining_ms >= operation_budget_ms


# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

REQUEST_LATENCY = Histogram(
    "recommendation_request_latency_ms",
    "End-to-end recommendation request latency",
    ["page_context", "fallback_type"],
    buckets=[10, 20, 30, 40, 50, 60, 75, 100, 150, 200, 500],
)

COMPONENT_LATENCY = Histogram(
    "recommendation_component_latency_ms",
    "Per-component latency",
    ["component"],
    buckets=[1, 2, 5, 10, 15, 20, 25, 30, 50, 100],
)

REQUEST_COUNT = Counter(
    "recommendation_request_total",
    "Total recommendation requests",
    ["page_context", "status"],
)

FALLBACK_COUNT = Counter(
    "recommendation_fallback_total",
    "Fallback events by type",
    ["fallback_type", "reason"],
)

CANDIDATE_COUNT = Histogram(
    "recommendation_candidate_count",
    "Number of candidates retrieved",
    buckets=[100, 200, 500, 800, 1000, 1500],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "recommendation_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["service"],
)

CACHE_HIT = Counter(
    "recommendation_cache_hit_total",
    "Cache hit/miss events",
    ["cache_type", "hit"],
)


# ---------------------------------------------------------------------------
# Stub Clients (Replace with real gRPC clients in production)
# ---------------------------------------------------------------------------


class FeatureStoreClient:
    """
    gRPC client for the Feature Store service.
    In production, this wraps a generated gRPC stub with:
    - Connection pooling
    - Retry logic
    - Serialization/deserialization
    """

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = structlog.get_logger(component="feature_client")

    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """Fetch pre-joined user feature vector from Redis via Feature Service."""
        # In production: gRPC call to FeatureService.GetUserFeatures
        # Simulated response with realistic feature structure
        return {
            "user_id": user_id,
            "features": {
                "purchase_count_30d": 12,
                "avg_order_value": 45.50,
                "preferred_categories": ["electronics", "books"],
                "session_click_count": 5,
                "last_purchase_days_ago": 3,
                "user_segment": "high_value",
                "device_preference": "mobile",
                "price_sensitivity": 0.65,
            },
            "embedding": [0.1] * 128,  # 128-dim user embedding from Two-Tower
            "timestamp_ms": int(time.time() * 1000),
        }

    async def get_batch_item_features(self, item_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch batch item features. Uses Redis pipelining internally."""
        # In production: gRPC call to FeatureService.GetBatchItemFeatures
        return {
            item_id: {
                "item_id": item_id,
                "category": "electronics",
                "brand": "generic",
                "price": 29.99,
                "ctr_7d": 0.035,
                "stock_count": 150,
                "days_since_listing": 45,
                "avg_rating": 4.2,
                "review_count": 234,
            }
            for item_id in item_ids
        }


class RetrievalClient:
    """gRPC client for the ANN Retrieval Service (Milvus-backed)."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = structlog.get_logger(component="retrieval_client")

    async def retrieve_candidates(
        self,
        user_embedding: list[float],
        num_candidates: int,
        exclude_ids: list[str] | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Multi-source candidate retrieval:
        - ANN search (60% of candidates)
        - Collaborative filtering (20%)
        - Trending items (10%)
        - Rule-based (10%)
        """
        # In production: gRPC call to RetrievalService.RetrieveCandidates
        # Returns merged, deduplicated candidates from multiple sources
        candidates = []
        for i in range(min(num_candidates, 1000)):
            candidates.append(
                {
                    "item_id": f"item_{i:07d}",
                    "retrieval_score": max(0, 1.0 - (i * 0.001)),
                    "source": "ANN"
                    if i < 600
                    else ("CF" if i < 800 else ("TRENDING" if i < 900 else "RULE")),
                }
            )
        return candidates


class RankingClient:
    """gRPC client for the Ranking Service (Triton-backed)."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = structlog.get_logger(component="ranking_client")

    async def rank_candidates(
        self,
        user_features: dict[str, Any],
        item_features: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
        model_version: str = "dlrm-v2.3.1",
    ) -> list[dict[str, Any]]:
        """
        Score candidates using DLRM on Triton.
        Returns ranked items with calibrated P(click) scores.
        """
        # In production: gRPC call to RankingService.RankCandidates
        # Feature assembly → Triton inference → calibration
        ranked = []
        for i, candidate in enumerate(candidates):
            ranked.append(
                {
                    "item_id": candidate["item_id"],
                    "score": max(0, 0.95 - (i * 0.001)),
                    "sub_scores": {
                        "click_prob": 0.035 - (i * 0.00003),
                        "purchase_prob": 0.008 - (i * 0.000008),
                    },
                }
            )
        return sorted(ranked, key=lambda x: x["score"], reverse=True)


class ReRankingClient:
    """gRPC client for the Re-Ranking Service."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = structlog.get_logger(component="reranking_client")

    async def rerank(
        self,
        ranked_items: list[dict[str, Any]],
        diversity_lambda: float = 0.7,
        max_same_category: int = 3,
        output_size: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Apply MMR diversity + business rules.
        Returns final ordered list with position assignments.
        """
        # In production: gRPC call to ReRankingService.ReRank
        final_items = []
        for i, item in enumerate(ranked_items[:output_size]):
            final_items.append(
                {
                    "item_id": item["item_id"],
                    "position": i + 1,
                    "final_score": item["score"] * (1 - 0.01 * i),  # simulated MMR
                    "relevance_score": item["score"],
                    "diversity_score": 0.8 - (i * 0.02),
                }
            )
        return final_items


class ExperimentClient:
    """gRPC client for the Experimentation Service."""

    def __init__(self, config: ServiceConfig):
        self.config = config

    async def get_assignment(self, user_id: str) -> dict[str, Any]:
        """Get experiment variant assignment (deterministic by user_id hash)."""
        # In production: gRPC call to ExperimentationService.GetAssignment
        variant_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        return {
            "experiment_id": "exp-dlrm-v2.3.1",
            "variant": "treatment" if variant_hash < 10 else "control",
            "parameters": {
                "model_version": "dlrm-v2.3.1",
                "diversity_lambda": "0.7",
            },
        }


# ---------------------------------------------------------------------------
# Popularity Fallback Cache
# ---------------------------------------------------------------------------


class PopularityFallback:
    """
    Pre-computed popularity-based recommendations.
    Used as the last resort when the full pipeline is unavailable.
    Refreshed every 15 minutes from batch pipeline.
    """

    def __init__(self):
        # In production: loaded from Redis/S3, refreshed every 15 min
        self._global_popular: list[str] = [f"popular_{i:04d}" for i in range(100)]
        self._category_popular: dict[str, list[str]] = {
            "electronics": [f"elec_{i:04d}" for i in range(50)],
            "clothing": [f"cloth_{i:04d}" for i in range(50)],
            "books": [f"book_{i:04d}" for i in range(50)],
        }
        self._segment_recs: dict[str, list[str]] = {
            "high_value": [f"hv_{i:04d}" for i in range(50)],
            "new_user": [f"nu_{i:04d}" for i in range(50)],
            "default": [f"def_{i:04d}" for i in range(50)],
        }

    def get_fallback_recs(
        self,
        num_items: int = 20,
        category: str | None = None,
        user_segment: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get fallback recommendations with priority: segment > category > global."""
        if user_segment and user_segment in self._segment_recs:
            item_ids = self._segment_recs[user_segment][:num_items]
            source = f"segment:{user_segment}"
        elif category and category in self._category_popular:
            item_ids = self._category_popular[category][:num_items]
            source = f"category:{category}"
        else:
            item_ids = self._global_popular[:num_items]
            source = "global_popularity"

        return [
            {
                "item_id": item_id,
                "position": i + 1,
                "score": 1.0 - (i * 0.01),
                "explanation": f"Popular in {source}",
                "tracking": {"source": "fallback", "fallback_type": source},
            }
            for i, item_id in enumerate(item_ids)
        ]


# ---------------------------------------------------------------------------
# Main Recommendation Engine (Orchestrator)
# ---------------------------------------------------------------------------


class RecommendationEngine:
    """
    Core orchestrator that manages the full recommendation pipeline.

    Pipeline: Feature Fetch → Retrieval → Ranking → Re-Ranking

    Key features:
    - Parallel async execution where possible
    - Circuit breakers per downstream service
    - Latency budget management with adaptive degradation
    - Multi-level fallback chain
    - Full observability (metrics, logs, traces)
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.logger = structlog.get_logger(component="recommendation_engine")

        # Initialize clients with real gRPC stubs
        self.feature_client = FeatureStoreClient(
            config.feature_service.host,
            config.feature_service.port,
        )
        self.feature_client.set_stub_class(recommendation_pb2_grpc.FeatureServiceStub)

        self.retrieval_client = RetrievalClient(
            config.retrieval_service.host,
            config.retrieval_service.port,
        )
        self.retrieval_client.set_stub_class(recommendation_pb2_grpc.RetrievalServiceStub)

        self.ranking_client = RankingClient(
            config.ranking_service.host,
            config.ranking_service.port,
        )
        self.ranking_client.set_stub_class(recommendation_pb2_grpc.RankingServiceStub)

        self.reranking_client = ReRankingClient(
            config.reranking_service.host,
            config.reranking_service.port,
        )
        self.reranking_client.set_stub_class(recommendation_pb2_grpc.ReRankingServiceStub)

        self.experiment_client = ExperimentClient(config.experiment_service)

        # Initialize circuit breakers
        self.breakers = {
            "feature_store": CircuitBreaker(
                "feature_store",
                failure_threshold=config.feature_service.circuit_breaker_threshold,
                recovery_timeout_sec=config.feature_service.circuit_breaker_recovery_sec,
            ),
            "retrieval": CircuitBreaker(
                "retrieval",
                failure_threshold=config.retrieval_service.circuit_breaker_threshold,
                recovery_timeout_sec=config.retrieval_service.circuit_breaker_recovery_sec,
            ),
            "ranking": CircuitBreaker(
                "ranking",
                failure_threshold=config.ranking_service.circuit_breaker_threshold,
                recovery_timeout_sec=config.ranking_service.circuit_breaker_recovery_sec,
            ),
        }

        # Fallback
        self.popularity_fallback = PopularityFallback()

    async def get_recommendations(
        self,
        request: RecommendationRequest,
        request_id: str,
    ) -> RecommendationResponse:
        """
        Main entry point for generating recommendations.

        Flow:
        1. [Parallel] Experiment assignment + User feature fetch
        2. [Sequential] Candidate retrieval (requires user embedding)
        3. [Sequential] Batch item feature fetch (requires candidate IDs)
        4. [Sequential] Ranking (requires all features)
        5. [Sequential] Re-ranking (requires ranked list)

        Each step has circuit breaker protection and fallback logic.
        """
        budget = LatencyBudget(self.config.total_latency_budget_ms)
        fallback_info = FallbackInfoResponse()
        tracer = trace.get_tracer(__name__)

        # -----------------------------------------------------------------
        # Step 1: Parallel — Experiment Assignment + User Features
        # -----------------------------------------------------------------
        with tracer.start_as_current_span("parallel_init"):
            experiment_task = asyncio.create_task(self._safe_get_experiment(request.user_id))
            user_features_task = asyncio.create_task(self._safe_get_user_features(request.user_id))

            experiment, user_features = await asyncio.gather(experiment_task, user_features_task)

        COMPONENT_LATENCY.labels(component="parallel_init").observe(budget.elapsed_ms)

        # -----------------------------------------------------------------
        # Step 2: Candidate Retrieval
        # -----------------------------------------------------------------
        candidates = None
        if budget.can_afford(20):
            retrieval_start = budget.elapsed_ms
            try:
                with tracer.start_as_current_span("retrieval"):
                    # Adaptive: reduce candidates if budget is tight
                    num_candidates = self.config.num_candidates
                    if budget.remaining_ms < 50:
                        num_candidates = 200  # fast mode
                        self.logger.warning(
                            "adaptive_candidate_reduction", remaining_ms=budget.remaining_ms
                        )

                    candidates = await self.breakers["retrieval"].call(
                        self.retrieval_client.retrieve_candidates,
                        user_embedding=user_features.get("embedding", [0.0] * 128)
                        if user_features
                        else [0.0] * 128,
                        num_candidates=num_candidates,
                    )
                    CANDIDATE_COUNT.observe(len(candidates) if candidates else 0)

            except (CircuitBreakerOpenError, Exception) as e:
                self.logger.warning("retrieval_failed", error=str(e))
                FALLBACK_COUNT.labels(fallback_type="retrieval", reason=type(e).__name__).inc()

            COMPONENT_LATENCY.labels(component="retrieval").observe(
                budget.elapsed_ms - retrieval_start
            )

        # -----------------------------------------------------------------
        # Step 3: Item Feature Fetch
        # -----------------------------------------------------------------
        item_features = {}
        if candidates and budget.can_afford(15):
            feature_start = budget.elapsed_ms
            try:
                with tracer.start_as_current_span("item_features"):
                    item_ids = [c["item_id"] for c in candidates]
                    item_features = await self.breakers["feature_store"].call(
                        self.feature_client.get_batch_item_features,
                        item_ids=item_ids,
                    )
            except (CircuitBreakerOpenError, Exception) as e:
                self.logger.warning("item_feature_fetch_failed", error=str(e))
                # Continue with empty features — model can handle defaults

            COMPONENT_LATENCY.labels(component="item_features").observe(
                budget.elapsed_ms - feature_start
            )

        # -----------------------------------------------------------------
        # Step 4: Ranking
        # -----------------------------------------------------------------
        ranked_items = None
        if candidates and budget.can_afford(25):
            ranking_start = budget.elapsed_ms
            try:
                with tracer.start_as_current_span("ranking"):
                    model_version = (
                        experiment.get("parameters", {}).get("model_version", "dlrm-v2.3.1")
                        if experiment
                        else "dlrm-v2.3.1"
                    )

                    ranked_items = await self.breakers["ranking"].call(
                        self.ranking_client.rank_candidates,
                        user_features=user_features or {},
                        item_features=item_features,
                        candidates=candidates,
                        model_version=model_version,
                    )
            except (CircuitBreakerOpenError, Exception) as e:
                self.logger.warning("ranking_failed", error=str(e))
                # Fallback: use retrieval scores as ranking
                if candidates:
                    ranked_items = sorted(
                        candidates, key=lambda x: x.get("retrieval_score", 0), reverse=True
                    )
                    fallback_info.fallback_used = True
                    fallback_info.fallback_reason = "ranking_circuit_open"
                    fallback_info.fallback_type = "retrieval_scores"
                    FALLBACK_COUNT.labels(fallback_type="ranking", reason="circuit_open").inc()

            COMPONENT_LATENCY.labels(component="ranking").observe(budget.elapsed_ms - ranking_start)

        # -----------------------------------------------------------------
        # Step 5: Re-Ranking
        # -----------------------------------------------------------------
        final_items = None
        if ranked_items and budget.can_afford(5):
            rerank_start = budget.elapsed_ms
            try:
                with tracer.start_as_current_span("reranking"):
                    diversity_lambda = float(
                        experiment.get("parameters", {}).get("diversity_lambda", "0.7")
                        if experiment
                        else 0.7
                    )

                    final_items = await self.reranking_client.rerank(
                        ranked_items=ranked_items,
                        diversity_lambda=diversity_lambda,
                        output_size=request.num_items,
                    )
            except Exception as e:
                self.logger.warning("reranking_failed", error=str(e))
                # Fallback: skip re-ranking, use ranked list directly
                final_items = [
                    {
                        "item_id": item.get("item_id"),
                        "position": i + 1,
                        "final_score": item.get("score", item.get("retrieval_score", 0)),
                    }
                    for i, item in enumerate(ranked_items[: request.num_items])
                ]

            COMPONENT_LATENCY.labels(component="reranking").observe(
                budget.elapsed_ms - rerank_start
            )

        # -----------------------------------------------------------------
        # Final Fallback: Popularity
        # -----------------------------------------------------------------
        if not final_items:
            self.logger.warning("full_pipeline_fallback", user_id=request.user_id)
            fallback_info.fallback_used = True
            fallback_info.fallback_reason = "pipeline_failure"
            fallback_info.fallback_type = "popularity"
            FALLBACK_COUNT.labels(fallback_type="popularity", reason="pipeline_failure").inc()

            fallback_recs = self.popularity_fallback.get_fallback_recs(
                num_items=request.num_items,
            )
            final_items = fallback_recs

        # -----------------------------------------------------------------
        # Build Response
        # -----------------------------------------------------------------
        total_latency_ms = int(budget.elapsed_ms)

        response_items = [
            RecommendedItemResponse(
                item_id=item.get("item_id", ""),
                position=item.get("position", i + 1),
                score=round(item.get("final_score", item.get("score", 0)), 4),
                explanation=item.get("explanation"),
                tracking={
                    "request_id": request_id,
                    "experiment_id": experiment.get("experiment_id", "") if experiment else "",
                    "model_version": experiment.get("parameters", {}).get("model_version", "")
                    if experiment
                    else "",
                    "position": str(item.get("position", i + 1)),
                    **(item.get("tracking", {})),
                },
            )
            for i, item in enumerate(final_items[: request.num_items])
        ]

        REQUEST_LATENCY.labels(
            page_context=request.page_context.value,
            fallback_type=fallback_info.fallback_type or "none",
        ).observe(total_latency_ms)

        REQUEST_COUNT.labels(
            page_context=request.page_context.value,
            status="success",
        ).inc()

        self.logger.info(
            "recommendation_served",
            request_id=request_id,
            user_id=request.user_id,
            page_context=request.page_context.value,
            num_items_returned=len(response_items),
            latency_ms=total_latency_ms,
            fallback_used=fallback_info.fallback_used,
            experiment_id=experiment.get("experiment_id") if experiment else None,
        )

        return RecommendationResponse(
            items=response_items,
            request_id=request_id,
            experiment_id=experiment.get("experiment_id") if experiment else None,
            total_latency_ms=total_latency_ms,
            fallback=fallback_info,
        )

    async def _safe_get_experiment(self, user_id: str) -> dict[str, Any] | None:
        """Fetch experiment assignment. Non-critical — default to control on failure."""
        try:
            return await asyncio.wait_for(
                self.experiment_client.get_assignment(user_id),
                timeout=0.005,  # 5ms timeout
            )
        except Exception:
            return None

    async def _safe_get_user_features(self, user_id: str) -> dict[str, Any] | None:
        """Fetch user features with circuit breaker. Fallback to defaults."""
        try:
            return await self.breakers["feature_store"].call(
                self.feature_client.get_user_features,
                user_id=user_id,
            )
        except (CircuitBreakerOpenError, Exception) as e:
            self.logger.warning("user_feature_fallback", error=str(e))
            FALLBACK_COUNT.labels(fallback_type="features", reason=type(e).__name__).inc()
            # Return default feature vector for user segment
            return {
                "user_id": user_id,
                "features": {"user_segment": "default"},
                "embedding": [0.0] * 128,
                "timestamp_ms": int(time.time() * 1000),
            }


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

config = ServingConfig()
engine: RecommendationEngine | None = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global engine, start_time
    start_time = time.monotonic()

    # Validate required configuration at startup
    from config import verify_startup_config

    verify_startup_config()

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )

    # Initialize engine
    engine = RecommendationEngine(config)

    logger = structlog.get_logger()
    logger.info(
        "recommendation_service_started",
        config={
            "total_latency_budget_ms": config.total_latency_budget_ms,
            "num_candidates": config.num_candidates,
        },
    )

    yield

    logger.info("recommendation_service_shutdown")


app = FastAPI(
    title="Recommendation Service",
    description="Production-grade real-time product recommendation API",
    version="2.3.1",
    lifespan=lifespan,
)

# CORS
ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# OpenTelemetry instrumentation
FastAPIInstrumentor.instrument_app(app)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/v1/recommendations",
    response_model=RecommendationResponse,
    summary="Get personalized recommendations",
    tags=["Recommendations"],
)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Generate personalized product recommendations.

    Pipeline: Features → Retrieval (ANN) → Ranking (DLRM) → Re-Ranking (MMR)

    SLA: p99 < 75ms
    """
    request_id = str(uuid.uuid4())
    return await engine.get_recommendations(request, request_id)


@app.post(
    "/v1/similar-items",
    response_model=RecommendationResponse,
    summary="Get similar items",
    tags=["Recommendations"],
)
async def get_similar_items(request: SimilarItemsRequest) -> RecommendationResponse:
    """
    Get items similar to a given item (item-to-item).
    Uses item embedding similarity from the Two-Tower model.
    """
    request_id = str(uuid.uuid4())
    # Convert to standard recommendation flow with item context
    rec_request = RecommendationRequest(
        user_id=request.user_id or "anonymous",
        page_context=PageContextEnum.PDP,
        num_items=request.num_items,
        client_context={"seed_item_id": request.item_id},
    )
    return await engine.get_recommendations(rec_request, request_id)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
async def health_check() -> HealthResponse:
    """Service health check with dependency status."""
    uptime = time.monotonic() - start_time

    dependencies = {}
    for name, breaker in engine.breakers.items():
        dependencies[name] = breaker.state.value

    return HealthResponse(
        status="serving",
        version="2.3.1",
        uptime_seconds=round(uptime, 1),
        dependencies=dependencies,
    )


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    tags=["Operations"],
)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; charset=utf-8",
    )


@app.get(
    "/ready",
    summary="Readiness probe",
    tags=["Operations"],
)
async def readiness():
    """Kubernetes readiness probe."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Check critical dependencies
    critical_open = any(
        breaker.state == CircuitState.OPEN
        for name, breaker in engine.breakers.items()
        if name in ("feature_store", "retrieval")
    )

    if critical_open:
        raise HTTPException(
            status_code=503,
            detail="Critical circuit breaker open",
        )

    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        workers=int(os.getenv("WORKERS", "4")),
        log_level="info",
        access_log=False,  # Use structured logging instead
    )
