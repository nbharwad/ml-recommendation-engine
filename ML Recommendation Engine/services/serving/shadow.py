"""
Shadow Deployment Service
===================
Log shadow model predictions without serving them.

Use Case:
- Test new models in production traffic
- Compare predictions without risk
- Gradual rollout with canary

Implementation:
- Istio traffic splitting (e.g., 1% shadow)
- Log predictions from shadow model
- Compare offline with baseline
- No user-facing impact

Industry Standard:
- Used by Netflix, Amazon, Google for safe rollouts
- Typical: 1-10% shadow traffic
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum

import structlog
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger(component="shadow")


SHADOW_REQUESTS = Counter(
    "shadow_requests_total",
    "Requests to shadow model",
    ["model_version", "outcome"],
)

SHADOW_LATENCY = Histogram(
    "shadow_latency_ms",
    "Shadow model inference latency",
    buckets=[1, 2, 5, 10, 20, 50, 100],
)

SHADOW_PREDICTIONS = Counter(
    "shadow_predictions_logged",
    "Shadow predictions logged to storage",
    ["model_version"],
)

SHADOW_COMPARISON = Histogram(
    "shadow_prediction_diff",
    "Difference between shadow and production",
    buckets=[0, 0.01, 0.05, 0.1, 0.2, 0.5],
)

SHADOW_TRAFFIC = Gauge(
    "shadow_traffic_percent",
    "Current shadow traffic percentage",
)


class ShadowOutcome(str, Enum):
    """Outcome of shadow request."""
    PROD_SERVED = "prod_served"
    SHADOW_LOGGED = "shadow_logged"
    BOTH = "both"
    SKIPPED = "skipped"


@dataclass
class ShadowConfig:
    """Shadow deployment configuration."""
    shadow_percent: float = 1.0
    models: list[str] = field(default_factory=lambda: ["ranking-v2"])
    log_predictions: bool = True
    compare_predictions: bool = True
    storage_topic: str = "shadow-predictions"
    enable_metrics: bool = True


@dataclass
class ShadowResult:
    """Result from shadow prediction."""
    request_id: str
    model_version: str
    predictions: list[dict]
    latency_ms: float
    shadow_score: float
    prod_score: Optional[float] = None
    difference: Optional[float] = None


class ShadowService:
    """
    Shadow deployment handler.
    
    Flow:
    1. Request arrives
    2. Randomly decides shadow vs production (based on %)
    3. If shadow:
       - Log request to shadow pipeline
       - Run shadow model (async)
       - Log predictions for offline comparison
    4. Serve production response
    5. Update metrics
    """
    
    def __init__(self, config: ShadowConfig):
        self.config = config
        self.logger = structlog.get_logger(component="shadow_service")
        self._shadow_rng = random.Random()
        
        SHADOW_TRAFFIC.set(config.shadow_percent)
        
    def should_shadow(self, user_id: str) -> bool:
        """
        Decide if this request should go to shadow.
        
        Uses deterministic hashing for consistent user experience.
        """
        # Hash-based assignment (same user always gets same treatment)
        user_hash = hash(f"shadow:{user_id}")
        threshold = int(self.config.shadow_percent * 10000)
        
        return (user_hash % 10000) < threshold
    
    async def execute_shadow(
        self,
        request_id: str,
        user_id: str,
        model_executor: Callable,
        production_predictions: list[dict],
    ) -> ShadowResult:
        """
        Execute shadow prediction and logging.
        
        Args:
            request_id: unique request ID
            user_id: user making request
            model_executor: async function to get model predictions
            production_predictions: production model output
            
        Returns:
            ShadowResult with predictions and comparison metrics
        """
        start = time.perf_counter()
        
        # Run shadow model (production continues in parallel)
        shadow_predictions = None
        shadow_score = None
        
        if self.should_shadow(user_id):
            SHADOW_REQUESTS.labels(
                model_version=self.config.models[0],
                outcome="shadow_triggered",
            ).inc()
            
            try:
                # Execute shadow model
                shadow_predictions = await model_executor()
                
                if shadow_predictions:
                    # Get top score for comparison
                    shadow_score = shadow_predictions[0].get("score", 0.0)
                    
                    # Calculate difference
                    if production_predictions:
                        prod_score = production_predictions[0].get("score", 0.0)
                        difference = abs(shadow_score - prod_score)
                        
                        SHADOW_COMPARISON.observe(difference)
                        
                        self.logger.info(
                            "shadow_comparison",
                            request_id=request_id,
                            shadow_score=shadow_score,
                            prod_score=prod_score,
                            difference=difference,
                        )
                    
                    # Log to storage (Kafka/Kinesis for offline analysis)
                    if self.config.log_predictions:
                        await self._log_shadow_predictions(
                            request_id,
                            user_id,
                            shadow_predictions,
                        )
                        
            except Exception as e:
                self.logger.error("shadow_prediction_failed", error=str(e))
                SHADOW_REQUESTS.labels(
                    model_version=self.config.models[0],
                    outcome="shadow_error",
                ).inc()
        
        latency_ms = (time.perf_counter() - start) * 1000
        SHADOW_LATENCY.observe(latency_ms)
        
        return ShadowResult(
            request_id=request_id,
            model_version=self.config.models[0],
            predictions=shadow_predictions or [],
            latency_ms=latency_ms,
            shadow_score=shadow_score or 0.0,
        )
    
    async def _log_shadow_predictions(
        self,
        request_id: str,
        user_id: str,
        predictions: list[dict],
    ):
        """Log shadow predictions to storage for offline analysis."""
        if not predictions:
            return
            
        logEntry = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp_ms": int(time.time() * 1000),
            "model_version": self.config.models[0],
            "top_predictions": predictions[:10],
        }
        
        # In production: send to Kafka/Kinesis
        # await kafka_producer.send(self.config.storage_topic, logEntry)
        
        SHADOW_PREDICTIONS.labels(
            model_version=self.config.models[0],
        ).inc()
        
        self.logger.debug(
            "shadow_logged",
            request_id=request_id,
            num_predictions=len(predictions),
        )
    
    async def get_shadow_comparison(
        self,
        model_version: str,
    ) -> dict:
        """
        Get offline comparison metrics for shadow model.
        
        Returns metrics comparing shadow vs production.
        """
        # In production: query from analytics system
        return {
            "model_version": model_version,
            "shadow_requests": 0,
            "avg_difference": 0.0,
            "significant_diffs": 0,
            "better_count": 0,
            "worse_count": 0,
        }


class TrafficSplitter:
    """
    Istio-compatible traffic splitting.
    
    Used for canary rollouts:
    - 1% → new version
    - 99% → production
    """
    
    def __init__(self, canary_percent: float = 1.0):
        self.canary_percent = canary_percent
        self.logger = structlog.get_logger(component="traffic_splitter")
    
    def get_version(self, user_id: str) -> str:
        """Get version for user based on canary percentage."""
        # Consistent hashing for same user
        user_hash = hash(f"canary:{user_id}")
        threshold = int(self.canary_percent * 10000)
        
        return "canary" if (user_hash % 10000) < threshold else "production"


async def create_shadow_handler(
    shadow_percent: float = 1.0,
    model_version: str = "ranking-v2",
) -> ShadowService:
    """Factory function to create shadow handler."""
    config = ShadowConfig(
        shadow_percent=shadow_percent,
        models=[model_version],
    )
    return ShadowService(config)


# Analysis functions
async def analyze_shadow_results(
    shadow_results: list[ShadowResult],
    production_results: list[list[dict]],
) -> dict:
    """
    Analyze shadow vs production results.
    
    Returns comparison metrics.
    """
    if not shadow_results or not production_results:
        return {"error": "No results to analyze"}
    
    differences = []
    better = 0
    worse = 0
    same = 0
    
    for shadow, prod in zip(shadow_results, production_results):
        if shadow and prod:
            diff = abs(shadow.shadow_score - prod[0].get("score", 0.0))
            differences.append(diff)
            
            if diff > 0.05:
                if shadow.shadow_score > prod[0].get("score", 0.0):
                    better += 1
                else:
                    worse += 1
            else:
                same += 1
    
    return {
        "total_compared": len(differences),
        "avg_difference": sum(differences) / len(differences) if differences else 0,
        "better_than_prod": better,
        "worse_than_prod": worse,
        "within_5pct": same,
    }


STATS = {}


def get_shadow_stats() -> dict:
    """Get shadow deployment statistics."""
    return STATS