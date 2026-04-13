"""
Ranking Service
===============
Model inference service using NVIDIA Triton for DLRM ranking.

Architecture:
- Triton Inference Server handles model loading, batching, GPU management
- This service wraps Triton with feature assembly + calibration
- Fallback to XGBoost (CPU) when GPU is unavailable

Pipeline: Feature Assembly → Model Inference → Score Calibration

Models:
- Primary: DLRM (Deep Learning Recommendation Model) on TensorRT INT8
- Fallback: XGBoost gradient boosted trees (CPU, ~5% CTR drop)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# In production:
# import tritonclient.grpc.aio as triton_grpc
# import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankingConfig:
    # Triton configuration
    triton_host: str = os.getenv("TRITON_HOST", "triton-inference")
    triton_port: int = int(os.getenv("TRITON_PORT", "8001"))
    
    # Model configuration
    dlrm_model_name: str = "dlrm_ranking"
    dlrm_model_version: str = "2"    # Triton model version
    xgboost_model_path: str = os.getenv("XGBOOST_MODEL_PATH", "/models/xgboost_baseline.json")
    
    # Feature configuration
    num_dense_features: int = 26
    num_sparse_features: int = 50
    embedding_dim: int = 128
    max_batch_size: int = 1024
    
    # Calibration
    calibration_temperature: float = 1.15  # Platt scaling temperature
    calibration_bias: float = -0.02
    
    # Performance
    inference_timeout_ms: int = 25
    grpc_port: int = int(os.getenv("GRPC_PORT", "50053"))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

RANKING_LATENCY = Histogram(
    "ranking_latency_ms",
    "Ranking latency by stage",
    ["stage"],
    buckets=[1, 2, 5, 8, 10, 15, 20, 25, 30, 50],
)

RANKING_BATCH_SIZE = Histogram(
    "ranking_batch_size",
    "Batch size for ranking inference",
    buckets=[50, 100, 200, 500, 800, 1000, 1500],
)

MODEL_VERSION_COUNTER = Counter(
    "ranking_model_version_total",
    "Inference requests by model version",
    ["model", "version"],
)

FALLBACK_COUNTER = Counter(
    "ranking_fallback_total",
    "Ranking fallback events",
    ["reason"],
)

GPU_UTILIZATION = Gauge(
    "ranking_gpu_utilization_percent",
    "GPU utilization from Triton metrics",
)


# ---------------------------------------------------------------------------
# Feature Assembly
# ---------------------------------------------------------------------------

class FeatureAssembler:
    """
    Assembles and preprocesses features for model inference.
    
    Converts raw feature dictionaries into dense/sparse tensors
    that match the model's expected input format.
    
    Feature groups:
    1. Dense features (26): numerical features, normalized
    2. Sparse features (50): categorical features, embedding indices
    3. Cross features: user × item interaction features
    """
    
    # Numerical feature normalization parameters (from training statistics)
    FEATURE_STATS = {
        "price": {"mean": 45.0, "std": 35.0},
        "ctr_7d": {"mean": 0.03, "std": 0.02},
        "avg_order_value": {"mean": 50.0, "std": 40.0},
        "purchase_count_30d": {"mean": 5.0, "std": 8.0},
        "days_since_listing": {"mean": 90.0, "std": 60.0},
        "review_count": {"mean": 50.0, "std": 100.0},
        "avg_rating": {"mean": 3.8, "std": 0.8},
        "session_click_count": {"mean": 8.0, "std": 6.0},
        "price_sensitivity": {"mean": 0.5, "std": 0.25},
        "last_purchase_days_ago": {"mean": 15.0, "std": 20.0},
    }
    
    # Dense feature order (must match training)
    DENSE_FEATURE_ORDER = [
        "price", "ctr_7d", "avg_order_value", "purchase_count_30d",
        "avg_rating", "review_count", "days_since_listing", "stock_count",
        "session_click_count", "price_sensitivity", "last_purchase_days_ago",
        "cart_abandonment_rate", "click_count_7d", "avg_session_duration_sec",
        "registration_days_ago", "view_count_24h", "purchase_count_7d",
        # Cross features (computed at serving time)
        "price_vs_avg_spend",        # item price / user avg order value
        "category_affinity_score",   # user preference for item's category
        "brand_affinity_score",      # user preference for item's brand
        "recency_score",             # 1.0 / (1.0 + days_since_listing)
        "price_percentile",          # item price percentile in category
        "user_item_ctr_history",     # historical CTR for this user-item pair
        "hour_of_day",               # request hour (0-23)
        "day_of_week",               # request day (0-6)
        "is_weekend",                # boolean
    ]
    
    # Sparse feature → vocabulary mapping
    SPARSE_FEATURES = [
        "user_id_hash", "item_id_hash", "category", "brand", "city",
        "device_type", "os", "browser", "user_segment", "price_bucket",
    ]
    
    def __init__(self, config: RankingConfig):
        self.config = config
        self.logger = structlog.get_logger(component="feature_assembler")
    
    def assemble_batch(
        self,
        user_features: dict[str, Any],
        item_features_list: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Assemble features for a batch of (user, item) pairs.
        
        Returns:
            Dictionary with 'dense_features' and 'sparse_features' numpy arrays
            ready for Triton inference.
        """
        batch_size = len(item_features_list)
        dense_batch = np.zeros((batch_size, self.config.num_dense_features), dtype=np.float32)
        sparse_batch = np.zeros((batch_size, self.config.num_sparse_features), dtype=np.int64)
        
        user_feats = user_features.get("features", {})
        
        for i, item in enumerate(item_features_list):
            item_feats = item.get("features", item)
            
            # Dense features
            dense_vec = self._compute_dense_features(user_feats, item_feats, context)
            dense_batch[i] = dense_vec
            
            # Sparse features
            sparse_vec = self._compute_sparse_features(user_feats, item_feats)
            sparse_batch[i] = sparse_vec
        
        return {
            "dense_features": dense_batch,
            "sparse_features": sparse_batch,
        }
    
    def _compute_dense_features(
        self,
        user_feats: dict,
        item_feats: dict,
        context: dict | None,
    ) -> np.ndarray:
        """Compute and normalize dense features for one (user, item) pair."""
        features = np.zeros(self.config.num_dense_features, dtype=np.float32)
        
        # Raw features with normalization
        for idx, feat_name in enumerate(self.DENSE_FEATURE_ORDER):
            if idx >= self.config.num_dense_features:
                break
            
            raw_value = 0.0
            
            # User features
            if feat_name in user_feats:
                raw_value = float(user_feats[feat_name])
            # Item features
            elif feat_name in item_feats:
                raw_value = float(item_feats[feat_name])
            # Cross features (computed)
            elif feat_name == "price_vs_avg_spend":
                price = float(item_feats.get("price", 0))
                avg_spend = float(user_feats.get("avg_order_value", 50))
                raw_value = price / max(avg_spend, 1.0)
            elif feat_name == "recency_score":
                days = float(item_feats.get("days_since_listing", 90))
                raw_value = 1.0 / (1.0 + days)
            elif feat_name == "hour_of_day" and context:
                raw_value = float(context.get("hour", 12))
            elif feat_name == "is_weekend" and context:
                raw_value = 1.0 if context.get("is_weekend", False) else 0.0
            
            # Z-score normalization
            if feat_name in self.FEATURE_STATS:
                stats = self.FEATURE_STATS[feat_name]
                raw_value = (raw_value - stats["mean"]) / max(stats["std"], 1e-6)
            
            features[idx] = raw_value
        
        return features
    
    def _compute_sparse_features(
        self,
        user_feats: dict,
        item_feats: dict,
    ) -> np.ndarray:
        """Compute sparse feature indices for embedding lookup."""
        features = np.zeros(self.config.num_sparse_features, dtype=np.int64)
        
        for idx, feat_name in enumerate(self.SPARSE_FEATURES):
            if idx >= self.config.num_sparse_features:
                break
            
            if feat_name.endswith("_hash"):
                # Hash-based features (for high-cardinality like user_id, item_id)
                base_name = feat_name.replace("_hash", "")
                value = str(user_feats.get(base_name, item_feats.get(base_name, "")))
                features[idx] = hash(value) % 1_000_000  # vocabulary size
            else:
                # Categorical features — use vocabulary index
                value = str(user_feats.get(feat_name, item_feats.get(feat_name, "unknown")))
                features[idx] = hash(value) % 10_000
        
        return features


# ---------------------------------------------------------------------------
# Triton Model Client
# ---------------------------------------------------------------------------

class TritonRankingClient:
    """
    Wrapper around Triton Inference Server gRPC client.
    
    Handles:
    - Model input/output tensor management
    - Dynamic batching (Triton-side, configured in model config)
    - TensorRT INT8 inference
    - Version management for A/B testing
    """
    
    def __init__(self, config: RankingConfig):
        self.config = config
        self.logger = structlog.get_logger(component="triton_client")
        self._client = None
    
    async def initialize(self):
        """Connect to Triton and verify model is loaded."""
        self.logger.info("triton_connecting", host=self.config.triton_host)
        
        # In production:
        # self._client = triton_grpc.InferenceServerClient(
        #     url=f"{self.config.triton_host}:{self.config.triton_port}",
        # )
        # assert await self._client.is_model_ready(self.config.dlrm_model_name)
        
        self.logger.info("triton_connected")
    
    async def predict(
        self,
        dense_features: np.ndarray,
        sparse_features: np.ndarray,
        model_version: str = "",
    ) -> np.ndarray:
        """
        Run DLRM inference on Triton.
        
        Args:
            dense_features: (batch_size, 26) float32
            sparse_features: (batch_size, 50) int64
            model_version: Triton model version for A/B testing
        
        Returns:
            (batch_size,) float32 array of predicted P(click)
        """
        start = time.monotonic()
        batch_size = dense_features.shape[0]
        RANKING_BATCH_SIZE.observe(batch_size)
        
        try:
            # In production Triton inference:
            # inputs = [
            #     triton_grpc.InferInput("dense_features", dense_features.shape, "FP32"),
            #     triton_grpc.InferInput("sparse_features", sparse_features.shape, "INT64"),
            # ]
            # inputs[0].set_data_from_numpy(dense_features)
            # inputs[1].set_data_from_numpy(sparse_features)
            #
            # outputs = [triton_grpc.InferRequestedOutput("predictions")]
            #
            # response = await self._client.infer(
            #     model_name=self.config.dlrm_model_name,
            #     model_version=model_version or self.config.dlrm_model_version,
            #     inputs=inputs,
            #     outputs=outputs,
            #     timeout=self.config.inference_timeout_ms / 1000,
            # )
            #
            # predictions = response.as_numpy("predictions").flatten()
            
            # Simulated inference
            predictions = np.random.uniform(0.01, 0.10, size=batch_size).astype(np.float32)
            predictions = np.sort(predictions)[::-1]  # Sort descending for realistic output
            
            latency_ms = (time.monotonic() - start) * 1000
            RANKING_LATENCY.labels(stage="triton_inference").observe(latency_ms)
            MODEL_VERSION_COUNTER.labels(
                model="dlrm",
                version=model_version or self.config.dlrm_model_version,
            ).inc()
            
            return predictions
            
        except Exception as e:
            self.logger.error("triton_inference_failed", error=str(e), batch_size=batch_size)
            FALLBACK_COUNTER.labels(reason="triton_error").inc()
            raise


# ---------------------------------------------------------------------------
# XGBoost Fallback
# ---------------------------------------------------------------------------

class XGBoostFallback:
    """
    CPU-based XGBoost ranking model as fallback.
    
    Used when:
    - Triton/GPU is unavailable
    - Model load failure
    - GPU OOM
    
    Performance:
    - Latency: ~30ms for 1000 items (CPU inference)
    - Quality: AUC ~3-5% lower than DLRM
    - CTR impact: ~5-8% drop
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.logger = structlog.get_logger(component="xgboost_fallback")
        self._model = None
    
    def load(self):
        """Load XGBoost model."""
        # In production:
        # self._model = xgb.Booster()
        # self._model.load_model(self.model_path)
        self.logger.info("xgboost_loaded", path=self.model_path)
    
    def predict(self, dense_features: np.ndarray) -> np.ndarray:
        """
        Run XGBoost inference (CPU).
        
        Args:
            dense_features: (batch_size, 26) float32
        
        Returns:
            (batch_size,) float32 array of predicted P(click)
        """
        start = time.monotonic()
        
        # In production:
        # dmatrix = xgb.DMatrix(dense_features)
        # predictions = self._model.predict(dmatrix)
        
        # Simulated
        batch_size = dense_features.shape[0]
        predictions = np.random.uniform(0.01, 0.08, size=batch_size).astype(np.float32)
        
        latency_ms = (time.monotonic() - start) * 1000
        RANKING_LATENCY.labels(stage="xgboost_inference").observe(latency_ms)
        MODEL_VERSION_COUNTER.labels(model="xgboost", version="baseline").inc()
        
        return predictions


# ---------------------------------------------------------------------------
# Score Calibration
# ---------------------------------------------------------------------------

class ScoreCalibrator:
    """
    Platt scaling calibration for model scores.
    
    Ensures predicted probabilities are well-calibrated:
    - If model predicts P(click)=0.05, approximately 5% of those items are clicked
    - Temperature and bias learned from calibration dataset (held-out)
    - Recalibrated weekly after model retraining
    """
    
    def __init__(self, temperature: float = 1.15, bias: float = -0.02):
        self.temperature = temperature
        self.bias = bias
    
    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling: calibrated = sigmoid(raw / temperature + bias)."""
        logits = raw_scores / self.temperature + self.bias
        calibrated = 1.0 / (1.0 + np.exp(-logits))
        return calibrated.astype(np.float32)


# ---------------------------------------------------------------------------
# Ranking Service
# ---------------------------------------------------------------------------

class RankingService:
    """
    Main ranking service that orchestrates:
    1. Feature assembly
    2. Model inference (DLRM primary, XGBoost fallback)
    3. Score calibration
    4. Result sorting
    """
    
    def __init__(self, config: RankingConfig):
        self.config = config
        self.logger = structlog.get_logger(component="ranking_service")
        
        self.feature_assembler = FeatureAssembler(config)
        self.triton_client = TritonRankingClient(config)
        self.xgboost_fallback = XGBoostFallback(config.xgboost_model_path)
        self.calibrator = ScoreCalibrator(
            temperature=config.calibration_temperature,
            bias=config.calibration_bias,
        )
    
    async def initialize(self):
        await self.triton_client.initialize()
        self.xgboost_fallback.load()
    
    async def rank_candidates(
        self,
        user_features: dict[str, Any],
        item_features_map: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
        model_version: str = "",
        request_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rank candidates using DLRM with XGBoost fallback.
        
        Args:
            user_features: pre-fetched user feature vector
            item_features_map: item_id → feature dict
            candidates: list of {item_id, retrieval_score, source}
            model_version: for A/B testing (routes to specific Triton model version)
            request_context: time, device, etc.
        
        Returns:
            List of {item_id, score, sub_scores} sorted by score descending
        """
        start = time.monotonic()
        
        # Step 1: Feature assembly
        assembly_start = time.monotonic()
        
        item_features_list = []
        valid_item_ids = []
        for candidate in candidates:
            item_id = candidate["item_id"]
            if item_id in item_features_map:
                item_features_list.append(item_features_map[item_id])
                valid_item_ids.append(item_id)
            else:
                # Use default features for items not in feature store
                item_features_list.append({"features": {}})
                valid_item_ids.append(item_id)
        
        assembled = self.feature_assembler.assemble_batch(
            user_features=user_features,
            item_features_list=item_features_list,
            context=request_context,
        )
        
        RANKING_LATENCY.labels(stage="feature_assembly").observe(
            (time.monotonic() - assembly_start) * 1000
        )
        
        # Step 2: Model inference (DLRM → XGBoost fallback)
        try:
            raw_scores = await self.triton_client.predict(
                dense_features=assembled["dense_features"],
                sparse_features=assembled["sparse_features"],
                model_version=model_version,
            )
            used_model = "dlrm"
        except Exception as e:
            self.logger.warning("dlrm_fallback_to_xgboost", error=str(e))
            FALLBACK_COUNTER.labels(reason="dlrm_failed").inc()
            raw_scores = self.xgboost_fallback.predict(assembled["dense_features"])
            used_model = "xgboost"
        
        # Step 3: Calibration
        calibrated_scores = self.calibrator.calibrate(raw_scores)
        
        # Step 4: Build ranked results
        ranked_results = []
        for i, (item_id, score) in enumerate(zip(valid_item_ids, calibrated_scores)):
            ranked_results.append({
                "item_id": item_id,
                "score": float(score),
                "sub_scores": {
                    "click_prob": float(score),
                    "raw_score": float(raw_scores[i]),
                    "retrieval_score": float(
                        candidates[i].get("retrieval_score", 0) if i < len(candidates) else 0
                    ),
                },
            })
        
        # Sort by score descending
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        total_latency_ms = (time.monotonic() - start) * 1000
        RANKING_LATENCY.labels(stage="total").observe(total_latency_ms)
        
        self.logger.info(
            "ranking_complete",
            num_candidates=len(candidates),
            num_ranked=len(ranked_results),
            model=used_model,
            latency_ms=round(total_latency_ms, 2),
            top_score=round(ranked_results[0]["score"], 4) if ranked_results else 0,
        )
        
        return ranked_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def serve():
    config = RankingConfig()
    service = RankingService(config)
    await service.initialize()
    
    start_http_server(9092)
    
    logger = structlog.get_logger()
    logger.info("ranking_service_started", grpc_port=config.grpc_port)
    
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        pass


if __name__ == "__main__":
    asyncio.run(serve())
