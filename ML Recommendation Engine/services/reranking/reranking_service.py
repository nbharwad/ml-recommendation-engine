"""
Re-Ranking Service
==================
Post-ranking diversification and business rule application.

Components:
1. MMR (Maximal Marginal Relevance): diversity-aware re-ordering
2. Business Rules Engine: promotional slots, category limits, freshness boost
3. Fairness Constraints: brand diversity, price range coverage

Design: Runs in-process with the serving layer for zero network overhead.
Alternatively deployed as separate gRPC service for independent scaling.

Latency budget: <5ms p99
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog
from prometheus_client import Histogram, Counter, Gauge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReRankingConfig:
    """Re-ranking parameters (hot-reloadable via config service)."""
    diversity_lambda: float = 0.7        # MMR λ: 0=max diversity, 1=max relevance
    output_size: int = 20                # final list size
    max_same_category: int = 3           # in top 10
    max_same_brand: int = 2              # in top 10
    freshness_boost_days: int = 7        # boost items newer than N days
    freshness_boost_factor: float = 1.2  # multiplicative boost
    promotion_slots: list[int] = field(default_factory=lambda: [2, 6])  # 0-indexed
    max_promotions: int = 2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

RERANK_LATENCY = Histogram(
    "reranking_latency_ms", "Re-ranking latency",
    ["stage"],
    buckets=[0.5, 1, 2, 3, 4, 5, 8, 10],
)

DIVERSITY_SCORE = Gauge(
    "reranking_diversity_score",
    "Diversity metrics",
    ["metric"],
)

BUSINESS_RULE_APPLIED = Counter(
    "reranking_business_rule_applied_total",
    "Business rule applications",
    ["rule_type"],
)


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance)
# ---------------------------------------------------------------------------

class MMRReRanker:
    """
    Maximal Marginal Relevance for diversity-aware re-ranking.
    
    MMR(item) = λ × Relevance(item) - (1-λ) × max_sim(item, selected)
    
    where:
    - λ controls relevance vs diversity trade-off
    - Relevance is the ranking model score
    - Similarity is cosine similarity of item embeddings
    
    Greedy algorithm: O(n×k) where n=candidates, k=output_size
    At n=100, k=20: ~2000 operations → <1ms
    """
    
    def __init__(self, lambda_: float = 0.7):
        self.lambda_ = lambda_
        self.logger = structlog.get_logger(component="mmr")
    
    def rerank(
        self,
        items: list[dict[str, Any]],
        embeddings: dict[str, np.ndarray] | None = None,
        output_size: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Apply MMR re-ranking.
        
        Args:
            items: ranked items with 'item_id' and 'score'
            embeddings: item_id → embedding vector (for similarity)
            output_size: number of items to select
        
        Returns:
            MMR-reranked items with diversity and relevance scores
        """
        if not items:
            return []
        
        n = len(items)
        output_size = min(output_size, n)
        
        # Normalize relevance scores to [0, 1]
        scores = np.array([item["score"] for item in items], dtype=np.float32)
        if scores.max() > scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.ones(n, dtype=np.float32)
        
        # Precompute similarity matrix if embeddings available
        if embeddings:
            emb_matrix = np.array([
                embeddings.get(item["item_id"], np.zeros(128))
                for item in items
            ], dtype=np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            emb_matrix = emb_matrix / norms
            sim_matrix = emb_matrix @ emb_matrix.T
        else:
            # Fallback: use score difference as approximate similarity
            sim_matrix = np.zeros((n, n), dtype=np.float32)
        
        # Greedy MMR selection
        selected_indices: list[int] = []
        remaining = set(range(n))
        
        for _ in range(output_size):
            if not remaining:
                break
            
            best_idx = -1
            best_mmr = float("-inf")
            
            for idx in remaining:
                relevance = norm_scores[idx]
                
                # Max similarity to already selected items
                if selected_indices:
                    max_sim = max(sim_matrix[idx][j] for j in selected_indices)
                else:
                    max_sim = 0.0
                
                mmr_score = self.lambda_ * relevance - (1 - self.lambda_) * max_sim
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.discard(best_idx)
        
        # Build result with scores
        result = []
        for position, idx in enumerate(selected_indices):
            item = items[idx].copy()
            item["relevance_score"] = float(norm_scores[idx])
            item["diversity_score"] = float(
                1.0 - max(sim_matrix[idx][j] for j in selected_indices if j != idx)
                if len(selected_indices) > 1 else 1.0
            )
            item["position"] = position + 1
            result.append(item)
        
        return result


# ---------------------------------------------------------------------------
# Business Rules Engine
# ---------------------------------------------------------------------------

class BusinessRulesEngine:
    """
    Pre-compiled business rules that modify the final ranked list.
    
    Rules are applied in priority order:
    1. Filters (must remove): out-of-stock, blocked items
    2. Constraints (must satisfy): max same category, max same brand
    3. Promotions (may inject): sponsored items in specific slots
    4. Boosts (may reorder): freshness, editorial picks
    
    Rules are defined in config and compiled at startup.
    Hot-reloaded on config change (no restart needed).
    """
    
    def __init__(self, config: ReRankingConfig):
        self.config = config
        self.logger = structlog.get_logger(component="business_rules")
    
    def apply(
        self,
        items: list[dict[str, Any]],
        business_rules: list[dict[str, Any]] | None = None,
        user_segment: str = "default",
    ) -> list[dict[str, Any]]:
        """Apply all business rules in priority order."""
        result = list(items)
        
        # Rule 1: Category diversity constraint
        result = self._apply_category_constraint(result)
        
        # Rule 2: Brand diversity constraint
        result = self._apply_brand_constraint(result)
        
        # Rule 3: Freshness boost
        result = self._apply_freshness_boost(result)
        
        # Rule 4: Promoted items
        if business_rules:
            result = self._apply_promotions(result, business_rules)
        
        # Re-assign positions
        for i, item in enumerate(result):
            item["position"] = i + 1
        
        return result
    
    def _apply_category_constraint(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Ensure no more than max_same_category items from same category in top 10.
        Move excess items to later positions.
        """
        category_counts: dict[str, int] = {}
        top_items: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        
        for item in items:
            category = item.get("category", item.get("features", {}).get("category", "unknown"))
            
            if len(top_items) < 10:
                count = category_counts.get(category, 0)
                if count < self.config.max_same_category:
                    top_items.append(item)
                    category_counts[category] = count + 1
                else:
                    deferred.append(item)
                    BUSINESS_RULE_APPLIED.labels(rule_type="category_constraint").inc()
            else:
                top_items.append(item)
        
        return top_items + deferred
    
    def _apply_brand_constraint(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Ensure no more than max_same_brand items from same brand in top 10."""
        brand_counts: dict[str, int] = {}
        top_items: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        
        for item in items:
            brand = item.get("brand", item.get("features", {}).get("brand", "unknown"))
            
            if len(top_items) < 10:
                count = brand_counts.get(brand, 0)
                if count < self.config.max_same_brand:
                    top_items.append(item)
                    brand_counts[brand] = count + 1
                else:
                    deferred.append(item)
                    BUSINESS_RULE_APPLIED.labels(rule_type="brand_constraint").inc()
            else:
                top_items.append(item)
        
        return top_items + deferred
    
    def _apply_freshness_boost(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Boost score of recently listed items."""
        for item in items:
            days = item.get("days_since_listing", 
                          item.get("features", {}).get("days_since_listing", 999))
            if days < self.config.freshness_boost_days:
                item["score"] = item.get("score", 0) * self.config.freshness_boost_factor
                item.setdefault("annotations", {})["freshness_boosted"] = "true"
                BUSINESS_RULE_APPLIED.labels(rule_type="freshness_boost").inc()
        
        # Re-sort after boost
        items.sort(key=lambda x: x.get("score", 0), reverse=True)
        return items
    
    def _apply_promotions(
        self,
        items: list[dict[str, Any]],
        promotions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Inject promoted items at specific positions."""
        promoted_count = 0
        
        for promo in promotions:
            if promoted_count >= self.config.max_promotions:
                break
            
            slot = promo.get("position", 0)
            item_id = promo.get("item_id")
            
            if slot < len(items) and item_id:
                promo_item = {
                    "item_id": item_id,
                    "score": items[slot]["score"] if slot < len(items) else 0,
                    "annotations": {"promoted": "true", "promo_id": promo.get("promo_id", "")},
                }
                items.insert(slot, promo_item)
                promoted_count += 1
                BUSINESS_RULE_APPLIED.labels(rule_type="promotion").inc()
        
        return items[:self.config.output_size]


# ---------------------------------------------------------------------------
# Diversity Metrics
# ---------------------------------------------------------------------------

class DiversityMetrics:
    """Calculate diversity metrics for the final recommendation list."""
    
    @staticmethod
    def category_entropy(items: list[dict[str, Any]]) -> float:
        """Shannon entropy of category distribution (higher = more diverse)."""
        categories = [
            item.get("category", item.get("features", {}).get("category", "unknown"))
            for item in items
        ]
        if not categories:
            return 0.0
        
        from collections import Counter
        counts = Counter(categories)
        total = len(categories)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return float(entropy)
    
    @staticmethod
    def intra_list_similarity(embeddings: list[np.ndarray]) -> float:
        """
        Average pairwise cosine similarity (lower = more diverse).
        Target: < 0.6 for good diversity.
        """
        if len(embeddings) < 2:
            return 0.0
        
        emb_matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_matrix = emb_matrix / norms
        
        sim_matrix = emb_matrix @ emb_matrix.T
        n = len(embeddings)
        
        # Average off-diagonal
        total_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
        return float(total_sim)
    
    @staticmethod
    def coverage(items: list[dict[str, Any]], total_catalog_size: int) -> float:
        """Fraction of unique items surfaced."""
        unique_items = len({item["item_id"] for item in items})
        return unique_items / max(total_catalog_size, 1)


# ---------------------------------------------------------------------------
# Re-Ranking Service
# ---------------------------------------------------------------------------

class ReRankingService:
    """
    Complete re-ranking pipeline:
    1. MMR diversity re-ranking
    2. Business rules application
    3. Diversity metrics computation
    """
    
    def __init__(self, config: ReRankingConfig | None = None):
        self.config = config or ReRankingConfig()
        self.logger = structlog.get_logger(component="reranking_service")
        
        self.mmr = MMRReRanker(lambda_=self.config.diversity_lambda)
        self.rules_engine = BusinessRulesEngine(self.config)
        self.diversity_metrics = DiversityMetrics()
    
    def rerank(
        self,
        ranked_items: list[dict[str, Any]],
        embeddings: dict[str, np.ndarray] | None = None,
        business_rules: list[dict[str, Any]] | None = None,
        user_segment: str = "default",
        output_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Full re-ranking pipeline.
        
        Returns:
            {
                "items": [...],
                "diversity_metrics": {
                    "category_entropy": float,
                    "intra_list_similarity": float,
                },
                "rerank_latency_us": int,
            }
        """
        start = time.monotonic()
        target_size = output_size or self.config.output_size
        
        # Step 1: MMR re-ranking
        mmr_start = time.monotonic()
        mmr_items = self.mmr.rerank(
            items=ranked_items,
            embeddings=embeddings,
            output_size=min(target_size * 2, len(ranked_items)),  # over-retrieve for rules
        )
        RERANK_LATENCY.labels(stage="mmr").observe((time.monotonic() - mmr_start) * 1000)
        
        # Step 2: Business rules
        rules_start = time.monotonic()
        final_items = self.rules_engine.apply(
            items=mmr_items,
            business_rules=business_rules,
            user_segment=user_segment,
        )
        RERANK_LATENCY.labels(stage="business_rules").observe(
            (time.monotonic() - rules_start) * 1000
        )
        
        # Trim to target size
        final_items = final_items[:target_size]
        
        # Step 3: Compute diversity metrics
        metrics = {
            "category_entropy": self.diversity_metrics.category_entropy(final_items),
        }
        
        DIVERSITY_SCORE.labels(metric="category_entropy").set(metrics["category_entropy"])
        
        total_latency_us = int((time.monotonic() - start) * 1_000_000)
        RERANK_LATENCY.labels(stage="total").observe(total_latency_us / 1000)
        
        self.logger.info(
            "reranking_complete",
            input_size=len(ranked_items),
            output_size=len(final_items),
            category_entropy=round(metrics["category_entropy"], 3),
            latency_us=total_latency_us,
        )
        
        return {
            "items": final_items,
            "diversity_metrics": metrics,
            "rerank_latency_us": total_latency_us,
        }
