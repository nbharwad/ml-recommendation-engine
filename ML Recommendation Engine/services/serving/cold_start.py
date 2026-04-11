"""
Cold Start Solution using Thompson Sampling
==================================
Industry standard approach for handling cold-start recommendations.

Uses Thompson Sampling (Multi-Armed Bandit) to explore new items
while exploiting known high-performing items.

Industry Standard:
- YouTube, Netflix, Amazon, Spotify all use Thompson Sampling variants
- Balances exploration/exploitation naturally
- Adapts to uncertainty without manual tuning

Implementation:
- Prior: Beta(1, 1) uniform for new items
- Sample from posterior after each interaction
- Update posterior with reward (click/no-click)
- Select item with highest sampled value
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import structlog
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger(component="cold_start")


COLD_START_SELECTIONS = Counter(
    "cold_start_selections_total",
    "Cold start item selections",
    ["strategy", "item_type"],
)

COLD_START_LATENCY = Histogram(
    "cold_start_latency_ms",
    "Cold start selection latency",
    buckets=[1, 2, 5, 10, 20, 50],
)

EXPLORATION_RATE = Gauge(
    "exploration_rate",
    "Current exploration rate (epsilon)",
)


@dataclass
class ItemPosterior:
    """Beta posterior for Thompson Sampling."""
    alpha: float = 1.0
    beta: float = 1.0
    
    def sample(self) -> float:
        """Sample from Beta distribution."""
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, reward: float):
        """Update posterior with reward."""
        self.alpha += reward
        self.beta += (1.0 - reward)


@dataclass
class ColdStartConfig:
    """Cold start configuration."""
    exploration_rate: float = 0.1
    min_interactions: int = 10
    candidate_pool_size: int = 100
    default_embedding: Optional[list] = None
    enable_thompson: bool = True
    cache_ttl_sec: int = 3600


class ThompsonSampling:
    """
    Thompson Sampling for cold-start recommendations.
    
    Uses Beta distribution to model click-through rate uncertainty.
    Each item has its own posterior (alpha, beta) representing:
    - alpha: number of positive outcomes (clicks)
    - beta: number of negative outcomes (views - clicks)
    
    Selection process:
    1. Sample value from each item's posterior
    2. Select item with highest sampled value
    3. After interaction, update posterior with actual reward
    """
    
    def __init__(self, config: ColdStartConfig):
        self.config = config
        self.item_posters: dict[str, ItemPosterior] = {}
        self.item_embeddings: dict[str, list[float]] = {}
        self.last_update = time.time()
        
    def register_item(
        self,
        item_id: str,
        embedding: Optional[list[float]] = None,
        prior_clicks: int = 0,
        prior_views: int = 0,
    ):
        """Register a new item with optional prior knowledge."""
        self.item_posters[item_id] = ItemPosterior(
            alpha=1.0 + prior_clicks,
            beta=1.0 + max(0, prior_views - prior_clicks),
        )
        
        if embedding:
            self.item_embeddings[item_id] = embedding
        elif self.config.default_embedding:
            self.item_embeddings[item_id] = self.config.default_embedding
            
    def select_item(
        self,
        candidates: list[dict],
        user_features: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Select best item using Thompson Sampling.
        
        Args:
            candidates: list of {item_id, features, ...}
            user_features: user context for content-based fallback
            
        Returns:
            Selected item dict or None if no candidates
        """
        start = time.perf_counter()
        
        if not candidates:
            return None
            
        # Filter to candidates with enough data OR exploration rate
        exploration_pool = []
        exploitation_pool = []
        
        for candidate in candidates:
            item_id = candidate.get("item_id", "")
            views = candidate.get("views", 0)
            
            if self.config.enable_thompson and item_id in self.item_posters:
                if views >= self.config.min_interactions:
                    exploitation_pool.append(candidate)
                else:
                    exploration_pool.append(candidate)
            else:
                exploration_pool.append(candidate)
        
        # Exploration: random selection with epsilon probability
        should_explore = random.random() < self.config.exploration_rate
        
        if should_explore and exploration_pool:
            selected = random.choice(exploration_pool)
            COLD_START_SELECTIONS.labels(strategy="exploration", item_type="new").inc()
            selected["_strategy"] = "exploration"
        elif exploitation_pool:
            # Thompson Sampling for exploitation
            selected = self._thompson_sample(exploitation_pool)
            COLD_START_SELECTIONS.labels(strategy="thompson", item_type="known").inc()
            selected["_strategy"] = "thompson"
        elif exploration_pool:
            # Content-based for truly cold items
            selected = self._content_based_select(exploration_pool, user_features)
            COLD_START_SELECTIONS.labels(strategy="content", item_type="cold").inc()
            selected["_strategy"] = "content"
        else:
            selected = random.choice(candidates)
            COLD_START_SELECTIONS.labels(strategy="random", item_type="fallback").inc()
            selected["_strategy"] = "random"
        
        latency_ms = (time.perf_counter() - start) * 1000
        COLD_START_LATENCY.observe(latency_ms)
        
        return selected
    
    def _thompson_sample(self, candidates: list[dict]) -> dict:
        """Select using Thompson Sampling."""
        samples = []
        
        for candidate in candidates:
            item_id = candidate.get("item_id", "")
            posterior = self.item_posters.get(item_id)
            
            if posterior:
                # Sample from Beta distribution
                sample_value = posterior.sample()
            else:
                # Uniform prior for unknown items
                sample_value = random.random()
            
            samples.append((sample_value, candidate))
        
        # Sort by sampled value descending
        samples.sort(key=lambda x: x[0], reverse=True)
        
        return samples[0][1]
    
    def _content_based_select(
        self,
        candidates: list[dict],
        user_features: Optional[dict],
    ) -> dict:
        """Content-based selection for cold items."""
        if not user_features or not candidates:
            return random.choice(candidates)
        
        user_vec = user_features.get("embedding")
        if not user_vec:
            return random.choice(candidates)
        
        best_score = -float("inf")
        best_candidate = candidates[0]
        
        for candidate in candidates:
            item_id = candidate.get("item_id", "")
            item_vec = self.item_embeddings.get(item_id)
            
            if item_vec and user_vec:
                score = cosine_similarity(user_vec, item_vec)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
        
        return best_candidate
    
    def update_reward(self, item_id: str, reward: float):
        """Update posterior with reward (click=1, no-click=0)."""
        if item_id not in self.item_posters:
            self.item_posters[item_id] = ItemPosterior()
        
        self.item_posters[item_id].update(reward)
        self.last_update = time.time()
        
    def batch_update(self, interactions: list[dict]):
        """Batch update from interaction log."""
        for interaction in interactions:
            item_id = interaction.get("item_id")
            reward = interaction.get("reward", 0)
            
            if item_id:
                self.update_reward(item_id, reward)
    
    def get_stats(self) -> dict:
        """Get cold start statistics."""
        return {
            "num_items": len(self.item_posters),
            "num_with_embeddings": len(self.item_embeddings),
            "last_update_ms": int((time.time() - self.last_update) * 1000),
        }


class BanditArm:
    """
    Bandit arm for A/B test variant selection.
    Handles multiple exploration strategies in production.
    """
    
    def __init__(self, strategy: str = "thompson"):
        self.strategy = strategy
        self.arms: dict[str, ThompsonSampling] = {}
        
    def select_arm(self, experiment_id: str, candidates: list[dict]) -> Optional[dict]:
        """Select arm for A/B test."""
        if experiment_id not in self.arms:
            self.arms[experiment_id] = ThompsonSampling(ColdStartConfig())
            
        return self.arms[experiment_id].select_item(candidates)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
        
    a = np.array(a)
    b = np.array(b)
    
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(dot / (norm_a * norm_b))


async def get_cold_start_recommendations(
    user_id: str,
    candidates: list[dict],
    user_features: Optional[dict] = None,
    num_items: int = 10,
) -> list[dict]:
    """
    Get cold start recommendations for a user.
    
    Main entry point for cold-start recommendations.
    Uses Thompson Sampling to balance explore/exploit.
    """
    config = ColdStartConfig()
    sampler = ThompsonSampling(config)
    
    # Select top k items
    selected = []
    remaining = list(candidates)
    
    for _ in range(min(num_items, len(candidates))):
        item = sampler.select_item(remaining, user_features)
        if item:
            selected.append(item)
            remaining.remove(item)
            
    return selected


def create_exploration_policy(policy_type: str = "thompson") -> ThompsonSampling:
    """Factory function to create exploration policy."""
    config = ColdStartConfig(enable_thompson=(policy_type == "thompson"))
    return ThompsonSampling(config)


STATS = {}


def get_thompson_stats() -> dict:
    """Get Thompson Sampling statistics."""
    return STATS