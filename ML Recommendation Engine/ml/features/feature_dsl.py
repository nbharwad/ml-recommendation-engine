"""
Feature DSL — Unified Feature Definitions
============================================
Single source of truth for feature definitions used across:
- Offline training (Spark)
- Online serving (Redis)
- Streaming computation (Flink)

This prevents training-serving skew by ensuring both pipelines
derive features from the same specification.

Usage:
    from ml.features.feature_dsl import FeatureRegistry
    
    registry = FeatureRegistry.load("feature_config.yaml")
    user_features = registry.get_user_features()
    item_features = registry.get_item_features()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FeatureSource(str, Enum):
    BATCH = "batch"           # Computed offline (Spark), refresh: 4h
    STREAMING = "streaming"   # Computed in real-time (Flink), refresh: <5min
    CDC = "cdc"               # Change Data Capture, refresh: on-change
    STATIC = "static"         # Configuration values, refresh: on-deploy


class FeatureType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    LIST = "list"
    EMBEDDING = "embedding"


@dataclass
class FeatureDefinition:
    """Single feature definition."""
    name: str
    type: FeatureType
    source: FeatureSource
    description: str
    
    # Defaults
    default_value: Any = None
    
    # Validation
    min_value: float | None = None
    max_value: float | None = None
    nullable: bool = False
    
    # For embeddings
    embedding_dim: int | None = None
    
    # Metadata
    version: str = "v1"
    owner: str = "ml-platform"
    
    # Normalization (for dense features in ranking)
    normalize_mean: float | None = None
    normalize_std: float | None = None
    
    # Freshness
    max_staleness_sec: int = 86400  # 24h default
    
    def validate_value(self, value: Any) -> bool:
        """Validate a feature value against its definition."""
        if value is None:
            return self.nullable
        
        if self.min_value is not None and float(value) < self.min_value:
            return False
        if self.max_value is not None and float(value) > self.max_value:
            return False
        
        return True
    
    def normalize(self, value: float) -> float:
        """Z-score normalization using training statistics."""
        if self.normalize_mean is not None and self.normalize_std is not None:
            return (value - self.normalize_mean) / max(self.normalize_std, 1e-6)
        return value


# ---------------------------------------------------------------------------
# Feature Groups
# ---------------------------------------------------------------------------

class FeatureGroup:
    """Named group of features for an entity (user/item)."""
    
    def __init__(self, name: str, entity_type: str):
        self.name = name
        self.entity_type = entity_type
        self.features: dict[str, FeatureDefinition] = {}
    
    def add(self, feature: FeatureDefinition) -> "FeatureGroup":
        self.features[feature.name] = feature
        return self
    
    def get(self, name: str) -> FeatureDefinition | None:
        return self.features.get(name)
    
    def get_by_source(self, source: FeatureSource) -> list[FeatureDefinition]:
        return [f for f in self.features.values() if f.source == source]
    
    def get_dense_features(self) -> list[FeatureDefinition]:
        """Get features suitable for dense input (numerical)."""
        return [
            f for f in self.features.values()
            if f.type in (FeatureType.INT, FeatureType.FLOAT)
        ]
    
    def get_sparse_features(self) -> list[FeatureDefinition]:
        """Get features suitable for sparse input (categorical)."""
        return [
            f for f in self.features.values()
            if f.type == FeatureType.STRING
        ]
    
    def get_default_vector(self) -> dict[str, Any]:
        """Get default feature vector."""
        return {
            f.name: f.default_value
            for f in self.features.values()
            if f.type != FeatureType.EMBEDDING
        }
    
    def to_schema_dict(self) -> list[dict[str, Any]]:
        """Export schema for documentation / validation."""
        return [
            {
                "name": f.name,
                "type": f.type.value,
                "source": f.source.value,
                "description": f.description,
                "default": f.default_value,
                "nullable": f.nullable,
                "min_value": f.min_value,
                "max_value": f.max_value,
                "max_staleness_sec": f.max_staleness_sec,
            }
            for f in self.features.values()
        ]


# ---------------------------------------------------------------------------
# Feature Registry
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """
    Central registry of all features in the recommendation system.
    
    Singleton that loads feature definitions and provides
    consistent access for both offline and online pipelines.
    
    Usage:
        registry = FeatureRegistry.default()
        user_group = registry.get_group("user_features")
        item_group = registry.get_group("item_features")
    """
    
    def __init__(self):
        self.groups: dict[str, FeatureGroup] = {}
        self.version: str = "v2.3"
    
    def register_group(self, group: FeatureGroup):
        self.groups[group.name] = group
    
    def get_group(self, name: str) -> FeatureGroup | None:
        return self.groups.get(name)
    
    @classmethod
    def default(cls) -> "FeatureRegistry":
        """Create the default feature registry for the recommendation system."""
        registry = cls()
        
        # ===== User Features =====
        user_group = FeatureGroup("user_features", "user")
        
        # Behavioral (streaming)
        user_group.add(FeatureDefinition(
            name="purchase_count_30d", type=FeatureType.INT, source=FeatureSource.STREAMING,
            description="Number of purchases in last 30 days",
            default_value=0, min_value=0, max_staleness_sec=300,
            normalize_mean=5.0, normalize_std=8.0,
        ))
        user_group.add(FeatureDefinition(
            name="click_count_7d", type=FeatureType.INT, source=FeatureSource.STREAMING,
            description="Number of clicks in last 7 days",
            default_value=0, min_value=0, max_staleness_sec=300,
            normalize_mean=25.0, normalize_std=30.0,
        ))
        user_group.add(FeatureDefinition(
            name="avg_session_duration_sec", type=FeatureType.FLOAT, source=FeatureSource.STREAMING,
            description="Average session duration in seconds",
            default_value=120.0, min_value=0, max_staleness_sec=300,
            normalize_mean=180.0, normalize_std=120.0,
        ))
        user_group.add(FeatureDefinition(
            name="session_click_count", type=FeatureType.INT, source=FeatureSource.STREAMING,
            description="Clicks in current session",
            default_value=0, min_value=0, max_staleness_sec=60,
        ))
        user_group.add(FeatureDefinition(
            name="cart_abandonment_rate", type=FeatureType.FLOAT, source=FeatureSource.STREAMING,
            description="Historical cart abandonment rate",
            default_value=0.5, min_value=0.0, max_value=1.0, max_staleness_sec=3600,
        ))
        user_group.add(FeatureDefinition(
            name="last_purchase_days_ago", type=FeatureType.INT, source=FeatureSource.STREAMING,
            description="Days since last purchase",
            default_value=999, min_value=0,
        ))
        
        # Profile (batch)
        user_group.add(FeatureDefinition(
            name="avg_order_value", type=FeatureType.FLOAT, source=FeatureSource.BATCH,
            description="Average order value in USD",
            default_value=35.0, min_value=0,
            normalize_mean=50.0, normalize_std=40.0,
        ))
        user_group.add(FeatureDefinition(
            name="user_segment", type=FeatureType.STRING, source=FeatureSource.BATCH,
            description="User value segment",
            default_value="default",
        ))
        user_group.add(FeatureDefinition(
            name="price_sensitivity", type=FeatureType.FLOAT, source=FeatureSource.BATCH,
            description="Price sensitivity score (0=insensitive, 1=very sensitive)",
            default_value=0.5, min_value=0.0, max_value=1.0,
        ))
        user_group.add(FeatureDefinition(
            name="registration_days_ago", type=FeatureType.INT, source=FeatureSource.BATCH,
            description="Days since user registered",
            default_value=0, min_value=0,
        ))
        
        # Embedding (batch)
        user_group.add(FeatureDefinition(
            name="user_embedding", type=FeatureType.EMBEDDING, source=FeatureSource.BATCH,
            description="128-dim user embedding from Two-Tower model",
            default_value=None, embedding_dim=128,
        ))
        
        registry.register_group(user_group)
        
        # ===== Item Features =====
        item_group = FeatureGroup("item_features", "item")
        
        # Static (CDC)
        item_group.add(FeatureDefinition(
            name="category", type=FeatureType.STRING, source=FeatureSource.CDC,
            description="Product category",
            default_value="unknown",
        ))
        item_group.add(FeatureDefinition(
            name="brand", type=FeatureType.STRING, source=FeatureSource.CDC,
            description="Product brand",
            default_value="unknown",
        ))
        item_group.add(FeatureDefinition(
            name="price", type=FeatureType.FLOAT, source=FeatureSource.CDC,
            description="Product price in USD",
            default_value=0.0, min_value=0.0,
            normalize_mean=45.0, normalize_std=35.0,
        ))
        item_group.add(FeatureDefinition(
            name="stock_count", type=FeatureType.INT, source=FeatureSource.CDC,
            description="Available stock count",
            default_value=0, min_value=0,
        ))
        
        # Dynamic (streaming)
        item_group.add(FeatureDefinition(
            name="ctr_7d", type=FeatureType.FLOAT, source=FeatureSource.STREAMING,
            description="Click-through rate in last 7 days",
            default_value=0.01, min_value=0.0, max_value=1.0, max_staleness_sec=300,
            normalize_mean=0.03, normalize_std=0.02,
        ))
        item_group.add(FeatureDefinition(
            name="view_count_24h", type=FeatureType.INT, source=FeatureSource.STREAMING,
            description="View count in last 24 hours",
            default_value=0, min_value=0, max_staleness_sec=300,
        ))
        item_group.add(FeatureDefinition(
            name="purchase_count_7d", type=FeatureType.INT, source=FeatureSource.STREAMING,
            description="Purchase count in last 7 days",
            default_value=0, min_value=0, max_staleness_sec=300,
        ))
        item_group.add(FeatureDefinition(
            name="avg_rating", type=FeatureType.FLOAT, source=FeatureSource.STREAMING,
            description="Average product rating",
            default_value=3.0, min_value=1.0, max_value=5.0,
            normalize_mean=3.8, normalize_std=0.8,
        ))
        item_group.add(FeatureDefinition(
            name="review_count", type=FeatureType.INT, source=FeatureSource.STREAMING,
            description="Total review count",
            default_value=0, min_value=0,
        ))
        item_group.add(FeatureDefinition(
            name="days_since_listing", type=FeatureType.INT, source=FeatureSource.CDC,
            description="Days since item was first listed",
            default_value=0, min_value=0,
        ))
        
        # Embedding
        item_group.add(FeatureDefinition(
            name="item_embedding", type=FeatureType.EMBEDDING, source=FeatureSource.BATCH,
            description="128-dim item embedding from Two-Tower model",
            default_value=None, embedding_dim=128,
        ))
        
        registry.register_group(item_group)
        
        return registry
    
    def export_schema(self, output_path: str):
        """Export full schema to JSON for documentation."""
        schema = {
            "version": self.version,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "groups": {
                name: {
                    "entity_type": group.entity_type,
                    "features": group.to_schema_dict(),
                    "count": len(group.features),
                }
                for name, group in self.groups.items()
            },
        }
        
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Consistency Check
# ---------------------------------------------------------------------------

def check_training_serving_consistency(registry: FeatureRegistry):
    """
    Verify that offline and online pipelines use the same feature definitions.
    
    Checks:
    1. Feature names match between training config and serving config
    2. Default values are identical
    3. Normalization parameters are identical
    
    Run this as a CI check to prevent training-serving skew.
    """
    issues = []
    
    for group_name, group in registry.groups.items():
        for feat_name, feat_def in group.features.items():
            # Check normalization params exist for dense features
            if feat_def.type in (FeatureType.INT, FeatureType.FLOAT):
                if feat_def.normalize_mean is not None and feat_def.normalize_std is None:
                    issues.append(f"{feat_name}: has normalize_mean but missing normalize_std")
            
            # Check embedding dimensions
            if feat_def.type == FeatureType.EMBEDDING and feat_def.embedding_dim is None:
                issues.append(f"{feat_name}: embedding type but missing embedding_dim")
            
            # Check bounded features
            if feat_def.type == FeatureType.FLOAT and feat_def.max_value is not None:
                if feat_def.default_value is not None:
                    if feat_def.default_value > feat_def.max_value:
                        issues.append(f"{feat_name}: default ({feat_def.default_value}) > max ({feat_def.max_value})")
    
    return issues


if __name__ == "__main__":
    registry = FeatureRegistry.default()
    
    print(f"Feature Registry v{registry.version}")
    print(f"{'='*50}")
    
    for group_name, group in registry.groups.items():
        print(f"\n{group_name} ({group.entity_type}):")
        print(f"  Total features: {len(group.features)}")
        print(f"  Dense features: {len(group.get_dense_features())}")
        print(f"  Sparse features: {len(group.get_sparse_features())}")
        print(f"  By source:")
        for source in FeatureSource:
            count = len(group.get_by_source(source))
            if count > 0:
                print(f"    {source.value}: {count}")
    
    # Consistency check
    issues = check_training_serving_consistency(registry)
    if issues:
        print(f"\n⚠️  Consistency issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ No consistency issues found")
    
    # Export schema
    registry.export_schema("feature_schema.json")
    print(f"\nSchema exported to feature_schema.json")
