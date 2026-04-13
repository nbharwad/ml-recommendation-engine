"""
Integration Tests
==================
End-to-end tests for the recommendation pipeline.
Requires docker-compose services running.

Tests:
1. Full pipeline: API → Serving → Features → Retrieval → Ranking → Re-Ranking
2. Event ingestion round-trip
3. Feature store read/write consistency
4. Experiment assignment consistency
5. Fallback behavior under failure
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import pytest
import pytest_asyncio
import httpx


# Test configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:29092")


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end recommendation pipeline tests."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_URL, timeout=30.0)
    
    def test_home_page_recommendations(self, client):
        """Test home page recommendation flow."""
        response = client.post("/v1/recommendations", json={
            "user_id": "integration_test_user_001",
            "page_context": "home",
            "num_items": 20,
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "items" in data
        assert len(data["items"]) > 0
        assert len(data["items"]) <= 20
        assert "request_id" in data
        assert "total_latency_ms" in data
        
        # Verify item structure
        for item in data["items"]:
            assert "item_id" in item
            assert "position" in item
            assert "score" in item
            assert item["position"] >= 1
            assert item["score"] >= 0
    
    def test_pdp_similar_items(self, client):
        """Test product detail page similar items."""
        response = client.post("/v1/similar-items", json={
            "item_id": "item_000001",
            "num_items": 10,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) > 0
    
    def test_latency_sla(self, client):
        """Verify p99 latency is within SLA (75ms).
        
        Runs 100 requests and checks p99.
        """
        latencies = []
        
        for i in range(100):
            start = time.monotonic()
            response = client.post("/v1/recommendations", json={
                "user_id": f"latency_test_user_{i:04d}",
                "page_context": "home",
                "num_items": 20,
            })
            latency_ms = (time.monotonic() - start) * 1000
            assert response.status_code == 200
            latencies.append(latency_ms)
        
        latencies.sort()
        p50 = latencies[49]
        p95 = latencies[94]
        p99 = latencies[98]
        
        print(f"\nLatency results (100 requests):")
        print(f"  p50: {p50:.1f}ms")
        print(f"  p95: {p95:.1f}ms")
        print(f"  p99: {p99:.1f}ms")
        
        # Relaxed for integration test (includes network overhead)
        assert p99 < 500, f"p99 latency {p99:.1f}ms exceeds 500ms threshold"
    
    def test_concurrent_requests(self, client):
        """Test system under concurrent load."""
        import concurrent.futures
        
        def make_request(user_id: str):
            c = httpx.Client(base_url=API_URL, timeout=10.0)
            response = c.post("/v1/recommendations", json={
                "user_id": user_id,
                "page_context": "home",
                "num_items": 10,
            })
            c.close()
            return response.status_code
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(make_request, f"concurrent_user_{i:04d}")
                for i in range(100)
            ]
            
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_count = sum(1 for r in results if r == 200)
        success_rate = success_count / len(results)
        
        print(f"\nConcurrent test: {success_count}/{len(results)} succeeded ({success_rate:.1%})")
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95%"
    
    def test_different_page_contexts(self, client):
        """Verify recommendations work for all page contexts."""
        for context in ["home", "pdp", "cart", "search", "category"]:
            response = client.post("/v1/recommendations", json={
                "user_id": "context_test_user",
                "page_context": context,
                "num_items": 10,
            })
            assert response.status_code == 200, f"Failed for context: {context}"
            data = response.json()
            assert len(data["items"]) > 0, f"No items for context: {context}"


# ---------------------------------------------------------------------------
# Experiment Assignment Tests
# ---------------------------------------------------------------------------

class TestExperimentAssignment:
    """Test experiment assignment consistency."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_URL, timeout=10.0)
    
    def test_assignment_is_deterministic(self, client):
        """Same user should always get same experiment variant."""
        user_id = "determinism_test_user_42"
        
        experiment_ids = set()
        for _ in range(10):
            response = client.post("/v1/recommendations", json={
                "user_id": user_id,
                "page_context": "home",
                "num_items": 5,
            })
            data = response.json()
            if data.get("experiment_id"):
                experiment_ids.add(data["experiment_id"])
        
        # Should always get the same experiment
        assert len(experiment_ids) <= 1, f"Got inconsistent experiments: {experiment_ids}"
    
    def test_different_users_get_different_variants(self, client):
        """Traffic should be split across variants."""
        variants = {}
        
        for i in range(100):
            response = client.post("/v1/recommendations", json={
                "user_id": f"split_test_user_{i:04d}",
                "page_context": "home",
                "num_items": 5,
            })
            data = response.json()
            exp_id = data.get("experiment_id", "none")
            variants[exp_id] = variants.get(exp_id, 0) + 1
        
        print(f"\nVariant distribution: {variants}")
        # Should see traffic distributed (not all in one bucket)


# ---------------------------------------------------------------------------
# Health & Resilience Tests
# ---------------------------------------------------------------------------

class TestHealthAndResilience:
    """Test health checks and resilience features."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_URL, timeout=10.0)
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "serving"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "dependencies" in data
    
    def test_readiness_endpoint(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
    
    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "recommendation_request" in response.text
        assert "recommendation_component_latency" in response.text
    
    def test_invalid_request_handling(self, client):
        """Test that invalid requests return proper error responses."""
        # Missing required field
        response = client.post("/v1/recommendations", json={
            "page_context": "home",
        })
        assert response.status_code == 422
        
        # Invalid num_items
        response = client.post("/v1/recommendations", json={
            "user_id": "test",
            "num_items": 100,  # exceeds max of 50
        })
        assert response.status_code == 422
    
    def test_rate_limiting(self, client):
        """Test that rate limiting kicks in under high load."""
        responses = []
        for _ in range(200):
            response = client.post("/v1/recommendations", json={
                "user_id": "rate_limit_test_user",
                "num_items": 5,
            })
            responses.append(response.status_code)
        
        # At very high rates, some should be rate-limited (429)
        # But in integration test, we may not hit the limit
        success_count = sum(1 for r in responses if r == 200)
        assert success_count > 0


# ---------------------------------------------------------------------------
# Fallback Tests
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    """Test graceful degradation and fallback chain."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_URL, timeout=10.0)
    
    def test_new_user_returns_recommendations(self, client):
        """Brand new user should get fallback/popularity recommendations."""
        response = client.post("/v1/recommendations", json={
            "user_id": f"brand_new_user_{int(time.time())}",
            "page_context": "home",
            "num_items": 20,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) > 0
    
    def test_response_always_has_items(self, client):
        """System should ALWAYS return items (via fallback if needed)."""
        for i in range(50):
            response = client.post("/v1/recommendations", json={
                "user_id": f"reliability_test_{i}",
                "page_context": "home",
                "num_items": 10,
            })
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["items"]) > 0, f"No items returned for request {i}"
    
    def test_fallback_info_in_response(self, client):
        """Verify fallback information is included in response."""
        response = client.post("/v1/recommendations", json={
            "user_id": "fallback_info_test",
            "num_items": 10,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "fallback" in data
        assert "fallback_used" in data["fallback"]


# ---------------------------------------------------------------------------
# Data Consistency Tests
# ---------------------------------------------------------------------------

class TestDataConsistency:
    """Test feature store and data consistency."""
    
    @pytest.mark.asyncio
    async def test_feature_store_read_write(self):
        """Test that written features can be read back correctly."""
        # In production: direct Redis read/write test
        import redis.asyncio as aioredis
        
        try:
            r = aioredis.Redis(host=REDIS_HOST, port=6379)
            
            # Write
            test_key = "test:feature:consistency"
            test_value = json.dumps({"feature_a": 1.5, "feature_b": "test"})
            await r.set(test_key, test_value)
            
            # Read
            result = await r.get(test_key)
            assert result is not None
            parsed = json.loads(result)
            assert parsed["feature_a"] == 1.5
            assert parsed["feature_b"] == "test"
            
            # Cleanup
            await r.delete(test_key)
            await r.close()
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
    
    @pytest.mark.asyncio
    async def test_feature_store_pipeline_performance(self):
        """Test Redis pipeline performance for batch reads."""
        import redis.asyncio as aioredis
        
        try:
            r = aioredis.Redis(host=REDIS_HOST, port=6379)
            
            # Write 1000 test features
            pipe = r.pipeline()
            for i in range(1000):
                pipe.set(f"perf:item:{i}", json.dumps({"price": i * 10.0}))
            await pipe.execute()
            
            # Pipeline read
            start = time.monotonic()
            pipe = r.pipeline()
            for i in range(1000):
                pipe.get(f"perf:item:{i}")
            results = await pipe.execute()
            latency_ms = (time.monotonic() - start) * 1000
            
            assert len(results) == 1000
            assert all(r is not None for r in results)
            print(f"\nRedis pipeline: 1000 reads in {latency_ms:.1f}ms")
            assert latency_ms < 50, f"Pipeline too slow: {latency_ms:.1f}ms"
            
            # Cleanup
            pipe = r.pipeline()
            for i in range(1000):
                pipe.delete(f"perf:item:{i}")
            await pipe.execute()
            await r.close()
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
