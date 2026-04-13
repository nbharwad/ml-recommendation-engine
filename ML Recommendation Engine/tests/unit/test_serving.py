"""
Unit Tests — Serving Layer
===========================
Tests for the recommendation serving orchestrator.
Covers: circuit breaker, latency budget, fallback chain, request handling.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Import from serving layer
import sys
sys.path.insert(0, "services/serving")
from main import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    LatencyBudget,
    PopularityFallback,
    RecommendationEngine,
    RecommendationRequest,
    RecommendationResponse,
    ServingConfig,
    PageContextEnum,
    app,
)


# ---------------------------------------------------------------------------
# Circuit Breaker Tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Test circuit breaker state transitions."""
    
    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            name="test_service",
            failure_threshold=3,
            recovery_timeout_sec=1,
            success_threshold=2,
        )
    
    @pytest.mark.asyncio
    async def test_starts_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, breaker):
        """Circuit should open after N consecutive failures."""
        async def failing_func():
            raise Exception("Service down")
        
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_open_circuit_rejects_immediately(self, breaker):
        """Open circuit should reject without calling the function."""
        async def failing_func():
            raise Exception("Down")
        
        # Trip the breaker
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        
        # Should raise CircuitBreakerOpenError immediately
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self, breaker):
        """Circuit should transition to half-open after recovery timeout."""
        async def failing_func():
            raise Exception("Down")
        
        async def success_func():
            return "OK"
        
        # Trip the breaker
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should go through (half-open)
        result = await breaker.call(success_func)
        assert result == "OK"
    
    @pytest.mark.asyncio
    async def test_closes_after_success_threshold(self, breaker):
        """Circuit should close after N consecutive successes in half-open."""
        async def failing_func():
            raise Exception("Down")
        
        async def success_func():
            return "OK"
        
        # Trip → Open
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        
        # Wait → Half-Open
        await asyncio.sleep(1.1)
        
        # Succeed enough times → Closed
        for _ in range(2):
            await breaker.call(success_func)
        
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_successful_calls_reset_failure_count(self, breaker):
        """Successful call should reset failure counter."""
        call_count = 0
        
        async def intermittent_func():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise Exception("Occasional failure")
            return "OK"
        
        # Mix of successes and failures (should never trip)
        for _ in range(10):
            try:
                await breaker.call(intermittent_func)
            except Exception:
                pass
        
        # Should still be closed (never hit 3 consecutive failures)
        assert breaker.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Latency Budget Tests
# ---------------------------------------------------------------------------

class TestLatencyBudget:
    """Test latency budget tracking and enforcement."""
    
    def test_initial_budget(self):
        budget = LatencyBudget(total_budget_ms=75)
        assert budget.remaining_ms > 0
        assert not budget.is_expired
    
    def test_budget_decreases_over_time(self):
        budget = LatencyBudget(total_budget_ms=100)
        time.sleep(0.01)  # 10ms
        assert budget.elapsed_ms >= 10
        assert budget.remaining_ms < 100
    
    def test_budget_expiry(self):
        budget = LatencyBudget(total_budget_ms=10)
        time.sleep(0.015)  # 15ms
        assert budget.is_expired
    
    def test_can_afford(self):
        budget = LatencyBudget(total_budget_ms=100)
        assert budget.can_afford(50)  # 50ms within 100ms budget
        assert budget.can_afford(90)  # 90ms within fresh 100ms budget
    
    def test_cannot_afford_when_expired(self):
        budget = LatencyBudget(total_budget_ms=5)
        time.sleep(0.01)  # 10ms, budget expired
        assert not budget.can_afford(1)


# ---------------------------------------------------------------------------
# Popularity Fallback Tests
# ---------------------------------------------------------------------------

class TestPopularityFallback:
    """Test fallback recommendation generation."""
    
    @pytest.fixture
    def fallback(self):
        return PopularityFallback()
    
    def test_returns_correct_count(self, fallback):
        recs = fallback.get_fallback_recs(num_items=10)
        assert len(recs) == 10
    
    def test_returns_proper_format(self, fallback):
        recs = fallback.get_fallback_recs(num_items=5)
        for rec in recs:
            assert "item_id" in rec
            assert "position" in rec
            assert "score" in rec
            assert rec["position"] >= 1
            assert rec["score"] > 0
    
    def test_segment_based_fallback(self, fallback):
        recs = fallback.get_fallback_recs(user_segment="high_value")
        assert all("hv_" in r["item_id"] for r in recs)
    
    def test_category_based_fallback(self, fallback):
        recs = fallback.get_fallback_recs(category="electronics")
        assert all("elec_" in r["item_id"] for r in recs)
    
    def test_global_fallback(self, fallback):
        recs = fallback.get_fallback_recs()
        assert all("popular_" in r["item_id"] for r in recs)
    
    def test_positions_are_sequential(self, fallback):
        recs = fallback.get_fallback_recs(num_items=20)
        positions = [r["position"] for r in recs]
        assert positions == list(range(1, 21))


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------

class TestRecommendationAPI:
    """Test REST API endpoints."""
    
    @pytest_asyncio.fixture
    async def client(self):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "serving"
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "recommendation_request" in response.text
    
    @pytest.mark.asyncio
    async def test_recommendation_request(self, client):
        response = await client.post(
            "/v1/recommendations",
            json={
                "user_id": "test_user_123",
                "page_context": "home",
                "num_items": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "request_id" in data
        assert len(data["items"]) <= 10
    
    @pytest.mark.asyncio
    async def test_invalid_request(self, client):
        response = await client.post(
            "/v1/recommendations",
            json={
                "user_id": "",  # invalid: empty string
                "num_items": 100,  # invalid: > max 50
            },
        )
        assert response.status_code == 422  # validation error
    
    @pytest.mark.asyncio
    async def test_similar_items_endpoint(self, client):
        response = await client.post(
            "/v1/similar-items",
            json={
                "item_id": "item_001",
                "num_items": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 5


# ---------------------------------------------------------------------------
# Re-Ranking Tests
# ---------------------------------------------------------------------------

class TestMMRReRanking:
    """Test MMR diversity algorithm."""
    
    def test_import(self):
        sys.path.insert(0, "services/reranking")
        from reranking_service import MMRReRanker
        
        ranker = MMRReRanker(lambda_=0.7)
        items = [
            {"item_id": f"item_{i}", "score": 1.0 - i * 0.1}
            for i in range(10)
        ]
        
        result = ranker.rerank(items, output_size=5)
        assert len(result) == 5
        assert all("position" in item for item in result)
    
    def test_pure_relevance(self):
        """With λ=1.0, should return items in score order."""
        sys.path.insert(0, "services/reranking")
        from reranking_service import MMRReRanker
        
        ranker = MMRReRanker(lambda_=1.0)
        items = [
            {"item_id": f"item_{i}", "score": 1.0 - i * 0.1}
            for i in range(10)
        ]
        
        result = ranker.rerank(items, output_size=5)
        scores = [r["relevance_score"] for r in result]
        
        # With λ=1.0 (pure relevance), items should be in original score order
        assert scores == sorted(scores, reverse=True)
    
    def test_empty_input(self):
        sys.path.insert(0, "services/reranking")
        from reranking_service import MMRReRanker
        
        ranker = MMRReRanker(lambda_=0.7)
        result = ranker.rerank([], output_size=5)
        assert result == []
