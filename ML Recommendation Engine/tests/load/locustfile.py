"""
Load Test — Locust Configuration
================================
Production load testing for the recommendation API.

Scenarios:
1. Steady state: 10K QPS sustained
2. Peak traffic: 50K QPS burst
3. Spike test: sudden 5× traffic spike
4. Soak test: 24h sustained load (memory leak detection)

Usage:
  locust -f tests/load/locustfile.py --host=http://staging-url:8080
"""

from locust import HttpUser, task, between, events, tag
import json
import random
import time
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

USER_IDS = [f"user_{i:06d}" for i in range(100_000)]
ITEM_IDS = [f"item_{i:06d}" for i in range(10_000)]
PAGE_CONTEXTS = ["home", "pdp", "cart", "search", "category"]
DEVICES = ["mobile", "desktop", "tablet"]


# ---------------------------------------------------------------------------
# Recommendation User
# ---------------------------------------------------------------------------

class RecommendationUser(HttpUser):
    """
    Simulates a real user interacting with the recommendation API.
    
    Behavior mix:
    - 60% home page recommendations
    - 20% PDP similar items
    - 10% search recommendations
    - 10% category recommendations
    """
    
    wait_time = between(0.01, 0.05)  # 20-100 requests/second per user
    
    def on_start(self):
        """Called when user starts. Assign persistent identity."""
        self.user_id = random.choice(USER_IDS)
        self.session_id = f"session_{int(time.time())}_{random.randint(0, 999999)}"
    
    @task(60)
    @tag("home")
    def get_home_recommendations(self):
        """Home page recommendations — most common request type."""
        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "page_context": "home",
            "num_items": 20,
            "client_context": {
                "device": random.choice(DEVICES),
                "locale": "en-US",
            }
        }
        
        with self.client.post(
            "/v1/recommendations",
            json=payload,
            name="/v1/recommendations [home]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data.get("items", [])) < 1:
                    response.failure("No items returned")
                elif data.get("total_latency_ms", 999) > 75:
                    response.failure(f"Latency exceeded SLA: {data['total_latency_ms']}ms")
                else:
                    response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(20)
    @tag("pdp")
    def get_similar_items(self):
        """Product detail page — similar items."""
        payload = {
            "item_id": random.choice(ITEM_IDS),
            "num_items": 10,
            "user_id": self.user_id,
        }
        
        with self.client.post(
            "/v1/similar-items",
            json=payload,
            name="/v1/similar-items [pdp]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(10)
    @tag("search")
    def get_search_recommendations(self):
        """Search page recommendations."""
        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "page_context": "search",
            "num_items": 20,
            "client_context": {
                "search_query": random.choice(["shoes", "laptop", "book", "headphones"]),
            }
        }
        
        self.client.post(
            "/v1/recommendations",
            json=payload,
            name="/v1/recommendations [search]",
        )
    
    @task(10)
    @tag("category")
    def get_category_recommendations(self):
        """Category page recommendations."""
        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "page_context": "category",
            "num_items": 20,
        }
        
        self.client.post(
            "/v1/recommendations",
            json=payload,
            name="/v1/recommendations [category]",
        )
    
    @task(5)
    @tag("health")
    def health_check(self):
        """Health check — monitors service availability."""
        self.client.get("/health", name="/health")


# ---------------------------------------------------------------------------
# Event Hooks — Custom Metrics
# ---------------------------------------------------------------------------

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, **kwargs):
    """Track custom metrics per request."""
    if response and response.status_code == 200:
        try:
            data = response.json()
            fallback = data.get("fallback", {})
            if fallback.get("fallback_used"):
                logger.warning(
                    f"Fallback used: {fallback.get('fallback_type')} - "
                    f"reason: {fallback.get('fallback_reason')}"
                )
        except Exception:
            pass


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary statistics at test end."""
    stats = environment.stats
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    
    if total_requests > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"LOAD TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Requests:  {total_requests:,}")
        logger.info(f"Total Failures:  {total_failures:,}")
        logger.info(f"Error Rate:      {total_failures/total_requests*100:.2f}%")
        logger.info(f"Avg Response:    {stats.total.avg_response_time:.1f}ms")
        logger.info(f"p50 Response:    {stats.total.get_response_time_percentile(0.5):.1f}ms")
        logger.info(f"p95 Response:    {stats.total.get_response_time_percentile(0.95):.1f}ms")
        logger.info(f"p99 Response:    {stats.total.get_response_time_percentile(0.99):.1f}ms")
        logger.info(f"RPS:             {stats.total.current_rps:.1f}")
        logger.info(f"{'='*60}")
        
        # SLA check
        p99 = stats.total.get_response_time_percentile(0.99)
        error_rate = total_failures / total_requests
        
        if p99 > 75:
            logger.error(f"❌ SLA BREACH: p99 ({p99:.1f}ms) > 75ms")
        else:
            logger.info(f"✅ SLA MET: p99 ({p99:.1f}ms) <= 75ms")
        
        if error_rate > 0.001:
            logger.error(f"❌ ERROR RATE: {error_rate*100:.3f}% > 0.1%")
        else:
            logger.info(f"✅ ERROR RATE: {error_rate*100:.3f}% <= 0.1%")
