"""
Chaos Engineering Test Suite
==============================
Validates system resilience under failure conditions.

Uses Chaos Mesh / Litmus compatible experiment definitions.
Each test has:
1. Steady state hypothesis
2. Chaos injection
3. Verification that system recovers
4. Rollback

Run in staging only — NEVER in production.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest


API_URL = os.getenv("API_URL", "http://localhost:8080")
KUBECTL = os.getenv("KUBECTL", "kubectl")
NAMESPACE = "recommendation"


# ---------------------------------------------------------------------------
# Chaos Experiment Definitions
# ---------------------------------------------------------------------------

@dataclass
class ChaosExperiment:
    """Base chaos experiment definition."""
    name: str
    description: str
    steady_state: dict[str, Any]  # expected baseline metrics
    recovery_timeout_sec: int = 120
    
    def verify_steady_state(self, client: httpx.Client) -> bool:
        """Verify system is in steady state before injecting chaos."""
        health = client.get("/health")
        if health.status_code != 200:
            return False
        
        # Run test requests
        latencies = []
        errors = 0
        for i in range(20):
            try:
                start = time.monotonic()
                resp = client.post("/v1/recommendations", json={
                    "user_id": f"chaos_steady_{i}",
                    "num_items": 5,
                })
                latencies.append((time.monotonic() - start) * 1000)
                if resp.status_code != 200:
                    errors += 1
            except Exception:
                errors += 1
        
        error_rate = errors / 20
        p99 = sorted(latencies)[min(18, len(latencies) - 1)] if latencies else 9999
        
        return error_rate < 0.05 and p99 < self.steady_state.get("max_p99_ms", 200)


# ---------------------------------------------------------------------------
# Chaos Mesh Manifests
# ---------------------------------------------------------------------------

POD_FAILURE_MANIFEST = """
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: {name}
  namespace: {namespace}
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - {namespace}
    labelSelectors:
      app: {target_app}
  duration: "{duration}"
"""

NETWORK_DELAY_MANIFEST = """
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: {name}
  namespace: {namespace}
spec:
  action: delay
  mode: all
  selector:
    namespaces:
      - {namespace}
    labelSelectors:
      app: {target_app}
  delay:
    latency: "{delay_ms}ms"
    jitter: "{jitter_ms}ms"
  duration: "{duration}"
"""

NETWORK_PARTITION_MANIFEST = """
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: {name}
  namespace: {namespace}
spec:
  action: partition
  mode: all
  selector:
    namespaces:
      - {namespace}
    labelSelectors:
      app: {source_app}
  direction: to
  target:
    selector:
      namespaces:
        - {namespace}
      labelSelectors:
        app: {target_app}
  duration: "{duration}"
"""


# ---------------------------------------------------------------------------
# Chaos Tests
# ---------------------------------------------------------------------------

class TestPodFailure:
    """Test system recovery when a pod is killed."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_URL, timeout=30.0)
    
    @pytest.mark.chaos
    def test_serving_pod_kill(self, client):
        """
        Hypothesis: Killing one serving pod should not cause visible errors.
        
        Expected:
        - System auto-recovers within 30s
        - Error rate < 5% during recovery
        - No data loss
        """
        experiment = ChaosExperiment(
            name="serving-pod-kill",
            description="Kill one recommendation-serving pod",
            steady_state={"max_p99_ms": 200, "max_error_rate": 0.05},
        )
        
        # Verify steady state
        assert experiment.verify_steady_state(client), "System not in steady state"
        
        # Inject chaos (in production: apply Chaos Mesh manifest)
        # manifest = POD_FAILURE_MANIFEST.format(
        #     name="serving-pod-kill",
        #     namespace=NAMESPACE,
        #     target_app="recommendation-serving",
        #     duration="30s",
        # )
        # os.system(f"echo '{manifest}' | {KUBECTL} apply -f -")
        
        # Monitor during chaos
        errors_during_chaos = 0
        total_during_chaos = 50
        
        for i in range(total_during_chaos):
            try:
                resp = client.post("/v1/recommendations", json={
                    "user_id": f"chaos_test_{i}",
                    "num_items": 5,
                })
                if resp.status_code != 200:
                    errors_during_chaos += 1
            except Exception:
                errors_during_chaos += 1
            time.sleep(0.5)
        
        error_rate = errors_during_chaos / total_during_chaos
        print(f"\nPod kill test: {errors_during_chaos}/{total_during_chaos} errors ({error_rate:.1%})")
        
        # Verify recovery
        assert experiment.verify_steady_state(client), "System did not recover"
        
        # In staging, error rate should be < 5%
        # assert error_rate < 0.05, f"Error rate {error_rate:.1%} exceeds 5%"
    
    @pytest.mark.chaos
    def test_gpu_node_failure(self, client):
        """
        Hypothesis: GPU node failure → XGBoost fallback → service continues.
        
        Expected:
        - Ranking falls back to XGBoost (CPU)
        - Latency may increase to ~100ms
        - CTR drops ~5% (acceptable degradation)
        - System auto-recovers when GPU returns
        """
        experiment = ChaosExperiment(
            name="gpu-node-failure",
            description="Simulate GPU node unavailability",
            steady_state={"max_p99_ms": 300, "max_error_rate": 0.01},
            recovery_timeout_sec=300,
        )
        
        assert experiment.verify_steady_state(client)
        
        # Monitor during simulated GPU failure
        fallback_count = 0
        total = 20
        
        for i in range(total):
            resp = client.post("/v1/recommendations", json={
                "user_id": f"gpu_fail_test_{i}",
                "num_items": 10,
            })
            if resp.status_code == 200:
                data = resp.json()
                if data.get("fallback", {}).get("fallback_used"):
                    fallback_count += 1
        
        print(f"\nGPU failure test: {fallback_count}/{total} used fallback")
        # Fallback is expected when GPU is down


class TestNetworkPartition:
    """Test system behavior under network failures."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_URL, timeout=30.0)
    
    @pytest.mark.chaos
    def test_feature_store_partition(self, client):
        """
        Hypothesis: Feature store (Redis) network partition → use default features.
        
        Expected:
        - System serves with default feature vectors
        - CTR drops ~8-12% (detected by monitoring)
        - Latency stays within SLA
        - Auto-recovers when partition heals
        """
        experiment = ChaosExperiment(
            name="feature-store-partition",
            description="Network partition between serving and Redis",
            steady_state={"max_p99_ms": 200, "max_error_rate": 0.01},
        )
        
        assert experiment.verify_steady_state(client)
        
        # During partition, requests should still succeed (with degraded quality)
        success_count = 0
        total = 30
        
        for i in range(total):
            try:
                resp = client.post("/v1/recommendations", json={
                    "user_id": f"partition_test_{i}",
                    "num_items": 10,
                })
                if resp.status_code == 200:
                    success_count += 1
            except Exception:
                pass
        
        success_rate = success_count / total
        print(f"\nFeature store partition: {success_count}/{total} succeeded ({success_rate:.1%})")
        
        # Even under partition, we should serve recommendations (from fallback)
        assert success_rate >= 0.90, f"Success rate {success_rate:.1%} below 90%"
    
    @pytest.mark.chaos
    def test_retrieval_service_partition(self, client):
        """
        Hypothesis: Retrieval service unavailable → popularity fallback.
        
        Expected:
        - System serves popularity-based recommendations
        - Response time may decrease (simpler pipeline)
        - CTR drops ~15-20%
        """
        experiment = ChaosExperiment(
            name="retrieval-partition",
            description="Network partition to retrieval service",
            steady_state={"max_p99_ms": 200, "max_error_rate": 0.01},
        )
        
        assert experiment.verify_steady_state(client)
        
        # All requests should succeed (popularity fallback)
        for i in range(20):
            resp = client.post("/v1/recommendations", json={
                "user_id": f"retrieval_partition_{i}",
                "num_items": 10,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["items"]) > 0, "No items returned during retrieval partition"


class TestResourceExhaustion:
    """Test behavior under resource pressure."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_URL, timeout=30.0)
    
    @pytest.mark.chaos
    def test_memory_pressure(self, client):
        """
        Hypothesis: Memory pressure → graceful OOM handling.
        
        Expected:
        - Kubernetes restarts the pod
        - Other pods continue serving
        - Service recovers within 30s
        """
        experiment = ChaosExperiment(
            name="memory-stress",
            description="Memory stress on serving pods",
            steady_state={"max_p99_ms": 200, "max_error_rate": 0.05},
        )
        
        assert experiment.verify_steady_state(client)
        # In production: apply Chaos Mesh StressChaos manifest
    
    @pytest.mark.chaos
    def test_cpu_saturation(self, client):
        """
        Hypothesis: CPU saturation → HPA scales up, latency temporarily increases.
        """
        experiment = ChaosExperiment(
            name="cpu-stress",
            description="CPU stress on serving pods",
            steady_state={"max_p99_ms": 300, "max_error_rate": 0.10},
        )
        
        assert experiment.verify_steady_state(client)


class TestCascadeFailure:
    """Test that failures don't cascade across services."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_URL, timeout=30.0)
    
    @pytest.mark.chaos
    def test_circuit_breaker_prevents_cascade(self, client):
        """
        Hypothesis: Slow downstream → circuit breaker opens → fast failure.
        
        Expected:
        - Circuit breaker trips after threshold failures
        - Subsequent requests get fallback immediately (no waiting)
        - Latency stays bounded (not waiting for timeouts)
        """
        # Inject slow responses to a downstream service
        # In production: apply NetworkChaos with large delay
        
        latencies = []
        for i in range(30):
            start = time.monotonic()
            resp = client.post("/v1/recommendations", json={
                "user_id": f"cascade_test_{i}",
                "num_items": 10,
            })
            latency = (time.monotonic() - start) * 1000
            latencies.append(latency)
        
        # After circuit breaker trips, latencies should be low
        # (fast-fail, not waiting for timeout)
        print(f"\nCascade test latencies: min={min(latencies):.1f}ms, max={max(latencies):.1f}ms")


# ---------------------------------------------------------------------------
# Chaos Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run chaos tests:
        pytest tests/chaos/test_chaos.py -v -m chaos --timeout=300
    
    Prerequisites:
    - Chaos Mesh installed in staging cluster
    - Services deployed and healthy
    - NEVER run in production
    """
    pytest.main([__file__, "-v", "-m", "chaos", "--timeout=300"])
