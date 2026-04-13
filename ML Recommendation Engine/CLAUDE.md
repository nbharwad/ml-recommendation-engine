# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start full local stack (API, Grafana, Jaeger, Prometheus)
make dev

# Stop local stack
make dev-down

# Unit tests with coverage
make test
# or directly:
pytest tests/unit/ -v --cov=services --cov-report=term-missing

# Single test file
pytest tests/unit/test_serving.py -v -k "test_circuit_breaker"

# Integration tests (requires docker-compose up)
make test-integration

# Lint (ruff + black + mypy)
make lint

# Auto-format
make format

# Build Docker images
make build

# Train models
make train-two-tower
make train-dlrm

# Load test (10K QPS)
make load-test
```

Python version: **3.12**. Linters: `ruff`, `black`, `isort`, `mypy`.

## Architecture Overview

This is a **real-time product recommendation system** targeting 10M+ products, 1M+ DAU, 50K QPS, p99 < 75ms.

### Request Flow (75ms total budget)

```
Client → API Gateway → Serving (FastAPI) [services/serving/main.py]
                              │
         ┌────────────────────┼──────────────────┐
         │ parallel           │                  │
    Experiment           Feature Store      Retrieval
    Assignment           (Redis, ≤10ms)    (Milvus ANN +
    (hash-based)         [feature-store/]   CF + Trending, ≤15ms)
                                           [retrieval/]
                                                 │
                                            Ranking
                                            (DLRM via Triton, ≤25ms)
                                            [ranking/]
                                                 │
                                           Re-Ranking
                                           (MMR + business rules, ≤5ms)
                                           [reranking/]
```

### Graceful Degradation Chain
`Full Pipeline → XGBoost Fallback → Segment Recs → Global Popularity → CDN Cache`

Each downstream call is wrapped in a **circuit breaker** (`CircuitBreaker` class in `services/serving/main.py`). Services fail fast when open, recover through HALF_OPEN state.

### Service Layer (`services/`)

| Service | Protocol | Purpose |
|---------|----------|---------|
| `serving/main.py` | REST (port 8080) | Orchestrator; all inbound traffic |
| `feature-store/feature_service.py` | gRPC (50051) | L1/L2 cache; pre-joined feature blobs from Redis |
| `retrieval/retrieval_service.py` | gRPC (50052) | ANN (60%) + CF (20%) + Trending (10%) + Rules (10%) |
| `ranking/ranking_service.py` | gRPC (50053) | DLRM via NVIDIA Triton; XGBoost fallback |
| `reranking/reranking_service.py` | gRPC (50054) | MMR diversity + business rule injection |
| `ingestion/ingestion_service.py` | REST | Event ingestion → Kafka → DLQ |
| `experimentation/experimentation_service.py` | gRPC (50055) | Deterministic A/B via consistent hash |

Internal service addresses are read from env vars (e.g., `FEATURE_SERVICE_HOST`, `RANKING_SERVICE_HOST`).

### ML Layer (`ml/`)

- **Two-Tower** (`ml/models/two_tower/model.py`): PyTorch Lightning dual-encoder. User tower runs **online** (real-time inference); item tower runs **offline** (batch every 4h, stored in Milvus). Output: 128-dim L2-normalized embeddings.
- **DLRM** (`ml/models/dlrm/model.py`): Ranking model. Exported to TensorRT INT8 for Triton serving. Calibration in `ml/models/dlrm/calibrate.py`.
- **XGBoost Baseline** (`ml/models/xgboost_baseline/model.py`): CPU fallback; ~30ms for 1K items.
- **Feature DSL** (`ml/features/feature_dsl.py`): **Single source of truth** for feature definitions across offline training (Spark), online serving (Redis), and streaming (Flink). Always add new features here first to avoid training-serving skew.
- **Evaluator** (`ml/evaluation/evaluator.py`): Offline evaluation with quality gates. Must pass before model promotion.
- **Training Pipeline** (`ml/pipelines/training_pipeline.py`): Airflow DAG for daily retraining with MLflow model registry.

### Streaming Layer (`streaming/`)

Apache Flink jobs (Java + Python stubs) for real-time feature computation:
- `session_features/`: Session-window user features → Redis
- `item_stats/`: Rolling item click/purchase stats → Redis
- `user_embeddings/`: Online user embedding updates → Redis + Milvus
- `trending/`: Tumbling-window trending items → Redis
- `enrichment/`: Event enrichment before Kafka downstream

Java source under `streaming/<job>/src/main/java/com/recsys/flink/`.

### Infrastructure

- **Vector DB**: Milvus (HNSW index, 10M × 128-dim vectors, 8 shards, 3 replicas)
- **Feature cache**: Redis Cluster with `volatile-lfu` eviction
- **Message queue**: Kafka (KRaft mode, no ZooKeeper)
- **Model serving**: NVIDIA Triton (GPU pods in Kubernetes)
- **Orchestration**: EKS + Istio service mesh + ArgoCD GitOps
- **IaC**: Terraform in `infrastructure/terraform/` (dev/staging/prod via `-var="environment=..."`)

### Proto Contract

All service API schemas defined in `protos/recommendation/v1/recommendation.proto`. Generate stubs with `grpcio-tools`.

### CI/CD (`.github/workflows/ci-cd.yml`)

6 stages: Lint → Unit Test → Integration Test (main branch only) → Build & Push to ECR → Deploy Staging → Deploy Production (manual approval). Production uses ArgoCD canary rollout at 5% → 100%.

### Observability

- **Metrics**: Prometheus scraping all services; 11 SLO-based alert rules in `monitoring/alerts/`
- **Tracing**: OpenTelemetry → Jaeger (every request gets a trace ID)
- **Dashboards**: Grafana at `localhost:3000` (admin/admin); 15-panel overview in `monitoring/dashboards/`
- **Runbooks**: `monitoring/runbooks/incident-response.md` for P1-P4 incidents

### Key Patterns

- **Config via frozen dataclasses**: All service configs use `@dataclass(frozen=True)` with env var defaults. No mutable global state.
- **Feature DSL prevents skew**: `ml/features/feature_dsl.py` is the canonical definition; both training and serving import from it. Never hardcode feature transformations outside this file.
- **MMR runs in-process**: Diversity re-ranking is intentionally kept in the serving pod (not a separate service) to avoid network latency.
- **Go sidecar**: `services/go/sidecar/main.go` handles low-latency path operations alongside the Python serving layer.
