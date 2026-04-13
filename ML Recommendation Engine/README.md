# Real-Time Product Recommendation System

> **Production-Grade ML Recommendation Engine**  
> Scale: 10M+ products · 1M+ DAU · 50K QPS · p99 < 75ms  
> FAANG-level architecture with 2026 engineering standards

---

## Architecture

```
                     ┌──────────────────────────────────────────────┐
                     │             75ms Latency Budget               │
                     ├──────┬───────┬───────┬───────┬──────┬───────┤
                     │ Net  │ Feat  │ Retrv │ Rank  │ ReRk │ Buffer│
                     │ 5ms  │ 10ms  │ 15ms  │ 25ms  │ 5ms  │ 15ms │
                     └──────┴───────┴───────┴───────┴──────┴───────┘

    Client ──→ API Gateway ──→ Serving Layer (FastAPI)
                                       │
                      ┌────────────────┼──────────────────┐
                      │ parallel       │                  │
                ┌─────▼──────┐  ┌──────▼──────┐   ┌──────▼──────┐
                │ Experiment │  │  Feature    │   │  Retrieval  │
                │ Assignment │  │  Store      │   │  (Milvus)   │
                │ (~0.1ms)   │  │  (Redis)    │   │  (ANN+CF+   │
                └────────────┘  │  (≤10ms)    │   │   Trending) │
                                └─────────────┘   └──────┬──────┘
                                                         │
                                                   ┌─────▼──────┐
                                                   │   Ranking   │
                                                   │  (Triton/   │
                                                   │   DLRM)     │
                                                   │  (≤25ms)    │
                                                   └──────┬──────┘
                                                          │
                                                   ┌──────▼──────┐
                                                   │  Re-Ranking  │
                                                   │  (MMR+Rules) │
                                                   │  (≤5ms)      │
                                                   └─────────────┘
```

### Graceful Degradation Chain

```
Full Pipeline → XGBoost Fallback → Segment Recs → Global Popularity → CDN Cache
   (normal)     (GPU failure)     (retrieval down) (all fails)       (last resort)
```

---

## Repository Structure — 38 Files

```
recommendation-system/
│
├── 📄 README.md
├── 📄 implementation.md                            # 6-phase, 24-week roadmap
├── 📄 Makefile                                     # 20+ build/deploy targets
├── 📄 docker-compose.yml                           # Full local dev stack
│
├── 📁 protos/recommendation/v1/
│   └── recommendation.proto                        # All service API contracts
│
├── 📁 services/                                    # 7 microservices
│   ├── serving/main.py                             # FastAPI orchestrator (800+ LOC)
│   ├── serving/requirements.txt                    # Production dependencies
│   ├── feature-store/feature_service.py            # L1/L2 caching, Redis pipelining
│   ├── retrieval/retrieval_service.py              # Multi-source ANN (Milvus+CF+Trending)
│   ├── ranking/ranking_service.py                  # DLRM via Triton + XGBoost fallback
│   ├── reranking/reranking_service.py              # MMR diversity + business rules
│   ├── ingestion/ingestion_service.py              # Event validation + Kafka + DLQ
│   └── experimentation/experimentation_service.py  # Deterministic A/B + statistics
│
├── 📁 ml/                                          # ML layer
│   ├── models/two_tower/model.py                   # Two-Tower embeddings (PyTorch Lightning)
│   ├── models/dlrm/model.py                        # DLRM ranking + TensorRT export
│   ├── models/xgboost_baseline/model.py            # Gradient boosted trees fallback
│   ├── evaluation/evaluator.py                     # Offline evaluation + quality gates
│   ├── features/feature_dsl.py                     # Unified feature definitions (SSoT)
│   ├── features/data_validation.py                 # Pre-training data validation (PSI)
│   └── pipelines/training_pipeline.py              # Airflow DAG orchestration
│
├── 📁 streaming/                                   # Real-time pipelines
│   ├── session_features/session_features_job.py    # Flink session feature computation
│   └── trending/trending_job.py                    # Flink trending items (tumbling window)
│
├── 📁 infrastructure/
│   ├── docker/Dockerfile.serving                   # Multi-stage production build
│   ├── docker/Dockerfile.ranking                   # GPU-enabled (CUDA 12.4)
│   ├── kubernetes/serving-deployment.yaml          # HPA, PDB, anti-affinity, probes
│   ├── kubernetes/ranking-deployment.yaml          # GPU scheduling + Triton sidecar
│   ├── kubernetes/services-deployment.yaml         # All services + namespace + quotas
│   └── terraform/main.tf                           # EKS + ElastiCache (dev/staging/prod)
│
├── 📁 .github/workflows/
│   └── ci-cd.yml                                   # 6-stage: Lint → Test → Build → Deploy
│
├── 📁 monitoring/
│   ├── prometheus.yml                              # Scrape config (all services)
│   ├── alerts/recommendation-alerts.yml            # 11 SLO-based alert rules
│   ├── dashboards/recommendation-overview.json     # Grafana dashboard (15 panels)
│   └── runbooks/incident-response.md               # P1-P4 incident response procedures
│
└── 📁 tests/
    ├── unit/test_serving.py                        # Circuit breaker, API, MMR tests
    ├── integration/test_e2e.py                     # Full pipeline + latency SLA tests
    ├── load/locustfile.py                          # 10K-50K QPS load testing
    └── chaos/test_chaos.py                         # Pod failure, network partition, cascade
```

---

## Quick Start

```bash
# 1. Start local development environment (all services)
make dev
#    → API:        http://localhost:8080
#    → Grafana:    http://localhost:3000
#    → Jaeger:     http://localhost:16686
#    → Prometheus: http://localhost:9090

# 2. Run unit tests
make test

# 3. Run integration tests
make test-integration

# 4. Load test (10K QPS)
make load-test

# 5. Train models
make train-two-tower
make train-dlrm

# 6. Deploy to staging
make deploy-staging

# 7. Infrastructure provisioning
make tf-plan
make tf-apply
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Serving** | FastAPI + gRPC | Async orchestration + low-latency internal RPC |
| **Feature Store** | Redis Cluster | Pre-joined vectors, sub-ms L1 cache |
| **ANN Search** | Milvus (HNSW) | 95%+ recall@100, distributed, <10ms |
| **Model Serving** | NVIDIA Triton | INT8 TensorRT, dynamic batching, multi-model |
| **Ranking Model** | DLRM (PyTorch) | Explicit dot-product feature interactions |
| **Candidate Model** | Two-Tower | In-batch negatives, separate online/offline towers |
| **Fallback Model** | XGBoost | CPU inference, 30ms for 1K items |
| **Streaming** | Apache Flink | Event-time processing, exactly-once state |
| **Message Queue** | Apache Kafka | 100K events/sec, exactly-once production |
| **Orchestration** | Kubernetes (EKS) | GPU scheduling, HPA, PDB, Istio mesh |
| **IaC** | Terraform | Multi-env (dev/staging/prod), S3 state |
| **Monitoring** | Prometheus + Grafana | SLO-based alerting, 11 alert rules |
| **Tracing** | OpenTelemetry + Jaeger | Distributed trace per request |
| **CI/CD** | GitHub Actions + ArgoCD | GitOps deployment, canary rollouts |
| **ML Pipeline** | Airflow + MLflow | Daily retraining, model registry |
| **Experiments** | Custom (consistent hash) | Deterministic A/B with SRM detection |

---

## Key Design Decisions

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| HNSW over IVF_PQ | +3ms latency, +10% recall | Higher recall worth 3ms at our budget |
| Pre-joined features | More Redis memory, fewer round trips | Single GET per entity vs 15+ per-feature reads |
| In-process MMR | CPU in serving pods | Eliminates network hop, <3ms for 20 items |
| XGBoost fallback | Lower quality, guaranteed availability | 5% CTR drop is better than complete outage |
| Feature DSL | More upfront work, zero skew | Prevents training-serving skew permanently |
| Circuit breakers | False positives possible | Prevents cascade failures and thundering herd |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Implementation Plan](./implementation.md) | 6-phase, 24-week roadmap with risks & rollback |
| [Incident Response](./monitoring/runbooks/incident-response.md) | P1-P4 runbook with kubectl commands |
| [Proto Schema](./protos/recommendation/v1/recommendation.proto) | All service API contracts |
| [Alert Rules](./monitoring/alerts/recommendation-alerts.yml) | 11 production alerts |

---

## Contributing

```bash
# Format code
make format

# Run full lint suite
make lint

# Run all tests
make test-all
```

---

**License**: Internal · **Team**: ML Platform · **Contact**: ml-platform@company.com
