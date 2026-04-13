# 🏗️ Implementation Plan — Real-Time Product Recommendation System

> **Version:** 1.0.0
> **Author:** Principal ML Systems Architect
> **Date:** 2026-04-11
> **Scale:** 10M+ products · 1M+ DAU · 50K QPS · p99 < 75ms
> **Timeline:** 24 weeks (6 phases)

---

## Phase 0: Infrastructure + Design (Weeks 1–2)

### Tasks

- [ ] **0.1** Provision AWS accounts (dev, staging, prod) with proper IAM boundaries
- [ ] **0.2** Set up Terraform modules for EKS, VPC, subnets, security groups
- [ ] **0.3** Deploy EKS cluster (3 node groups: general, GPU, memory-optimized)
- [ ] **0.4** Install Istio service mesh with mTLS enabled
- [ ] **0.5** Deploy monitoring stack (Prometheus + Grafana + Thanos)
- [ ] **0.6** Deploy ELK stack for structured logging
- [ ] **0.7** Deploy Jaeger for distributed tracing
- [ ] **0.8** Set up Kafka cluster (Confluent Cloud) with schema registry
- [ ] **0.9** Design and finalize all Protobuf schemas (.proto files)
- [ ] **0.10** Set up GitHub Actions CI pipeline (lint, test, build, push)
- [ ] **0.11** Set up ArgoCD for GitOps deployment
- [ ] **0.12** Set up HashiCorp Vault for secrets management
- [ ] **0.13** Create development environment with docker-compose (local dev)

### Deliverables

- Running EKS cluster with Istio, monitoring, logging, tracing
- CI/CD pipeline (GitHub Actions → ArgoCD)
- All Protobuf schemas versioned in `protos/` directory
- Terraform modules in `infrastructure/terraform/`
- Local development environment (docker-compose)

### Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| AWS account quota limits | Medium | High | Request quota increases early |
| Istio performance overhead | Low | Medium | Benchmark with/without, tune envoy proxy |
| Kafka partition strategy wrong | Medium | High | Start with 64, can add but not remove |

### Success Metrics

- All infrastructure provisioned via IaC (zero manual resources)
- CI/CD deploys sample service to staging in <5 minutes
- Monitoring shows cluster health (CPU, memory, network)
- Local docker-compose runs full stack

### Rollback Plan

- Every resource created via Terraform → `terraform destroy`
- ArgoCD: revert to previous git commit for any deployment
- Keep manual notes for any manual steps (should be zero)

---

## Phase 1: MVP — Popularity-Based Recommendations (Weeks 3–5)

### Tasks

- [ ] **1.1** Implement Serving Layer (FastAPI orchestrator)
  - REST endpoint: `POST /v1/recommendations`
  - Health check: `GET /health`
  - OpenAPI documentation
- [ ] **1.2** Implement Go gRPC sidecar for downstream fan-out
- [ ] **1.3** Set up Redis Cluster (ElastiCache, 16 nodes)
  - Seed with item catalog features
  - Seed with precomputed popularity rankings
- [ ] **1.4** Implement Feature Service (gRPC)
  - `GetUserFeatures`, `GetItemFeatures`, `BatchGetItemFeatures`
  - Redis client with connection pooling
- [ ] **1.5** Implement Popularity Ranker
  - Item popularity by category (7-day rolling)
  - Global trending items
- [ ] **1.6** Implement basic event ingestion
  - Kafka producer (Python client)
  - Event schema validation
- [ ] **1.7** Implement API Gateway (Kong/Istio ingress)
  - Rate limiting, authentication
- [ ] **1.8** Deploy MVP to staging with synthetic traffic
- [ ] **1.9** Write integration tests for full MVP flow
- [ ] **1.10** Load test MVP at 1K QPS (baseline)

### Deliverables

- Working recommendation API returning popularity-based results
- Feature store with item features and popularity scores
- Event ingestion pipeline (events → Kafka)
- Staging deployment with 1K QPS load test passing

### Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Redis sizing wrong | Medium | Medium | Start small, monitor memory, scale up |
| API latency >200ms | Low | High | Profile and optimize hot paths |
| Event data quality issues | High | Medium | Schema validation, DLQ from day 1 |

### Success Metrics

- API returns recommendations within 200ms p99 (relaxed for MVP)
- Redis feature reads <5ms p99
- Events flowing through Kafka (end-to-end verified)
- Zero deployment manual steps

### Rollback Plan

- Feature flag: `recs_enabled = false` → return empty recommendations
- Redis: snapshot + restore from backup
- Full rollback: ArgoCD revert

---

## Phase 2: Retrieval + Ranking Models (Weeks 6–10)

### Tasks

- [ ] **2.1** Prepare training data
  - Export interaction events from Kafka → S3 (Parquet)
  - Label generation: positive (click/purchase), negative (impression, no click)
  - Train/validation/test split (time-based: 80/10/10)
  - Data quality validation (Great Expectations)
- [ ] **2.2** Train Two-Tower model for candidate generation
  - User tower: user features → 128-dim embedding
  - Item tower: item features → 128-dim embedding
  - Negative sampling: in-batch + hard negatives
  - Framework: PyTorch + PyTorch Lightning
- [ ] **2.3** Generate item embeddings (all 10M items)
  - Batch inference on GPU cluster
  - Export embeddings to Milvus-compatible format
- [ ] **2.4** Deploy Milvus cluster (8 shards, 3 replicas each)
  - Load 10M item embeddings
  - Benchmark: recall@100 and latency at target QPS
- [ ] **2.5** Implement Retrieval Service (gRPC)
  - ANN search via Milvus client
  - Multi-source retrieval (ANN + popularity + trending)
  - Merge and deduplicate candidates
- [ ] **2.6** Train XGBoost ranking baseline
  - ~50 features (start simple)
  - Optimize AUC on holdout set
- [ ] **2.7** Train DLRM ranking model
  - ~200 features (full feature set)
  - TensorRT optimization → INT8 quantization
- [ ] **2.8** Deploy Triton Inference Server with DLRM
  - Dynamic batching configuration
  - GPU memory management
  - Model versioning
- [ ] **2.9** Implement Ranking Service (gRPC)
  - Feature assembly pipeline
  - Triton client for inference
  - Fallback to XGBoost
- [ ] **2.10** Implement Re-Ranking Service
  - MMR algorithm
  - Business rules engine
  - Diversity metrics
- [ ] **2.11** Integrate full pipeline: Retrieval → Ranking → Re-Ranking
- [ ] **2.12** Offline evaluation: Compare vs popularity baseline
- [ ] **2.13** A/B test framework setup (experiment assignment service)
- [ ] **2.14** Deploy to staging, load test at 10K QPS

### Deliverables

- Trained Two-Tower model with item embeddings in Milvus
- Trained DLRM model deployed on Triton
- Full retrieval → ranking → re-ranking pipeline
- Offline evaluation showing lift over popularity baseline
- Staging deployment passing 10K QPS load test at <75ms p99

### Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Poor model quality (AUC < 0.7) | Medium | High | Fall back to XGBoost, improve features iteratively |
| Milvus cluster instability | Medium | Medium | Zilliz Cloud managed option as backup |
| GPU OOM on Triton | Low | High | Model quantization, reduce batch size |
| Feature engineering bottleneck | High | Medium | Start with proven features from literature |

### Success Metrics

- ANN recall@100 > 20%
- DLRM AUC > 0.75 on holdout
- Full pipeline p99 < 75ms at 10K QPS
- Offline CTR prediction accuracy > 70%

### Rollback Plan

- Model rollback: Triton loads previous model version (<30s)
- Retrieval rollback: bypass ANN, use popularity fallback
- Full pipeline rollback: serve MVP (popularity) via feature flag

---

## Phase 3: Real-Time Streaming System (Weeks 11–14)

### Tasks

- [ ] **3.1** Implement Flink job: Session Feature Aggregation
  - Session window (30-min gap)
  - Output: last-N viewed items, session click count, session dwell time
  - Sink: Redis
- [ ] **3.2** Implement Flink job: Item Statistics
  - Sliding window (1h, 5-min slide)
  - Output: real-time CTR, view count, cart-add rate
  - Sink: Redis
- [ ] **3.3** Implement Flink job: Trending Items
  - Tumbling window (5 min)
  - Output: top-1000 trending items by category
  - Sink: Redis
- [ ] **3.4** Implement Flink job: Real-Time User Embeddings
  - Sliding window (30 min, 5-min slide)
  - Output: updated user embedding (Two-Tower user inference)
  - Sink: Milvus (for retrieval query) + Redis (for ranking features)
- [ ] **3.5** Implement streaming item onboarding
  - Catalog update → content embedding → Milvus insert
  - Target: new item searchable within 15 minutes
- [ ] **3.6** Implement event enrichment pipeline
  - Raw events → enriched events (join with item metadata, user segment)
- [ ] **3.7** Configure Flink checkpointing (S3 state backend)
  - Exactly-once for purchase events
  - At-least-once for feature aggregation
- [ ] **3.8** Implement late event handling (side-outputs)
- [ ] **3.9** Monitor Flink jobs: checkpoint size, processing lag, throughput
- [ ] **3.10** Integration test: end-to-end streaming (event → feature → recommendation)

### Deliverables

- 4 Flink streaming jobs running in production
- Real-time features updating in Redis within 5 seconds
- New items searchable within 15 minutes
- Streaming monitoring dashboard

### Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Flink checkpoint failures | Medium | High | RocksDB state backend, incremental checkpoints |
| Consumer lag during peak | Medium | Medium | Auto-scale Flink task managers |
| State store OOM | Low | High | State TTL (24h), compaction |
| Late event volume > expected | Low | Low | Increase watermark delay if needed |

### Success Metrics

- Feature freshness < 5 minutes (streaming features)
- New item onboarding < 15 minutes
- Flink job uptime > 99.9%
- Consumer lag < 1 minute during peak

### Rollback Plan

- Disable streaming features: serve batch-computed features (4h old)
- Flink jobs can be restarted from latest checkpoint
- If checkpoint corrupted: restart from beginning of Kafka offsets (replay last 24h)

---

## Phase 4: Scaling to 50K QPS (Weeks 15–17)

### Tasks

- [ ] **4.1** Load test full pipeline at 25K QPS → fix bottlenecks
- [ ] **4.2** Load test full pipeline at 50K QPS → fix bottlenecks
- [ ] **4.3** Deploy to 3 regions (us-east-1, us-west-2, eu-west-1)
  - GeoDNS routing (Route53)
  - Cross-region data replication (Kafka MirrorMaker, Redis sentinel)
- [ ] **4.4** Configure HPA for all services
  - Custom metrics: QPS, GPU utilization, Redis latency
- [ ] **4.5** Implement connection pooling optimization
  - gRPC multiplexed connections
  - Redis connection pool (keepalive)
- [ ] **4.6** Implement L1 process-local cache
  - User features: 10s TTL
  - Item features: 60s TTL
  - Expected hit rate: 60% for user features
- [ ] **4.7** Implement request coalescing
  - Deduplicate concurrent identical requests
- [ ] **4.8** Implement hedged requests for retrieval
  - Send to 2 replicas, take first response
- [ ] **4.9** Performance profiling (flame graphs, latency breakdown)
- [ ] **4.10** Off-peak auto-scaling configuration

### Deliverables

- System handles 50K QPS at <75ms p99
- 3-region active-active deployment
- Auto-scaling tested and validated
- Performance report with latency breakdown

### Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Cross-region latency too high | Medium | High | Ensure all reads are region-local |
| Auto-scaling too slow | Medium | Medium | Pre-warm during known peaks |
| Network bottleneck | Low | High | Profile with tcpdump, optimize serialization |

### Success Metrics

- 50K QPS sustained for 1 hour, p99 < 75ms
- Cross-region failover completes in <60s
- Auto-scale up from 10K → 50K QPS in <3 minutes
- Off-peak cost reduction: 40%+

### Rollback Plan

- Multi-region: fall back to single region
- Auto-scaling: switch to fixed replica count
- Caching: disable if invalidation bugs discovered

---

## Phase 5: ML Optimization + Experimentation (Weeks 18–20)

### Tasks

- [ ] **5.1** Model quantization deep-dive
  - INT8 quantization with TensorRT
  - Accuracy validation: AUC drop < 1%
  - Latency benchmark: target 2× speedup
- [ ] **5.2** Feature pre-joining
  - Denormalize user + item features into single Redis key
  - Benchmark: feature fetch time reduction
- [ ] **5.3** Implement full A/B testing platform
  - Experiment definition API
  - Consistent user assignment (hash-based)
  - Statistical analysis pipeline (Spark)
  - Guardrail metrics (latency, error rate, revenue)
- [ ] **5.4** Run A/B test: DLRM vs XGBoost
  - 2-week test, 10% traffic per variant
  - Measure: CTR, conversion, revenue, latency
- [ ] **5.5** Cold start optimization
  - Thompson Sampling for exploration slots
  - Content-based warm-start for new items
  - Bandit-based new user strategy
- [ ] **5.6** Model retraining pipeline
  - Weekly automated retraining (Airflow/Kubeflow)
  - Data validation gate (Great Expectations)
  - Auto-deploy if quality gate passes
- [ ] **5.7** Drift detection pipeline
  - Feature drift: PSI per feature (hourly)
  - Prediction drift: score distribution monitoring
  - Auto-retrain trigger
- [ ] **5.8** Shadow deployment infrastructure
  - Log predictions from shadow model alongside production
  - Compare offline without serving shadow predictions

### Deliverables

- Quantized model deployed (2× faster, <1% accuracy loss)
- A/B testing platform operational with first experiment results
- Automated retraining pipeline with quality gates
- Drift detection with alerting
- Cold start optimization deployed

### Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Quantization accuracy loss > 1% | Low | Medium | Use calibration dataset, tune quantization |
| A/B test insufficient sample size | Medium | Low | Run for 3 weeks instead of 2 |
| Retraining pipeline failures | Medium | Medium | Manual fallback, checkpoint-based restart |

### Success Metrics

- Quantized model: 2× inference speedup, AUC drop < 1%
- A/B test statistical significance (p < 0.05) within 2 weeks
- Drift detection catches injected synthetic drift within 1 hour
- Cold start: new user engagement +10% vs popularity baseline

### Rollback Plan

- Quantization: revert to FP32 model
- A/B tests: kill experiment, route 100% to control
- Retraining: use previous model version (manual trigger)

---

## Phase 6: Enterprise Hardening (Weeks 21–24)

### Tasks

- [ ] **6.1** Implement circuit breakers for all downstream services
  - Configuration per service (failure threshold, recovery timeout)
  - Dashboard: circuit breaker state visualization
- [ ] **6.2** Implement retry strategy
  - Exponential backoff with jitter
  - Retry budgets (prevent cascade)
- [ ] **6.3** Implement comprehensive rate limiting
  - Per-user, per-API-key, per-IP, global
  - Token bucket algorithm
- [ ] **6.4** Set up RBAC
  - Kubernetes RBAC for cluster access
  - API-level RBAC via JWT claims
  - 5 roles: reader, admin, operator, data_scientist, super_admin
- [ ] **6.5** Secrets management hardening
  - Vault integration for all services
  - Dynamic database credentials
  - 90-day rotation for API keys
- [ ] **6.6** Data privacy compliance
  - User data purge pipeline (GDPR right-to-forget)
  - PII pseudonymization in logs
  - Differential privacy for analytics
- [ ] **6.7** Security audit
  - Penetration test (external vendor)
  - Dependency vulnerability scan (Snyk/Trivy)
  - Container image scanning
- [ ] **6.8** Chaos engineering test suite
  - Pod failure, node failure, AZ failure, region failure
  - Redis failure, Kafka failure, GPU failure
  - Model corruption, config corruption
  - All tests pass in staging
- [ ] **6.9** Operational runbooks
  - P1 incident response (< 30 min resolution)
  - Scaling procedures
  - Model rollback procedures
  - Data recovery procedures
- [ ] **6.10** Load test the hardened system
  - 50K QPS with chaos experiments running simultaneously
- [ ] **6.11** Production readiness review
  - Checklist: SLO defined, alerts configured, runbooks written, on-call rotation
- [ ] **6.12** Documentation
  - Architecture decision records (ADRs)
  - API documentation (Swagger/gRPC docs)
  - Onboarding guide for new engineers

### Deliverables

- Circuit breakers, retries, rate limiting implemented
- RBAC with Vault integration
- GDPR-compliant data handling
- Chaos engineering test suite (all passing)
- Operational runbooks and documentation
- Production readiness review completed

### Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Circuit breaker over-triggers | Medium | Medium | Careful threshold tuning in staging |
| Security audit findings | High | Varies | Budget 2 weeks for remediation |
| Chaos tests reveal unknown failures | Medium | High | Fix before prod launch |

### Success Metrics

- All circuit breakers tested and validated
- RBAC blocks unauthorized access (tested)
- Data purge completes within 72 hours
- Chaos tests: system recovers from all failure scenarios
- On-call can resolve P1 from runbook in <30 minutes
- Zero critical/high security findings (or remediated)

### Rollback Plan

- Circuit breakers: disable (passthrough mode) if causing issues
- RBAC: emergency break-glass role for full access
- Chaos tests are staging-only (no prod risk)
- Full system rollback: blue-green deployment, switch traffic

---

## Timeline Summary

```
Week  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
      ├──────┤
      Phase 0: Infra
               ├────────┤
               Phase 1: MVP
                        ├──────────────────┤
                        Phase 2: Retrieval + Ranking
                                          ├──────────────┤
                                          Phase 3: Real-Time
                                                        ├────────┤
                                                        Phase 4: Scale
                                                                 ├────────┤
                                                                 Phase 5: ML Opt
                                                                          ├──────────────┤
                                                                          Phase 6: Enterprise
```

## Team Allocation

| Role | Count | Primary Phase | Ongoing |
|------|-------|--------------|---------|
| ML Engineer | 2 | Phase 2, 5 | Model iteration, retraining |
| Backend Engineer | 2 | Phase 1, 3, 4 | API, streaming, scaling |
| Data Engineer | 1 | Phase 0, 1, 3 | Pipelines, data quality |
| Infrastructure/SRE | 1 | Phase 0, 4, 6 | Reliability, monitoring |
| Product Manager | 1 | All phases | Requirements, A/B tests |
| **Total** | **7** | — | — |

## Cost Projection

| Phase | Duration | Monthly Cost | Phase Total |
|-------|----------|-------------|------------|
| Phase 0 | 2 weeks | $5,000 | $2,500 |
| Phase 1 | 3 weeks | $15,000 | $11,250 |
| Phase 2 | 5 weeks | $35,000 | $43,750 |
| Phase 3 | 4 weeks | $45,000 | $45,000 |
| Phase 4 | 3 weeks | $55,000 | $41,250 |
| Phase 5 | 3 weeks | $57,000 | $42,750 |
| Phase 6 | 4 weeks | $57,000 | $57,000 |
| **Total** | **24 weeks** | — | **~$243,500** |
| **Ongoing (post-launch)** | — | **$57,000/mo** | — |
