# ============================================================================
# Operational Runbook — Recommendation System
# For on-call engineers responding to production incidents
# ============================================================================

# 🔴 P1 — Recommendation Service Down (< 30 min resolution)

## Symptoms
- Alert: `RecommendationErrorRateHigh` (>0.5% error rate)
- Users seeing empty recommendation sections
- Revenue dashboard shows sudden CTR drop

## Triage (2 minutes)
```bash
# 1. Check service health
kubectl -n recommendation get pods | grep -v Running

# 2. Check circuit breakers
curl http://recommendation-serving.recommendation.svc/health | jq '.dependencies'

# 3. Check recent deployments
kubectl -n recommendation rollout history deployment/recommendation-serving

# 4. Check logs for errors
kubectl -n recommendation logs -l app=recommendation-serving --tail=100 | grep ERROR
```

## Common Causes & Fixes

### Cause 1: Bad deployment (most common)
```bash
# Rollback to previous version
kubectl -n recommendation rollout undo deployment/recommendation-serving

# Verify
kubectl -n recommendation rollout status deployment/recommendation-serving
```

### Cause 2: Redis (Feature Store) down
```bash
# Check Redis cluster
aws elasticache describe-replication-groups --replication-group-id rec-system-prod-features

# If Redis is down, serving layer uses fallback features automatically
# CTR will drop ~8-12% but service stays up

# Manual failover if needed
aws elasticache modify-replication-group \
  --replication-group-id rec-system-prod-features \
  --primary-cluster-id rec-system-prod-features-002  # promote replica
```

### Cause 3: GPU/Triton failure
```bash
# Check GPU pods
kubectl -n recommendation get pods -l app=ranking-service

# Check Triton health
kubectl -n recommendation exec -it ranking-service-xxx -c triton -- curl localhost:8000/v2/health/ready

# Ranking service auto-falls back to XGBoost (CPU)
# CTR drops ~5% but service stays up

# If all ranking pods are down, restart:
kubectl -n recommendation rollout restart deployment/ranking-service
```

### Cause 4: Traffic spike (overload)
```bash
# Check HPA status
kubectl -n recommendation get hpa

# Force scale up
kubectl -n recommendation scale deployment/recommendation-serving --replicas=60

# Check rate limiting
kubectl -n recommendation logs -l app=recommendation-serving | grep "rate_limit"
```

---

# 🟡 P2 — Latency SLA Breach (p99 > 75ms)

## Symptoms
- Alert: `RecommendationP99LatencyHigh`
- Grafana dashboard shows p99 creeping up

## Triage
```bash
# 1. Check per-component latency
curl http://recommendation-serving.recommendation.svc/metrics | grep component_latency

# 2. Identify slow component
# Look for:
#   - feature_fetch_ms > 10ms → Redis issue
#   - retrieval_ms > 15ms → Milvus issue  
#   - ranking_ms > 25ms → GPU contention
#   - reranking_ms > 5ms → Complex rule set

# 3. Check resource utilization
kubectl -n recommendation top pods
```

## Fixes

### Slow feature fetch
```bash
# Check Redis latency
redis-cli -h $REDIS_HOST --latency-history -i 1

# Check Redis memory
redis-cli -h $REDIS_HOST info memory

# If memory pressure → flush low-priority caches
redis-cli -h $REDIS_HOST config set maxmemory-policy volatile-lfu
```

### Slow retrieval
```bash
# Check Milvus query latency
kubectl -n recommendation logs -l app=retrieval-service | grep latency

# If ANN index is stale → force rebuild
kubectl -n recommendation exec -it retrieval-pod -- python -c "
from retrieval_service import RetrievalService
# force_rebuild_index()
"
```

### Slow ranking
```bash
# Check GPU utilization
kubectl -n recommendation exec -it ranking-pod -c triton -- nvidia-smi

# If GPU util > 90% → scale up
kubectl -n recommendation scale deployment/ranking-service --replicas=8
```

---

# 🟡 P2 — Model Quality Degradation

## Symptoms
- Alert: `RecommendationHighFallbackRate` (>5% fallback)
- Alert: `ANNRecallDegraded` (<80% recall)
- Business dashboard: CTR drop >10% from baseline

## Triage
```bash
# 1. Check model version
curl http://recommendation-serving.recommendation.svc/health | jq '.dependencies'

# 2. Check if new model was deployed recently
kubectl -n recommendation get deployment ranking-service -o jsonpath='{.metadata.annotations}'

# 3. Check feature drift
# Look at Grafana dashboard: "Feature Drift (PSI)"

# 4. Check ANN recall
curl http://retrieval-service.recommendation.svc/metrics | grep ann_recall
```

## Fixes

### Bad model version
```bash
# Rollback model in Triton
kubectl -n recommendation exec -it ranking-pod -c triton -- \
  ls /models/dlrm_ranking/

# Switch to previous version by updating model config
# Or rollback Triton deployment
kubectl -n recommendation rollout undo deployment/ranking-service
```

### ANN index corruption
```bash
# Check index health
kubectl -n recommendation logs -l app=retrieval-service | grep "recall"

# Restore previous index snapshot
# Index snapshots are in S3: s3://rec-system/index-snapshots/
aws s3 ls s3://rec-system/index-snapshots/ --recursive

# Load previous snapshot
kubectl -n recommendation exec -it retrieval-pod -- python -c "
# restore_index('s3://rec-system/index-snapshots/2026-04-10/')
"
```

### Feature drift
```bash
# Check PSI for top features
# Grafana dashboard: Recommendation System → Feature Drift

# If PSI > 0.25 for critical features:
# 1. Investigate data pipeline
# 2. Check for catalog changes
# 3. Consider emergency retrain
```

---

# 🟢 P3 — Kafka Consumer Lag

## Symptoms
- Alert: `KafkaConsumerLagHigh` (>10 min lag)
- Real-time features are stale

## Triage
```bash
# Check consumer lag
kafka-consumer-groups --bootstrap-server $KAFKA_BOOTSTRAP \
  --group session-feature-consumer --describe

# Check Flink job status
kubectl -n recommendation get pods -l app=flink-session-features
```

## Fixes
```bash
# Scale Flink consumers
kubectl -n recommendation scale deployment/flink-session-features --replicas=32

# If Flink checkpoint failed, restart from last good checkpoint
kubectl -n recommendation delete pod -l app=flink-session-features
# (Flink auto-restarts from checkpoint)

# If lag is persistent, check for data schema changes
kubectl -n recommendation logs -l app=flink-session-features --tail=200 | grep ERROR
```

---

# 📋 Escalation Matrix

| Severity | Response Time | Escalation After | Contacts |
|----------|-------------- |-------------------|----------|
| P1 (Service Down) | 5 min | 15 min | ML Platform On-Call → Eng Manager |
| P2 (Degraded) | 15 min | 1 hour | ML Platform On-Call → Tech Lead |
| P3 (Warning) | 1 hour | 4 hours | ML Platform On-Call |
| P4 (Cosmetic) | Next business day | — | Queue |

---

# 📅 Regular Maintenance

## Weekly
- [ ] Review model retrain job logs
- [ ] Check feature drift dashboards
- [ ] Verify backup integrity
- [ ] Review error rate trends

## Monthly
- [ ] Chaos engineering test run (staging)
- [ ] Load test validation
- [ ] Certificate rotation check
- [ ] Cost review and optimization

## Quarterly
- [ ] Security audit / pen test
- [ ] Disaster recovery drill
- [ ] Capacity planning review
- [ ] On-call retrospective
