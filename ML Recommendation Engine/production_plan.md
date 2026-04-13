# Production Plan: ML Recommendation Engine

## Context

This plan bridges the gap between the current ~49% implementation state and a fully enterprise-grade production system. The core serving pipeline is well-built (FastAPI orchestrator, circuit breakers, 3-tier feature store, DLRM/XGBoost/Two-Tower models, Triton serving, MMR re-ranking). However, a comprehensive gap analysis against `implementation.md` and `gap_analysis.md` identified **36 specific gaps** spanning security blockers, missing streaming infrastructure, incomplete GitOps wiring, absent operational procedures, and compliance deficits.

The system targets 10M+ products, 1M+ DAU, 50K QPS, p99 < 75ms. At that scale, every gap is a potential incident.

**Plan structure:** 5 sequential phases over 8-10 weeks. Phases 1-3 are strictly gated. Phases 4-5 run partly in parallel after Phase 3 completes.

**Team assumption:** 2-3 engineers. ~70-100 hours total implementation.

---

## Phase 1: Security Blockers (Weeks 1-2)

**Gate:** No phase 2 begins until all 6 items below pass their acceptance criteria.

### 1.1 Remove Hardcoded Credentials
**Files:** `docker-compose.yml`, new `.env.example`, `.gitignore`

MinIO uses `MINIO_ACCESS_KEY: minioadmin` (lines 162-163). Grafana uses `GF_SECURITY_ADMIN_PASSWORD: admin` (line 203).

- Replace all hardcoded values with `${VAR}` environment variable references
- Create `.env.example` with documented placeholder values (committed)
- Create `.env` with real dev secrets (gitignored — add to `.gitignore`)
- Add `detect-secrets` pre-commit hook and CI step in the lint stage
- `make dev` should fail with a clear error if `.env` is missing

**Acceptance:** `git grep "minioadmin"` returns zero results. `docker compose up` fails without `.env`.

---

### 1.2 Fix CORS Wildcard
**File:** `services/serving/main.py` (line 917-921)

Current: `allow_origins=["*"]` — this is a browser CORS spec violation when `allow_credentials=True`.

Replace with:
```python
ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "https://www.example.com").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "X-Request-ID"])
```
Add `CORS_ALLOWED_ORIGINS` to `serving-deployment.yaml` env block. Local dev sets to `http://localhost:3000`.

**Acceptance:** `curl -H "Origin: https://malicious.com"` returns no CORS header. Allowlisted origin succeeds.

---

### 1.3 Add Startup Config Validation
**Files:** Create `services/serving/config.py`; update `services/serving/main.py` lifespan

Services currently start with missing env vars and fail at request time. Use `pydantic_settings.BaseSettings`:
```python
class AppSettings(BaseSettings):
    feature_service_host: str
    jwt_issuer: str
    jwt_jwks_uri: str
    cors_allowed_origins: str
    pseudonymization_salt: str  # required for GDPR
    
    @model_validator(mode="after")
    def validate_critical(self):
        if len(self.pseudonymization_salt) < 32:
            raise ValueError("PSEUDONYMIZATION_SALT must be >= 32 chars")
        return self
```
Instantiate as first action in the lifespan function. A `ValidationError` exits with code 1 and lists all missing variables.

**Acceptance:** Missing `JWT_ISSUER` causes process exit with human-readable error. All required vars documented in `.env.example`.

---

### 1.4 Fix Kafka Authentication (docker-compose only — MSK is correct)
**File:** `docker-compose.yml`

Production MSK uses SASL/IAM correctly. Local dev uses `PLAINTEXT` for all listeners, exposing all user event data.

- Switch `docker-compose.yml` to `SASL_PLAINTEXT` for the HOST listener
- Add `kafka_server_jaas.conf` to `infrastructure/docker/kafka/` reading credentials from `.env`
- Add a top-of-file comment block: `# ⚠️ LOCAL DEVELOPMENT ONLY — Production uses AWS MSK with SASL/IAM`

**Acceptance:** `kafka-console-consumer` without credentials is rejected. Consumer with credentials succeeds.

---

### 1.5 Add Redis Authentication
**File:** `docker-compose.yml`; `services/feature-store/feature_service.py`

Redis runs without a password (line 92-93). Production ElastiCache has transit encryption; local dev doesn't match.

- Add `--requirepass ${REDIS_PASSWORD}` to Redis command in docker-compose
- Add `REDIS_PASSWORD` env var to feature-store and serving service blocks
- Update `feature_service.py` to read `REDIS_PASSWORD` and pass to Redis client constructor
- Ensure `REDIS_PASSWORD` is never logged

**Acceptance:** `redis-cli ping` returns `NOAUTH`. Feature store connects with credentials.

---

### 1.6 Complete JWT Signature Verification
**File:** `services/serving/auth.py` (lines 208-237)

Current `validate_jwt` decodes the JWT payload using base64 **without signature verification**. Any crafted JWT with `"role": "super_admin"` is accepted.

Replace with JWKS-based validation using `python-jose` (add to `requirements.txt`):
- Fetch JWKS from `JWT_JWKS_URI` env var with 1-hour TTL cache
- Validate signature (RS256), expiry, issuer, audience
- Support key rotation: refresh cache on 401 from downstream

**Acceptance:** Tampered payload → 401. Expired token → 401. Unknown issuer → 401. Unit tests for all 4 rejection cases.

### Phase 1 Verification
```bash
detect-secrets scan --baseline .secrets.baseline
pytest tests/unit/test_auth.py tests/unit/test_config_validation.py -v
```

---

## Phase 2: Vault Integration and Secret Lifecycle (Weeks 2-3)

Phase 1 removed hardcoded secrets. Phase 2 establishes proper secret lifecycle.

### 2.1 Deploy Vault with Helm and Kubernetes Auth
**Files:** Create `infrastructure/kubernetes/vault/vault-helm-values.yaml`, `infrastructure/kubernetes/vault/vault-auth-setup.sh`

- Deploy Vault in HA mode (3 replicas with Raft storage) via Helm
- Enable Kubernetes auth method pointing to EKS cluster service account JWT issuer
- Apply `infrastructure/vault/policies.hcl` (already exists with correct least-privilege policies)
- Create Vault roles binding service accounts to policies (e.g., `recommendation-serving` SA → `recommendation-services` policy)

**Acceptance:** `vault status` shows `Initialized: true, Sealed: false`. Test pod with correct SA can read `secret/data/recommendation/redis`. Pod without SA is denied.

---

### 2.2 Wire Vault Agent Sidecar Injector
**Files:** `infrastructure/kubernetes/serving-deployment.yaml`, `infrastructure/kubernetes/services-deployment.yaml`

Add Vault Agent annotations to pod templates:
```yaml
annotations:
  vault.hashicorp.com/agent-inject: "true"
  vault.hashicorp.com/role: "recommendation-serving"
  vault.hashicorp.com/agent-inject-secret-redis: "secret/data/recommendation/redis"
```
Update `main.py` lifespan to read `/vault/secrets/redis` file and inject into `os.environ`.

**Acceptance:** Serving pod has `vault-agent` sidecar. `/vault/secrets/redis` is mounted. Vault audit log shows every secret read.

---

### 2.3 Populate Vault and Wire Alerting
**File:** Create `scripts/bootstrap/03_vault_secrets.sh`

Populate all paths referenced in `policies.hcl` including:
- `secret/recommendation/redis`, `kafka`, `milvus`, `triton`
- `secret/common/slack` (Slack webhook — alertmanager currently has `${SLACK_WEBHOOK_*}` as broken env vars)
- `secret/pagerduty` (routing key)

This fixes the broken alert routing to humans (Gap #22 in gap analysis).

**Acceptance:** `vault kv list secret/recommendation/` shows all expected paths. Test alert delivered to Slack #ml-platform-alerts and PagerDuty.

### Phase 2 Verification
```bash
kubectl -n vault exec vault-0 -- vault status
kubectl -n recommendation exec -it <serving-pod> -- ls /vault/secrets/
amtool alert add severity=critical alertname=TestAlert  # verify Slack delivery
```

---

## Phase 3: Streaming Layer Completion (Weeks 3-5)

The streaming layer is the **critical weakness** — only 20% implemented. Without real-time feature freshness, user embeddings are hours-old (batch-only) and new items are invisible for up to 24 hours.

### 3.1 Complete User Embedding Job
**File:** `streaming/user_embeddings/user_embedding_job.py`

The `build_pipeline()` method is empty (line 89-94). `_load_model()` returns `None` (line 46).

- Implement `_load_model()` with ONNX Runtime: `ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])`
- Implement `encode_user_history()` to tokenize item sequences and run real inference (replace random stub)
- Implement `build_pipeline()` with full PyFlink graph:
  - Kafka source → filter → `process_session_update` → Redis sink + Milvus sink
  - `env.enable_checkpointing(60_000)` with S3 backend
  - Parallelism: 16

**Acceptance:** Synthetic session event → embedding in Redis within 30s. Milvus shows updated vector. Flink checkpoint written to S3. Job recovers from TaskManager kill within 90s.

---

### 3.2 Complete Item Statistics Pipeline
**File:** `streaming/item_stats/item_stats_job.py`

`build_pipeline()` is empty (line 139-152). `ItemStatsState` and `ItemStatsProcessor` are well-built — just need wiring.

- Sliding window: 1h/5m and 24h/15m variants
- WatermarkStrategy with 30s out-of-orderness
- Late events → dead letter topic (`item-stats-late-events`) via side output
- Redis sink with `is:{item_id}` key prefix (matching feature store convention)

**Acceptance:** `VIEW` events for `test-item-001` produce `is:test-item-001` in Redis within 5 minutes. Late events visible in dead letter topic.

---

### 3.3 Implement Item Onboarding and Event Enrichment Jobs
**Files:** Create `streaming/item_onboarding/onboarding_job.py`; create `streaming/enrichment/enrichment_job.py`

**Onboarding job:**
- Source: `item-catalog-updates` topic
- Compute initial item embedding using item tower of Two-Tower model
- Write to Milvus (ANN searchable) + Redis feature store with `if:` prefix
- Initialize Thompson Sampling priors in cold-start module (`services/serving/cold_start.py`)
- Target: new item searchable in ANN within 60 seconds

**Enrichment job:**
- Source: `raw-user-events`; output: `enriched-user-events`
- Enrich with: user segment (Redis lookup), item category (Redis lookup), session context
- This enriched topic feeds Item Stats and Session Features jobs
- Also add late event side output to `session_features_job.py`

**Acceptance:** Catalog event → item in Milvus within 60s. Enriched events have `user_segment` + `item_category` fields.

---

### 3.4 Milvus Production K8s Deployment
**Files:** `infrastructure/kubernetes/milvus/milvus-ha-values.yaml` (exists); create `infrastructure/argocd/apps/staging/milvus.yaml`

The Helm values file is complete but no ArgoCD Application exists — Milvus is only in docker-compose. Create ArgoCD Application manifests for staging and prod. Wire into root app-of-apps structure.

**Acceptance:** `kubectl get pods -l app.kubernetes.io/name=milvus` shows all components Running. ArgoCD shows Synced. 10M-vector query < 10ms p99.

### Phase 3 Verification
```bash
pytest tests/integration/test_streaming_e2e.py -v
kubectl -n flink get jobs
redis-cli get "is:item_0000001"   # verify item stats
redis-cli get "ue:user_0000001"   # verify user embedding
kafka-console-consumer --topic item-stats-late-events --max-messages 10
```

---

## Phase 4: GitOps, Observability, and Scale Hardening (Weeks 5-7)

### 4.1 Complete ArgoCD App-of-Apps Structure
**Files:** `infrastructure/argocd/application.yaml` (exists, incomplete); create `infrastructure/argocd/apps/staging/` and `infrastructure/argocd/apps/prod/`

The root ArgoCD app references `apps/${ENVIRONMENT}` but that directory doesn't exist. Currently, staging deploys via `kubectl apply -f` (not GitOps).

Create per-service Application manifests:
```
infrastructure/argocd/apps/
  staging/   → automated sync (recommendation-services, milvus, monitoring, flink, vault)
  prod/      → manual sync only (ArgoCD creates audit log entry)
```
Add ArgoCD Image Updater so `staging-*` ECR tag pushes trigger automatic staging sync.

**Acceptance:** All 10 applications visible in ArgoCD. PR merge → staging sync within 2 minutes. Prod requires manual sync button. ArgoCD fires Slack notification on sync events.

---

### 4.2 Deploy ELK Stack and Wire Structured Logging
**Files:** `infrastructure/kubernetes/logging/elk-stack.yaml` (StatefulSet exists); create `infrastructure/kubernetes/logging/filebeat-daemonset.yaml` and `logstash-pipeline.conf`

ELK StatefulSet exists but Filebeat (log collector) is missing. Currently, `kubectl logs` across 30+ pods is the only log access — unusable during incidents.

- Create Filebeat DaemonSet mounting `/var/log/containers`
- Create Logstash pipeline parsing structlog JSON, indexing `request_id`, `user_id`, `latency_ms`, `experiment_id`
- Route `level=ERROR` logs to `recommendation-errors` index
- Add ELK to ArgoCD

**Acceptance:** Kibana shows logs from all pods. `request_id` search traces a request across all services. Pod deletion does not lose logs.

---

### 4.3 Wire Request Coalescing into main.py
**Files:** `services/serving/coalescing.py` (stub); `services/serving/main.py`

`RequestCoalescer` is fully implemented. `CoalescingMiddleware.__call__` is empty (lines 350-355). `math` is imported after its use (line 163 vs line 395 — bug).

- Move `import math` to top of `coalescing.py`
- Complete `CoalescingMiddleware.__call__` to intercept POST `/v1/recommendations`, extract `user_id`, and coalesce concurrent identical requests
- Add middleware to `main.py` after CORS middleware: `CoalescingConfig(window_ms=5, max_waiters=100)`

**Acceptance:** 1000 concurrent requests for same `user_id` → < 10% actual ranking calls. `requests_coalesced_total` counter incrementing in Prometheus. No p99 regression.

---

### 4.4 Add pyproject.toml, Lock Files, and Coverage Gate
**Files:** Create `pyproject.toml` at repo root

- Use `uv` workspace with all service packages as members
- Add `--cov-fail-under=80` to pytest addopts (fixes missing coverage threshold)
- Run `uv lock` to generate `uv.lock`, commit it
- Update CI to use `uv run pytest` and enforce lockfile freshness

**Acceptance:** `uv sync` is deterministic. CI fails below 80% coverage. `uv.lock` freshness enforced in CI.

---

### 4.5 Add SBOM Generation and Supply Chain Security
**File:** Create/complete `.github/workflows/security-scan.yml`

The file exists but appears empty. Add:
- `anchore/sbom-action` to generate SPDX SBOM per image build
- `aquasecurity/trivy-action` with `exit-code: 1` on CRITICAL/HIGH CVEs
- `sigstore/cosign-installer` to sign images with Cosign
- Upload SBOM as build artifact and Trivy results to GitHub Security tab

**Acceptance:** Every image build produces an SBOM. Critical CVE fails PR. Images are Cosign-signed and verifiable.

### Phase 4 Verification
```bash
kubectl -n argocd get applications -o jsonpath='{range .items[*]}{.metadata.name}: {.status.health.status}\n{end}'
curl -s http://elasticsearch.logging.svc:9200/_cluster/health | jq '.status'
curl -s http://recommendation-serving:8080/metrics | grep coalescing
uv run pytest --cov-fail-under=80
```

---

## Phase 5: GDPR, Compliance, and Operational Procedures (Weeks 7-10)

### 5.1 Complete GDPR Deletion with Audit Trail
**Files:** `ml/pipelines/data_purge.py` (stub exists); create `services/serving/gdpr_api.py`

`execute_purge` method exists but has no implementation. The current code has metrics and config but no actual deletion logic, no confirmation, and no audit trail — a regulator cannot verify compliance.

Complete `execute_purge`:
1. Redis purge: delete `uf:{user_id}`, `ue:{user_id}`, all session keys
2. Milvus purge: delete embedding vector
3. Kafka: publish tombstone to `user-deletions` topic (consumed by Flink to drop in-flight events)
4. S3 training data: submit async Spark job to remove user rows from Parquet datasets
5. Audit record: write to append-only Elasticsearch index (no delete policy) or DynamoDB with `prevent_destroy = true`

Add REST endpoint: `DELETE /v1/users/{user_id}/data` — requires `manage_users` permission.

**Acceptance:** Deletion request → all Redis keys gone, Milvus embedding deleted, audit record written. PURGE_PROGRESS metric reaches 100%. Purge completes within 72-hour SLA (Prometheus alert fires if it doesn't).

---

### 5.2 Complete Missing Runbooks
**Files:** `monitoring/runbooks/` — add 4 new runbooks

Current runbooks cover P1 (service down) and P2 (latency). Missing:

| Runbook | Content |
|---------|---------|
| `model-rollback.md` | How to roll back a bad model: check MLflow registry, `kubectl rollout undo`, verify XGBoost fallback |
| `flink-recovery.md` | How to restore Flink from checkpoint: S3 path, `flink savepoint`, TaskManager restart |
| `scaling-procedure.md` | Manual HPA override, adding node group capacity, Redis cluster rebalancing |
| `data-quality-incident.md` | PSI breach response, disabling a feature, emergency data freeze |

Each runbook must include: symptoms, 2-minute triage steps, common causes with fixes, kubectl/AWS CLI commands, escalation path.

**Acceptance:** On-call engineer can resolve a model rollback within 30 minutes using only the runbook. All commands tested in staging.

---

### 5.3 API Documentation and OpenAPI Spec
**Files:** Create `docs/api/openapi.yaml`; update `services/serving/main.py`

FastAPI auto-generates OpenAPI but it's not exported or published. 

- Add `response_model` to all endpoints in `main.py` (some missing)
- Add example requests/responses in docstrings
- Export spec: `python -c "import json; from main import app; print(json.dumps(app.openapi()))" > docs/api/openapi.yaml`
- Add to CI: if spec changes without a version bump in `info.version`, the build fails (prevents silent breaking changes)
- Document rate limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`) and error code catalog

**Acceptance:** `docs/api/openapi.yaml` is committed and up to date. Error codes cataloged. CI fails on spec drift.

---

### 5.4 On-Call Rotation and PagerDuty Wiring
**Files:** `monitoring/alerts/recommendation-alerts.yml`; Vault secret `secret/pagerduty`

After Phase 2 populates the PagerDuty routing key in Vault:
- Update `alertmanager-config.yaml` to route P1 alerts to PagerDuty (`pagerduty_configs` receiver)
- P2 alerts go to Slack `#ml-platform-alerts`
- Document on-call rotation in `docs/oncall.md` (who is primary, secondary; escalation to manager)
- Set up monthly fire drills: simulate a P1 (kill the serving deployment) and measure MTTD/MTTR

**Acceptance:** `amtool alert add severity=critical` triggers PagerDuty within 2 minutes. On-call doc complete. First fire drill scheduled.

---

### 5.5 Multi-Region Terraform (Post-Launch Optimization)
**Files:** `infrastructure/terraform/multi-region.tf` (referenced but empty)

Deploy to 3 AWS regions with GeoDNS (Route53 latency routing). Requires:
- Cross-region Kafka Mirror Maker 2 (replicate `enriched-user-events` topic)
- Redis Global Datastore (ElastiCache cross-region replication)
- Milvus: independent per-region cluster seeded from S3 snapshot
- ArgoCD: multi-cluster registration

**Note:** This is a post-launch optimization. Do after the system is stable in single-region. Treat as a separate project (4-6 weeks).

### Phase 5 Verification
```bash
# GDPR deletion test
curl -X DELETE http://localhost:8080/v1/users/test_user_001/data -H "Authorization: Bearer $ADMIN_TOKEN"
redis-cli keys "uf:test_user_001*"  # should return empty
# Check audit record
curl "http://elasticsearch:9200/gdpr-audit-*/_search?q=user_id:test_user_001"

# Verify alert routing
amtool alert add severity=critical alertname=RecommendationServiceDown
# Verify PagerDuty incident created within 2 min
```

---

## Complete Gap Summary Table

| # | Gap | Phase | Files | Priority |
|---|-----|-------|-------|----------|
| 1 | Hardcoded MinIO/Grafana credentials | 1.1 | `docker-compose.yml` | CRITICAL |
| 2 | CORS wildcard `allow_origins=["*"]` | 1.2 | `services/serving/main.py:917` | CRITICAL |
| 3 | No startup config validation | 1.3 | Create `services/serving/config.py` | CRITICAL |
| 4 | Kafka PLAINTEXT in docker-compose | 1.4 | `docker-compose.yml` | CRITICAL |
| 5 | Redis unauthenticated | 1.5 | `docker-compose.yml`, `feature_service.py` | CRITICAL |
| 6 | JWT no signature verification | 1.6 | `services/serving/auth.py:208-237` | CRITICAL |
| 7 | Vault deployed but never wired | 2.1-2.3 | `infrastructure/kubernetes/vault/` | HIGH |
| 8 | Alert routing broken (Slack/PD) | 2.3 | `scripts/bootstrap/03_vault_secrets.sh` | HIGH |
| 9 | User embedding job empty | 3.1 | `streaming/user_embeddings/user_embedding_job.py:89` | HIGH |
| 10 | Item stats pipeline empty | 3.2 | `streaming/item_stats/item_stats_job.py:139` | HIGH |
| 11 | Item onboarding missing | 3.3 | Create `streaming/item_onboarding/onboarding_job.py` | HIGH |
| 12 | Event enrichment missing | 3.3 | Create `streaming/enrichment/enrichment_job.py` | HIGH |
| 13 | Late event handling missing | 3.3 | `streaming/session_features/session_features_job.py` | HIGH |
| 14 | Milvus no K8s/ArgoCD manifest | 3.4 | Create `infrastructure/argocd/apps/staging/milvus.yaml` | HIGH |
| 15 | ArgoCD app-of-apps incomplete | 4.1 | Create `infrastructure/argocd/apps/staging/` and `prod/` | HIGH |
| 16 | ELK Filebeat missing | 4.2 | Create `infrastructure/kubernetes/logging/filebeat-daemonset.yaml` | HIGH |
| 17 | Coalescing middleware empty | 4.3 | `services/serving/coalescing.py:350-355`, `main.py` | HIGH |
| 18 | No pyproject.toml or lock files | 4.4 | Create root `pyproject.toml` | MEDIUM |
| 19 | No coverage threshold | 4.4 | `pyproject.toml` pytest addopts | MEDIUM |
| 20 | Security scan workflow empty | 4.5 | `.github/workflows/security-scan.yml` | MEDIUM |
| 21 | No SBOM generation | 4.5 | `.github/workflows/security-scan.yml` | MEDIUM |
| 22 | GDPR delete logic not implemented | 5.1 | `ml/pipelines/data_purge.py` | HIGH |
| 23 | No GDPR audit trail | 5.1 | Create `services/serving/gdpr_api.py` | HIGH |
| 24 | Model rollback runbook missing | 5.2 | Create `monitoring/runbooks/model-rollback.md` | MEDIUM |
| 25 | Flink recovery runbook missing | 5.2 | Create `monitoring/runbooks/flink-recovery.md` | MEDIUM |
| 26 | Scaling procedure runbook missing | 5.2 | Create `monitoring/runbooks/scaling-procedure.md` | MEDIUM |
| 27 | Data quality incident runbook | 5.2 | Create `monitoring/runbooks/data-quality-incident.md` | MEDIUM |
| 28 | No published OpenAPI spec | 5.3 | Create `docs/api/openapi.yaml` | MEDIUM |
| 29 | PagerDuty not wired | 5.4 | `alertmanager-config.yaml`, Vault secret | HIGH |
| 30 | No on-call rotation documented | 5.4 | Create `docs/oncall.md` | MEDIUM |
| 31 | Multi-region not built | Post-launch | `infrastructure/terraform/multi-region.tf` | LOW |
| 32 | Hedged requests not fully activated | Optimization | `services/retrieval/retrieval_service.py` | LOW |
| 33 | Thompson Sampling cold start stub | Optimization | `services/serving/cold_start.py` | LOW |
| 34 | No Renovate/Dependabot | Optimization | Create `.github/renovate.json` | LOW |
| 35 | Combined chaos + load tests | Optimization | `tests/chaos/` | LOW |
| 36 | Data catalog / lineage missing | Post-launch | Apache Atlas or Datahub | LOW |

---

## Production Readiness Verification

The existing `docs/production-readiness-review.md` template already has the checklist structure. After all 5 phases, verify:

### Functional
```bash
# Full pipeline under load
make load-test  # 50K QPS, p99 < 75ms
# Chaos resilience
pytest tests/chaos/test_chaos.py -v
# All Flink jobs running
kubectl -n flink get jobs
# End-to-end streaming
pytest tests/integration/test_streaming_e2e.py -v
```

### Security
```bash
detect-secrets scan --baseline .secrets.baseline
trivy image $ECR_REGISTRY/recommendation-serving:latest --exit-code 1 --severity CRITICAL
cosign verify $ECR_REGISTRY/recommendation-serving:latest
pytest tests/unit/test_auth.py -v
```

### Compliance
```bash
# GDPR deletion test
curl -X DELETE localhost:8080/v1/users/test_user_001/data -H "Authorization: Bearer $ADMIN_TOKEN"
# Verify audit record exists in Elasticsearch
# Coverage gate
uv run pytest --cov-fail-under=80
```

### Observability
- [ ] Grafana dashboard loads at `localhost:3000`
- [ ] Kibana shows logs from all 7 service types
- [ ] Jaeger traces show request_id across all service hops
- [ ] Test alert reaches Slack #ml-platform-alerts AND PagerDuty within 2 minutes
- [ ] All 11 SLO alert rules active in Prometheus

### Operations
- [ ] On-call rotation documented and staffed
- [ ] All 4 new runbooks tested in staging with a stopwatch
- [ ] ArgoCD shows all applications Synced/Healthy
- [ ] Vault unsealed and audit log flowing
