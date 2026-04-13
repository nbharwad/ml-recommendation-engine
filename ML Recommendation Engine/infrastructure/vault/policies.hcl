# ============================================================================
# Vault Policies — Least Privilege Secret Access
# Each service gets exactly the secrets it needs, nothing more
# ============================================================================
#
# Auth method: Kubernetes (IRSA-based, no static credentials)
# Secret engine: KV v2 at path "secret/"
# Lease duration: 1 hour (auto-renewed by Vault Agent)
#
# Secret paths:
#   secret/recommendation/redis    → Redis connection (host, port, password)
#   secret/recommendation/kafka    → Kafka connection (brokers)
#   secret/recommendation/triton   → Triton server address
#   secret/recommendation/milvus   → Milvus connection
#   secret/mlflow/*                → MLflow tracking server
#   secret/common/slack            → Slack webhook (for alerts)
# ============================================================================

# ---------------------------------------------------------------------------
# Policy: recommendation-services (read-only to recommendation secrets)
# Applied to: serving, feature-store, retrieval, ranking, reranking, ingestion, experimentation
# ---------------------------------------------------------------------------
path "secret/data/recommendation/*" {
  capabilities = ["read"]
}

path "secret/metadata/recommendation/*" {
  capabilities = ["list"]
}

path "secret/data/common/slack" {
  capabilities = ["read"]
}

# Allow lease renewal (so Vault Agent can rotate before TTL expires)
path "auth/token/renew-self" {
  capabilities = ["update"]
}

# ---------------------------------------------------------------------------
# Policy: flink-jobs (read-only to streaming dependencies)
# Applied to: Flink task managers
# ---------------------------------------------------------------------------
path "secret/data/recommendation/kafka" {
  capabilities = ["read"]
}

path "secret/data/recommendation/redis" {
  capabilities = ["read"]
}

path "secret/data/recommendation/milvus" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

# ---------------------------------------------------------------------------
# Policy: ml-engineers (for model registry, MLflow, S3 access)
# Applied to: training pods, Airflow workers
# ---------------------------------------------------------------------------
path "secret/data/mlflow/*" {
  capabilities = ["read", "create", "update"]
}

path "secret/data/recommendation/s3" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

# ---------------------------------------------------------------------------
# Policy: ml-platform-admin (operators)
# Applied to: CI/CD pipeline, SRE team
# ---------------------------------------------------------------------------
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "auth/*" {
  capabilities = ["read", "list"]
}

path "sys/health" {
  capabilities = ["read"]
}
