#!/usr/bin/env bash
# ============================================================================
# Vault Secrets Bootstrap
# Populates all required secrets for the recommendation system
# ============================================================================

set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-https://vault.vault.svc.cluster.local:8200}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Populating Vault secrets..."

export VAULT_TOKEN="$1"

log "Redis credentials..."
vault kv put secret/recommendation/redis \
    host="redis.recommendation.svc.cluster.local" \
    port="6379" \
    password="${REDIS_PASSWORD}"

log "Kafka credentials..."
vault kv put secret/recommendation/kafka \
    bootstrap_servers="kafka.recommendation.svc.cluster.local:9092" \
    username="${KAFKA_USERNAME}" \
    password="${KAFKA_PASSWORD}"

log "Milvus credentials..."
vault kv put secret/recommendation/milvus \
    host="milvus.recommendation.svc.cluster.local" \
    port="19530"

log "Triton credentials..."
vault kv put secret/recommendation/triton \
    server="triton.recommendation.svc.cluster.local:8001"

log "Slack webhook..."
vault kv put secret/common/slack \
    webhook_url="${SLACK_WEBHOOK_URL}"

log "PagerDuty..."
vault kv put secret/pagerduty \
    routing_key="${PAGERDUTY_ROUTING_KEY}"

log "Vault secrets populated successfully"