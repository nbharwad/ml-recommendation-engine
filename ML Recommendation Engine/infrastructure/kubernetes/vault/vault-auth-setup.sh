#!/usr/bin/env bash
# ============================================================================
# Vault Authentication Setup — Kubernetes Auth Method
# Configures Vault to authenticate via Kubernetes Service Accounts
# ============================================================================

set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-https://vault.vault.svc.cluster.local:8200}"
KUBERNETES_CERT_PATH="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
VAULT_TOKEN_FILE="/vault/userconfig/token"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Configuring Vault Kubernetes Auth..."

export VAULT_TOKEN="$1"

vault auth enable kubernetes

vault write auth/kubernetes/config \
    kubernetes_host="https://$KUBERNETES_CERT_PATH" \
    kubernetes_ca_cert=@$KUBERNETES_CERT_PATH \
    token_reviewer_jwt="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)"

log "Creating Vault policies..."

vault policy write recommendation-services - <<EOF
path "secret/data/recommendation/*" {
  capabilities = ["read"]
}
path "secret/metadata/recommendation/*" {
  capabilities = ["list"]
}
path "secret/data/common/slack" {
  capabilities = ["read"]
}
path "auth/token/renew-self" {
  capabilities = ["update"]
}
EOF

vault policy write flink-jobs - <<EOF
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
EOF

log "Creating Vault roles..."

vault write auth/kubernetes/role/recommendation-serving \
    bound_service_account_names=recommendation-serving \
    bound_service_account_namespaces=recommendation \
    policies=recommendation-services \
    ttl=1h

vault write auth/kubernetes/role/flink-jobs \
    bound_service_account_names=flink-taskmanager \
    bound_service_account_namespaces=flink \
    policies=flink-jobs \
    ttl=1h

log "Vault Kubernetes Auth configured successfully"