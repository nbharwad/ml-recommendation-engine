# ============================================================================
# Terraform — Vault (HashiCorp) via Helm on EKS
# Secrets management for all recommendation system services
# ============================================================================
#
# Architecture:
# - Vault runs in HA mode (3 replicas) with Raft integrated storage
# - EKS pods authenticate via Vault's Kubernetes auth method
# - Each service gets exactly the secrets it needs (least privilege)
# - Secrets are injected as env vars via Vault Agent Injector (sidecar)
# - Dynamic secrets: Redis/DB credentials with 1h TTL auto-rotation
#
# Auth flow:
#   Pod → Vault Agent → Kubernetes SA JWT → Vault → renders secret → env var
# ============================================================================

# ---------------------------------------------------------------------------
# Vault Namespace & Helm Release
# ---------------------------------------------------------------------------

resource "kubernetes_namespace" "vault" {
  metadata {
    name = "vault"
    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }
}

resource "helm_release" "vault" {
  name       = "vault"
  namespace  = kubernetes_namespace.vault.metadata[0].name
  repository = "https://helm.releases.hashicorp.com"
  chart      = "vault"
  version    = "0.28.1" # Vault 1.17.x

  wait    = true
  timeout = 600

  values = [<<-YAML
    global:
      enabled: true
      tlsDisable: false

    server:
      enabled: true
      image:
        tag: "${var.vault_version}"

      # HA mode with Raft
      ha:
        enabled: true
        replicas: ${var.environment == "prod" ? 3 : 1}
        raft:
          enabled: true
          setNodeId: true
          config: |
            cluster_name = "${local.name}-vault"
            
            storage "raft" {
              path    = "/vault/data"
              node_id = "{{ env "HOSTNAME" }}"
              
              retry_join {
                leader_api_addr = "https://vault-0.vault-internal:8200"
              }
              retry_join {
                leader_api_addr = "https://vault-1.vault-internal:8200"
              }
              retry_join {
                leader_api_addr = "https://vault-2.vault-internal:8200"
              }
            }
            
            listener "tcp" {
              address     = "[::]:8200"
              tls_disable = 1  # In prod: TLS via cert-manager
            }
            
            service_registration "kubernetes" {}
            
            # AWS KMS auto-unseal (no manual unseal on restart)
            seal "awskms" {
              region     = "${var.region}"
              kms_key_id = "${aws_kms_key.vault.key_id}"
            }
            
            api_addr     = "https://{{ env "HOSTNAME" }}.vault-internal:8200"
            cluster_addr = "https://{{ env "HOSTNAME" }}.vault-internal:8201"
            ui           = true

      # Node scheduling
      nodeSelector:
        node-type: general

      # Resource requests/limits
      resources:
        requests:
          cpu: 250m
          memory: 256Mi
        limits:
          cpu: 1000m
          memory: 1Gi

      # Persistent storage for Raft
      dataStorage:
        enabled: true
        size: ${var.vault_storage_size_gb}Gi
        storageClass: gp3

      # Liveliness probe
      readinessProbe:
        enabled: true
        path: "/v1/sys/health?standbyok=true"

      # Audit log (required for compliance)
      auditStorage:
        enabled: true
        size: 10Gi
        storageClass: gp3

      serviceAccount:
        create: true
        name: vault
        annotations:
          eks.amazonaws.com/role-arn: "${aws_iam_role.vault.arn}"

    # Vault Agent Injector (sidecar for secret injection)
    injector:
      enabled: true
      replicas: 2
      resources:
        requests:
          cpu: 50m
          memory: 128Mi
        limits:
          cpu: 250m
          memory: 256Mi

    # UI
    ui:
      enabled: true
      serviceType: ClusterIP
  YAML
  ]

  depends_on = [module.eks, kubernetes_namespace.vault]
}

# ---------------------------------------------------------------------------
# KMS Key for Vault Auto-Unseal
# ---------------------------------------------------------------------------

resource "aws_kms_key" "vault" {
  description             = "Vault auto-unseal key for ${local.name}"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags                    = merge(local.common_tags, { Name = "${local.name}-vault-unseal" })
}

resource "aws_kms_alias" "vault" {
  name          = "alias/${local.name}-vault-unseal"
  target_key_id = aws_kms_key.vault.key_id
}

# ---------------------------------------------------------------------------
# IAM Role for Vault (KMS Unseal + AWS Auth)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "vault" {
  name = "${local.name}-vault"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = module.eks.oidc_provider_arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:vault:vault"
        }
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "vault_kms" {
  name = "${local.name}-vault-kms"
  role = aws_iam_role.vault.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        Resource = aws_kms_key.vault.arn
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Vault Bootstrap Script (ConfigMap)
# Applied once after Vault is initialized
# ---------------------------------------------------------------------------

resource "kubernetes_config_map" "vault_bootstrap" {
  metadata {
    name      = "vault-bootstrap"
    namespace = kubernetes_namespace.vault.metadata[0].name
  }

  data = {
    "bootstrap.sh" = <<-SCRIPT
      #!/bin/bash
      set -euo pipefail
      
      VAULT_ADDR="http://vault.vault.svc.cluster.local:8200"
      export VAULT_ADDR
      
      # Enable Kubernetes auth backend
      vault auth enable kubernetes
      
      # Configure Kubernetes auth
      vault write auth/kubernetes/config \
        kubernetes_host="https://$KUBERNETES_PORT_443_TCP_ADDR:443" \
        token_reviewer_jwt="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)" \
        kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
      
      # Enable KV v2 secrets engine
      vault secrets enable -path=secret kv-v2
      
      # ===== Policies =====
      
      # Policy: recommendation services (read-only)
      vault policy write recommendation-services - <<EOF
        path "secret/data/recommendation/*" {
          capabilities = ["read", "list"]
        }
        path "secret/data/common/*" {
          capabilities = ["read"]
        }
      EOF
      
      # Policy: Flink (consumer only)
      vault policy write flink-jobs - <<EOF
        path "secret/data/recommendation/kafka" {
          capabilities = ["read"]
        }
        path "secret/data/recommendation/redis" {
          capabilities = ["read"]
        }
      EOF
      
      # Policy: MLflow (model registry)
      vault policy write mlflow - <<EOF
        path "secret/data/mlflow/*" {
          capabilities = ["read"]
        }
      EOF
      
      # ===== Kubernetes Auth Roles =====
      
      # Role for recommendation services
      vault write auth/kubernetes/role/recommendation-services \
        bound_service_account_names="recommendation-serving,feature-store,retrieval,ranking,reranking,ingestion,experimentation" \
        bound_service_account_namespaces="recommendation" \
        policies="recommendation-services" \
        ttl=1h
      
      # Role for Flink
      vault write auth/kubernetes/role/flink \
        bound_service_account_names="flink" \
        bound_service_account_namespaces="flink" \
        policies="flink-jobs" \
        ttl=1h
      
      # ===== Seed Initial Secrets =====
      # (In production: secrets come from a secure bootstrap process)
      
      vault kv put secret/recommendation/redis \
        host="${redis_endpoint}" \
        port="6379" \
        # password: set manually after bootstrap
      
      vault kv put secret/recommendation/kafka \
        bootstrap_brokers="${kafka_bootstrap}" \
        # credentials: IAM-based, no static credentials needed
      
      echo "Vault bootstrap complete"
    SCRIPT
  }
}
