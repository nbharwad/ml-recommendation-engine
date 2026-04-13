#!/usr/bin/env bash
# ============================================================================
# Bootstrap Script — Phase 0 Complete Infrastructure Setup
# Provisions everything needed before any service is deployed
#
# Prerequisites (must exist before running):
#   - AWS CLI configured with admin access
#   - Terraform >= 1.9 installed
#   - kubectl installed
#   - helm >= 3.15 installed
#   - jq, envsubst installed
#
# Usage:
#   export TF_VAR_environment=staging
#   export TF_VAR_aws_account_id=$(aws sts get-caller-identity --query Account --output text)
#   export TF_VAR_grafana_admin_password="your-secure-password"
#   bash scripts/bootstrap/01_bootstrap_infrastructure.sh
# ============================================================================

set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# Colors & logging
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
err()  { echo -e "${RED}❌ $*${NC}" >&2; exit 1; }
step() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════${NC}"; echo -e "${BOLD}$*${NC}"; echo -e "${BOLD}${BLUE}══════════════════════════════════════${NC}"; }

# ---------------------------------------------------------------------------
# Prerequisites validation
# ---------------------------------------------------------------------------

step "Step 0: Validate Prerequisites"

check_tool() {
  if ! command -v "$1" &>/dev/null; then
    err "Required tool not found: $1. Install it and retry."
  fi
  ok "$1 found: $(command -v "$1")"
}

check_tool terraform
check_tool kubectl
check_tool helm
check_tool aws
check_tool jq

# Validate env vars
: "${TF_VAR_environment:?TF_VAR_environment must be set (dev|staging|prod)}"
: "${TF_VAR_aws_account_id:?TF_VAR_aws_account_id must be set}"
: "${TF_VAR_grafana_admin_password:?TF_VAR_grafana_admin_password must be set}"

ENVIRONMENT="${TF_VAR_environment}"
REGION="${TF_VAR_region:-us-east-1}"

log "Environment: $ENVIRONMENT"
log "Region:      $REGION"
log "Account:     $TF_VAR_aws_account_id"

# ---------------------------------------------------------------------------
# Step 1: Terraform State Bootstrap
# ---------------------------------------------------------------------------

step "Step 1: Bootstrap Terraform State (S3 + DynamoDB)"

BUCKET_NAME="rec-system-terraform-state-${TF_VAR_aws_account_id}"
TABLE_NAME="terraform-locks"

# Create state bucket if it doesn't exist
if ! aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
  log "Creating Terraform state bucket: $BUCKET_NAME"
  aws s3api create-bucket \
    --bucket "$BUCKET_NAME" \
    --region "$REGION" \
    $([ "$REGION" != "us-east-1" ] && echo "--create-bucket-configuration LocationConstraint=$REGION" || echo "") \
    --no-cli-pager

  # Enable versioning (required for state safety)
  aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

  # Enable encryption
  aws s3api put-bucket-encryption \
    --bucket "$BUCKET_NAME" \
    --server-side-encryption-configuration '{
      "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
    }'

  # Block all public access
  aws s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

  ok "State bucket created: $BUCKET_NAME"
else
  ok "State bucket already exists: $BUCKET_NAME"
fi

# Create DynamoDB lock table if it doesn't exist
if ! aws dynamodb describe-table --table-name "$TABLE_NAME" --region "$REGION" 2>/dev/null; then
  log "Creating DynamoDB lock table: $TABLE_NAME"
  aws dynamodb create-table \
    --table-name "$TABLE_NAME" \
    --region "$REGION" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --no-cli-pager
  ok "DynamoDB lock table created"
else
  ok "DynamoDB lock table already exists"
fi

# ---------------------------------------------------------------------------
# Step 2: Terraform — Core Infrastructure
# ---------------------------------------------------------------------------

step "Step 2: Provision Core Infrastructure (EKS, VPC, Redis, MSK)"

cd infrastructure/terraform

# Update backend bucket name
sed -i "s/rec-system-terraform-state/$BUCKET_NAME/" main.tf

terraform init \
  -backend-config="bucket=$BUCKET_NAME" \
  -backend-config="region=$REGION" \
  -backend-config="dynamodb_table=$TABLE_NAME"

terraform validate
ok "Terraform configuration valid"

terraform plan \
  -var="environment=$ENVIRONMENT" \
  -var="region=$REGION" \
  -out=tfplan

log "Review the plan above. Applying in 10 seconds... (Ctrl+C to cancel)"
sleep 10

terraform apply tfplan
ok "Core infrastructure provisioned"

# Export outputs for use in subsequent steps
CLUSTER_NAME=$(terraform output -raw eks_cluster_name)
REDIS_ENDPOINT=$(terraform output -raw redis_primary_endpoint)
KAFKA_BROKERS=$(terraform output -raw kafka_bootstrap_brokers)

ok "Cluster: $CLUSTER_NAME"

cd - > /dev/null

# ---------------------------------------------------------------------------
# Step 3: Configure kubectl
# ---------------------------------------------------------------------------

step "Step 3: Configure kubectl"

aws eks update-kubeconfig \
  --region "$REGION" \
  --name "$CLUSTER_NAME"

kubectl cluster-info
ok "kubectl configured successfully"

# ---------------------------------------------------------------------------
# Step 4: Install Core Cluster Services
# ---------------------------------------------------------------------------

step "Step 4: Install Core Cluster Services"

# NVIDIA GPU device plugin
log "Installing NVIDIA GPU device plugin..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.0/deployments/static/nvidia-device-plugin.yml
ok "NVIDIA device plugin installed"

# Metrics Server (required for HPA)
log "Installing metrics-server..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
ok "Metrics server installed"

# AWS Load Balancer Controller (required for ALB ingress)
log "Installing AWS Load Balancer Controller..."
helm repo add eks https://aws.github.io/eks-charts --force-update
helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --set clusterName="$CLUSTER_NAME" \
  --set serviceAccount.create=true \
  --wait
ok "Load balancer controller installed"

# EBS CSI Driver (for persistent volumes)
log "Installing EBS CSI StorageClass..."
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF
ok "gp3 StorageClass installed"

# ---------------------------------------------------------------------------
# Step 5: Install Istio
# ---------------------------------------------------------------------------

step "Step 5: Install Istio Service Mesh"

# Download Istio CLI
ISTIO_VERSION="1.23.2"
if ! command -v istioctl &>/dev/null; then
  log "Downloading Istio $ISTIO_VERSION..."
  curl -sSL https://istio.io/downloadIstio | ISTIO_VERSION="$ISTIO_VERSION" sh -
  export PATH="$PWD/istio-${ISTIO_VERSION}/bin:$PATH"
fi

# Install Istio with production profile
istioctl install --set profile=default \
  --set values.global.meshID="rec-system-mesh" \
  --set values.global.multiCluster.clusterName="$CLUSTER_NAME" \
  --set values.gateways.istio-ingressgateway.autoscaleMin=3 \
  --set values.gateways.istio-ingressgateway.autoscaleMax=10 \
  -y

# Apply service mesh configuration
kubectl apply -f infrastructure/kubernetes/istio/service-mesh.yaml
ok "Istio installed and configured"

# ---------------------------------------------------------------------------
# Step 6: Install Monitoring Stack (Prometheus + Grafana)
# ---------------------------------------------------------------------------

step "Step 6: Install Monitoring Stack"

kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
helm repo add grafana https://grafana.github.io/helm-charts --force-update
helm repo update

# kube-prometheus-stack (Prometheus + Grafana + Alertmanager + node-exporter)
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --version "65.1.1" \
  --set prometheus.prometheusSpec.retention="15d" \
  --set prometheus.prometheusSpec.retentionSize="50GB" \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=gp3 \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage="100Gi" \
  --set grafana.adminPassword="$TF_VAR_grafana_admin_password" \
  --set grafana.sidecar.dashboards.enabled=true \
  --set grafana.sidecar.dashboards.searchNamespace=ALL \
  --wait --timeout=10m

# Apply custom alert rules
kubectl apply -f monitoring/alerts/recommendation-alerts.yml

# Import Grafana dashboard
kubectl create configmap recommendation-dashboard \
  --from-file=monitoring/dashboards/recommendation-overview.json \
  --namespace monitoring \
  --dry-run=client -o yaml | kubectl apply -f -

ok "Monitoring stack installed"

# ---------------------------------------------------------------------------
# Step 7: Install ELK Stack
# ---------------------------------------------------------------------------

step "Step 7: Install ELK Logging Stack"

kubectl apply -f infrastructure/kubernetes/logging/elk-stack.yaml

# Wait for Elasticsearch to be ready
log "Waiting for Elasticsearch (may take ~3 minutes)..."
kubectl rollout status statefulset/elasticsearch -n logging --timeout=300s
ok "Elasticsearch ready"

# ---------------------------------------------------------------------------
# Step 8: Install Jaeger Tracing
# ---------------------------------------------------------------------------

step "Step 8: Install Jaeger Distributed Tracing"

kubectl apply -f infrastructure/kubernetes/tracing/jaeger.yaml

log "Waiting for Jaeger operator..."
kubectl rollout status deployment/jaeger-operator -n tracing --timeout=120s
ok "Jaeger installed"

# ---------------------------------------------------------------------------
# Step 9: Install Vault
# ---------------------------------------------------------------------------

step "Step 9: Install HashiCorp Vault"

kubectl create namespace vault --dry-run=client -o yaml | kubectl apply -f -

helm repo add hashicorp https://helm.releases.hashicorp.com --force-update

helm upgrade --install vault hashicorp/vault \
  --namespace vault \
  --version "0.28.1" \
  --set server.ha.enabled="$([ "$ENVIRONMENT" = 'prod' ] && echo 'true' || echo 'false')" \
  --set server.ha.replicas=3 \
  --wait --timeout=5m

log "Vault installed. Next steps:"
log "1. Initialize Vault:  kubectl exec -n vault vault-0 -- vault operator init"
log "2. Save the unseal keys and root token securely"
log "3. Run bootstrap: kubectl exec -n vault vault-0 -- sh /vault-bootstrap/bootstrap.sh"
warn "Vault is NOT automatically initialized — this requires a human operator"

# ---------------------------------------------------------------------------
# Step 10: Install ArgoCD
# ---------------------------------------------------------------------------

step "Step 10: Install ArgoCD GitOps"

kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

log "Waiting for ArgoCD..."
kubectl rollout status deployment/argocd-server -n argocd --timeout=180s

# Retrieve initial admin password
ARGOCD_INITIAL_PW=$(kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d)

log "ArgoCD initial password: $ARGOCD_INITIAL_PW"
warn "Change this password immediately after first login"

# Apply ArgoCD applications (pointing to this repo)
kubectl apply -f infrastructure/argocd/application.yaml

ok "ArgoCD installed and configured"

# ---------------------------------------------------------------------------
# Step 11: Create Kafka Topics
# ---------------------------------------------------------------------------

step "Step 11: Create Kafka Topics"

log "Bootstrap brokers: $KAFKA_BROKERS"
log "Creating topics via a temporary pod..."

kubectl run kafka-setup \
  --image=confluentinc/cp-kafka:7.7.0 \
  --restart=Never \
  --namespace=default \
  --env="KAFKA_BROKERS=$KAFKA_BROKERS" \
  --command -- bash -c "
    TOPICS=(
      'user-events:64:3'
      'user-events-dlq:16:3'
      'session-features:32:3'
      'item-stats:32:3'
      'experiment-exposures:16:3'
      'model-predictions:32:3'
      'item-onboarding:8:3'
    )
    
    for t in \${TOPICS[@]}; do
      IFS=':' read -r topic partitions replication <<< \$t
      kafka-topics --create \
        --if-not-exists \
        --bootstrap-server \$KAFKA_BROKERS \
        --topic \$topic \
        --partitions \$partitions \
        --replication-factor \$replication \
        --command-config /tmp/client.properties
      echo \"Created: \$topic (\$partitions partitions)\"
    done
  "

kubectl wait --for=condition=complete pod/kafka-setup --timeout=120s
kubectl delete pod kafka-setup --ignore-not-found
ok "Kafka topics created"

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------

step "🎉 Phase 0 Infrastructure Bootstrap Complete"

echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│              INFRASTRUCTURE SUMMARY                             │"
echo "├─────────────────────────────────────────────────────────────────┤"
printf "│ %-30s %-34s │\n" "EKS Cluster:"      "$CLUSTER_NAME"
printf "│ %-30s %-34s │\n" "Redis Endpoint:"   "$REDIS_ENDPOINT"
printf "│ %-30s %-34s │\n" "Environment:"      "$ENVIRONMENT"
printf "│ %-30s %-34s │\n" "Region:"           "$REGION"
echo "├─────────────────────────────────────────────────────────────────┤"
echo "│  Services Installed:                                            │"
echo "│    ✅ EKS Cluster (3 node groups: general, GPU, memory)         │"
echo "│    ✅ Istio Service Mesh (mTLS, traffic management)             │"
echo "│    ✅ Prometheus + Grafana (monitoring)                         │"
echo "│    ✅ ELK Stack (centralized logging)                           │"
echo "│    ✅ Jaeger + OTel Collector (distributed tracing)             │"
echo "│    ✅ HashiCorp Vault (secrets management)                      │"
echo "│    ✅ ArgoCD (GitOps deployment)                                │"
echo "│    ✅ MSK Kafka + 7 topics                                      │"
echo "│    ✅ ElastiCache Redis (feature store)                         │"
echo "├─────────────────────────────────────────────────────────────────┤"
echo "│  Next Steps:                                                    │"
echo "│    1. Initialize Vault (manual — requires human operator)       │"
echo "│    2. Change ArgoCD admin password                              │"
echo "│    3. Configure DNS records for api.recommendation.internal     │"
echo "│    4. Install TLS cert via cert-manager                         │"
echo "│    5. Proceed to Phase 1: Deploy recommendation services        │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
echo "Access UIs (after port-forwarding):"
echo "  kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring"
echo "  kubectl port-forward svc/recommendation-tracing-query 16686:16686 -n tracing"
echo "  kubectl port-forward svc/argocd-server 8080:443 -n argocd"
echo "  kubectl port-forward svc/kibana 5601:5601 -n logging"
