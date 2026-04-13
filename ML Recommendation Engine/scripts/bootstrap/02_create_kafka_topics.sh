#!/usr/bin/env bash
# ============================================================================
# Kafka Topic Management Script
# Creates, describes, and configures all topics for the recommendation system
#
# Usage:
#   bash scripts/bootstrap/02_create_kafka_topics.sh create   [create all topics]
#   bash scripts/bootstrap/02_create_kafka_topics.sh describe [describe all topics]
#   bash scripts/bootstrap/02_create_kafka_topics.sh verify   [verify config matches]
# ============================================================================

set -euo pipefail

: "${KAFKA_BOOTSTRAP_SERVERS:?Set KAFKA_BOOTSTRAP_SERVERS env var}"
: "${ENVIRONMENT:?Set ENVIRONMENT env var (dev|staging|prod)}"

# Determine replication factor based on environment
if [ "$ENVIRONMENT" = "prod" ]; then
  REPLICATION_FACTOR=3
else
  REPLICATION_FACTOR=1
fi

# ---------------------------------------------------------------------------
# Topic Definitions
# Format: "name partitions retention_ms description"
# ---------------------------------------------------------------------------

declare -A TOPICS
TOPICS=(
  # Core event stream — highest throughput topic
  ["user-events"]="64 604800000"          # 64 partitions, 7 days

  # Dead letter queue — events that failed validation
  ["user-events-dlq"]="16 2592000000"     # 16 partitions, 30 days

  # Flink session feature output
  ["session-features"]="32 86400000"      # 32 partitions, 24 hours

  # Real-time item statistics from Flink
  ["item-stats"]="32 86400000"            # 32 partitions, 24 hours

  # Item onboarding pipeline  
  ["item-onboarding"]="8 86400000"        # 8 partitions, 24 hours

  # A/B test exposure events
  ["experiment-exposures"]="16 1209600000" # 16 partitions, 14 days

  # Logged model predictions (for training + shadow mode)
  ["model-predictions"]="32 604800000"    # 32 partitions, 7 days

  # User embedding updates from real-time pipeline
  ["user-embeddings"]="32 3600000"        # 32 partitions, 1 hour
)

# Client properties for IAM auth (MSK)
CLIENT_PROPS=$(mktemp)
cat > "$CLIENT_PROPS" <<EOF
security.protocol=SASL_SSL
sasl.mechanism=AWS_MSK_IAM
sasl.jaas.config=software.amazon.msk.auth.iam.IAMLoginModule required;
sasl.client.callback.handler.class=software.amazon.msk.auth.iam.IAMClientCallbackHandler
EOF

CMD="${1:-create}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

create_topics() {
  log "Creating Kafka topics for environment: $ENVIRONMENT"
  
  for topic in "${!TOPICS[@]}"; do
    read -r partitions retention_ms <<< "${TOPICS[$topic]}"
    
    kafka-topics.sh \
      --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
      --command-config "$CLIENT_PROPS" \
      --create \
      --if-not-exists \
      --topic "$topic" \
      --partitions "$partitions" \
      --replication-factor "$REPLICATION_FACTOR" \
      --config retention.ms="$retention_ms" \
      --config min.insync.replicas=2 \
      --config compression.type=lz4 \
      && log "✅ Created: $topic ($partitions partitions, ${retention_ms}ms retention)"

  done
  
  log "All topics created"
}

describe_topics() {
  log "Describing all recommendation topics..."
  
  for topic in "${!TOPICS[@]}"; do
    echo ""
    echo "── $topic ──────────────────────────────────────────────"
    kafka-topics.sh \
      --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
      --command-config "$CLIENT_PROPS" \
      --describe \
      --topic "$topic" 2>/dev/null || echo "  [Not found]"
  done
}

verify_topics() {
  log "Verifying topic configuration..."
  FAILED=0
  
  for topic in "${!TOPICS[@]}"; do
    read -r expected_partitions _ <<< "${TOPICS[$topic]}"
    
    actual_partitions=$(kafka-topics.sh \
      --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
      --command-config "$CLIENT_PROPS" \
      --describe \
      --topic "$topic" 2>/dev/null | grep "PartitionCount" | awk '{print $2}' || echo "0")
    
    if [ "$actual_partitions" = "$expected_partitions" ]; then
      echo "✅ $topic: $actual_partitions partitions"
    else
      echo "❌ $topic: expected $expected_partitions, got $actual_partitions"
      FAILED=$((FAILED + 1))
    fi
  done
  
  if [ "$FAILED" -gt 0 ]; then
    log "❌ Verification FAILED: $FAILED topics misconfigured"
    exit 1
  else
    log "✅ All topics verified correctly"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "$CMD" in
  create)   create_topics ;;
  describe) describe_topics ;;
  verify)   verify_topics ;;
  *) echo "Usage: $0 {create|describe|verify}"; exit 1 ;;
esac

rm -f "$CLIENT_PROPS"
