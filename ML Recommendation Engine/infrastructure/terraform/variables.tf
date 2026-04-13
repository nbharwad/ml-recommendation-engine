# ============================================================================
# Terraform — Variables
# Central variable definitions for all modules
# ============================================================================

variable "project" {
  type        = string
  default     = "rec-system"
  description = "Short project name used across all resource names"
}

variable "aws_account_id" {
  type        = string
  description = "AWS Account ID (used in IAM ARNs)"
  sensitive   = true
}

variable "domain_name" {
  type        = string
  default     = "recommendation.internal"
  description = "Internal domain name for service discovery"
}

variable "alert_email" {
  type        = string
  default     = "ml-platform-alerts@company.com"
  description = "Email address for infrastructure alerts"
}

# ---------------------------------------------------------------------------
# Kafka / MSK
# ---------------------------------------------------------------------------

variable "kafka_version" {
  type        = string
  default     = "3.6.0"
  description = "Kafka version for MSK cluster"
}

variable "kafka_topics" {
  type = map(object({
    partitions         = number
    replication_factor = number
    retention_hours    = number
    description        = string
  }))
  default = {
    "user-events" = {
      partitions         = 64
      replication_factor = 3
      retention_hours    = 168 # 7 days
      description        = "Raw user interaction events (views, clicks, purchases)"
    }
    "user-events-dlq" = {
      partitions         = 16
      replication_factor = 3
      retention_hours    = 720 # 30 days — longer for debugging
      description        = "Dead letter queue for invalid/rejected events"
    }
    "session-features" = {
      partitions         = 32
      replication_factor = 3
      retention_hours    = 24
      description        = "Session-level feature aggregations from Flink"
    }
    "item-stats" = {
      partitions         = 32
      replication_factor = 3
      retention_hours    = 24
      description        = "Real-time item statistics (CTR, views, cart rate)"
    }
    "experiment-exposures" = {
      partitions         = 16
      replication_factor = 3
      retention_hours    = 336 # 14 days
      description        = "A/B test exposure events for statistical analysis"
    }
    "model-predictions" = {
      partitions         = 32
      replication_factor = 3
      retention_hours    = 168
      description        = "Logged model predictions (for shadow mode + training)"
    }
    "item-onboarding" = {
      partitions         = 8
      replication_factor = 3
      retention_hours    = 24
      description        = "New item catalog events for embedding pipeline"
    }
  }
}

# ---------------------------------------------------------------------------
# Redis / ElastiCache
# ---------------------------------------------------------------------------

variable "redis_version" {
  type    = string
  default = "7.1"
}

variable "redis_config" {
  type = map(string)
  default = {
    maxmemory_policy = "volatile-lfu" # Evict LFU items with TTL
    tcp_keepalive    = "300"
    timeout          = "0"
  }
}

# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------

variable "vault_version" {
  type    = string
  default = "1.17.2"
}

variable "vault_storage_size_gb" {
  type    = number
  default = 50
}

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

variable "prometheus_retention_days" {
  type    = number
  default = 15
}

variable "grafana_admin_password" {
  type      = string
  sensitive = true
  default   = "" # Must be set via TF_VAR_grafana_admin_password
}

variable "slack_webhook_url" {
  type      = string
  sensitive = true
  default   = "" # For Alertmanager Slack notifications
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "enable_flow_logs" {
  type        = bool
  default     = true
  description = "Enable VPC flow logs (required for security audit)"
}

# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------

variable "enable_multi_region" {
  type        = bool
  default     = false
  description = "Phase 4: Enable multi-region active-active deployment"
}

variable "enable_shadow_mode" {
  type        = bool
  default     = false
  description = "Phase 5: Log shadow model predictions alongside production"
}
