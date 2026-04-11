# ============================================================================
# Terraform — Multi-Region Deployment
# Recommendation system deployment across 3 AWS regions
# ============================================================================
#
# Architecture:
# - Primary: us-east-1
# - Secondary: us-west-2
# - Tertiary: eu-west-1
#
# Data Replication:
# - Kafka: MirrorMaker for cross-region replication
# - Redis: Global tables for cross-region cache
# - Milvus: Separate clusters per region (vector embeddings)
#
# Traffic Routing:
# - Route53: GeoDNS with health checks
# - failover: automatic on region failure
# ============================================================================

terraform {
  required_version = ">= 1.9.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
}

# ---------------------------------------------------------------------------
# Multi-Region Variables
# ---------------------------------------------------------------------------

variable "primary_region" {
  type        = string
  default     = "us-east-1"
  description = "Primary region (traffic origin)"
}

variable "secondary_region" {
  type        = string
  default     = "us-west-2"
  description = "Secondary region for failover"
}

variable "tertiary_region" {
  type        = string
  default     = "eu-west-1"
  description = "Tertiary region for DR"
}

variable "enable_multi_region" {
  type        = bool
  default   = false
  description = "Enable multi-region deployment"
}

# ---------------------------------------------------------------------------
# Region Aliases
# ---------------------------------------------------------------------------

locals {
  regions = {
    primary   = var.primary_region
    secondary = var.secondary_region
    tertiary  = var.tertiary_region
  }
  
  region_aliases = {
    us-east-1  = "primary"
    us-west-2   = "secondary"
    eu-west-1   = "tertiary"
  }
}

# ---------------------------------------------------------------------------
# EKS Cluster per Region
# ---------------------------------------------------------------------------

resource "aws_eks_cluster" "this" {
  for_each = var.enable_multi_region ? local.regions : {}
  
  name     = "rec-system-${var.environment}-${each.value}"
  role_arn = aws_iam_role.eks_cluster[each.key].arn
  version = var.cluster_version
  
  vpc_config {
    subnet_ids =/aws_subnet.private[*].id
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]
  
  provider = aws[each.value]
}

# ---------------------------------------------------------------------------
# IAM Roles per Region
# ---------------------------------------------------------------------------

resource "aws_iam_role" "eks_cluster" {
  for_each = var.enable_multi_region ? local.regions : {}
  
  name = "rec-system-eks-cluster-${each.key}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "eks.${each.value}.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
  
  provider = aws[each.value]
}

# ---------------------------------------------------------------------------
# Route53 Health Check & Failover
# ---------------------------------------------------------------------------

resource "aws_route53_health_check" "this" {
  for_each = var.enable_multi_region ? local.regions : {}
  
  fqdn              = "api.recSystem.${each.value}.example.com"
  port              = 443
  type              = "HTTPS"
  resource_path    = "/health"
  failure_threshold = 3
  request_interval = 30
  
  provider = aws.primary
}

resource "aws_route53_record" "primary" {
  zone_id = aws_route53_zone.this.zone_id
  name    = "api.recSystem.example.com"
  type    = "A"
  
  failover_routing_policy {
    type = "PRIMARY"
  }
  
  set_identifier  = "primary"
  health_check_id = aws_route53_health_check.primary.id
  
  alias {
    name                   = aws_lb.primary.dns_name
    zone_id              = aws_lb.primary.zone_id
    evaluate_target_health = true
  }
  
  provider = aws.primary
}

resource "aws_route53_record" "secondary" {
  zone_id = aws_route53_zone.this.zone_id
  name    = "api.recSystem.example.com"
  type    = "A"
  
  failover_routing_policy {
    type = "SECONDARY"
  }
  
  set_identifier  = "secondary"
  health_check_id = aws_route53_health_check.secondary.id
  
  alias {
    name                   = aws_lb.secondary.dns_name
    zone_id              = aws_lb.secondary.zone_id
    evaluate_target_health = true
  }
  
  provider = aws.primary
}

# ---------------------------------------------------------------------------
# Cross-Region Data Replication
# ---------------------------------------------------------------------------

resource "aws_msk_connect_custom_plugin" "mirror_maker" {
  for_each = var.enable_multi_region ? local.regions : toset([var.primary_region])
  
  name = "rec-system-mirror-maker-${each.value}"
  
  runtime_properties = {
    "connector.config" = jsonencode({
      "replication.factor" = 3
      "sync.topic.configs.enabled" = true
      "check.interval.ms" = 10000
      "sync.group.offsets.enabled" = true
    })
  }
  
  provider = aws[each.value]
}

# ---------------------------------------------------------------------------
# Redis Global Tables (Cross-Region Cache)
# ---------------------------------------------------------------------------

resource "aws_elasticache_global_replication_group" "this" {
  for_each = var.enable_multi_region ? toset(["global"]) : toset([])
  
  global_replication_group_id = "rec-system-global-cache"
  
  primary_replication_group_id = aws_elasticache_replication_group.primary.id
  
  provider = aws.primary
}

resource "aws_elasticache_replication_group" "primary" {
  for_each = var.enable_multi_region ? local.regions : {}
  
  replication_group_id = "rec-system-cache-${each.value}"
  description     = "Recommendation system cache - ${each.key}"
  
  engine               = "redis"
  engine_version      = "7.0"
  node_type          = "cache.r6g.xlarge"
  num_cache_clusters = 2
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  provider = aws[each.value]
}

# ---------------------------------------------------------------------------
# Multi-Region Outputs
# ---------------------------------------------------------------------------

output "primary_region" {
  value = var.primary_region
}

output "secondary_region" {
  value = var.secondary_region
}

output "tertiary_region" {
  value = var.tertiary_region
}

output "cluster_endpoints" {
  description = "EKS cluster endpoints by region"
  value = {
    for key, cluster in aws_eks_cluster.this :
    key => cluster.endpoint
  }
}

output "cache_endpoints" {
  description = "Redis endpoints by region"
  value = {
    for key, rg in aws_elasticache_replication_group.primary :
    key => rg.primary_endpoint
  }
}

output "route53_health_checks" {
  description = "Route53 health check IDs"
  value = {
    for key, hc in aws_route53_health_check.this :
    key => hc.id
  }
}