# ============================================================================
# Terraform — Outputs
# Exported values for use by Kubernetes manifests, CI/CD, and application config
# ============================================================================

# ---------------------------------------------------------------------------
# EKS
# ---------------------------------------------------------------------------

output "eks_cluster_name" {
  description = "EKS cluster name (used in kubectl config)"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS API server endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "eks_cluster_ca" {
  description = "Base64-encoded cluster CA certificate"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "eks_oidc_issuer" {
  description = "OIDC issuer URL for IRSA (IAM Roles for Service Accounts)"
  value       = module.eks.cluster_oidc_issuer_url
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs for application workloads"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "Public subnet IDs for load balancers"
  value       = module.vpc.public_subnets
}

# ---------------------------------------------------------------------------
# Redis (ElastiCache)
# ---------------------------------------------------------------------------

output "redis_primary_endpoint" {
  description = "Redis primary endpoint for write operations"
  value       = aws_elasticache_replication_group.feature_store.primary_endpoint_address
  sensitive   = true
}

output "redis_reader_endpoint" {
  description = "Redis reader endpoint for read operations (load balanced)"
  value       = aws_elasticache_replication_group.feature_store.reader_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "Redis port"
  value       = 6379
}

# ---------------------------------------------------------------------------
# Kafka / MSK
# ---------------------------------------------------------------------------

output "kafka_bootstrap_brokers" {
  description = "MSK bootstrap brokers (IAM auth)"
  value       = aws_msk_cluster.main.bootstrap_brokers_sasl_iam
  sensitive   = true
}

output "kafka_cluster_arn" {
  description = "MSK cluster ARN"
  value       = aws_msk_cluster.main.arn
}

# ---------------------------------------------------------------------------
# IAM
# ---------------------------------------------------------------------------

output "msk_producer_policy_arn" {
  description = "IAM policy ARN for Kafka producer (attach to service accounts)"
  value       = aws_iam_policy.msk_producer.arn
}

output "msk_consumer_policy_arn" {
  description = "IAM policy ARN for Kafka consumer (attach to Flink service accounts)"
  value       = aws_iam_policy.msk_consumer.arn
}

# ---------------------------------------------------------------------------
# Convenience: kubeconfig update command
# ---------------------------------------------------------------------------

output "kubeconfig_command" {
  description = "Run this to configure kubectl access"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}
