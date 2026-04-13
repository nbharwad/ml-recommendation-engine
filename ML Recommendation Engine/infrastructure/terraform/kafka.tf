# ============================================================================
# Terraform — Amazon MSK (Managed Kafka)
# Production Kafka cluster for event ingestion and streaming pipelines
# ============================================================================
#
# Design decisions:
# - MSK Serverless not used: too expensive at 50K QPS
# - Provisioned MSK: predictable cost, fine-grained partition control
# - 3 brokers across 3 AZs: HA + supports up to 100K events/sec
# - 64 partitions on user-events topic: parallelism for Flink consumers
# - Schema Registry: Glue Schema Registry (managed, no extra infra)
# - Encryption: TLS in-transit + KMS at-rest
# ============================================================================

# ---------------------------------------------------------------------------
# MSK Security Group
# ---------------------------------------------------------------------------

resource "aws_security_group" "msk" {
  name_prefix = "${local.name}-msk-"
  vpc_id      = module.vpc.vpc_id
  description = "MSK Kafka cluster security group"

  # Kafka plaintext (disabled — TLS only)
  # ingress port 9092 intentionally omitted

  # Kafka TLS
  ingress {
    from_port       = 9094
    to_port         = 9094
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
    description     = "Kafka TLS from EKS"
  }

  # Kafka IAM Auth (SASL/IAM)
  ingress {
    from_port       = 9098
    to_port         = 9098
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
    description     = "Kafka SASL/IAM from EKS"
  }

  # Zookeeper (internal only)
  ingress {
    from_port = 2181
    to_port   = 2181
    protocol  = "tcp"
    self      = true
    description = "Zookeeper internal"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${local.name}-msk" })

  lifecycle {
    create_before_destroy = true
  }
}

# ---------------------------------------------------------------------------
# MSK Cluster Configuration
# ---------------------------------------------------------------------------

resource "aws_msk_configuration" "main" {
  name              = "${local.name}-config"
  kafka_versions    = ["3.6.0"]
  description       = "Recommendation system Kafka config"

  server_properties = <<-EOF
    # Retention
    log.retention.hours=168
    log.retention.bytes=107374182400
    
    # Performance
    num.io.threads=16
    num.network.threads=8
    socket.send.buffer.bytes=102400
    socket.receive.buffer.bytes=102400
    socket.request.max.bytes=104857600
    
    # Replication
    default.replication.factor=3
    min.insync.replicas=2
    
    # Consumer groups
    offsets.topic.replication.factor=3
    offsets.retention.minutes=10080
    
    # Compression
    compression.type=lz4
    
    # Auto topic creation (disabled — explicit only)
    auto.create.topics.enable=false
    
    # Log cleanup
    log.cleanup.policy=delete
    log.segment.bytes=1073741824
    log.roll.hours=24
  EOF
}

# ---------------------------------------------------------------------------
# MSK Cluster
# ---------------------------------------------------------------------------

resource "aws_msk_cluster" "main" {
  cluster_name           = "${local.name}-kafka"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = var.environment == "prod" ? 6 : 3

  broker_node_group_info {
    instance_type   = var.environment == "prod" ? "kafka.m5.4xlarge" : "kafka.m5.large"
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.msk.id]

    storage_info {
      ebs_storage_info {
        volume_size = var.environment == "prod" ? 2000 : 100 # GB per broker
      }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }

  # IAM authentication (recommended over SASL/SCRAM)
  client_authentication {
    sasl {
      iam = true
    }
    tls {}
  }

  # TLS in-transit encryption
  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
    encryption_at_rest_kms_key_arn = aws_kms_key.msk.arn
  }

  # Enhanced monitoring
  enhanced_monitoring = var.environment == "prod" ? "PER_TOPIC_PER_PARTITION" : "PER_BROKER"

  open_monitoring {
    prometheus {
      jmx_exporter {
        enabled_in_broker = true
      }
      node_exporter {
        enabled_in_broker = true
      }
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
      s3 {
        enabled = true
        bucket  = aws_s3_bucket.msk_logs.id
        prefix  = "msk-broker-logs/"
      }
    }
  }

  tags = local.common_tags

  lifecycle {
    prevent_destroy = true # Prevent accidental deletion
  }
}

# ---------------------------------------------------------------------------
# KMS Key for MSK Encryption
# ---------------------------------------------------------------------------

resource "aws_kms_key" "msk" {
  description             = "MSK encryption key for ${local.name}"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags                    = local.common_tags
}

resource "aws_kms_alias" "msk" {
  name          = "alias/${local.name}-msk"
  target_key_id = aws_kms_key.msk.key_id
}

# ---------------------------------------------------------------------------
# CloudWatch Log Group for MSK
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${local.name}"
  retention_in_days = var.environment == "prod" ? 30 : 7
  tags              = local.common_tags
}

# ---------------------------------------------------------------------------
# S3 Bucket for MSK Logs
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "msk_logs" {
  bucket = "${local.name}-msk-logs-${data.aws_caller_identity.current.account_id}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "msk_logs" {
  bucket = aws_s3_bucket.msk_logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "msk_logs" {
  bucket = aws_s3_bucket.msk_logs.id

  rule {
    id     = "expire-old-logs"
    status = "Enabled"
    expiration {
      days = var.environment == "prod" ? 90 : 14
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "msk_logs" {
  bucket = aws_s3_bucket.msk_logs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.msk.arn
    }
  }
}

# ---------------------------------------------------------------------------
# MSK Topics (via Kafka provider — applied after cluster is running)
# ---------------------------------------------------------------------------

# NOTE: Topics are managed via Helm chart (Strimzi) or direct kafka-topics.sh
# commands in the bootstrap script. Terraform doesn't directly manage topics.
# See: scripts/bootstrap/create-topics.sh

# ---------------------------------------------------------------------------
# IAM Policy for EKS → MSK Access
# ---------------------------------------------------------------------------

resource "aws_iam_policy" "msk_producer" {
  name        = "${local.name}-msk-producer"
  description = "Allow Kafka producer access for recommendation services"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:Connect",
          "kafka-cluster:AlterCluster",
          "kafka-cluster:DescribeCluster",
        ]
        Resource = aws_msk_cluster.main.arn
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:*Topic*",
          "kafka-cluster:WriteData",
          "kafka-cluster:ReadData",
        ]
        Resource = "arn:aws:kafka:${var.region}:${data.aws_caller_identity.current.account_id}:topic/${local.name}-kafka/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:AlterGroup",
          "kafka-cluster:DescribeGroup",
        ]
        Resource = "arn:aws:kafka:${var.region}:${data.aws_caller_identity.current.account_id}:group/${local.name}-kafka/*"
      },
    ]
  })
}

resource "aws_iam_policy" "msk_consumer" {
  name        = "${local.name}-msk-consumer"
  description = "Allow Kafka consumer access for Flink jobs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:Connect",
          "kafka-cluster:DescribeCluster",
        ]
        Resource = aws_msk_cluster.main.arn
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:ReadData",
          "kafka-cluster:DescribeTopicDynamicConfiguration",
          "kafka-cluster:DescribeTopic",
        ]
        Resource = "arn:aws:kafka:${var.region}:${data.aws_caller_identity.current.account_id}:topic/${local.name}-kafka/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:AlterGroup",
          "kafka-cluster:DescribeGroup",
        ]
        Resource = "arn:aws:kafka:${var.region}:${data.aws_caller_identity.current.account_id}:group/${local.name}-kafka/*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "msk_bootstrap_brokers_tls" {
  description = "TLS bootstrap brokers for Kafka clients"
  value       = aws_msk_cluster.main.bootstrap_brokers_sasl_iam
  sensitive   = true
}

output "msk_cluster_arn" {
  value = aws_msk_cluster.main.arn
}

output "msk_zookeeper_connect" {
  value     = aws_msk_cluster.main.zookeeper_connect_string
  sensitive = true
}
