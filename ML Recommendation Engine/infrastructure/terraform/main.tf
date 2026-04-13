# ============================================================================
# Terraform — EKS Cluster Module
# Production Kubernetes cluster for recommendation system
# ============================================================================

terraform {
  required_version = ">= 1.9.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.16"
    }
  }
  
  backend "s3" {
    bucket         = "rec-system-terraform-state"
    key            = "eks/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "environment" {
  type        = string
  description = "Environment name (dev, staging, prod)"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "cluster_version" {
  type    = string
  default = "1.31"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

# Environment-specific configurations
locals {
  env_config = {
    dev = {
      general_instance_types = ["m6i.xlarge"]
      general_min_size       = 2
      general_max_size       = 5
      general_desired_size   = 3
      gpu_instance_types     = ["g5.xlarge"]
      gpu_min_size           = 0
      gpu_max_size           = 2
      gpu_desired_size       = 1
      memory_instance_types  = ["r6i.2xlarge"]
      memory_min_size        = 2
      memory_max_size        = 4
      memory_desired_size    = 2
    }
    staging = {
      general_instance_types = ["m6i.2xlarge"]
      general_min_size       = 3
      general_max_size       = 10
      general_desired_size   = 5
      gpu_instance_types     = ["g5.2xlarge"]
      gpu_min_size           = 2
      gpu_max_size           = 4
      gpu_desired_size       = 2
      memory_instance_types  = ["r6i.2xlarge"]
      memory_min_size        = 4
      memory_max_size        = 8
      memory_desired_size    = 4
    }
    prod = {
      general_instance_types = ["m6i.2xlarge"]
      general_min_size       = 10
      general_max_size       = 30
      general_desired_size   = 15
      gpu_instance_types     = ["g5.2xlarge"]
      gpu_min_size           = 4
      gpu_max_size           = 16
      gpu_desired_size       = 8
      memory_instance_types  = ["r6i.4xlarge"]
      memory_min_size        = 8
      memory_max_size        = 24
      memory_desired_size    = 12
    }
  }
  
  config    = local.env_config[var.environment]
  name      = "rec-system-${var.environment}"
  
  common_tags = {
    Project     = "recommendation-system"
    Environment = var.environment
    ManagedBy   = "terraform"
    Team        = "ml-platform"
  }
}

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

provider "aws" {
  region = var.region
  
  default_tags {
    tags = local.common_tags
  }
}

# ---------------------------------------------------------------------------
# VPC
# ---------------------------------------------------------------------------

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.16"
  
  name = "${local.name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.region}a", "${var.region}b", "${var.region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "dev"  # cost optimization for dev
  enable_dns_hostnames = true
  
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# ---------------------------------------------------------------------------
# EKS Cluster
# ---------------------------------------------------------------------------

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.31"
  
  cluster_name    = local.name
  cluster_version = var.cluster_version
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_public_access = var.environment == "dev"
  
  # Managed node groups
  eks_managed_node_groups = {
    # General compute (serving, orchestration, monitoring)
    general = {
      name            = "general"
      instance_types  = local.config.general_instance_types
      min_size        = local.config.general_min_size
      max_size        = local.config.general_max_size
      desired_size    = local.config.general_desired_size
      
      labels = {
        node-type = "general"
      }
    }
    
    # GPU nodes (ranking model inference)
    gpu = {
      name            = "gpu"
      instance_types  = local.config.gpu_instance_types
      min_size        = local.config.gpu_min_size
      max_size        = local.config.gpu_max_size
      desired_size    = local.config.gpu_desired_size
      
      ami_type = "AL2_x86_64_GPU"
      
      labels = {
        node-type                  = "gpu"
        "nvidia.com/gpu.present"   = "true"
      }
      
      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          effect = "NO_SCHEDULE"
        }
      }
    }
    
    # Memory-optimized (Redis, Milvus, feature store)
    memory = {
      name            = "memory"
      instance_types  = local.config.memory_instance_types
      min_size        = local.config.memory_min_size
      max_size        = local.config.memory_max_size
      desired_size    = local.config.memory_desired_size
      
      labels = {
        node-type = "memory"
      }
    }
  }
  
  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# ElastiCache (Redis) for Feature Store
# ---------------------------------------------------------------------------

resource "aws_elasticache_replication_group" "feature_store" {
  replication_group_id = "${local.name}-features"
  description          = "Feature store Redis cluster"
  
  node_type            = var.environment == "prod" ? "cache.r6g.4xlarge" : "cache.r6g.xlarge"
  num_cache_clusters   = var.environment == "prod" ? 3 : 1
  
  engine               = "redis"
  engine_version       = "7.1"
  port                 = 6379
  parameter_group_name = "default.redis7.cluster.on"
  
  automatic_failover_enabled = var.environment == "prod"
  multi_az_enabled           = var.environment == "prod"
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  snapshot_retention_limit = var.environment == "prod" ? 7 : 1
  
  tags = local.common_tags
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name}-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name}-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.feature_store.primary_endpoint_address
}

output "vpc_id" {
  value = module.vpc.vpc_id
}
