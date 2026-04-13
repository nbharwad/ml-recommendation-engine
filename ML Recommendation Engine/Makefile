.PHONY: help dev test lint build deploy-staging deploy-prod load-test clean

# Default target
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ===== Development =====

dev: ## Start local development environment
	docker compose up -d
	@echo "Services starting..."
	@echo "API:        http://localhost:8080"
	@echo "Grafana:    http://localhost:3000"
	@echo "Jaeger:     http://localhost:16686"
	@echo "Prometheus: http://localhost:9090"

dev-down: ## Stop local development environment
	docker compose down -v

# ===== Testing =====

test: ## Run unit tests
	pytest tests/unit/ -v --cov=services --cov-report=term-missing

test-integration: ## Run integration tests (requires docker-compose up)
	pytest tests/integration/ -v --timeout=120

test-all: test test-integration ## Run all tests

# ===== Load Testing =====

load-test: ## Run load test (10K QPS steady state)
	locust -f tests/load/locustfile.py \
		--headless \
		--host=http://localhost:8080 \
		--users=5000 \
		--spawn-rate=100 \
		--run-time=5m \
		--csv=reports/load-test

load-test-peak: ## Run peak load test (50K QPS)
	locust -f tests/load/locustfile.py \
		--headless \
		--host=http://localhost:8080 \
		--users=25000 \
		--spawn-rate=500 \
		--run-time=15m \
		--csv=reports/peak-load-test

# ===== Code Quality =====

lint: ## Run linters
	ruff check services/ ml/ streaming/
	black --check services/ ml/ streaming/
	mypy services/ --ignore-missing-imports

format: ## Format code
	black services/ ml/ streaming/
	isort services/ ml/ streaming/
	ruff check --fix services/ ml/ streaming/

# ===== Build =====

build: ## Build all Docker images
	docker compose build

build-serving: ## Build serving image
	docker build -t rec-serving -f infrastructure/docker/Dockerfile.serving .

build-ranking: ## Build ranking (GPU) image
	docker build -t rec-ranking -f infrastructure/docker/Dockerfile.ranking .

# ===== Deployment =====

deploy-staging: ## Deploy to staging (via ArgoCD)
	@echo "Deploying to staging..."
	kubectl apply -f infrastructure/kubernetes/ -n recommendation --context staging

deploy-prod: ## Deploy to production (requires approval)
	@echo "⚠️  Production deployment requires manual approval in GitHub"
	@echo "Push to main branch and approve the deployment in GitHub Actions"

# ===== ML =====

train-two-tower: ## Train Two-Tower model
	python ml/models/two_tower/model.py \
		--train-data data/train/ \
		--val-data data/val/ \
		--output-dir models/two_tower/ \
		--epochs 10

train-dlrm: ## Train DLRM ranking model
	python ml/models/dlrm/model.py \
		--train-data data/train/ \
		--val-data data/val/ \
		--output-dir models/dlrm/ \
		--epochs 5

# ===== Infrastructure =====

tf-plan: ## Terraform plan
	cd infrastructure/terraform && terraform plan -var="environment=staging"

tf-apply: ## Terraform apply (staging)
	cd infrastructure/terraform && terraform apply -var="environment=staging" -auto-approve

# ===== Cleanup =====

clean: ## Clean up artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf reports/ .pytest_cache/ htmlcov/ .mypy_cache/
