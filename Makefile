# Music Genre Classification Project Makefile
# This Makefile provides convenient commands for development, testing, and deployment

.PHONY: help install setup test test-unit test-integration test-coverage clean lint format docker-build docker-up docker-down docker-clean ci

# Default target
help: ## Show this help message
	@echo "🎵 Music Genre Classification Project"
	@echo "====================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🔍 For more detailed help on testing: python run_tests.py --help"
	@echo "🔍 For more detailed help on code quality: python check_code_quality.py --help"

# Installation and Setup
install: ## Install all project dependencies
	@echo "📦 Installing project dependencies..."
	@bash setup_tests.sh

setup: install ## Alias for install (same as install)

install-dev: ## Install development dependencies only (faster)
	@echo "📦 Installing development dependencies..."
	@pip install -r requirements-quality.txt
	@pip install -r tests/requirements-test.txt

install-api: ## Install API dependencies only
	@echo "📦 Installing API dependencies..."
	@pip install -r api/requirements.txt

install-monitoring: ## Install monitoring dependencies
	@echo "📊 Installing monitoring dependencies..."
	@if [ -f "monitoring/requirements.txt" ]; then \
		pip install -r monitoring/requirements.txt; \
	else \
		echo "⚠️  monitoring/requirements.txt not found"; \
	fi

install-classifier: ## Install classifier dependencies
	@echo "🤖 Installing classifier dependencies..."
	@if [ -f "classifier/requirements.txt" ]; then \
		pip install -r classifier/requirements.txt; \
	else \
		echo "⚠️  classifier/requirements.txt not found"; \
	fi

# Testing Commands
test: ## Run all tests
	@echo "🧪 Running all tests..."
	@python run_tests.py

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	@python run_tests.py unit

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	@python run_tests.py integration

test-service: ## Run API service tests
	@echo "🧪 Running service tests..."
	@python run_tests.py service

test-server: ## Run server tests
	@echo "🧪 Running server tests..."
	@python run_tests.py server

test-monitoring: ## Run monitoring tests
	@echo "🧪 Running monitoring tests..."
	@python run_tests.py monitoring

test-classifier: ## Run classifier tests
	@echo "🧪 Running classifier tests..."
	@python run_tests.py classifier

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	@python run_tests.py -c

test-verbose: ## Run tests with verbose output
	@echo "🧪 Running tests (verbose)..."
	@python run_tests.py -v

# Code Quality
lint: ## Run code quality checks (isort, black, pylint)
	@echo "🔍 Running code quality checks..."
	@python check_code_quality.py

format: ## Fix code formatting issues (isort, black)
	@echo "🔧 Fixing code formatting..."
	@python check_code_quality.py --fix

format-quick: ## Quick format without pylint (faster)
	@echo "🔧 Quick formatting (skip pylint)..."
	@python check_code_quality.py --skip-pylint --fix

pre-commit-install: ## Install pre-commit hooks
	@echo "🪝 Installing pre-commit hooks..."
	@pip install pre-commit
	@pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "🪝 Running pre-commit hooks..."
	@pre-commit run --all-files

# Docker Commands
docker-build: ## Build Docker containers
	@echo "🐳 Building Docker containers..."
	@docker-compose build

docker-up: ## Start Docker services
	@echo "🐳 Starting Docker services..."
	@docker-compose up -d

docker-up-build: ## Build and start Docker services
	@echo "🐳 Building and starting Docker services..."
	@docker-compose up --build -d

docker-down: ## Stop Docker services
	@echo "🐳 Stopping Docker services..."
	@docker-compose down

docker-logs: ## Show Docker service logs
	@echo "🐳 Showing Docker logs..."
	@docker-compose logs -f

docker-clean: ## Clean Docker containers, images, and volumes
	@echo "🐳 Cleaning Docker resources..."
	@docker-compose down -v --remove-orphans
	@docker system prune -f

# Development Commands  
run-local: ## Run Flask server locally
	@echo "🚀 Starting Flask server locally..."
	@cd api && python server.py

run-client: ## Run test client
	@echo "🎵 Running test client..."
	@python client.py --file audio_files_test/

train-model: ## Train the classification model
	@echo "🤖 Training classification model..."
	@cd classifier && python train.py

# Jupyter Commands
jupyter: ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter notebook server..."
	@jupyter notebook notebooks/

jupyter-lab: ## Start JupyterLab server
	@echo "📓 Starting JupyterLab server..."
	@jupyter lab notebooks/

# Monitoring Commands
monitoring-up: ## Start monitoring services (Prometheus, Grafana)
	@echo "📊 Starting monitoring services..."
	@docker-compose up -d prometheus grafana alertmanager monitoring

monitoring-down: ## Stop monitoring services
	@echo "📊 Stopping monitoring services..."
	@docker-compose stop prometheus grafana alertmanager monitoring

# MLflow Commands
mlflow-up: ## Start MLflow server
	@echo "🔬 Starting MLflow server..."
	@docker-compose up -d mlflow

mlflow-ui: ## Open MLflow UI (requires mlflow to be running)
	@echo "🔬 MLflow UI available at: http://localhost:5000"

# Prefect Commands
prefect-up: ## Start Prefect server and worker
	@echo "🌊 Starting Prefect services..."
	@docker-compose up -d postgres prefect-server prefect-worker

prefect-ui: ## Open Prefect UI (requires prefect to be running)
	@echo "🌊 Prefect UI available at: http://localhost:4200"

prefect-deploy: ## Start Prefect deploy flows
	@docker exec -it prefect-worker prefect deploy --all
	@echo "✅ Prefect deployment ready!"

# Grafana Commands
grafana-ui: ## Open Grafana UI (requires monitoring to be running)
	@echo "📊 Grafana UI available at: http://localhost:3000 (admin/admin)"


# Cleanup Commands
clean: ## Clean up temporary files and artifacts
	@echo "🧹 Cleaning up temporary files..."
	@python run_tests.py --clean
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete
	@rm -rf htmlcov/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@echo "✅ Cleanup completed"

clean-all: clean docker-clean ## Deep clean including Docker resources
	@echo "🧹 Deep cleaning all resources..."
	@rm -rf mlruns/
	@rm -rf artifacts/
	@rm -rf .prefect/
	@echo "✅ Deep cleanup completed"


# Quick Start Commands
dev-setup: install pre-commit-install ## Complete development setup
	@echo "🚀 Development environment setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "  make test          # Run tests"
	@echo "  make run-local     # Start local server"
	@echo "  make docker-up     # Start with Docker"



# Status Commands
status: ## Show project status and service health
	@echo "📊 Project Status"
	@echo "=================="
	@echo ""
	@echo "🐳 Docker Services:"
	@docker-compose ps 2>/dev/null || echo "Docker Compose not running"
	@echo ""
	@echo "📁 Project Structure:"
	@ls -la
	@echo ""
	@echo "🔍 Available make targets:"
	@echo "  make help          # Show all available commands"
	@echo "  make dev-setup     # Complete development setup"
	@echo "  make test          # Run all tests"
	@echo "  make docker-up     # Start services with Docker"