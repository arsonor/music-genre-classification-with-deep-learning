# CI/CD Pipeline - Music Genre Classification

## Overview

Automated CI/CD pipeline that tests, validates, and ensures quality of the music genre classification API on every push and pull request.

## Pipeline Workflow

### Triggers
- **Push** to `main` or `develop` branches
- **Pull requests** to `main` or `develop` branches  
- **Manual** workflow dispatch

### Jobs Sequence

```
Code Quality → Unit Tests → Integration Tests → Security Scan → Docker Build → Performance Tests → Notify
```

## Job Details

### 1. Code Quality & Linting
- Runs `check_code_quality.py` (isort, black, pylint)
- Installs all project dependencies for proper linting
- Validates code standards before testing

### 2. Unit Tests (Matrix Strategy)
- **Parallel execution** across test categories:
  - `service` - Genre prediction logic tests
  - `server` - Flask endpoint tests  
  - `monitoring` - Prometheus metrics tests
  - `classifier` - ML pipeline & model tests
- Generates coverage reports
- Uses existing `run_tests.py` script

### 3. Integration Tests
- Full application stack testing
- Creates test audio files for realistic testing
- Tests with Redis service for caching
- Validates end-to-end workflows

### 4. Security Scanning
- **Bandit** - Python security linter
- **Safety** - Dependency vulnerability scanner
- **pip-audit** - Package security auditing
- Skips B104 (bind all interfaces) and B113 (missing timeout) for development patterns

### 5. Docker Build & Test
- Tests Docker Compose setup with core services (`api`, `nginx`, `mlflow`)
- Validates container builds and service communication
- Tests API endpoints with real audio files
- Includes MLflow model artifacts for functional testing

### 6. Performance Tests (Main Branch Only)
- **Cold Start Testing** - Initial model loading (≤30s)
- **Warm Request Testing** - Subsequent requests (≤5s)
- **Concurrent Load Testing** - 5 simultaneous requests
- **Response Consistency** - Performance stability validation
- Measures: response times, success rates, throughput

### 7. Notification
- Reports overall pipeline status
- Summarizes job results
- Fails pipeline if any critical job fails

## Performance Benchmarks

| Metric | Threshold |
|--------|-----------|
| Cold Start | ≤30 seconds |
| Warm Requests | ≤5 seconds |
| Concurrent Average | ≤15 seconds |
| Consistency Outliers | <3x average |

## Key Features

- **Dependency Caching** - Speeds up builds
- **Parallel Testing** - Matrix strategy for unit tests
- **MLflow Integration** - Uses committed model artifacts
- **Docker Testing** - Validates production setup
- **Security Validation** - Multiple security tools
- **Performance Validation** - Real-world load testing

## Environment Requirements

- **Python**: 3.11
- **Docker**: Latest with Compose support
- **Services**: Redis, MLflow, PostgreSQL
- **Dependencies**: Listed in respective `requirements.txt` files

## Artifacts Generated

- **Coverage Reports** - HTML coverage in `htmlcov/`
- **Security Reports** - JSON reports for bandit, safety, pip-audit
- **Performance Metrics** - Response time statistics
- **Docker Logs** - Service debugging information

## Pipeline Success Criteria

✅ All code quality checks pass  
✅ All unit tests pass with >80% coverage  
✅ Integration tests validate full stack  
✅ No high-severity security issues  
✅ Docker containers build and communicate  
✅ API meets performance benchmarks  

## Usage

Pipeline runs automatically on push/PR. For manual testing:

```bash
# Run specific test categories locally
python run_tests.py service --coverage
python run_tests.py server --verbose

# Run code quality checks
python check_code_quality.py

# Test Docker setup
docker-compose up -d api nginx mlflow
```