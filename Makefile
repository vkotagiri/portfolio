# Makefile
# Agentic Portfolio - Development Commands
# =========================================

.PHONY: help install dev test lint format check clean run report

# Default target
help:
	@echo "Agentic Portfolio - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install dev dependencies + pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make test        Run test suite"
	@echo "  make test-cov    Run tests with coverage report"
	@echo "  make lint        Run linters (ruff, mypy)"
	@echo "  make format      Format code with black"
	@echo "  make check       Run all checks (lint + test)"
	@echo ""
	@echo "Application:"
	@echo "  make run         Start API server"
	@echo "  make report      Generate today's report"
	@echo "  make refresh     Run daily price refresh"
	@echo ""
	@echo "Database:"
	@echo "  make init-db     Initialize database tables"
	@echo "  make backup-db   Backup database to backups/"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       Remove cache files and build artifacts"

# Python environment
PYTHON := python3
VENV := env_portfolio
ACTIVATE := source $(VENV)/bin/activate &&

# Installation
install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) pip install --upgrade pip
	$(ACTIVATE) pip install -r requirements.txt

dev: install
	$(ACTIVATE) pip install -e ".[dev]"
	$(ACTIVATE) pre-commit install

# Testing
test:
	$(ACTIVATE) PYTHONPATH=. pytest tests/ -v

test-cov:
	$(ACTIVATE) PYTHONPATH=. pytest tests/ -v --cov=app --cov-report=html --cov-report=term

test-fast:
	$(ACTIVATE) PYTHONPATH=. pytest tests/ -v -m "not slow"

# Code Quality
lint:
	$(ACTIVATE) ruff check app/ tests/
	$(ACTIVATE) mypy app/ --ignore-missing-imports

format:
	$(ACTIVATE) black app/ tests/
	$(ACTIVATE) ruff check app/ tests/ --fix

check: lint test

# Application
run:
	$(ACTIVATE) uvicorn app.server.api:app --reload --host 0.0.0.0 --port 8000

report:
	$(ACTIVATE) PYTHONPATH=. python -m app.server.cli report today

report-ai:
	$(ACTIVATE) PYTHONPATH=. python -m app.server.cli report today --ai-summary

refresh:
	./scripts/daily_refresh.sh

# Database
init-db:
	$(ACTIVATE) PYTHONPATH=. python -m app.server.cli init-db

backup-db:
	@mkdir -p backups
	cp portfolio.db backups/portfolio_$$(date +%Y%m%d_%H%M%S).db
	@echo "Database backed up to backups/"

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned up cache and build artifacts"
