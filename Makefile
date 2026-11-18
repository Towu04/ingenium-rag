.PHONY: help install test test-cov lint format check clean

# Automatically resolve src/ imports so tests work out-of-the-box
PYTHONPATH := src:.
PYTEST := PYTHONPATH=$(PYTHONPATH) pytest

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

test: ## Run all unit tests quickly (quiet output, short tracebacks)
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run all tests with terminal coverage report
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

lint: ## Run ruff linter and type checks
	ruff check src/ tests/

format: ## Auto-format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

check: lint test ## Run both linter and unit tests (run this before opening a PR!)

clean: ## Remove pytest, ruff, and Python cache artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +