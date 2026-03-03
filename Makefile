.PHONY: help install install-dev install-test install-all test test-functional test-unit check tidy

TEST ?=

help: ## Show available make targets and descriptions.
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install base project dependencies.
	uv sync

install-dev: ## Install project with development dependencies.
	uv sync --extra dev

install-test: ## Install project with test dependencies.
	uv sync --extra test

install-all: ## Install project with all optional dependencies.
	uv sync --all-extras

test: ## Run all tests, or one test when TEST=/path/to/test is provided.
	uv run pytest --cov=applpy --cov-report=term-missing $(if $(strip $(TEST)),$(TEST),test_applpy)

test-functional: ## Run functional tests, or TEST=/path/to/test to target one test.
	uv run pytest --cov=applpy --cov-report=term-missing $(if $(strip $(TEST)),$(TEST),test_applpy/functional)

test-unit: ## Run unit tests, or TEST=/path/to/test to target one test.
	uv run pytest --cov=applpy --cov-report=term-missing $(if $(strip $(TEST)),$(TEST),test_applpy/unit)

check: ## Run Ruff lint checks.
	uv run ruff check applpy test_applpy

tidy: ## Run Ruff autoformatter.
	uv run ruff check --fix applpy test_applpy
	uv run ruff format applpy test_applpy
