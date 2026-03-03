.PHONY: help install install-dev install-test install-all check tidy

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

check: ## Run Ruff lint checks.
	uv run ruff check applpy

tidy: ## Run Ruff autoformatter.
	uv run ruff check --fix applpy
	uv run ruff format applpy
