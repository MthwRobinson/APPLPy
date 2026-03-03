.PHONY: help install install-dev lint tidy

help: ## Show available make targets and descriptions.
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install base project dependencies.
	python -m pip install .

install-dev: ## Install project with development dependencies.
	python -m pip install ".[dev]"

lint: ## Run Ruff lint checks.
	ruff check .

tidy: ## Run Ruff autoformatter.
	ruff format .
