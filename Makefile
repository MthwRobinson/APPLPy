.PHONY: help install install-dev install-test install-rust install-all rust-develop rust-check-import test test-functional test-unit check tidy docker-build docker-run docker-run-jupter

TEST ?=

help: ## Show available make targets and descriptions.
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install base project dependencies.
	uv sync

install-dev: ## Install project with development dependencies.
	uv sync --extra dev --extra rust

install-test: ## Install project with test dependencies.
	uv sync --extra test --extra dev --extra rust

install-all: ## Install project with all optional dependencies.
	uv sync --all-extras

rust-develop: ## Build and install the Rust extension into the active uv environment.
	uv run --no-sync maturin develop -m rust/Cargo.toml

rust-check-import: ## Verify APPLPy can call the Rust dummy function.
	uv run --no-sync python -c "from applpy.rust_bindings import dummy_ping; print(dummy_ping())"

test: ## Run all tests, or one test when TEST=/path/to/test is provided.
	PYTHONPATH=. \
		uv run --no-sync \
		pytest --cov=applpy --cov-report=term-missing \
		$(if $(strip $(TEST)),$(TEST),test_applpy)

test-functional: ## Run functional tests, or TEST=/path/to/test to target one test.
	PYTHONPATH = . \
		uv run --no-sync \
		pytest --cov=applpy --cov-report=term-missing \
		$(if $(strip $(TEST)),$(TEST),test_applpy/functional)

test-unit: ## Run unit tests, or TEST=/path/to/test to target one test.
	PYTHONPATH=. \
		uv run --no-sync \
		pytest --cov=applpy --cov-report=term-missing \
		$(if $(strip $(TEST)),$(TEST),test_applpy/unit)

ipython: ## Runs iPython with --no-sync to ensure rust bindings are preserved
	uv run --no-sync ipython

check: ## Run Ruff lint checks.
	uv run ruff check applpy test_applpy

tidy: ## Run Ruff autoformatter.
	uv run ruff check --fix applpy test_applpy
	uv run ruff format applpy test_applpy

docker-build: ## Builds the docker image for the project
	docker build -f Dockerfile -t applpy:latest .

docker-run: ## Runs the docker image and runs an interactive python session
	docker run --rm -it applpy:latest

docker-run-sh: ## Runs the docker image and runs an interactive shell session
	docker run --rm -it --entrypoint /bin/sh applpy:latest

docker-run-cmd: ## Runs the docker image and runs the specified command
	docker run --rm -it --entrypoint /usr/bin/uv applpy:latest run $(CMD)

docker-run-jupyter: ## Runs the docker container and lauches a jupyter notebook
	docker run --rm -it \
		--entrypoint /usr/bin/uv \
		-p 8888:8888 \
		applpy:latest \
		run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=''
