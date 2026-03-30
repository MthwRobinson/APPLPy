TEST ?=

.PHONY: help
help: ## Show available make targets and descriptions.
	@awk 'BEGIN {FS = ":.*## "} \
		/^[a-zA-Z0-9_.-]+:.*## / { \
			target=$$1; \
			desc=$$2; \
			targets[++count]=target; \
			descs[count]=desc; \
			if (length(target) > maxlen) maxlen=length(target); \
		} \
		END { \
			for (i=1; i<=count; i++) printf "  %-" maxlen "s  %s\n", targets[i], descs[i]; \
		}' $(MAKEFILE_LIST)

.PHONY: install
install: ## Install base project dependencies.
	uv sync

.PHONY: install-dev
install-dev: ## Install project with development dependencies.
	uv sync --extra dev --extra rust

.PHONY: install-test
install-test: ## Install project with test dependencies.
	uv sync --extra test --extra dev --extra rust

.PHONY: install-all
install-all: ## Install project with all optional dependencies.
	uv sync --all-extras

.PHONY: install-rust
install-rust: ## Install Rust extension build tooling.
	uv sync --extra rust

.PHONY: build-dist-rust
build-dist-rust: ## Build Python distribution and Rust extension wheels into dist/.
	rm -rf dist
	uv build --out-dir dist
	uv run --no-sync maturin build -m rust/Cargo.toml --features extension-module --release --out dist

.PHONY: rust-develop
rust-develop: ## Build and install the Rust extension into the active uv environment.
	uv run --no-sync maturin develop -m rust/Cargo.toml --features extension-module

.PHONY: rust-check-import
rust-check-import: ## Verify APPLPy can call the Rust dummy function.
	uv run --no-sync python -c "from applpy.rust_bindings import dummy_ping; print(dummy_ping())"

.PHONY: cargo-test
cargo-test: ## Run rust cargo tests with Python runtime paths derived from the active uv environment.
	@set -eu; \
		pybin="$$(uv run --no-sync python -c 'import sys; print(sys.executable)')"; \
		libdir="$$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
		sitepkgs="$$(uv run --no-sync python -c 'import os, sysconfig; paths=[]; purelib=sysconfig.get_path("purelib"); platlib=sysconfig.get_path("platlib"); [paths.append(p) for p in (purelib, platlib) if p and p not in paths]; print(os.pathsep.join(paths))')"; \
		PYO3_PYTHON="$$pybin" \
		LD_LIBRARY_PATH="$${libdir}$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
		PYTHONPATH="$${sitepkgs}$${PYTHONPATH:+:$${PYTHONPATH}}" \
		cargo test --manifest-path rust/Cargo.toml

.PHONY: cargo-check
cargo-check: ## Run rust formatter check and clippy lint checks.
	cargo fmt --manifest-path rust/Cargo.toml --check
	cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings

.PHONY: cargo-tidy
cargo-tidy: ## Format rust code.
	cargo fmt --manifest-path rust/Cargo.toml

.PHONY: test
test: ## Run all tests, or one test when TEST=/path/to/test is provided.
	PYTHONPATH=. \
		uv run --no-sync \
		pytest --cov=applpy --cov-report=term-missing \
		$(if $(strip $(TEST)),$(TEST),test_applpy)

.PHONY: test-functional
test-functional: ## Run functional tests, or TEST=/path/to/test to target one test.
	PYTHONPATH = . \
		uv run --no-sync \
		pytest --cov=applpy --cov-report=term-missing \
		$(if $(strip $(TEST)),$(TEST),test_applpy/functional)

.PHONY: test-unit
test-unit: ## Run unit tests, or TEST=/path/to/test to target one test.
	PYTHONPATH=. \
		uv run --no-sync \
		pytest --cov=applpy --cov-report=term-missing \
		$(if $(strip $(TEST)),$(TEST),test_applpy/unit)

.PHONY: ipython
ipython: ## Runs iPython with --no-sync to ensure rust bindings are preserved
	uv run --no-sync ipython

.PHONY: check
check: ## Run Ruff lint checks.
	uv run --no-sync ruff check applpy test_applpy

.PHONY: tidy
tidy: ## Run Ruff autoformatter.
	uv run ruff check --fix applpy test_applpy
	uv run ruff format applpy test_applpy

.PHONY: docker-build
docker-build: ## Builds the docker image for the project
	docker build -f Dockerfile -t applpy:latest .

.PHONY: docker-run
docker-run: ## Runs the docker image and runs an interactive python session
	docker run --rm -it applpy:latest

.PHONY: docker-run-sh
docker-run-sh: ## Runs the docker image and runs an interactive shell session
	docker run --rm -it --entrypoint /bin/sh applpy:latest

.PHONY: docker-run-cmd
docker-run-cmd: ## Runs the docker image and runs the specified command
	docker run --rm -it --entrypoint /usr/bin/uv applpy:latest run $(CMD)

.PHONY: docker-run-jupyter
docker-run-jupyter: ## Runs the docker container and lauches a jupyter notebook
	docker run --rm -it \
		--entrypoint /usr/bin/uv \
		-p 8888:8888 \
		applpy:latest \
		run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=''
