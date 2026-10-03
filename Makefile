UV ?= uv
RUN = $(UV) run --locked

.PHONY: help sync lint format typecheck protos test test-all integration coverage check build audit benchmark

help:
	@echo 'make sync      Install the locked development environment'
	@echo 'make check     Run lint, types, protobuf, coverage, and package checks'
	@echo 'make test      Run fast tests with bounded fixtures'
	@echo 'make test-all  Run fast tests and integration checks'
	@echo 'make integration Run package, example, and process checks'
	@echo 'make format    Format project Python files'
	@echo 'make audit     Audit locked runtime dependencies for known vulnerabilities'
	@echo 'make benchmark Measure performance and save a local baseline'

sync:
	$(UV) sync --locked --all-extras

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .

typecheck:
	$(RUN) ty check

protos:
	$(RUN) python scripts/generate_protos.py --check

test:
	$(RUN) pytest

test-all:
	$(RUN) pytest -m 'not performance'

integration:
	$(RUN) pytest -m integration

coverage:
	$(RUN) pytest -m 'not performance' --cov --cov-config=pyproject.toml --cov-report=term-missing --cov-report=xml --junitxml=reports/tests.xml

check: lint typecheck protos coverage build

build:
	$(UV) build --out-dir reports/dist
	$(RUN) twine check --strict reports/dist/*

audit:
	mkdir -p reports
	$(UV) export --locked --all-extras --no-dev --no-emit-project --output-file reports/requirements-audit.txt
	$(RUN) pip-audit --requirement reports/requirements-audit.txt --require-hashes --disable-pip --strict

benchmark:
	$(RUN) pytest tests/benchmarks -m performance --benchmark-enable --benchmark-only --benchmark-save=local
