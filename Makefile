UV ?= uv
UVX ?= uvx
RUN = $(UV) run --locked

.PHONY: help install shell sync lint format typecheck protos test test-all integration coverage check build publish audit benchmark

help:
	@echo 'make shell     Run the local checkout with uvx'
	@echo 'make install   Install the checkout as an editable CLI tool'
	@echo 'make sync      Install the locked development environment'
	@echo 'make check     Run lint, types, protobuf, coverage, and package checks'
	@echo 'make test      Run fast tests with bounded fixtures'
	@echo 'make test-all  Run fast tests and integration checks'
	@echo 'make integration Run package, example, and process checks'
	@echo 'make format    Format project Python files'
	@echo 'make build     Build and validate fresh wheel and source distributions'
	@echo 'make publish   Build, validate, and publish to PyPI'
	@echo 'make audit     Audit locked runtime dependencies for known vulnerabilities'
	@echo 'make benchmark Measure performance and save a local baseline'

install:
	$(UV) tool install --editable .

shell:
	$(UVX) --from . premixdb shell

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
	rm -rf reports/dist
	$(UV) build --out-dir reports/dist
	$(RUN) twine check --strict reports/dist/*

publish:
	@test -n "$$UV_PUBLISH_TOKEN" || (echo 'Set UV_PUBLISH_TOKEN to a PyPI API token.' >&2; exit 1)
	$(MAKE) build
	env -u UV_PUBLISH_USERNAME -u UV_PUBLISH_PASSWORD $(UV) publish --trusted-publishing never reports/dist/*

audit:
	mkdir -p reports
	$(UV) export --locked --all-extras --no-dev --no-emit-project --output-file reports/requirements-audit.txt
	$(RUN) pip-audit --requirement reports/requirements-audit.txt --require-hashes --disable-pip --strict

benchmark:
	$(RUN) pytest tests/benchmarks -n 0 -m performance --benchmark-enable --benchmark-only --benchmark-save=local
