.PHONY: install test lint typecheck format clean build release-test release verify-release prepare-release version dev-full quickstart run-golden onboarding-check

install:
	pip install -e .[dev]

dev-full:
	pip install -e .[dev,dashboard,tuning]
	@echo ""
	@echo "========================================="
	@echo "Atropos development environment complete!"
	@echo "========================================="
	@echo ""
	@echo "Optional pruning framework setup:"
	@echo "  Wanda pruning: make setup-wanda"
	@echo "  Note: Wanda dependencies may conflict with main Atropos dependencies."
	@echo "        Consider using a separate environment for pruning experiments."

setup-wanda:
	python scripts/setup_wanda.py

test:
	pytest tests/ -v --cov=atropos --cov-report=term-missing

test-ci:
	pytest tests/ -v --cov=atropos --cov-report=term-missing -m "not integration"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

release-test: build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: build
	twine upload dist/*

verify-release: build
	pip install dist/*.whl
	python scripts/smoke_test.py

prepare-release:
	python scripts/prepare_release.py

version:
	@echo "Current version:"
	@grep '^version =' pyproject.toml | cut -d '"' -f2
	@grep '__version__' src/atropos/__init__.py | cut -d '"' -f2


quickstart:
	python -m venv .venv
	.venv/bin/python -m pip install --upgrade pip wheel
	.venv/bin/pip install -e .

run-golden:
	.venv/bin/atropos-llm preset medium-coder --strategy mild_pruning --report markdown

onboarding-check:
	.venv/bin/python scripts/smoke_test.py
