.PHONY: install test lint typecheck format clean build release-test release verify-release prepare-release version

install:
	pip install -e .[dev]

setup-wanda:
	python scripts/setup_wanda.py

test:
	pytest tests/ -v --cov=atropos --cov-report=term-missing

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
