.PHONY: install test lint typecheck format clean build

install:
	pip install -e .[dev]

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
