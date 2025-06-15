.PHONY: help install install-dev test lint format clean build upload docs

help:
	@echo "Available commands:"
	@echo "  make install      Install the package"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linters"
	@echo "  make format       Format code"
	@echo "  make clean        Clean build artifacts"
	@echo "  make build        Build distribution packages"
	@echo "  make upload       Upload to PyPI"
	@echo "  make docs         Generate documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test]"
	pre-commit install

test:
	pytest tests/ -v --cov=lesynthesis --cov-report=html --cov-report=term

lint:
	ruff check lesynthesis/
	mypy lesynthesis/
	black --check lesynthesis/
	isort --check-only lesynthesis/

format:
	black lesynthesis/
	isort lesynthesis/
	ruff check --fix lesynthesis/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

docs:
	@echo "Documentation generation not yet implemented" 