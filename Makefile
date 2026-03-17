.PHONY: train eval test lint format clean setup

setup:
	pip install -r requirements.txt

train:
	python scripts/train.py --config configs/default.yaml

eval:
	python scripts/evaluate.py --config configs/default.yaml

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/ scripts/
	ruff check src/ tests/ scripts/ --fix

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov
