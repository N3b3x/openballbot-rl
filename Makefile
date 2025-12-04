.PHONY: install install-dev test lint format type-check clean docs

install:
	bash scripts/setup/install.sh

install-dev:
	bash scripts/setup/install.sh --dev

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=ballbot_gym --cov=ballbot_rl --cov-report=html

lint:
	ruff check ballbot_gym ballbot_rl tests

format:
	black ballbot_gym ballbot_rl tests scripts

format-check:
	black --check ballbot_gym ballbot_rl tests scripts

type-check:
	mypy ballbot_gym ballbot_rl --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

docs:
	# Add documentation generation command if using Sphinx/MkDocs
	@echo "Documentation generation not yet configured"

train:
	ballbot-train --config configs/train/ppo_directional.yaml

eval:
	ballbot-eval --algo ppo --path outputs/models/example_model.zip --n_test 5

