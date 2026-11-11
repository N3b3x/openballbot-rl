.PHONY: install install-dev test test-cov lint format format-check type-check clean docs test-ros2 check-docs check-site

# Targets install, install-dev, test, test-cov, lint, format, format-check, type-check, train and eval belong to the legacy SB3 path
# (ballbot_gym, ballbot_rl, configs/). The current GPU pipeline is run from experiments/warp_ppo; see docs/pipeline.md.
# test-ros2, check-docs and check-site cover the current ROS 2 package, the documentation and the website.

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

test-ros2:
	cd ros2/ballbot_control && python -m pytest -q

check-docs:
	python scripts/docs/check_docs.py

check-site:
	python scripts/site/check_links.py

docs: check-docs

train:
	ballbot-train --config configs/train/ppo_directional.yaml

eval:
	ballbot-eval --algo ppo --path outputs/experiments/archived_models/2025-12-03_ppo-perlin-directional-5.2M-steps/best_model.zip --n_test 5

