SHELL := /bin/bash

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip

.PHONY: help venv install freeze run-notebook lint format test clean

help:
	@echo "Targets:"
	@echo "  make venv        - create virtual environment and upgrade pip"
	@echo "  make install     - install dependencies from requirements.txt"
	@echo "  make freeze      - freeze installed packages to requirements.txt"
	@echo "  make run-notebook- start Jupyter (optional)"
	@echo "  make lint        - ruff lint (optional)"
	@echo "  make format      - black formatting (optional)"
	@echo "  make test        - run pytest (optional)"
	@echo "  make clean       - remove caches"

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

freeze:
	$(PIP) freeze > requirements.txt

run-notebook: venv
	$(PIP) install -r requirements.txt
	$(PYTHON) -m jupyter notebook --ip=0.0.0.0 --no-browser

lint: venv
	$(PIP) install ruff
	$(PYTHON) -m ruff check .

format: venv
	$(PIP) install black
	$(PYTHON) -m black .

test: venv
	$(PIP) install pytest
	$(PYTHON) -m pytest -q

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
