SHELL := /bin/bash

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip

SRC := src
NOTEBOOKS := notebooks

.PHONY: help venv install install-dev freeze run-notebook lint format test check \
        format-code format-notebooks lint-notebooks check-code check-notebooks \
        check-notebooks-format check-notebooks-lint clean clean-venv

help:
	@echo "Targets:"
	@echo "  make venv               - create virtual environment (if missing) and upgrade pip"
	@echo "  make install            - install runtime dependencies from requirements.txt"
	@echo "  make install-dev        - install dev tools (ruff/black/nbqa/pytest)"
	@echo "  make run-notebook       - start Jupyter"
	@echo "  make lint               - ruff check src"
	@echo "  make format             - format src + notebooks (nbqa)"
	@echo "  make test               - run pytest"
	@echo "  make check              - run CI-like checks (lint + format-check + notebook-check + tests)"
	@echo "  make clean              - remove caches"
	@echo "  make clean-venv         - remove virtual environment"

venv:
	@test -x $(PYTHON) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-dev: install
	$(PIP) install -r requirements-dev.txt

install: venv
	$(PIP) install -r requirements.txt

run-notebook: install-dev
	$(PYTHON) -m jupyter notebook --ip=0.0.0.0 --no-browser

# ---------- Lint / format (code) ----------
lint: install-dev
	$(PYTHON) -m ruff check $(SRC)

format-code: install-dev
	$(PYTHON) -m black $(SRC)

check-code: install-dev
	$(PYTHON) -m black --check $(SRC)
	$(PYTHON) -m ruff check $(SRC)

# ---------- Lint / format (notebooks) ----------
format-notebooks: install-dev
	nbqa black $(NOTEBOOKS)
	nbqa ruff $(NOTEBOOKS) --fix

check-notebooks-format: install-dev
	nbqa black --check $(NOTEBOOKS)

check-notebooks-lint: install-dev
	nbqa ruff $(NOTEBOOKS)

check-notebooks: check-notebooks-format check-notebooks-lint

# ---------- Tests ----------
test: install-dev
	$(PYTHON) -m pytest -q

# ---------- Meta targets ----------
format: format-code format-notebooks

check: check-code check-notebooks test

# ---------- Clean ----------
clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-venv:
	rm -rf $(VENV)

freeze: venv
	@echo "Warning: this freezes ALL installed packages into requirements.txt"
	$(PIP) freeze > requirements.txt

test: install-dev
	PYTHONPATH=. $(PYTHON) -m pytest -q
