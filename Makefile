.PHONY: setup up down ingest run test lint eval fmt

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(VENV)/bin/pre-commit install || true

up:
	docker compose up -d qdrant redis postgres

down:
	docker compose down

ingest:
	PYTHONPATH=src $(PY) scripts/ingest.py --all

run:
	PYTHONPATH=src $(VENV)/bin/uvicorn nexora_rag.api.main:app --reload --app-dir src

test:
	PYTHONPATH=src $(VENV)/bin/pytest tests/ -v --cov=src/nexora_rag --cov-report=term-missing

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src

fmt:
	$(VENV)/bin/ruff format src tests

eval:
	PYTHONPATH=src $(PY) scripts/evaluate.py
