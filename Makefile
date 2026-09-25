# Configuration
#--------------

PYTHON := uv run python

MAIN := src

MYPY_FLAGS := --python-version 3.13 \
			  --warn-return-any \
              --warn-unused-ignores \
              --ignore-missing-imports \
              --disallow-untyped-defs \
              --check-untyped-defs


# Rules
#------

install:
	uv sync

run:
	$(PYTHON) -m $(MAIN)

debug:
	$(PYTHON) -m pdb -m $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 src
	uv run mypy src $(MYPY_FLAGS)

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

.PHONY: install run debug clean lint lint-strict