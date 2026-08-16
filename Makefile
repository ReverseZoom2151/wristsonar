# POSIX make. Works with GNU make under Git Bash on Windows and under a normal
# shell on Linux and macOS. Run every recipe through sh so that Windows does
# not hand the commands to cmd.exe.
SHELL := /bin/sh
.SHELLFLAGS := -eu -c

# Pick an interpreter that actually runs. Windows ships a "python3" alias stub
# that exists on PATH but fails when executed, so test by running it rather
# than by looking it up. Override with `make PYTHON=... <target>`.
PYTHON ?= $(shell python -c "" >/dev/null 2>&1 && echo python || echo python3)
PIP := $(PYTHON) -m pip

.DEFAULT_GOAL := check
.PHONY: help install lint typecheck test check format

help:
	@echo "install    install the package in editable mode with its dev extra"
	@echo "lint       ruff check"
	@echo "typecheck  mypy"
	@echo "test       pytest, no dataset required"
	@echo "check      lint, typecheck and test"
	@echo "format     ruff format and ruff check --fix"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

check: lint typecheck test

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .
