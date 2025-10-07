# Makefile to help automate key steps

.DEFAULT_GOAL := help
# Will likely fail on Windows, but Makefiles are in general not Windows
# compatible so we're not too worried
TEMP_FILE := $(shell mktemp)

VENV_DIR ?= venv

# A helper script to get short descriptions of each target in the Makefile
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([\$$\(\)a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-30s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


.PHONY: help
help:  ## print short description of each target
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: checks $(VENV_DIR)  ## run all the checks
checks:  ## run all the linting checks of the codebase
	@echo "=== pre-commit ==="; $(VENV_DIR)/bin/pre-commit run --all-files || echo "--- pre-commit failed ---" >&2; \
		echo "=== mypy ==="; MYPYPATH=stubs $(VENV_DIR)/bin/mypy src || echo "--- mypy failed ---" >&2; \
		echo "======"

.PHONY: ruff-fixes $(VENV_DIR) 
ruff-fixes:  ## fix the code using ruff
    # format before and after checking so that the formatted stuff is checked and
    # the fixed stuff is formatted
	$(VENV_DIR)/bin/ruff format src tests scripts docs
	$(VENV_DIR)/bin/ruff check src tests scripts docs --fix
	$(VENV_DIR)/bin/ruff format src tests scripts docs

.PHONY: test
test:  ## run the tests
	$(VENV_DIR)/bin/pytest src tests -r a -v --doctest-modules --doctest-report ndiff --cov=src

# Note on code coverage and testing:
# You must specify cov=src.
# Otherwise, funny things happen when doctests are involved.
# If you want to debug what is going on with coverage,
# we have found that adding COVERAGE_DEBUG=trace
# to the front of the below command
# can be very helpful as it shows you
# if coverage is tracking the coverage
# of all of the expected files or not.
# We are sure that the coverage maintainers would appreciate a PR
# that improves the coverage handling when there are doctests
# and a `src` layout like ours.

.PHONY: docs $(VENV_DIR)
docs:  ## build the docs
	$(VENV_DIR)/bin/mkdocs build

.PHONY: docs-strict $(VENV_DIR)
docs-strict:  ## build the docs strictly (e.g. raise an error on warnings, this most closely mirrors what we do in the CI)
	$(VENV_DIR)/bin/mkdocs build --strict

.PHONY: docs-serve $(VENV_DIR)
docs-serve:  ## serve the docs locally
	$(VENV_DIR)/bin/mkdocs serve

.PHONY: changelog-draft $(VENV_DIR)
changelog-draft:  ## compile a draft of the next changelog
	$(VENV_DIR)/bin/towncrier build --draft --version draft

.PHONY: licence-check $(VENV_DIR)
licence-check:  ## Check that licences of the dependencies are suitable
	# Will likely fail on Windows, but Makefiles are in general not Windows
	# compatible so we're not too worried
	$(VENV_DIR)/bin/pip freeze > $(TEMP_FILE)
	$(VENV_DIR)/bin/liccheck -r $(TEMP_FILE) -R licence-check.txt
	rm -f $(TEMP_FILE)

virtual-environment: $(VENV_DIR)  ## update venv, create a new venv if it doesn't exist make
	[ -d $(VENV_DIR) ] || python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip wheel
	$(VENV_DIR)/bin/pip install -e .[all-dev]
	touch $(VENV_DIR)

clean: $(VENV_DIR)
	touch pyptoject.toml

first-venv: ## create a new virtual environment for the very first repo setup
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install setuptools.scm
	# don't touch here as we don't want this venv to persist anyway