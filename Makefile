# MouseDroid — developer entry points.
#
# Thin, discoverable wrappers over the existing tooling. Nothing here
# reimplements a gate: `scripts/ci.sh` stays the authoritative local superset
# and `.github/workflows/ci.yml` stays authoritative for CI. A target that
# drifted from its script would be worse than no target at all, so each one is
# a single delegation.
#
# The ordered ladder, its rationale, and the failure-triage table live in
# `.claude/skills/gate-ladder/SKILL.md`.

# Resolve Python exactly as scripts/ci.sh does, in the same order: an explicit
# MOUSEDROID_PYTHON wins, then the project venv (Windows layout first, matching
# ci.sh), then whatever is on PATH. Diverging here would let `make lint` and
# `bash scripts/ci.sh` run different interpreters — and therefore different
# pinned ruff/mypy versions — on the same checkout.
PYTHON ?= $(shell \
	if [ -n "$$MOUSEDROID_PYTHON" ]; then echo "$$MOUSEDROID_PYTHON"; \
	elif [ -x ./.venv/Scripts/python.exe ]; then echo ./.venv/Scripts/python.exe; \
	elif [ -x ./.venv/bin/python ]; then echo ./.venv/bin/python; \
	else command -v python3 2>/dev/null || command -v python 2>/dev/null; fi)

ifeq ($(strip $(PYTHON)),)
$(error No Python interpreter found. Set MOUSEDROID_PYTHON or install Python.)
endif

# Directories linted / formatted by CI. Kept as variables so a target and the
# workflow cannot disagree about scope by accident.
LINT_DIRS := src/ tests/ tools/
COV_MIN   := 90

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-fast smoke regression \
        behaviour coverage branch-coverage validate skills hooks gates ci clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

install: ## Install with the extras CI uses (NOT a bare [dev] — see gate-ladder)
	$(PYTHON) -m pip install -e ".[dev,telemetry,mcp]"

lint: ## ruff check over src/ tests/ tools/ (+ scripts/)
	$(PYTHON) -m ruff check $(LINT_DIRS)
	$(PYTHON) -m ruff check scripts/

format: ## ruff format --check (CI runs the same scope)
	$(PYTHON) -m ruff format --check $(LINT_DIRS)

typecheck: ## mypy --strict over src/ and the workforce hook package
	$(PYTHON) -m mypy src/ --strict --ignore-missing-imports
	MYPYPATH=. $(PYTHON) -m mypy tools/claude_hooks/ --strict \
		--ignore-missing-imports --explicit-package-bases

test-fast: ## Unit + property + integration, no coverage (quickest real signal)
	$(PYTHON) -m pytest tests/unit tests/property tests/integration \
		-m "not hardware" --import-mode=importlib --no-cov -q

test: ## The same tiers with the repo coverage gate applied
	$(PYTHON) -m pytest tests/unit tests/property tests/integration \
		-m "not hardware" --import-mode=importlib \
		--cov=src/mousedroid --cov-report=term-missing --cov-fail-under=$(COV_MIN)

smoke: ## Sub-10s import/parse sanity tier
	$(PYTHON) -m pytest tests/smoke -m "not hardware and not slow" \
		--import-mode=importlib --no-cov -q

regression: ## Regression + AQA tier
	$(PYTHON) -m pytest tests/regression/ -m "not hardware" --import-mode=importlib -q

behaviour: ## Functional + user-journey + security tiers (~2.5s, F-028)
	$(PYTHON) -m pytest tests/functional tests/user_journey tests/security \
		-m "not hardware and not slow" --import-mode=importlib --no-cov -q

coverage: test ## Alias for the coverage-gated test run

branch-coverage: ## Changed-lines branch-coverage gate (needs a git diff base)
	$(PYTHON) scripts/check_branch_coverage.py --min $(COV_MIN) \
		--tests tests/unit tests/property tests/integration

validate: ## Spec-harness fast tier + the standalone value/settings gates
	$(PYTHON) scripts/validate.py --tier fast
	$(PYTHON) scripts/check_no_hardcoded_values.py
	$(PYTHON) scripts/check_settings_identity.py

skills: ## Validate .claude/skills/<name>/SKILL.md
	$(PYTHON) tools/validate_skill_commands.py

hooks: ## Workforce hook tests under their own coverage gate
	$(PYTHON) -m pytest tests/unit/tools/claude_hooks -o addopts="" \
		--cov=tools/claude_hooks --cov-branch --cov-report=term-missing -q

# Deliberately excludes branch-coverage: that target runs the full unit +
# property + integration suite to collect coverage (~4 min), which defeats the
# point of a fast fail-first bundle. Run `make branch-coverage` separately, or
# `make ci` for the authoritative superset.
gates: lint format typecheck skills validate ## Fast gates only, fail-fast (no test suite)

ci: ## The authoritative local superset (scripts/ci.sh)
	bash scripts/ci.sh

clean: ## Remove caches and coverage artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov \
		.coverage .coverage.* coverage.xml coverage.json coverage-branch.json
	find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} +
