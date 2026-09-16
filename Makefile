# MouseDroid — developer entry points.
#
# Thin, discoverable wrappers over the existing tooling. Nothing here
# reimplements a gate: `scripts/ci.sh` stays the authoritative local superset
# and `.github/workflows/ci.yml` stays authoritative for CI. A target that
# drifted from its script would be worse than no target at all, so each one is
# either a single delegation or an ordered composite of those delegations
# (`test`, `gates`) — never a second, hand-maintained copy of a command.
#
# Two targets are held to a named CI job and must stay faithful to it:
#   `make test`  ≡ the four pytest steps of the blocking `test` job
#   `make gates` ⊇ every gate in the `local-gates` job
# Thresholds are read from their source of truth (COV_MIN below,
# .claude/workforce.yaml for `hooks`), never restated as a second literal.
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
.PHONY: help install install-isaac lint format typecheck test test-cov test-fast smoke \
        regression behaviour coverage branch-coverage validate skills boundaries \
        doc-budgets hooks gates ci clean

# `make test` is an ordered composite of the four pytest steps below, and they
# all write the same .coverage / .pytest_cache state. Under `-j` make would
# start them concurrently and they would corrupt each other's data files, so
# parallel execution is refused repo-wide: nothing here is CPU-bound work that
# -j would help (pytest-xdist inside a step is the way to parallelise).
.NOTPARALLEL:

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

install: ## Install with the extras CI uses (NOT a bare [dev] — see gate-ladder)
	$(PYTHON) -m pip install -e ".[dev,telemetry,mcp]"

install-isaac: ## Workstation Isaac Lab extra (NOT in CI; Linux + Isaac Sim)
	$(PYTHON) -m pip install -e ".[dev,telemetry,mcp,isaac]"

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

# The four pytest steps of the blocking `test` job in .github/workflows/ci.yml,
# one target each, plus `test` as the ordered composite CLAUDE.md advertises.
#
# `make test` used to run ONLY the coverage step, so a green `make test` was no
# evidence at all — it never executed the regression + e2e step, which is
# exactly the step PR #223 fails in CI after passing locally. Splitting the job
# into named targets (rather than one long recipe) keeps each command string in
# this file exactly once and lets a failing step be re-run in isolation during
# triage.
#
# The recipe text is deliberately literal rather than hidden behind tier/marker
# variables: tests/regression/test_ci_gate_wiring_aqa.py DISCOVERS every site
# that runs the orphan tiers by matching `pytest tests/functional` in the file
# text and pins its `-m` expression against ci.yml's. Behind a variable this
# Makefile would silently drop out of that parity check — the target would
# still be correct today and unguarded tomorrow.
test: test-cov regression smoke behaviour ## All 4 pytest steps of the blocking CI `test` job

test-cov: ## `test` step 1/4 — unit+property+integration under the coverage gate
	$(PYTHON) -m pytest tests/unit tests/property tests/integration \
		-m "not hardware" --import-mode=importlib \
		--cov=src/mousedroid --cov-report=term-missing --cov-fail-under=$(COV_MIN) -x

# Supersets CI's step, which filters `not slow`: ci.sh deliberately keeps the
# slow mypy-clean regression (test_pr105b_mypy_clean.py) in the local loop, and
# the slow arm e2e tests importorskip("mujoco") and skip cleanly without the
# [arm] extras. Running more than CI here is safe; running less is what the old
# `regression`-without-e2e target did.
regression: ## `test` step 2/4 — regression + e2e tiers (no coverage gate)
	$(PYTHON) -m pytest tests/regression tests/e2e -m "not hardware" \
		--import-mode=importlib --no-cov -q

smoke: ## `test` step 3/4 — sub-10s import/parse sanity tier
	$(PYTHON) -m pytest tests/smoke -m "not hardware and not slow" \
		--import-mode=importlib --no-cov -q

behaviour: ## `test` step 4/4 — parked-autonomous functional/user-journey + security (~2.5s)
	$(PYTHON) -m pytest tests/functional tests/user_journey tests/security \
		-m "not hardware and not slow" --import-mode=importlib --no-cov -q

coverage: test-cov ## Alias for the coverage-gated step alone (not the whole job)

branch-coverage: ## Changed-lines branch-coverage gate (needs a git diff base)
	$(PYTHON) scripts/check_branch_coverage.py --min $(COV_MIN) \
		--tests tests/unit tests/property tests/integration

validate: ## Spec-harness fast tier + the standalone value/settings gates
	$(PYTHON) scripts/validate.py --tier fast
	$(PYTHON) scripts/check_no_hardcoded_values.py
	$(PYTHON) scripts/check_settings_identity.py

skills: ## Validate .claude/skills/<name>/SKILL.md
	$(PYTHON) tools/validate_skill_commands.py

boundaries: ## Protocol-based DI subsystem boundary gate, full tree (local-gates parity)
	$(PYTHON) scripts/check_subsystem_boundaries.py

# The first of the three used to be the one gate in this file that NO CI job ran:
# only scripts/ci.sh checked root CLAUDE.md against DocsConfig.core_max_lines,
# and test_docs_trimmer.py exercises the tool against synthetic fixtures rather
# than the real file. It is now a real `local-gates` step in ci.yml, and
# tests/regression/test_ci_gate_wiring_aqa.py::TestEveryCiShGateReachesCi sweeps
# for the next gate that tries to live here alone. Kept in `make gates` so the
# local ladder still matches CI rather than deferring to it.
doc-budgets: ## CLAUDE.md size + doc hygiene + ratchet budgets, all strict
	$(PYTHON) -m tools.claude_hooks.docs_trimmer
	$(PYTHON) tools/doc_hygiene.py NEXT_STEPS.md --strict
	$(PYTHON) -m tools.ratchet_budgets --strict

# The threshold is READ from .claude/workforce.yaml (coverage.tools_line_min),
# exactly as the local-gates job does, instead of being restated here. Passing
# no --cov-fail-under silently applied pyproject's repo-wide fail_under = 90,
# so this target was enforcing a different number than CI by accident — and a
# literal 85 here would just move the accident one edit into the future.
# -m "not hardware" mirrors CI too: the hook suite has no hardware marks today,
# and the filter is what keeps that true by construction.
hooks: ## Workforce hook tests under workforce.yaml's own coverage gate
	min="$$($(PYTHON) -c 'from tools.claude_hooks.config import load_config; \
	print(load_config().coverage.tools_line_min)')"; \
	$(PYTHON) -m pytest tests/unit/tools/claude_hooks \
		-m "not hardware" --import-mode=importlib -q -o addopts="" \
		--cov=tools/claude_hooks --cov-branch --cov-report=term-missing \
		--cov-fail-under="$$min"

# Everything the `local-gates` CI job runs that is not already covered by the
# targets above (boundaries, doc-budgets, hooks), so a clean `make gates` means
# that job will be clean too. `validate` additionally runs `validate.py --tier
# fast`, which belongs to harness.yml rather than local-gates — a superset, not
# a divergence.
#
# Deliberately excludes branch-coverage: that target runs the full unit +
# property + integration suite to collect coverage (~4 min), which defeats the
# point of a fast fail-first bundle. Run `make branch-coverage` separately, or
# `make ci` for the authoritative superset. `hooks` is in scope despite being a
# pytest run: 272 tests in ~2s, and local-gates treats it as a deterministic
# gate rather than a test tier.
gates: lint format typecheck skills validate boundaries doc-budgets hooks ## Fast gates only, fail-fast (no src test suite)

ci: ## The authoritative local superset (scripts/ci.sh)
	bash scripts/ci.sh

clean: ## Remove caches and coverage artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov \
		.coverage .coverage.* coverage.xml coverage.json coverage-branch.json
	find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} +
