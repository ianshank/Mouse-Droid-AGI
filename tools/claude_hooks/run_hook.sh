#!/usr/bin/env bash
# Run a Claude Code hook under an interpreter that can actually import it.
#
# Why this wrapper exists
# ----------------------
# `.claude/settings.json` used to wire its hooks as `python3 -m
# tools.claude_hooks.<mod>`. `python3` is whichever interpreter is first on the
# hook process's PATH -- on this project's containers, the *system* one, which
# does not have the hook package's dependencies (pydantic, PyYAML) installed.
# The import raised, the hook exited 1, and Claude Code treats a non-2 exit from
# a PreToolUse hook as a non-blocking hook *error*: the gate failed open. The
# F-008 freeze gate and the edit-time secret scan were both bypassed with no
# visible symptom, because a gate that never blocks looks exactly like a gate
# with nothing to block.
#
# So the interpreter is resolved by *capability*, not by PATH order: probe
# candidates until one can import `tools.claude_hooks.config` -- the module every
# wired hook imports, and the one that pulls in pydantic and PyYAML. Probing the
# real import beats pointing at a hardcoded `.venv/bin/python`, because it also
# rejects a virtualenv that exists but never had the dev extra installed.
#
# This is a transparent `python` shim: every argument is forwarded untouched, so
# the wired command still reads `-m tools.claude_hooks.<mod>` and the hook keeps
# the repository root on `sys.path`.
#
#   bash "$CLAUDE_PROJECT_DIR/tools/claude_hooks/run_hook.sh" \
#       -m tools.claude_hooks.freeze_gate
#
# Override: set MOUSEDROID_PYTHON and it is used verbatim, with no probe -- an
# operator naming an interpreter is an instruction, not a hint.
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CLAUDE_PROJECT_DIR is what Claude Code exports; the script-relative fallback is
# what makes this runnable by hand and from the test suite.
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "${_script_dir}/../.." && pwd)}"
cd "${PROJECT_DIR}"

_warn() {
	printf 'run_hook.sh: %s\n' "$1" >&2
}

# The canary import, not a dependency name list: it stays correct when the hook
# package's dependencies change. stdin is redirected because the caller's stdin
# carries the hook payload JSON and must reach the real hook unread.
_can_import() {
	"$1" -c 'import tools.claude_hooks.config' </dev/null >/dev/null 2>&1
}

_resolve_interpreter() {
	if [ -n "${MOUSEDROID_PYTHON:-}" ]; then
		printf '%s\n' "${MOUSEDROID_PYTHON}"
		return 0
	fi

	local -a candidates=()
	# Guarded rather than interpolated unconditionally: with VIRTUAL_ENV unset,
	# "${VIRTUAL_ENV:-}/bin/python" is "/bin/python", which on some hosts exists
	# and is emphatically not a project virtualenv.
	if [ -n "${VIRTUAL_ENV:-}" ]; then
		candidates+=("${VIRTUAL_ENV}/bin/python")
	fi
	candidates+=("${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/venv/bin/python")
	local on_path
	for on_path in python3 python; do
		if command -v "${on_path}" >/dev/null 2>&1; then
			candidates+=("$(command -v "${on_path}")")
		fi
	done

	local candidate first_executable=""
	for candidate in "${candidates[@]}"; do
		[ -x "${candidate}" ] || continue
		if [ -z "${first_executable}" ]; then
			first_executable="${candidate}"
		fi
		if _can_import "${candidate}"; then
			printf '%s\n' "${candidate}"
			return 0
		fi
	done

	if [ -n "${first_executable}" ]; then
		# Preserves the pre-existing behaviour (the hook runs, fails to import,
		# and the gate fails open) but says so out loud instead of silently.
		_warn "no interpreter could import tools.claude_hooks.config; falling back to ${first_executable} -- gates will fail open. Install the dev extra into a virtualenv, or set MOUSEDROID_PYTHON."
		printf '%s\n' "${first_executable}"
		return 0
	fi
	return 1
}

if [ "$#" -eq 0 ]; then
	# Without this an argument-less wiring would exec a bare interpreter, which
	# reads the hook payload from stdin as a REPL script and then blocks until
	# Claude Code's hook timeout — a hang, not an error anyone can read.
	_warn "no arguments: expected an interpreter argv such as '-m tools.claude_hooks.freeze_gate'"
	exit 1
fi

if ! interpreter="$(_resolve_interpreter)"; then
	_warn "no Python interpreter found in a project virtualenv or on PATH"
	exit 1
fi

exec "${interpreter}" "$@"
