# `tools/agent_reviewer/` - read-only Agent-SDK peer reviewer

F-059 WS-B (the reviewer) and WS-D (its guardrails). An adversarial reviewer that runs on the
workstation, reads a change set (`--base...--head`) through a fixed set of read-only tools, and
writes a review in the house shape. It spends real money, so every run is capped and ledgered.
Nothing here runs on the rover or is imported by it.

Architecture and trust boundary: `docs/architecture/c4-claude-workforce.md`. The containment
itself is `policy.py`; `tests/unit/tools/test_agent_reviewer.py` pins it.

## What it may touch

| Surface | Allowed |
|---|---|
| Files | `Read`, `Grep`, `Glob` anywhere the user can read, **except** `*.env`, `docker.env*` and the usual credential directories (deny rules) |
| Shell | `git diff`, `git log`, `git show`; `python -m pytest <paths> --collect-only` - one plain command, no pipes, redirection, chaining, substitution or variables |
| MCP | `system_info`, `lidar_diagnostics`, `esp32_diagnostics` of the `mousedroid` server, built from `.mcp.json` with `MOUSEDROID_MOCK_HARDWARE=true` forced |
| Writes | none by the agent. The runner writes one review under `reports/agent_reviewer/<date>/wsb-NNNN.md` and one row in `reports/credit/ledger.jsonl` |
| Network | the Anthropic API, through the SDK's bundled Claude Code CLI, plus one free `models.retrieve` lookup |

## What it never does

- No `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, `WebFetch`, `WebSearch` or subagent tool.
- No actuation tool (`set_velocity`, `emergency_stop`, `translate_nl_mission`, `export_experience`,
  `tensorrt_compile`, `calibrate_ultrasonic`) - and the MCP server is started with
  `MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS=false` as a second lock.
- No `github` MCP server (it carries a token and can write), no rover host, no hardware.
- No credential on the command line - there is no such option. The key is read from
  `ANTHROPIC_API_KEY` in the environment and nowhere else, and is never printed, logged, placed in
  the options, or written to the artifact or the ledger.
- No user, project or local settings file (`setting_sources=[]`): nothing on this machine can widen the
  allowlist. No `@path` expansion in prompts (`verbatim_prompts`).
- No spend without a pre-check: `charge()` runs before any client exists and raises `CapExceeded`.

## Run the first review (after the key is rotated)

Prerequisite: `NEXT_STEPS.md` item 1 - the chat-exposed key revoked and replaced. Do not run this with
the old key.

1. Install into the venv: `uv pip install --native-tls -e ".[agent_reviewer,anthropic]"`
   (`anthropic` is optional: without it the model alias is passed to the CLI unresolved and the row
   says `resolved_via: config`).
2. Export the new key in your shell from your own secret store: `ANTHROPIC_API_KEY`. Not in chat, not on a
   command line, not in a file this tool reads (it reads none).
3. Run from an activated venv (the mock MCP server is started as `python -m mousedroid`) in a checkout
   or worktree **at the head under review**. If the working tree is elsewhere the reviewer is told to read
   files with `git show <head>:<path>`, which is slower and costs more turns.
4. Rehearse - builds everything, constructs no client, spends nothing, prints the allowlist,
   denylist, prompt sizes, the artifact path and the credit headroom:

   ```bash
   python -m tools.agent_reviewer.run_review \
       --base origin/claude/markdown-implementation-plan-aVJ2l --head <PR-B branch> \
       --cap-usd 10 --opening "R3-1,R3-2,R3-3,R3-4" --dry-run
   ```

5. Run it: the same command without `--dry-run`. `--cap-usd` is this run's budget: it must fit in what is
   left of the workstream cap (`credit.caps_usd.B`), and the SDK stops the run at it.
6. `python -m tools.credit_ledger status`; read the artifact; commit the artifact and the ledger row.

## How a run is costed and recorded

The cost is `ResultMessage.total_cost_usd` (the CLI's whole-tree figure, subagents included; present on error
results too). It, and the SDK's `max_budget_usd` that stops the run, are the CLI's client-side estimates, not the
invoice - and the stop lands after the response that crosses the budget, so a run can end a little over its cap.
Compare the ledger with the console after the first run.

When there is no trustworthy number the row is charged the run cap (`cost_source: cap_assumed`) rather than 0:
the cost is missing, the run died after starting with no result, or the CLI reports 0 after a session crash (it
zeroes its cost fields there) or beside token usage (a model missing from its price table). An over-count is
fixed by reading the console; an under-count lets the next run overspend. A run that never started costs 0
(`not_started`). A review that looks like it contains a credential is **withheld** (status `output_withheld`,
no text saved) but still charged.

Exit codes: `0` reviewed and recorded; `1` the review did not succeed, or its ledger row could not be written
(the row is printed - append it by hand); `2` refused before any spend; `3` the cap would be exceeded; `130`
interrupted. Statuses: `ok`, `error_max_turns`, `error_max_budget_usd`, `error_during_execution`, `error_api`,
`error`, `error:<ExceptionType>`, `timeout`, `interrupted`, `output_withheld`.

`max_turns`, the model alias, the caps and the ledger path live in `.claude/workforce.yaml`
(`agent_reviewer:` and `credit:`), never in code.

## Known limits

- `Read` is not confined to the repository. The deny rules cover env files and credential directories; run the
  review from an account that holds nothing it should not read.
- The Bash guard is a whitelist over a small character set, not a shell parser; it fails closed.
- `python -m pytest --collect-only` imports the repository's test modules, i.e. runs its import-time code with the
  API key in the environment. Only review branches you would run.
- `model: opus` is an alias. `models.retrieve` may not accept a bare family alias; then the run falls back to
  `resolved_via: config` and the CLI resolves `opus` itself. The ids actually served are recorded in the artifact
  (`models served`). Set `agent_reviewer.model` to a full alias in `workforce.yaml` to get a `models.retrieve` row.
- The real-SDK pins in the test file need the extra and are skipped in CI (no job installs it).
