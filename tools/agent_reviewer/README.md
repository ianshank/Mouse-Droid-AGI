# `tools/agent_reviewer/` - read-only Agent-SDK peer reviewer

F-059 WS-B (the reviewer) and WS-D (its guardrails). An adversarial reviewer that runs on the
workstation, reads a change set (`--base...--head`) through a fixed set of read-only tools, and
writes a review in the house shape. It spends real money, so every run is capped and ledgered.
Nothing here runs on the rover or is imported by it.

Architecture and trust boundary: `docs/architecture/c4-claude-workforce.md`. The containment
itself is `policy.py`; `tests/unit/tools/test_agent_reviewer.py` pins it.

## Trust model

The reviewer runs with an API key in its environment, against a head commit that may be **hostile**:
it controls every file in the working tree. So nothing the head controls may reach an exec sink, a process that
holds the key, or the reviewer's own instructions:

- the agent prompt (`.claude/agents/peer-reviewer.md`), the MCP declaration (`.mcp.json`) and the list of files
  `git show <rev>:<path>` may name are all read from the **base ref**, never the working tree;
- the MCP server entry is built **in code** from constants (`sys.executable -m mousedroid --config
  config/default.yaml`, a fixed minimal environment); `.mcp.json` at the base only has to *confirm* it, and a
  mismatch refuses the run;
- no interpreter is on the shell allowlist (`pytest --collect-only` imports the head's `conftest.py`);
- the MCP server must not be the branch's own code: the plan names where the interpreter finds `mousedroid`
  (`mcp_package:`), and a run is refused when that is the checkout under review (an editable install) and the
  checkout is not at the base commit (see step 3 below).

## What it may touch

| Surface | Allowed |
|---|---|
| Files | `Read`, `Grep`, `Glob` anywhere the user can read, **except** `*.env`, `docker.env*` and the usual credential directories (deny rules) |
| Shell | `git diff`, `git log`, `git show` - one plain command: no pipes, redirection, chaining, substitution, variables or unquoted globs. Pathspecs go after `--` and are literal. `git show <rev>:<path>` names only a file tracked at the base ref (read a file the change adds from the diff). No `--all`, `--reflog`, `--branches`, `--remotes`, `--output`, `--no-index` |
| MCP | `system_info`, `lidar_diagnostics`, `esp32_diagnostics` of the mock-hardware `mousedroid` server |
| Writes | none by the agent. The runner writes one review under `reports/agent_reviewer/<date>/wsb-NNNN.md` and one row in `reports/credit/ledger.jsonl` |
| Network | the Anthropic API, through the SDK's bundled Claude Code CLI, plus one free `models.retrieve` lookup |

## What it never does

- No `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, `WebFetch`, `WebSearch` or subagent tool; no `python`, no `pytest`.
- No actuation tool (`set_velocity`, `emergency_stop`, `translate_nl_mission`, `export_experience`,
  `tensorrt_compile`, `calibrate_ultrasonic`) - and the MCP server is started with
  `MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS=false` as a second lock.
- No `github` MCP server (it carries a token and can write), no rover host, no hardware.
- No credential on the command line - there is no such option. The key is read from
  `ANTHROPIC_API_KEY` in the environment and nowhere else, and is never printed, logged, placed in
  the options, or written to the artifact or the ledger.
- No user, project or local settings file (`setting_sources=[]`): nothing on this machine, and none of the head's
  `.claude/settings.json` hooks, can widen the allowlist. No `@path` expansion in prompts (`verbatim_prompts`).
- No spend without a pre-check: `charge()` runs before any client exists and raises `CapExceeded`.

## Run the first review (after the key is rotated)

Prerequisite: `NEXT_STEPS.md` item 1 - the chat-exposed key revoked and replaced. Do not run this with
the old key.

1. Install into the venv: `uv pip install --native-tls -e ".[agent_reviewer,anthropic]"`
   (`anthropic` is optional: without it the model alias is passed to the CLI unresolved and the row
   says `resolved_via: config`).
2. Export the new key in your shell from your own secret store: `ANTHROPIC_API_KEY`. Not in chat, not on a
   command line, not in a file this tool reads (it reads none).
3. Choose the interpreter and the checkout. The mock MCP server is started with the interpreter you run this with
   (`.venv/Scripts/python.exe -m ...`), so `mousedroid` must be installed in it - and **not as an editable install
   of the checkout under review**: `python -m mousedroid` would then run the branch's own code beside the CLI that
   holds the key. The dry-run prints `mcp_package:` (where the interpreter finds it) and a run is refused when that is
   the checkout and the checkout is not at the base commit. Two setups are fine: a venv whose `mousedroid` is a
   non-editable install built from the base ref (`uv pip install --native-tls ".[mcp]"` in a clean checkout of the
   base), used from a checkout at the head; or a checkout that is itself at the base commit, where the reviewer
   reads the head through git only. `--allow-in-tree-mcp` accepts the risk for a branch you would run anyway.
   Run from a checkout **at the head under review** when you can: if the working tree is elsewhere the reviewer is
   told not to trust `Read` for the head's files, which is slower and costs more turns.
4. Rehearse - builds everything, constructs no client, spends nothing, prints the allowlist,
   denylist, the MCP entry and process environment, prompt sizes, the artifact path and the credit headroom:

   ```bash
   .venv/Scripts/python.exe -m tools.agent_reviewer.run_review \
       --base origin/claude/markdown-implementation-plan-aVJ2l --head <PR-B branch> \
       --cap-usd 10 --opening "R3-1,R3-2,R3-3,R3-4" --dry-run
   ```

   A refusal that talks about the MCP server importing `mousedroid` from this checkout means step 3 is not
   satisfied yet; read the `mcp_package:` and `base_tracked_files:` lines of an accepted plan before a live run.

5. Run it: the same command without `--dry-run`. `--cap-usd` is this run's budget: it must fit in what is
   left of the workstream cap (`credit.caps_usd.B`), and the SDK stops the run at it. Consider a first run with
   a small cap (for example `--cap-usd 1`) to see the tool surface work before spending $10.
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
- The Bash guard is a whitelist over a small character set, not a shell parser; it fails closed. `git log -p` over the
  named revisions shows every file those commits touched, including a secret committed in the past and still
  reachable; env files are gitignored and CI scans history.
- **The key stays in the CLI's own environment.** The SDK lays `ClaudeAgentOptions.env` over the inherited
  environment, so the allowlisted `env` the reviewer adds (`PATH`, `HOME`, `LANG`, literal git pathspecs, mock hardware)
  cannot remove `ANTHROPIC_API_KEY` from it - and the CLI needs the key. What keeps it away from tree-controlled code is
  that no process the head can steer runs with it: the shell allowlist runs only `git`, and the MCP entry blanks the
  credential variables and is started with `PYTHONSAFEPATH=1` (honoured on Python 3.11+). The bundled CLI also has
  `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`, which strips credentials from Bash and MCP children; it is **not** enabled
  because the CLI ties it to forcing the permission mode down and to a subprocess sandbox, neither of which can be
  rehearsed without spending. Try it on the first small-cap run if you want the extra layer.
- The MCP server reads `config/default.yaml` from the working tree (data, not code).
  `.gitleaks.toml` is also read from the working tree; it feeds only the extra prompt screens, not the built-in
  credential-shape checks. So is `.claude/workforce.yaml`: its fields are validated (plain names, repo-relative
  paths) and none is an exec sink, but a hostile head can choose the artifact directory and the ledger path inside
  the checkout and the caps the pre-check reads - the per-run `--cap-usd` still bounds each run.
- On Python 3.10 `PYTHONSAFEPATH` is ignored, so a module planted in the repository root would shadow the
  standard library when the MCP server starts; run the reviewer on 3.11 or later.
- `git` runs with your own git configuration. A diff or text-conversion driver you have defined globally
  and that a head's `.gitattributes` names would run; the guard refuses `--ext-diff` and `--textconv`
  but git applies configured drivers by default.
- `model: opus` is an alias. `models.retrieve` may not accept a bare family alias; then the run falls back to
  `resolved_via: config` and the CLI resolves `opus` itself. The ids actually served are recorded in the artifact
  (`models served`). Set `agent_reviewer.model` to a full alias in `workforce.yaml` to get a `models.retrieve` row.
- The real-SDK pins in the test file need the extra and are skipped in CI (no job installs it).
