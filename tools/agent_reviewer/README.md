# `tools/agent_reviewer/` - read-only Agent-SDK peer reviewer

F-059 WS-B (the reviewer) and WS-D (its guardrails). An adversarial reviewer that runs on the
workstation, reads a change set (`--base...--head`) through a fixed set of read-only tools, and
writes a review in the house shape. It spends real money, so every run is capped and ledgered.
Nothing here runs on the rover or is imported by it.

Architecture and trust boundary: `docs/architecture/c4-claude-workforce.md`. The containment
itself is `policy.py`; `tests/unit/tools/test_agent_reviewer.py` pins it.

## Trust model

The reviewer runs with an API key in its environment, against a head commit that may be **hostile**:
it controls every file in the working tree. So the trust boundary is the **base ref**, and nothing the
head controls may reach an exec sink, a process that holds the key, or the reviewer's own instructions.

- **Everything the reviewer configures itself from is read from the base commit** (`git cat-file`), never from
  the working tree: `.claude/workforce.yaml` (it selects the agent file, the MCP declaration, the secret-scan
  program and its policy file, the caps, and where the review and the ledger row are written), the agent prompt
  `.claude/agents/peer-reviewer.md`, `.mcp.json`, the MCP server's `config/default.yaml` and `.gitleaks.toml`.
  The files `git show <commit>:<path>` may name are the ones tracked there.
- **The MCP server entry is built in code** from constants (`sys.executable -I`, a fixed bootstrap, a fixed
  minimal environment). The base's `.mcp.json` only has to *confirm* the pinned invocation; a mismatch refuses
  the run. `-I` keeps the working directory and every `PYTHON*` variable off the import path (on every Python),
  and the bootstrap runs the server from a scratch directory outside the checkout, with a copy of the
  base's configuration.
- **The shell is `git diff|log|show` and nothing else** - no interpreter (`pytest --collect-only` imports the
  head's `conftest.py`). Commits are named only by the two pinned ids, so a bare blob or tree id cannot be used
  to print an object whatever its path.
- **`git` and `gitleaks` are found in absolute `PATH` directories only.** Windows (and `shutil.which`) try the
  working directory first, so a `git.exe` committed to the head would otherwise run with the key in its
  environment. The CLI is told `NoDefaultCurrentDirectoryInExePath=1` for the programs it starts.
- **The code that runs beside the key must not be the branch's.** That is the reviewer's own modules
  (`tools/agent_reviewer`), the interpreter's environment (its `site-packages` and `.pth` files run at startup,
  even under `-I`) and the `mousedroid` package the MCP server imports (the checkout's own source under an
  editable install). When any of them lives inside the checkout under review and the checkout is not at the
  base commit, the run is refused before any spend unless you pass `--trust-checkout`. An interpreter
  environment inside the checkout in which the head *tracks* files is refused regardless: it is not yours.

## What it may touch

| Surface | Allowed |
|---|---|
| Files | `Read`, `Grep`, `Glob` anywhere the user can read, **except** `*.env`, `docker.env*` and the usual credential directories (deny rules) |
| Shell | `git diff`, `git log`, `git show` - one plain command: no pipes, redirection, chaining, substitution, variables or unquoted globs. Commits are the two full ids pinned in the task, with `^`, `^N` or `~N` (`A..B` ranges too): not `HEAD`, a branch, an abbreviation or a blob id. Option values are attached (`-n5`, `--grep=text`). Pathspecs go after `--` and are literal. `git show <commit>:<path>` names only a file tracked at the base (read a file the change adds from the diff). No `--all`, `--reflog`, `--branches`, `--remotes`, `--raw`, `--abbrev`, `--output`, `--no-index` |
| MCP | `system_info`, `lidar_diagnostics`, `esp32_diagnostics` of the mock-hardware `mousedroid` server |
| Writes | none by the agent. The runner writes one review under `reports/agent_reviewer/<date>/wsb-NNNN.md` and one row in `reports/credit/ledger.jsonl` (paths from the base's `workforce.yaml`) |
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

1. **Set up three directories that do not contain one another.** The trusted code, the head under review, and
   the virtualenv:

   ```bash
   git worktree add ../reviewer-code <trusted-commit>   # a commit YOU trust that has this tool
   git worktree add ../review-head <head-branch>        # the head under review (Read/Grep see it)
   uv venv ../reviewer-venv
   uv pip install --native-tls --python ../reviewer-venv/Scripts/python.exe \
       "../reviewer-code[agent_reviewer,anthropic,mcp]"  # NOT editable: mousedroid from the trusted code
   ```

   `<trusted-commit>` is today the tip of this PR's branch (or any commit you have read), and trunk once the tool
   is merged; it need not be the review's base. (`anthropic` is optional: without it the model alias is passed to
   the CLI unresolved and the row says `resolved_via: config`.) Nothing the head controls is then the reviewer's
   code, its interpreter or the MCP server's package, and no flag is needed. Two shortcuts exist. A checkout that
   is itself at the review's base commit is trusted wholesale (the reviewer then reads the head through git only).
   Running from the head's own checkout and `.venv` works with `--trust-checkout` - only for a branch you would run
   anyway, since that branch's `tools/agent_reviewer`, `.venv` and `mousedroid` then run beside the key.
2. Export the new key in your shell from your own secret store: `ANTHROPIC_API_KEY`. Not in chat, not on a
   command line, not in a file this tool reads (it reads none).
3. Rehearse - builds everything, constructs no client, spends nothing, prints the allowlist, denylist, the MCP
   entry and process environment, where each piece of code comes from, prompt sizes, the artifact path and the
   credit headroom:

   ```bash
   cd ../reviewer-code
   ../reviewer-venv/Scripts/python.exe -m tools.agent_reviewer.run_review \
       --repo-root ../review-head \
       --base origin/claude/markdown-implementation-plan-aVJ2l --head <PR-B branch> \
       --cap-usd 10 --opening "R3-1,R3-2,R3-3,R3-4" --dry-run
   ```

   Read the `workforce_config:`, `trust_controller:`, `trust_interpreter:` and `trust_package:` lines of the plan
   before a live run. A refusal that talks about code inside this checkout means step 1 is not satisfied yet.
4. Run it: the same command without `--dry-run`. `--cap-usd` is this run's budget: it must fit in what is
   left of the workstream cap (`credit.caps_usd.B` in the base's `workforce.yaml`), and the SDK stops the run at
   it. The base ref must therefore carry the `credit:` block (and `agent_reviewer:`, the agent file, `.mcp.json`
   and `config/default.yaml`): a base without `workforce.yaml` gets the schema defaults, which cap nothing and so
   refuse to spend. Consider a first run with a small cap (for example `--cap-usd 1`) to see the tool surface work
   before spending $10.
5. Read the artifact and the new row of `reports/credit/ledger.jsonl` (both under `../review-head`; the run prints
   them), compare the cost with the console, and commit them on the head branch. Do not run the head's own
   `tools/credit_ledger.py` while the key is exported.

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
(`agent_reviewer:` and `credit:`), never in code - and are read from the base ref.

## Known limits

- `Read` is not confined to the repository. The deny rules cover env files and credential directories; run the
  review from an account that holds nothing it should not read. They are matches on the path as written, so other
  spellings of the same file (Windows 8.3 short names, `::$DATA` streams, a symlink the head commits) are not
  enumerated.
- The Bash guard is a whitelist over a small character set, not a shell parser; it fails closed. `git log -p` over the
  pinned revisions shows every file those commits touched, including a secret committed in the past and still
  reachable; env files are gitignored and CI scans history.
- **The key stays in the CLI's own environment.** The SDK lays `ClaudeAgentOptions.env` over the inherited
  environment, so the allowlisted `env` the reviewer adds (`PATH`, `HOME`, `LANG`, literal git pathspecs, mock hardware)
  cannot remove `ANTHROPIC_API_KEY` from it - and the CLI needs the key. What keeps it away from tree-controlled code is
  that no process the head can steer runs with it: the shell allowlist runs only `git`, and the MCP server runs trusted
  code (see the trust model) with the credential variables blanked in its entry. The bundled CLI also has
  `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`, which strips credentials from Bash and MCP children; it is **not** enabled
  because the CLI ties it to forcing the permission mode down and to a subprocess sandbox, neither of which can be
  rehearsed without spending. Try it on the first small-cap run if you want the extra layer.
- The base ref is trusted: if it is itself compromised, so is everything above. The ledger file is the exception to
  "read from the base" - it is the spend history, so it is read from the checkout. A hostile head can edit its rows, so
  the cumulative workstream cap is a guard against your own mistakes; the per-run `--cap-usd` (also the SDK's hard stop)
  bounds each run whatever the ledger says.
- `git` runs with your own git configuration. A diff or text-conversion driver you have defined globally
  and that a head's `.gitattributes` names would run; the guard refuses `--ext-diff` and `--textconv`
  but git applies configured drivers by default.
- If the installed SDK wheel lacks its bundled Claude Code CLI, the SDK falls back to a `PATH` search for `claude`
  (which on Windows looks in the working directory first). Install the platform wheel from PyPI.
- `model: opus` is an alias. `models.retrieve` may not accept a bare family alias; then the run falls back to
  `resolved_via: config` and the CLI resolves `opus` itself. The ids actually served are recorded in the artifact
  (`models served`). Set `agent_reviewer.model` to a full alias in `workforce.yaml` to get a `models.retrieve` row.
- The real-SDK pins in the test file need the extra and are skipped in CI (no job installs it).
