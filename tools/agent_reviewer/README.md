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
  environment. The CLI is told `NoDefaultCurrentDirectoryInExePath=1` for the programs it starts. For the
  programs the *CLI* looks up by bare name there is no such switch to rely on alone, so a run is also refused
  before any spend when the head (or the checkout) tracks a file in its root named like one of them - `git`,
  `bash`, `sh`, `claude`, `rg`, `node`, `npm`, `npx`, `gitleaks`, `cmd`, `powershell`, `pwsh`, `python*` - with
  any Windows executable extension (`.exe`, `.cmd`, `.bat`, ...).
- **Reads stay inside the checkout.** A second PreToolUse hook (`policy.ReadGuard`, matcher `Read|Grep|Glob`)
  resolves every path the tool is given (`file_path` and `path` - the CLI reads a Grep `file_path` as its
  `path`) with `os.path.realpath`, which follows symlinks and junctions and expands Windows 8.3 short names,
  and requires the result to be inside the checkout under review. It also refuses `.git` internals (use
  `git show`), secret-file and private-key names *after* the link is followed, Windows device names (`NUL`,
  `CON`, `COM1`, ... with or without an extension), paths the shell would expand (`~`, `$VAR`, `%VAR%`), NTFS
  data streams (`::$DATA`) and glob characters in a path field. Your shell profile (where keys get exported),
  other worktrees and `~/.claude/projects` transcripts are therefore out of reach. It answers with a reason
  and never raises; it fails closed, like the Bash guard.
- **A glob is judged by a strict grammar, not by rewriting it into text.** The CLI splits a Grep `glob` on
  whitespace and commas and hands the pieces to ripgrep, which reads character classes, braces, `?`, `!` and
  backslash escapes; a screen that read the string as one name would be reading something else. So a glob is
  written with letters, digits, `. _ @ + -`, `/` and `*` and nothing else: no `[..]`, `{..}`, `?`, `!`, `\`,
  comma or space (one glob per call). A directory part is a literal name, `*` or `**`; the file part is a
  literal name or `*.<extension>` (`**/*.py`, `src/**/*.md`, `**/pyproject.toml`; not `test_*.py`, `*` or
  `src/**/x*.py` - a wildcard inside a file name is refused by design: search `src/**/*.py` and narrow in the
  pattern). A glob or a path that starts with `-` (`-g`, `--no-ignore`, `-uuu`) is refused, because a search
  command would read it as an option; write a path as `./name`.
  **The extension must be on a fixed list**, `agent_reviewer.glob_extensions` of the base's `workforce.yaml`
  (default `py md toml json yml yaml txt sh cfg ini rst`; the plan prints it): a wildcard cannot be shown clear
  of the secret patterns, because `.env.*` and `docker.env*` end in `*` and overlap every extension, so which
  extensions the reviewer may search by is decided by a list, not by a sample of names. `*.c` is refused until
  `c` is listed. Behind the list, as a belt, the file part is matched against concrete names made from the
  secret patterns (`.env`, `prod.env`, `docker.env.local`, `.env.production`, `id_rsa`, ...), so an operator
  who lists `env` still finds `*.env` refused. A glob that can step through `.git` - no directory part (ripgrep
  matches such a glob at any depth) or a wildcard one - may not name git's own files (`config`, `HEAD`,
  `ORIG_HEAD`, `FETCH_HEAD`, `packed-refs`, `index`, `COMMIT_EDITMSG`, `*.pack`, `*.idx`): `*/config` and
  `**/HEAD` are refused. The literal start of the glob (`link` in `link/*.py`) is resolved against the search
  directory and must stay inside the checkout. The guard refuses rather than rewrites a call; and the CLI is
  told `Read(**/.git/**)` besides the secret-file deny rules, which it applies to Grep and Glob as ripgrep
  exclusions placed after the model's own globs, so `.git` stays out of what they find whatever the hook accepts.
  What is left: `*.py` can still match a file called `.env.py` (a secret name by the patterns), which the CLI's
  exclusion and the read guard refuse to show.
- **A head that tracks a symbolic link out of the checkout is refused.** `git ls-tree -r` of the head and of
  the checkout, and `git ls-files -s` of the checkout's index (a link that is staged and not committed is
  already on disk), list the mode-`120000` entries; a target that is absolute, climbs above the repository
  root, or has a `..` after a name (which only the file system can resolve once links chain) refuses the run
  before any spend, as a planted program does.
- **The reviewer's own `git` and `gitleaks` run without any credential variable** - any name with `KEY`, `TOKEN`,
  `SECRET`, `PASSWORD` or `CREDENTIAL(S)` as one of its words (`API_KEY`, a bare `TOKEN`, `AWS_ACCESS_KEY_ID`,
  `GOOGLE_APPLICATION_CREDENTIALS`, ...), plus `AWS_SECRET_ACCESS_KEY` and the password variables the rule cannot
  see: they read repository objects and scan text, and only the CLI needs the key.
- **The spend ledger is the controller's, not the head's.** It lives in the checkout the reviewer's own code runs
  from (override with `--ledger-root DIR`), is refused if it is a symbolic link or resolves outside that
  checkout, and the spend the cap sees is `max(the file, what the base ref's committed copy records)`: a head
  cannot delete rows to widen the cap. The cap is **per workstream**, not per head worktree - every review
  charges the same ledger. A `--ledger-root` inside the checkout under review is refused unless the checkout is at
  the base commit or you pass `--trust-checkout` (the plan prints `trust_ledger:`): the file would be one the head
  can rewrite, and the base's committed copy only floors the total.
- **The code that runs beside the key must not be the branch's.** That is the reviewer's own modules
  (`tools/agent_reviewer`), the interpreter's environment (its `site-packages` and `.pth` files run at startup,
  even under `-I`) and the `mousedroid` package the MCP server imports (the checkout's own source under an
  editable install). When any of them lives inside the checkout under review and the checkout is not at the
  base commit, the run is refused before any spend unless you pass `--trust-checkout`. An interpreter
  environment inside the checkout in which the head *tracks* files is refused regardless: it is not yours.

## What it may touch

| Surface | Allowed |
|---|---|
| Files | `Read`, `Grep`, `Glob` **inside the checkout under review only** (`ReadGuard`), and not `.git`, `*.env`, `docker.env*`; the CLI's deny rules cover `.git` and the usual credential directories too. A wildcard file part is `*.<ext>` with the extension on `agent_reviewer.glob_extensions` |
| Shell | `git diff`, `git log`, `git show` - one plain command: no pipes, redirection, chaining, substitution, variables or unquoted globs. Commits are the two full ids pinned in the task, with `^`, `^N` or `~N` (`A..B` ranges too): not `HEAD`, a branch, an abbreviation or a blob id. Option values are attached (`-n5`, `--grep=text`). Pathspecs go after `--` and are literal. `git show <commit>:<path>` names only a file tracked at the base (read a file the change adds from the diff). No `--all`, `--reflog`, `--branches`, `--remotes`, `--raw`, `--abbrev`, `--output`, `--no-index`, and no short option bundle that contains a capital `O` (`-O<file>` reads a file; so does `-pO<file>`) |
| MCP | `system_info`, `lidar_diagnostics`, `esp32_diagnostics` of the mock-hardware `mousedroid` server |
| Writes | none by the agent. The runner writes one review under `reports/agent_reviewer/<date>/wsb-NNNN.md` in the checkout under review and one row in `reports/credit/ledger.jsonl` of the **controller's** checkout (paths from the base's `workforce.yaml`) |
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
   git worktree add --detach ../reviewer-code <trusted-commit>   # a commit YOU trust that has this tool
   git worktree add --detach ../review-head <head-sha>           # the head under review (Read/Grep see it)
   uv venv ../reviewer-venv
   uv pip install --native-tls --python ../reviewer-venv/Scripts/python.exe \
       "../reviewer-code[agent_reviewer,anthropic,mcp]"  # NOT editable: mousedroid from the trusted code
   ```

   `--detach` with the head's commit id (`git rev-parse <head-branch>`) always works: `git worktree add` refuses
   a branch that is checked out in another worktree, which the head branch usually is. A fresh detached worktree
   also holds nothing but tracked files - no untracked secrets for `Grep` to find.
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

   Read the `workforce_config:`, `ledger:`, `ledger_floor:`, `trust_controller:`, `trust_interpreter:`,
   `trust_package:` and `trust_ledger:` lines of the plan before a live run. A refusal that talks about code inside this checkout means step 1 is not satisfied yet.
4. Run it: the same command without `--dry-run`. `--cap-usd` is this run's budget: it must fit in what is
   left of the workstream cap (`credit.caps_usd.B` in the base's `workforce.yaml`), and the SDK stops the run at
   it. The base ref must therefore carry the `credit:` block (and `agent_reviewer:`, the agent file, `.mcp.json`
   and `config/default.yaml`): a base without `workforce.yaml` gets the schema defaults, which cap nothing and so
   refuse to spend. Until PR-M (the `credit:` block and the ledger) is merged into the base you review against,
   the dry run therefore exits 3 ("no spend cap is configured for workstream 'B'"): review against a base
   that has it. Consider a first run with a small cap (for example `--cap-usd 1`) to see the tool surface work
   before spending $10.
5. Read the artifact (under `../review-head`) and the new row of `reports/credit/ledger.jsonl` (under
   `../reviewer-code`, the controller's checkout; the run prints both), compare the cost with the console, and
   commit the artifact on the head branch and the row where the ledger is tracked. Until the row is committed it
   counts from the controller's file only, so a second controller checkout would not see it. Do not run the
   head's own `tools/credit_ledger.py` while the key is exported.

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

The workstream cap is shared: it is **per workstream (`B`), not per head worktree**. Reviews of different heads
from the same controller checkout all charge the one ledger there. The spend the pre-check sees is the larger of
that file's sum and the sum in the base ref's committed copy of the ledger (`ledger_floor:` in the plan), so a file
that lost rows - a stale checkout, a head that deleted them - cannot widen the cap below what the base records.

`max_turns`, `max_prompt_chars` (a run whose system and task prompts together pass it is refused before any
spend), `glob_extensions` (the extensions a wildcard glob may use), the model alias, the caps and the ledger
path live in `.claude/workforce.yaml` (`agent_reviewer:` and `credit:`), never in code - and are read from the
base ref.

## Known limits

- `Read`, `Grep` and `Glob` are confined to the checkout by `ReadGuard`, which resolves symlinks, junctions and
  short names before it judges a path. INSIDE the checkout, a `Grep` over a directory can still print lines of a
  secret file whose name neither the CLI's deny rules nor the guard recognise (the secret names are the
  project's patterns and the `ssh-keygen` key names; a glob is matched against concrete names made from them, not
  proven harmless): a head that *tracks* a secret has it in the diff anyway. A symbolic link or directory junction that
  is untracked (in neither the head, the checkout's HEAD nor its index), or made after the run starts, does not
  refuse the run; a path that goes through one is still resolved and judged. Review from a checkout that holds
  no real secrets - a fresh `git worktree add --detach` has none.
- A short option bundle containing a capital `O` is refused wherever it sits (`-pO<file>`), so a pickaxe or regex
  string whose leading letters include one (`-SOrange`, `-GFOO`) is refused too: search for the lower-case part,
  or use `Grep` on the checkout.
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
  "read from the base" - it is the spend history, so it is read from the controller's checkout, which the head cannot
  edit, and floored by the base's committed copy. Whoever can write to the controller's checkout, or a base whose
  committed ledger is stale, still decides the cumulative total: it is a guard against your own mistakes, and the
  per-run `--cap-usd` (also the SDK's hard stop) bounds each run whatever the ledger says.
- The egress secret screen (credential-shape patterns, gitleaks over the text that leaves the machine, strict mode)
  is implemented here (`prompt.py`) and again in PR-O's batch tool (`tools/dead_code_triage_batch.py`). One shared
  helper is a follow-up once both are merged.
- `git` runs with your own git configuration. A diff or text-conversion driver you have defined globally
  and that a head's `.gitattributes` names would run; the guard refuses `--ext-diff` and `--textconv`
  but git applies configured drivers by default.
- If the installed SDK wheel lacks its bundled Claude Code CLI, the SDK falls back to a `PATH` search for `claude`
  (which on Windows looks in the working directory first). Install the platform wheel from PyPI.
- `model: opus` is an alias. `models.retrieve` may not accept a bare family alias; then the run falls back to
  `resolved_via: config` and the CLI resolves `opus` itself. The ids actually served are recorded in the artifact
  (`models served`). Set `agent_reviewer.model` to a full alias in `workforce.yaml` to get a `models.retrieve` row.
- The real-SDK pins in the test file need the extra and are skipped in CI (no job installs it).
