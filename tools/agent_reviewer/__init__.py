"""F-059 WS-B / WS-D: a read-only Agent-SDK peer reviewer with a spend ledger.

Developer tooling only - nothing here runs on the rover and nothing here is
imported by it. ``claude_agent_sdk`` (the ``agent_reviewer`` extra) and
``anthropic`` are imported inside the functions that need them, never at module
scope, so importing this package costs nothing and works on a machine that has
neither.

* :mod:`tools.agent_reviewer.policy` - the allowlist, the Bash guard, the mock-only
  MCP config and the SDK options. The reviewer's containment.
* :mod:`tools.agent_reviewer.prompt` - the system prompt, and the refusal to put
  a secret into any prompt.
* :mod:`tools.agent_reviewer.ledger` - the pre-call cap check and the spend row.
* :mod:`tools.agent_reviewer.run_review` - the CLI:
  ``python -m tools.agent_reviewer.run_review --base REF --head REF --cap-usd N``.

See ``tools/agent_reviewer/README.md`` for what the reviewer may touch, what it
never does, and how to run the first review.
"""
