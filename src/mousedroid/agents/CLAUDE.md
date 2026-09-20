# Agents Subsystem — Surface Contract

> Navigation agents selecting latent-space actions under ``SafetyContext`` override
> (``AgentProtocol``; concrete ``MouseDroidNavigationAgent``).

## Invariants & Agent Rules

1. **Protocol Surface**: Application code depends on ``AgentProtocol`` (``name``, ``act``,
   ``reset``). ``act`` returns shape ``(action_dim,)`` in ``[-1, 1]``.
2. **Safety Override Input**: ``act`` receives a ``SafetyContext``; the navigation agent
   clamps planned actions to ``SafetyConfig.action_min`` / ``action_max`` and applies
   human-radius policy from ``ThreeLawsConfig``.
3. **Planner Injection**: ``MouseDroidNavigationAgent`` is constructed with an injected
   ``MCTSPlanner`` and root ``Settings`` — no planner construction inside the package.

## Key Files

- `base.py::AgentProtocol` — runtime-checkable agent interface.
- `navigation.py::MouseDroidNavigationAgent` — MCTS-based action selection with safety override.
- `_planning.py::compute_mcts_budget` — MCTS compute-budget helper for the navigation agent.
- `tests/unit/agents/` — subsystem unit tests.
