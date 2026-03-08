"""In-context learner — prompt-based adaptation without weight updates."""

from __future__ import annotations

from typing import Any

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class InContextLearner:
    """Prompt-based adaptation that prepends demonstration examples.

    Constructs a context window from support examples so that a downstream
    model can adapt to a new task purely through in-context conditioning.

    Args:
        max_context_examples: Maximum number of examples in the prompt.
    """

    def __init__(self, max_context_examples: int) -> None:
        self._max_examples = max_context_examples
        self._context: list[dict[str, Any]] = []

        _log.info("in_context_init", max_examples=max_context_examples)

    def add_example(self, input_data: Any, output_data: Any) -> None:
        """Add a demonstration example to the context window.

        Oldest examples are evicted when the window is full.

        Args:
            input_data: Example input (observation, query, etc.).
            output_data: Expected output (action, answer, etc.).
        """
        entry: dict[str, Any] = {"input": input_data, "output": output_data}
        self._context.append(entry)
        if len(self._context) > self._max_examples:
            self._context.pop(0)

    def build_prompt(self, query: Any) -> list[dict[str, Any]]:
        """Build a prompt from context examples plus the new query.

        Args:
            query: Current input to condition on.

        Returns:
            List of context dicts followed by the query dict.
        """
        prompt: list[dict[str, Any]] = list(self._context)
        prompt.append({"input": query, "output": None})
        return prompt

    def adapt(self, support_data: list[tuple[Any, Any]]) -> None:
        """Bulk-load support examples for a new task.

        Args:
            support_data: List of ``(input, output)`` demonstration pairs.
        """
        self.clear()
        for inp, out in support_data:
            self.add_example(inp, out)
        _log.debug("in_context_adapted", n_examples=len(self._context))

    def clear(self) -> None:
        """Clear the context window."""
        self._context.clear()

    def __len__(self) -> int:
        """Number of examples currently in the context."""
        return len(self._context)
