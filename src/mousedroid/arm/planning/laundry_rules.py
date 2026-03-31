"""Rule engine for laundry sorting task.

Maps detected garment (type, color) tuples to basket assignments
using configurable sorting rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmTaskConfig

_log = get_logger(__name__)


@dataclass(frozen=True)
class SortingRule:
    """Single sorting rule mapping garment attributes to a basket.

    Attributes:
        color_category: Color category ("white", "dark", "color", "*" for any).
        garment_type: Garment type ("shirt", "pants", "towel", "*" for any).
        basket_index: Target basket index (0-based).
    """

    color_category: str
    garment_type: str
    basket_index: int


# Default 3-basket sorting rules: whites, darks, colours
DEFAULT_RULES: list[SortingRule] = [
    SortingRule(color_category="white", garment_type="*", basket_index=0),
    SortingRule(color_category="dark", garment_type="*", basket_index=1),
    SortingRule(color_category="color", garment_type="*", basket_index=2),
]


class LaundryRuleEngine:
    """Rule-based laundry sorting engine.

    Matches garment attributes against an ordered rule list and
    returns the target basket index.

    Args:
        task_cfg: Task configuration with basket count.
        rules: Optional custom rule list (defaults to 3-basket colour sort).
    """

    def __init__(
        self,
        task_cfg: ArmTaskConfig,
        rules: list[SortingRule] | None = None,
    ) -> None:
        """Initialise laundry rule engine.

        Args:
            task_cfg: Task config with num_baskets.
            rules: Custom rules or None for defaults.
        """
        self._task_cfg = task_cfg
        self._rules = rules if rules is not None else DEFAULT_RULES
        _log.info(
            "laundry_rules_init",
            num_rules=len(self._rules),
            num_baskets=task_cfg.num_baskets,
        )

    def classify_color(self, rgb_mean: tuple[float, float, float]) -> str:
        """Classify colour category from mean RGB values.

        Uses simple thresholds on brightness and saturation to
        categorise into white/dark/colour.

        Args:
            rgb_mean: Mean (R, G, B) values in [0, 255].

        Returns:
            Color category: "white", "dark", or "color".
        """
        r, g, b = rgb_mean
        brightness = (r + g + b) / 3.0
        max_channel = max(r, g, b)
        min_channel = min(r, g, b)
        saturation = (max_channel - min_channel) / max(max_channel, 1.0)

        if brightness > 200.0 and saturation < 0.15:
            return "white"
        if brightness < 80.0:
            return "dark"
        return "color"

    def sort(self, garment_type: str, color_category: str) -> int:
        """Determine target basket for a garment.

        Matches against rules in order, returning the first match.
        Falls back to the last basket if no rule matches.

        Args:
            garment_type: Garment type (e.g., "shirt", "pants").
            color_category: Colour category ("white", "dark", "color").

        Returns:
            Target basket index (0-based).
        """
        for rule in self._rules:
            type_match = rule.garment_type == "*" or rule.garment_type == garment_type
            color_match = rule.color_category == "*" or rule.color_category == color_category
            if type_match and color_match:
                _log.debug(
                    "laundry_sorted",
                    garment=garment_type,
                    color=color_category,
                    basket=rule.basket_index,
                )
                return rule.basket_index

        # Fallback to last basket
        fallback = self._task_cfg.num_baskets - 1
        _log.warning(
            "laundry_no_rule_match",
            garment=garment_type,
            color=color_category,
            fallback_basket=fallback,
        )
        return fallback
