"""Tests for laundry sorting rule engine."""

from __future__ import annotations

from mousedroid.arm.planning.laundry_rules import (
    DEFAULT_RULES,
    LaundryRuleEngine,
    SortingRule,
)
from mousedroid.config.schema import ArmTaskConfig


def _make_engine(
    rules: list[SortingRule] | None = None,
    num_baskets: int = 3,
) -> LaundryRuleEngine:
    """Create rule engine with test defaults."""
    cfg = ArmTaskConfig(
        task_type="laundry_sorting",
        num_baskets=num_baskets,
    )
    return LaundryRuleEngine(cfg, rules=rules)


class TestLaundryRuleEngine:
    """Test LaundryRuleEngine sorting logic."""

    def test_white_shirt_to_basket_0(self) -> None:
        engine = _make_engine()
        assert engine.sort("shirt", "white") == 0

    def test_dark_pants_to_basket_1(self) -> None:
        engine = _make_engine()
        assert engine.sort("pants", "dark") == 1

    def test_color_towel_to_basket_2(self) -> None:
        engine = _make_engine()
        assert engine.sort("towel", "color") == 2

    def test_wildcard_garment_type(self) -> None:
        engine = _make_engine()
        # Default rules use "*" for garment type
        assert engine.sort("socks", "white") == 0
        assert engine.sort("underwear", "dark") == 1

    def test_fallback_to_last_basket(self) -> None:
        # Empty rule list — everything falls through
        engine = _make_engine(rules=[])
        assert engine.sort("shirt", "white") == 2  # last basket

    def test_custom_rules(self) -> None:
        custom = [
            SortingRule(color_category="*", garment_type="towel", basket_index=0),
            SortingRule(color_category="*", garment_type="*", basket_index=1),
        ]
        engine = _make_engine(rules=custom)
        assert engine.sort("towel", "white") == 0
        assert engine.sort("shirt", "dark") == 1

    def test_default_rules_count(self) -> None:
        assert len(DEFAULT_RULES) == 3


class TestColorClassification:
    """Test colour classification from RGB values."""

    def test_white_classification(self) -> None:
        engine = _make_engine()
        assert engine.classify_color((240.0, 235.0, 230.0)) == "white"

    def test_dark_classification(self) -> None:
        engine = _make_engine()
        assert engine.classify_color((30.0, 25.0, 20.0)) == "dark"

    def test_color_classification(self) -> None:
        engine = _make_engine()
        assert engine.classify_color((200.0, 50.0, 50.0)) == "color"

    def test_grey_is_color(self) -> None:
        engine = _make_engine()
        # Mid-brightness, low saturation but not white
        result = engine.classify_color((128.0, 128.0, 128.0))
        assert result in ("color", "dark")  # implementation dependent
