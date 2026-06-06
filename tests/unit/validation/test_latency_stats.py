"""Unit tests for the latency-sample summariser."""

from __future__ import annotations

import pytest

from mousedroid.validation.latency_stats import (
    LatencySummary,
    intervals_ms,
    percentile,
    summarize,
)


class TestPercentile:
    def test_single_sample_returns_that_sample(self) -> None:
        assert percentile([42.0], 95.0) == 42.0

    def test_p0_and_p100_are_min_and_max(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        assert percentile(data, 0.0) == 1.0
        assert percentile(data, 100.0) == 4.0

    def test_p50_interpolates_between_middle_ranks(self) -> None:
        # rank = 0.5 * (4-1) = 1.5 -> between data[1]=2 and data[2]=3 -> 2.5
        assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)

    def test_known_p95_linear_interpolation(self) -> None:
        data = [float(i) for i in range(1, 101)]  # 1..100
        # rank = 0.95 * 99 = 94.05 -> data[94]=95 + 0.05*(96-95) = 95.05
        assert percentile(data, 95.0) == pytest.approx(95.05)

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            percentile([], 50.0)

    @pytest.mark.parametrize("q", [-1.0, 101.0])
    def test_out_of_range_q_raises(self, q: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            percentile([1.0], q)


class TestSummarize:
    def test_summary_fields_on_known_sample(self) -> None:
        summary = summarize([10.0, 20.0, 30.0, 40.0, 50.0])
        assert summary.n == 5
        assert summary.min_ms == 10.0
        assert summary.max_ms == 50.0
        assert summary.mean_ms == pytest.approx(30.0)
        assert summary.p50_ms == pytest.approx(30.0)

    def test_unsorted_input_is_sorted_internally(self) -> None:
        a = summarize([50.0, 10.0, 30.0, 20.0, 40.0])
        b = summarize([10.0, 20.0, 30.0, 40.0, 50.0])
        assert a.model_dump() == b.model_dump()

    def test_monotonic_percentile_ordering(self) -> None:
        summary = summarize([float(i) for i in range(1, 101)])
        assert summary.min_ms <= summary.p50_ms <= summary.p95_ms
        assert summary.p95_ms <= summary.p99_ms <= summary.max_ms

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            summarize([])

    def test_render_text_is_single_line(self) -> None:
        text = summarize([1.0, 2.0, 3.0]).render_text()
        assert "\n" not in text
        assert "p95=" in text
        assert "p99=" in text

    def test_summary_is_pydantic_serialisable(self) -> None:
        summary = summarize([1.0, 2.0])
        assert isinstance(summary, LatencySummary)
        assert set(summary.model_dump()) == {
            "n",
            "min_ms",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
        }


class TestIntervalsMs:
    def test_gaps_in_milliseconds(self) -> None:
        # 0.0, 0.1, 0.25 s -> gaps of 100 ms and 150 ms.
        assert intervals_ms([0.0, 0.1, 0.25]) == pytest.approx([100.0, 150.0])

    def test_count_is_n_minus_one(self) -> None:
        assert len(intervals_ms([1.0, 2.0, 3.0, 4.0])) == 3

    @pytest.mark.parametrize("stamps", [[], [42.0]])
    def test_fewer_than_two_yields_empty(self, stamps: list[float]) -> None:
        assert intervals_ms(stamps) == []

    def test_composes_with_summarize(self) -> None:
        summary = summarize(intervals_ms([0.0, 0.1, 0.2, 0.3]))
        assert summary.n == 3
        assert summary.mean_ms == pytest.approx(100.0)
