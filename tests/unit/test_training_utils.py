"""Tests for ``training/training_utils`` reusable SGD utilities."""

from __future__ import annotations

import numpy as np
import pytest

from training.training_utils import iter_batches, log_epoch_loss, sgd_step


# ---------------------------------------------------------------------------
# iter_batches
# ---------------------------------------------------------------------------


class TestIterBatches:
    def test_yields_correct_batch_size(self) -> None:
        rng = np.random.default_rng(0)
        batches = list(iter_batches(100, 10, rng))
        assert all(len(b) == 10 for b in batches)

    def test_yields_correct_number_of_batches(self) -> None:
        rng = np.random.default_rng(0)
        batches = list(iter_batches(100, 10, rng))
        assert len(batches) == 10

    def test_drops_incomplete_final_batch(self) -> None:
        rng = np.random.default_rng(0)
        # 105 samples / 10 batch size → 10 full batches, 5 leftover dropped
        batches = list(iter_batches(105, 10, rng))
        assert len(batches) == 10

    def test_covers_all_indices(self) -> None:
        rng = np.random.default_rng(42)
        n, batch_size = 50, 5
        batches = list(iter_batches(n, batch_size, rng))
        all_indices = np.concatenate(batches)
        assert len(all_indices) == n
        assert set(all_indices.tolist()) == set(range(n))

    def test_shuffled_across_epochs(self) -> None:
        rng = np.random.default_rng(0)
        epoch1 = list(iter_batches(50, 5, rng))
        epoch2 = list(iter_batches(50, 5, rng))
        # At least one batch should differ between epochs
        changed = any(
            not np.array_equal(a, b) for a, b in zip(epoch1, epoch2)
        )
        assert changed

    def test_single_batch(self) -> None:
        rng = np.random.default_rng(0)
        batches = list(iter_batches(10, 10, rng))
        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_empty_when_n_less_than_batch_size(self) -> None:
        rng = np.random.default_rng(0)
        batches = list(iter_batches(5, 10, rng))
        assert batches == []

    def test_indices_within_range(self) -> None:
        rng = np.random.default_rng(0)
        n = 80
        for idx in iter_batches(n, 16, rng):
            assert idx.min() >= 0
            assert idx.max() < n

    def test_seeded_rng_is_reproducible(self) -> None:
        batches_a = list(iter_batches(100, 10, np.random.default_rng(7)))
        batches_b = list(iter_batches(100, 10, np.random.default_rng(7)))
        for a, b in zip(batches_a, batches_b):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# sgd_step
# ---------------------------------------------------------------------------


class TestSgdStep:
    def test_updates_weights_in_place(self) -> None:
        weights = {"w": np.ones((2, 2), dtype=np.float32)}
        grads = {"w": np.ones((2, 2), dtype=np.float32)}
        sgd_step(weights, grads, lr=0.1)
        np.testing.assert_allclose(weights["w"], np.full((2, 2), 0.9, dtype=np.float32))

    def test_zero_lr_leaves_weights_unchanged(self) -> None:
        weights = {"w": np.array([1.0, 2.0])}
        grads = {"w": np.array([100.0, 200.0])}
        original = weights["w"].copy()
        sgd_step(weights, grads, lr=0.0)
        np.testing.assert_array_equal(weights["w"], original)

    def test_multiple_params_updated(self) -> None:
        weights = {
            "w": np.array([1.0, 2.0]),
            "b": np.array([0.5]),
        }
        grads = {
            "w": np.array([1.0, 1.0]),
            "b": np.array([0.5]),
        }
        sgd_step(weights, grads, lr=1.0)
        np.testing.assert_allclose(weights["w"], [0.0, 1.0])
        np.testing.assert_allclose(weights["b"], [0.0])

    def test_keys_not_in_grads_are_untouched(self) -> None:
        weights = {
            "w": np.array([1.0]),
            "b": np.array([2.0]),  # no gradient provided
        }
        grads = {"w": np.array([1.0])}
        sgd_step(weights, grads, lr=1.0)
        np.testing.assert_array_equal(weights["b"], [2.0])

    def test_lr_scales_update_correctly(self) -> None:
        weights = {"w": np.array([10.0])}
        grads = {"w": np.array([2.0])}
        sgd_step(weights, grads, lr=0.5)
        np.testing.assert_allclose(weights["w"], [9.0])

    def test_negative_gradient_increases_weights(self) -> None:
        weights = {"w": np.array([1.0])}
        grads = {"w": np.array([-1.0])}
        sgd_step(weights, grads, lr=1.0)
        np.testing.assert_allclose(weights["w"], [2.0])


# ---------------------------------------------------------------------------
# log_epoch_loss
# ---------------------------------------------------------------------------


class TestLogEpochLoss:
    def _make_stub_logger(self) -> tuple[object, list[dict]]:
        calls: list[dict] = []

        class _StubLogger:
            def info(self, event: str, **kw: object) -> None:
                calls.append({"event": event, **kw})

        return _StubLogger(), calls

    def test_logs_at_correct_interval(self) -> None:
        logger, calls = self._make_stub_logger()
        for epoch in range(1, 41):
            log_epoch_loss(logger, "test_epoch", epoch, 10.0, 5, log_every=20)
        assert len(calls) == 2  # epochs 20 and 40

    def test_does_not_log_between_intervals(self) -> None:
        logger, calls = self._make_stub_logger()
        log_epoch_loss(logger, "test", 19, 10.0, 5, log_every=20)
        assert calls == []

    def test_logged_loss_is_mean(self) -> None:
        logger, calls = self._make_stub_logger()
        log_epoch_loss(logger, "ev", 20, 100.0, 5, log_every=20)
        assert calls[0]["loss"] == pytest.approx(20.0)

    def test_logged_epoch_matches(self) -> None:
        logger, calls = self._make_stub_logger()
        log_epoch_loss(logger, "ev", 20, 1.0, 1, log_every=20)
        assert calls[0]["epoch"] == 20

    def test_event_name_forwarded(self) -> None:
        logger, calls = self._make_stub_logger()
        log_epoch_loss(logger, "my_model_epoch", 10, 5.0, 1, log_every=10)
        assert calls[0]["event"] == "my_model_epoch"

    def test_handles_zero_batches(self) -> None:
        """n_batches=0 should not raise — uses max(n_batches, 1) guard."""
        logger, calls = self._make_stub_logger()
        log_epoch_loss(logger, "ev", 20, 5.0, 0, log_every=20)
        assert calls[0]["loss"] == pytest.approx(5.0)
