"""Smoke tests for AI vision, audio, and fusion pipelines.

These tests verify correct behaviour using mock data — no real hardware,
GPU, or model weights are required.  They exercise:
  * AI protocol dataclasses
  * Bundle computed properties (human_detected, human_dist_m, gesture/voice stop)
  * Three Laws wiring via the bundle properties
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from mousedroid.ai.vision.protocols import Detection, Gesture, VisionAIResult
from mousedroid.ai.audio.protocols import Transcription, AudioAIResult
from mousedroid.sensing.bundle import MouseDroidObservationBundle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_person_detection(confidence: float = 0.9) -> Detection:
    return Detection(
        bbox=(10, 20, 100, 200),
        class_id=0,
        class_name="person",
        confidence=confidence,
    )


def _make_dog_detection() -> Detection:
    return Detection(
        bbox=(5, 5, 50, 50),
        class_id=16,
        class_name="dog",
        confidence=0.8,
    )


def _make_stop_gesture() -> Gesture:
    return Gesture(label="stop", confidence=0.95, hand="Right")


def _make_vision_result(detections=(), gestures=()) -> VisionAIResult:
    return VisionAIResult(
        detections=list(detections),
        embedding=np.zeros(512, dtype=np.float32),
        faces=[],
        gestures=list(gestures),
        frame_shape=(480, 640, 3),
        timestamp=0.0,
    )


def _make_transcription(text: str) -> Transcription:
    return Transcription(text=text, language="en", confidence=0.9, duration_s=1.0)


def _make_audio_result(transcription: Transcription | None = None) -> AudioAIResult:
    return AudioAIResult(
        wake_detected=False,
        transcription=transcription,
        sound_events=[],
        voice_command=None,
        timestamp=0.0,
    )


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------


class TestDetectionDataclass:
    def test_fields_accessible(self):
        d = _make_person_detection()
        assert d.class_name == "person"
        assert d.class_id == 0
        assert d.confidence == pytest.approx(0.9)
        assert d.bbox == (10, 20, 100, 200)

    def test_non_person_class(self):
        d = _make_dog_detection()
        assert d.class_name == "dog"
        assert d.class_id == 16


class TestGestureDataclass:
    def test_stop_gesture(self):
        g = _make_stop_gesture()
        assert g.label == "stop"
        assert g.confidence == pytest.approx(0.95)
        assert g.hand == "Right"


class TestVisionAIResult:
    def test_empty_result(self):
        result = _make_vision_result()
        assert result.detections == []
        assert result.gestures == []

    def test_with_detections_and_gestures(self):
        result = _make_vision_result(
            detections=[_make_person_detection()],
            gestures=[_make_stop_gesture()],
        )
        assert len(result.detections) == 1
        assert len(result.gestures) == 1


class TestAudioAIResult:
    def test_empty_result(self):
        result = _make_audio_result()
        assert result.transcription is None
        assert result.wake_detected is False
        assert result.sound_events == []

    def test_with_transcription(self):
        t = _make_transcription("hello world")
        result = _make_audio_result(transcription=t)
        assert result.transcription is not None
        assert result.transcription.text == "hello world"


# ---------------------------------------------------------------------------
# Bundle — human_detected property
# ---------------------------------------------------------------------------


class TestBundleHumanDetected:
    def test_no_ai_result_returns_false(self):
        bundle = MouseDroidObservationBundle()
        assert bundle.human_detected is False

    def test_no_persons_returns_false(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision_result(detections=[_make_dog_detection()])
        )
        assert bundle.human_detected is False

    def test_person_detected_returns_true(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision_result(
                detections=[_make_person_detection()]
            )
        )
        assert bundle.human_detected is True

    def test_mixed_detections_returns_true(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision_result(
                detections=[_make_dog_detection(), _make_person_detection()]
            )
        )
        assert bundle.human_detected is True


# ---------------------------------------------------------------------------
# Bundle — human_dist_m property
# ---------------------------------------------------------------------------


class TestBundleHumanDistM:
    def test_no_human_returns_inf(self):
        bundle = MouseDroidObservationBundle()
        assert bundle.human_dist_m == math.inf

    def test_no_human_with_dogs_returns_inf(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision_result(detections=[_make_dog_detection()])
        )
        assert bundle.human_dist_m == math.inf

    def test_falls_back_to_ultrasonic(self):
        bundle = MouseDroidObservationBundle(
            _distance_m=1.5,
            _vision_ai_result=_make_vision_result(
                detections=[_make_person_detection()]
            ),
        )
        # No fused depth — falls back to raw ultrasonic
        assert bundle.human_dist_m == pytest.approx(1.5)

    def test_uses_fused_depth_when_available(self):
        from mousedroid.ai.fusion.sensor_fusion import FusedDepthResult
        import time as _time
        fused = FusedDepthResult(
            depth_map=np.ones((64, 64), dtype=np.float32) * 2.0,
            center_distance_m=2.0,
            ultrasonic_distance_m=1.5,
            timestamp=_time.time(),
        )
        bundle = MouseDroidObservationBundle(
            _distance_m=1.5,
            _vision_ai_result=_make_vision_result(
                detections=[_make_person_detection()]
            ),
            _fused_depth=fused,
        )
        # Should prefer fused depth (2.0) over ultrasonic (1.5)
        assert bundle.human_dist_m == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Bundle — gesture_stop_commanded property
# ---------------------------------------------------------------------------


class TestBundleGestureStop:
    def test_no_ai_result_returns_false(self):
        bundle = MouseDroidObservationBundle()
        assert bundle.gesture_stop_commanded is False

    def test_non_stop_gesture_returns_false(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision_result(
                gestures=[Gesture(label="thumbs_up", confidence=0.9, hand="Left")]
            )
        )
        assert bundle.gesture_stop_commanded is False

    def test_stop_gesture_returns_true(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision_result(
                gestures=[_make_stop_gesture()]
            )
        )
        assert bundle.gesture_stop_commanded is True

    def test_mixed_gestures_with_stop_returns_true(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision_result(
                gestures=[
                    Gesture(label="point", confidence=0.8, hand="Right"),
                    _make_stop_gesture(),
                ]
            )
        )
        assert bundle.gesture_stop_commanded is True


# ---------------------------------------------------------------------------
# Bundle — voice_stop_commanded property
# ---------------------------------------------------------------------------


class TestBundleVoiceStop:
    def test_no_ai_result_returns_false(self):
        bundle = MouseDroidObservationBundle()
        assert bundle.voice_stop_commanded is False

    def test_benign_transcription_returns_false(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio_result(
                transcription=_make_transcription("hello how are you")
            )
        )
        assert bundle.voice_stop_commanded is False

    @pytest.mark.parametrize("text", ["stop", "halt", "freeze", "no", "danger"])
    def test_stop_keywords_return_true(self, text: str):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio_result(
                transcription=_make_transcription(text)
            )
        )
        assert bundle.voice_stop_commanded is True

    def test_keyword_in_sentence_returns_true(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio_result(
                transcription=_make_transcription("please stop what you are doing")
            )
        )
        assert bundle.voice_stop_commanded is True

    def test_case_insensitive(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio_result(
                transcription=_make_transcription("STOP MOVING NOW")
            )
        )
        assert bundle.voice_stop_commanded is True

    def test_substring_false_positive_avoided(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio_result(
                transcription=_make_transcription("that motor is unstoppable")
            )
        )
        assert bundle.voice_stop_commanded is False

    def test_no_transcription_returns_false(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio_result(transcription=None)
        )
        assert bundle.voice_stop_commanded is False


# ---------------------------------------------------------------------------
# Three Laws integration — Law 1 (human proximity via bundle)
# ---------------------------------------------------------------------------


class TestThreeLawsLaw1Integration:
    """Verify Three Laws checker receives correct context from bundle properties."""

    def _run_check(self, bundle: MouseDroidObservationBundle) -> list:
        from mousedroid.safety.three_laws import RoboticsLawChecker

        checker = RoboticsLawChecker(
            human_safety_radius_m=0.5,
            emergency_stop_dist_m=0.15,
            enabled=True,
        )
        action = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        context = {
            "human_detected": bundle.human_detected,
            "human_dist_m": bundle.human_dist_m,
            "obstacle_dist_m": bundle.distance_m,
        }
        _, violations = checker.check(action, context)
        return violations

    def test_no_human_no_violations(self):
        bundle = MouseDroidObservationBundle(_distance_m=2.0)
        violations = self._run_check(bundle)
        law1_violations = [v for v in violations if v.law.value == 1]
        assert law1_violations == []

    def test_close_person_triggers_law1(self):
        bundle = MouseDroidObservationBundle(
            _distance_m=0.3,
            _vision_ai_result=_make_vision_result(
                detections=[_make_person_detection()]
            ),
        )
        violations = self._run_check(bundle)
        law1_violations = [v for v in violations if v.law.value == 1]
        assert len(law1_violations) >= 1
        assert law1_violations[0].severity > 0.0

    def test_distant_person_no_law1_violation(self):
        bundle = MouseDroidObservationBundle(
            _distance_m=2.0,
            _vision_ai_result=_make_vision_result(
                detections=[_make_person_detection()]
            ),
        )
        violations = self._run_check(bundle)
        law1_violations = [v for v in violations if v.law.value == 1]
        assert law1_violations == []


# ---------------------------------------------------------------------------
# Three Laws integration — Law 2 (gesture/voice stop)
# ---------------------------------------------------------------------------


class TestThreeLawsLaw2Integration:
    """Verify stop gesture / voice command routes to commanded_action = zeros."""

    def _run_check_with_command(
        self,
        bundle: MouseDroidObservationBundle,
        action_dim: int = 3,
    ) -> tuple[np.ndarray, list]:
        from mousedroid.safety.three_laws import RoboticsLawChecker

        checker = RoboticsLawChecker(command_blend_weight=0.8, enabled=True)
        action = np.array([0.5, 0.0, 0.0], dtype=np.float64)[:action_dim]

        commanded = (
            np.zeros(action_dim, dtype=np.float32).tolist()
            if (bundle.gesture_stop_commanded or bundle.voice_stop_commanded)
            else None
        )
        context = {
            "human_detected": bundle.human_detected,
            "human_dist_m": bundle.human_dist_m,
            "obstacle_dist_m": bundle.distance_m,
            "commanded_action": commanded,
        }
        safe_action, violations = checker.check(action, context)
        return safe_action, violations

    def test_no_stop_command_no_law2_violations(self):
        bundle = MouseDroidObservationBundle()
        safe, violations = self._run_check_with_command(bundle)
        law2_violations = [v for v in violations if v.law.value == 2]
        assert law2_violations == []

    def test_gesture_stop_triggers_law2_blend(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision_result(gestures=[_make_stop_gesture()])
        )
        safe, violations = self._run_check_with_command(bundle)
        # After blend toward zeros, speed should be reduced
        assert float(safe[0]) < 0.5

    def test_voice_stop_halt_triggers_law2_blend(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio_result(
                transcription=_make_transcription("halt")
            )
        )
        safe, violations = self._run_check_with_command(bundle)
        assert float(safe[0]) < 0.5
