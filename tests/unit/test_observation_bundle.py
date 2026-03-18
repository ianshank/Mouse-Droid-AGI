from __future__ import annotations

import time

import numpy as np

from mousedroid.sensing.bundle import MouseDroidObservationBundle


def test_default_values():
    obs = MouseDroidObservationBundle()
    assert obs.distance_m == 4.0
    assert obs.n_modalities == 4


def test_timestamp_is_monotonic():
    before = time.monotonic()
    obs = MouseDroidObservationBundle()
    after = time.monotonic()
    assert before <= obs.timestamp <= after


def test_vision_features_shape():
    obs = MouseDroidObservationBundle()
    assert obs.vision_features.shape == (256,)


def test_vision_features_dtype():
    obs = MouseDroidObservationBundle()
    assert obs.vision_features.dtype == np.float32


def test_motor_state_shape():
    obs = MouseDroidObservationBundle()
    assert obs.motor_state.shape == (4,)


def test_valid_mask_shape():
    obs = MouseDroidObservationBundle()
    assert obs.valid_mask.shape == (4,)


def test_valid_mask_defaults_to_ones():
    obs = MouseDroidObservationBundle()
    np.testing.assert_array_equal(obs.valid_mask, np.ones(4, dtype=np.float32))


def test_custom_values():
    vision = np.ones(128, dtype=np.float32)
    motor = np.array([0.1, 0.2, 0.3, 12.0], dtype=np.float32)
    mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    obs = MouseDroidObservationBundle(
        _vision_features=vision,
        _distance_m=1.5,
        _motor_state=motor,
        _valid_mask=mask,
    )
    assert obs.distance_m == 1.5
    np.testing.assert_array_equal(obs.vision_features, vision)
    np.testing.assert_array_equal(obs.motor_state, motor)


def test_n_modalities_equals_4():
    obs = MouseDroidObservationBundle()
    assert obs.n_modalities == 4


def test_audio_chunk_shape():
    obs = MouseDroidObservationBundle()
    assert obs.audio_chunk.shape == (1024,)


def test_audio_chunk_dtype():
    obs = MouseDroidObservationBundle()
    assert obs.audio_chunk.dtype == np.float32


def test_audio_chunk_custom():
    audio = np.ones(512, dtype=np.float32) * 0.5
    obs = MouseDroidObservationBundle(_audio_chunk=audio)
    np.testing.assert_array_equal(obs.audio_chunk, audio)


# ---------------------------------------------------------------------------
# Configurable frozenset fields (Phase 2 refactor)
# ---------------------------------------------------------------------------

from mousedroid.ai.vision.protocols import Detection, Gesture, VisionAIResult
from mousedroid.ai.audio.protocols import Transcription, AudioAIResult


def _make_detection(class_name: str) -> Detection:
    return Detection(bbox=(0, 0, 10, 10), class_id=0, class_name=class_name, confidence=0.9)


def _make_gesture(label: str) -> Gesture:
    return Gesture(label=label, confidence=0.9, hand="Right")


def _make_vision(detections=(), gestures=()):
    return VisionAIResult(
        detections=list(detections),
        embedding=np.zeros(512, dtype=np.float32),
        faces=[],
        gestures=list(gestures),
        frame_shape=(480, 640, 3),
        timestamp=0.0,
    )


def _make_audio(transcription_text: str | None = None):
    t = (
        Transcription(text=transcription_text, language="en", confidence=0.9, duration_s=1.0)
        if transcription_text
        else None
    )
    return AudioAIResult(
        wake_detected=False, transcription=t, sound_events=[], voice_command=None, timestamp=0.0
    )


class TestBundlePersonClassNamesConfig:
    def test_default_person_class_triggers_human_detected(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision(detections=[_make_detection("person")])
        )
        assert bundle.human_detected is True

    def test_non_person_class_not_detected_by_default(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision(detections=[_make_detection("robot")])
        )
        assert bundle.human_detected is False

    def test_custom_person_class_triggers_human_detected(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision(detections=[_make_detection("rider")]),
            _person_class_names=frozenset({"person", "rider"}),
        )
        assert bundle.human_detected is True

    def test_empty_person_class_names_never_detects(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision(detections=[_make_detection("person")]),
            _person_class_names=frozenset(),
        )
        assert bundle.human_detected is False


class TestBundleLaw2GestureLabelsConfig:
    def test_default_stop_gesture_triggers(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision(gestures=[_make_gesture("stop")])
        )
        assert bundle.gesture_stop_commanded is True

    def test_non_stop_gesture_not_triggered_by_default(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision(gestures=[_make_gesture("wave")])
        )
        assert bundle.gesture_stop_commanded is False

    def test_custom_law2_label_triggers(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision(gestures=[_make_gesture("freeze")]),
            _law2_gesture_labels=frozenset({"stop", "freeze"}),
        )
        assert bundle.gesture_stop_commanded is True

    def test_empty_law2_labels_never_triggers(self):
        bundle = MouseDroidObservationBundle(
            _vision_ai_result=_make_vision(gestures=[_make_gesture("stop")]),
            _law2_gesture_labels=frozenset(),
        )
        assert bundle.gesture_stop_commanded is False


class TestBundleStopKeywordsConfig:
    def test_default_stop_keyword_triggers(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio("halt everything now")
        )
        assert bundle.voice_stop_commanded is True

    def test_unrelated_speech_not_triggered(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio("move forward please")
        )
        assert bundle.voice_stop_commanded is False

    def test_custom_stop_keyword_triggers(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio("abort mission"),
            _stop_keywords=frozenset({"abort", "cancel"}),
        )
        assert bundle.voice_stop_commanded is True

    def test_empty_stop_keywords_never_triggers(self):
        bundle = MouseDroidObservationBundle(
            _audio_ai_result=_make_audio("stop right now"),
            _stop_keywords=frozenset(),
        )
        assert bundle.voice_stop_commanded is False
