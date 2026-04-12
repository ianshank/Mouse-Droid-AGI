"""Rocky phrase bank — maps robot events to Rocky-style English phrases.

Each event maps to a list of candidate phrases. One is chosen at random
per utterance, giving the droid personality variation.
"""

from __future__ import annotations

DEFAULT_PHRASES: dict[str, list[str]] = {
    "task_complete": [
        "Good good good!",
        "Is done! Happy!",
        "Work finish! Very satisfy!",
    ],
    "obstacle_detected": [
        "Is problem! Object ahead!",
        "Stop stop! Thing in way!",
        "Careful careful! Obstacle!",
    ],
    "emergency_stop": [
        "Bad bad bad! Emergency!",
        "Is danger! Stop now!",
        "Critical critical! All stop!",
    ],
    "path_clear": [
        "Path good! Go go go!",
        "Nothing ahead. Safe safe.",
        "Clear clear! Move forward!",
    ],
    "low_battery": [
        "Need energy! Is critical!",
        "Power low low. Must charge!",
        "Battery bad! Need fuel!",
    ],
    "new_object": [
        "Amaze! What is?",
        "New thing! Very interest!",
        "Never see before! Curious!",
    ],
    "navigation_success": [
        "You are good engineer!",
        "Navigation work! Happy happy!",
        "Arrive arrive! Success!",
    ],
    "error": [
        "Is not working! Bad!",
        "Problem problem. Not understand.",
        "Error error! Something wrong!",
    ],
    "idle": [
        "Hmm. Waiting waiting.",
        "Is boring. Want task!",
        "Nothing to do. Give command?",
    ],
    "startup": [
        "Hello hello! Rocky ready!",
        "Systems good! Ready for work!",
        "Am awake! What we do?",
    ],
    "shutdown": [
        "Goodbye goodbye! Sleep now.",
        "Shutting down. Rest rest.",
        "Power off. See you soon!",
    ],
}
