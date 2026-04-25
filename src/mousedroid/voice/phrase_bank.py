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
    "turn_left": [
        "Turning left! Watch out!",
        "Go left! Swivel swivel!",
        "Left turn. Adjust adjust.",
    ],
    "turn_right": [
        "Turning right! Make way!",
        "Go right! Pivot pivot!",
        "Right turn. Correct course.",
    ],
    "arrived": [
        "Is here! Destination reach!",
        "Arrive! Journey complete!",
        "We here! Very good!",
    ],
    "battery_low_warn": [
        "Warning! Battery getting low.",
        "Power drop. Find charge soon.",
        "Energy low. Suggest recharge.",
    ],
    "battery_critical": [
        "Critical! Battery almost dead!",
        "Emergency charge! Very low!",
        "Shutdown soon! Need power now!",
    ],
    "llm_translation_ack": [
        "Understand! Command receive.",
        "Got it! Process now.",
        "Acknowledge! Working on.",
    ],
    "llm_translation_failed": [
        "Not understand. Repeat please!",
        "Confusion confusion. Try again.",
        "Parse fail. Different words?",
    ],
    "greeting": [
        "Hello! Is Rocky! Very nice meet!",
        "Greetings greetings! How do?",
        "Oh! Is you! Welcome welcome!",
        "Hi hi hi! Rocky very happy see!",
        "Good day! Rocky at service!",
    ],
    "greeting_formal": [
        "Good greetings, esteemed person.",
        "Rocky present and functional. Hello.",
        "Salutations. Rocky report for duty.",
    ],
    "greeting_excited": [
        "Oh oh oh! Is person! Hello hello!",
        "Friend friend! Rocky so happy!",
        "Yes yes! You arrive! Wonderful!",
    ],
    "farewell": [
        "Goodbye! Come back soon yes?",
        "Safe travels! Rocky miss you!",
        "Until next time! Bye bye!",
        "Go well! Rocky wave goodbye!",
    ],
}
