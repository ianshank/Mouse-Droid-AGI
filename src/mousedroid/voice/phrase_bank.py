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
        "Task complete! Rocky proud!",
        "Finish finish! We did good!",
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
        "Rocky here. Ready when you ready.",
        "Quiet quiet. Rocky wait patient.",
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
        "Made it made it! Rocky happy!",
        "Stop here. Is right place!",
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
    # --- Conversational vocabulary (LLM answer_query path) ----------------- #
    # Spoken around the deliberative Q&A path so the operator hears Rocky
    # acknowledge / frame an answer. The answer text itself is dynamic (from
    # the LLM) and is spoken via ``RockyVoiceEngine.play_phrase``; these events
    # are the static framing the phrase bank supplies.
    "query_received": [
        "Question! Rocky think think.",
        "You ask? Rocky listen good!",
        "Ooh, good question. Let Rocky see.",
        "Hmm! Rocky consider this.",
    ],
    "query_answered": [
        "Rocky know this! Here answer:",
        "Aha! Rocky tell you:",
        "Is easy. Answer is:",
        "Rocky figure out! Listen:",
    ],
    "query_failed": [
        "Not sure. Rocky brain quiet.",
        "No answer come. Sorry sorry.",
        "Rocky not know this one. Ask different?",
        "Confusion! Rocky cannot answer.",
    ],
    "thinking": [
        "Hmm... compute compute.",
        "Wait wait. Rocky think hard.",
        "Processing... almost almost!",
        "One moment. Brain work work.",
    ],
    "acknowledge": [
        "Okay okay! Rocky understand.",
        "Got it got it!",
        "Roger roger! Rocky hear you.",
        "Yes! Rocky on it.",
    ],
    "affirmative": [
        "Yes yes! Is correct.",
        "Affirmative! Rocky agree.",
        "Yes! Very true very true.",
    ],
    "negative": [
        "No no. Is not so.",
        "Negative! Rocky disagree.",
        "No. Wrong wrong wrong.",
    ],
}
