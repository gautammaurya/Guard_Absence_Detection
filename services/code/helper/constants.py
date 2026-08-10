from datetime import datetime

# ── Service identity ─────────────────────────────────────────────────────────
SERVICE_DISPLAY_NAME = "Base Service"
EVENT_MESSAGE = "Event Detected"
UPDATED_CONFIDENCE = 0.7  # fallback confidence when the backend sends none

# ── Main-loop frame skipping ─────────────────────────────────────────────────
INFERENCE_SKIP = 3  # only every Nth consumed frame runs perform_prediction

# ── Model registry ───────────────────────────────────────────────────────────
# Declare every model this service loads. The COUNT IS ARBITRARY — main.py never
# inspects this; it just loads them into a {name: model} dict.
#   name    : the key you use as models["<name>"] inside service.py
#   weights : a path, or None to use the backend-managed active version
#   type    : loader key registered in model/models.py LOADERS ("yolo", "openvino", ...)
#
# Examples:
#   single detector        -> [{"name": "primary",   "weights": None, "type": "yolo"}]
#   person + PPE           -> [{"name": "person", ...}, {"name": "ppe", ...}]
#   detect + recognize     -> [{"name": "detector", ...}, {"name": "recognizer", "type": "openvino", ...}]
MODELS = [
    {"name": "primary", "weights": None, "type": "yolo"},
]

# ── Active version (weights=None models resolve to this; training updates it) ──
VERSION_DATA = {
    "version_name": "V1",
    "version": {
        "weights": "resources/weights/model.pt",
        "datetime": str(datetime.now()),
    },
    "version_count": 0,
}

# ── Class + alert config synced with the backend ──────────────────────────────
CLASS_LEVEL_DETAILS = {
    "classes_data": [
        {
            "class_name": "Event Detected",
            "ai_class_name": "object",
            "ai_class_index": 0,
            "value_type": "str",
        }
    ],
    "event_message": EVENT_MESSAGE,
}
