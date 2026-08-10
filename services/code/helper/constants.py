from datetime import datetime

# ── Service identity ─────────────────────────────────────────────────────────
SERVICE_DISPLAY_NAME = "Guard Absence"
EVENT_MESSAGE = "Guard Absence Detected"

# Person-detection confidence fallback when the backend sends none. Deliberately
# LOWER than the base-service default of 0.7: this service alerts on the ABSENCE
# of evidence, so a missed person detection becomes a FALSE ALARM. Recall on
# "someone is here" matters far more than precision.
UPDATED_CONFIDENCE = 0.45

# ── Main-loop frame skipping ─────────────────────────────────────────────────
INFERENCE_SKIP = 3  # only every Nth consumed frame runs perform_prediction

# ── Model registry ───────────────────────────────────────────────────────────
# Only the person DETECTOR lives here. It is a stock COCO-pretrained YOLO — this
# service intentionally does NOT train a "security guard" class, because there
# isn't enough guard data to train one. Guard identity is handled by few-shot
# re-identification against an enrolled gallery instead (see model/gallery.py).
#
# The re-ID embedder is deliberately NOT declared here: it is owned by the
# gallery resource built in model/models.py::initialise_resources(), because it
# must also serve the /enroll_guard/ HTTP endpoint, which runs outside the frame
# loop where the `models` registry isn't reachable. One handle, two callers.
MODELS = [
    {"name": "person", "weights": None, "type": "yolo"},
]

# COCO "person" index for the detector above. Set to None to keep every class.
PERSON_CLASS_INDEX = 0

# ── Re-ID embedder (owned by the gallery, not by MODELS) ─────────────────────
# OpenVINO IR pair (.xml + .bin) must both sit in resources/weights/.
# Recommended: person-reidentification-retail-0287 (256-d output).
# If the file is missing the service still runs — it degrades to PRESENCE_MODE
# "any_person" and logs a warning rather than crashing.
REID_WEIGHTS = "resources/weights/person-reidentification-retail-0287.xml"

# Where the enrolled guard gallery is persisted across restarts.
GALLERY_PATH = "resources/gallery/guard_gallery.npz"
# Optional cold-start seed: resources/gallery/seed/<camera_id>/<identity>/*.jpg
# Every image in there is embedded and pinned into the gallery at startup.
GALLERY_SEED_DIR = "resources/gallery/seed"

# ── Active version (weights=None models resolve to this; training updates it) ──
VERSION_DATA = {
    "version_name": "V1",
    "version": {
        "weights": "resources/weights/yolov8m.pt",
        "datetime": str(datetime.now()),
    },
    "version_count": 0,
}

# ── Class + alert config synced with the backend ──────────────────────────────
CLASS_LEVEL_DETAILS = {
    "classes_data": [
        {
            "class_name": "Guard Absent",
            "ai_class_name": "guard_absent",
            "ai_class_index": 0,
            "value_type": "str",
        }
    ],
    "event_message": EVENT_MESSAGE,
}

# ─────────────────────────────────────────────────────────────────────────────
# Backend-manageable tunables.
#
# Every constant below is a FALLBACK ONLY. service.py re-reads each one from
# CAMERAS_DATA[camera_id]["env_vars"] on every frame, under exactly the name
# used here, so ops can retune any of them per camera from the backend with no
# redeploy and no restart. See CONTRACT.md, "Backend-manageable parameters".
# ─────────────────────────────────────────────────────────────────────────────

# How the service decides "the guard is present":
#   "any_person"  — any person inside the guard-post zone counts. Needs no
#                   gallery at all. THE DAY-ONE DEFAULT: safe, immediately
#                   useful, and impossible to false-alarm from re-ID error.
#   "reid_guard"  — only a gallery match counts. Strictest; every re-ID miss
#                   becomes a false absence alert.
#   "hybrid"      — a gallery match resets the absence timer; an unrecognised
#                   person HOLDS the timer (neither resets nor accrues). Flip
#                   to this once a gallery is enrolled and proven.
# In "hybrid"/"reid_guard", a camera whose gallery is still empty automatically
# degrades to "any_person" — an unenrolled camera must never alert nonstop.
PRESENCE_MODE = "any_person"

# Absence duration that fires the alert (the "5 minutes" requirement).
GUARD_ABSENCE_SECONDS = 300.0
# Re-alert cadence while the post is STILL empty after the first alert.
# 0 disables repeats (one alert per absence episode).
GUARD_ABSENCE_REPEAT_SECONDS = 600.0

# Cosine similarity above which a person crop is accepted as an enrolled guard.
# Kept deliberately LOOSE. A missed match delays nothing but produces a false
# alarm; an over-permissive match only delays a true alert, which the absence
# accumulator absorbs. Tune down (0.45) if you get false alarms, up (0.65) if
# passers-by are being counted as the guard.
GUARD_MATCH_THRESHOLD = 0.55

# Track-level voting: a track's guard/not-guard label comes from the MEDIAN of
# its per-frame similarity scores, not from any single frame. This is the single
# biggest accuracy win available without more data — single-frame re-ID is noisy
# under pose change, motion blur and partial occlusion.
TRACK_SCORE_WINDOW = 15      # rolling scores kept per track
TRACK_VOTE_MIN_SAMPLES = 4   # scores needed before a track is labelled at all
TRACK_STATE_TTL_SECONDS = 120.0  # forget per-track state this long after last sight
TRACK_STATE_MAX_ENTRIES = 300     # hard cap on per-camera track state

# Crop-quality gate applied BEFORE embedding. Re-ID nets expect full-body
# pedestrian crops; tiny, squashed or blurred crops are where most false
# mismatches come from, so they are skipped rather than embedded badly.
MIN_CROP_WIDTH = 48
MIN_CROP_HEIGHT = 96
MIN_CROP_SHARPNESS = 25.0    # Laplacian variance; 0 disables the blur check
CROP_ASPECT_MIN = 1.2        # height / width
CROP_ASPECT_MAX = 4.5

# Bounded auto-enrolment: very confident matches are appended back into the
# gallery so it adapts to lighting, shift changes and seasons on its own. Capped
# and cooldown-throttled so it cannot drift away from the pinned seed crops.
GUARD_AUTO_ENROLL = 1              # 0 disables
AUTO_ENROLL_MIN_SCORE = 0.78       # must clear this, well above the match gate
AUTO_ENROLL_COOLDOWN_SECONDS = 120.0
GALLERY_MAX_PER_CAMERA = 64        # oldest un-pinned vectors evicted past this

# Identity label used when enrolment doesn't supply one.
GUARD_IDENTITY_DEFAULT = "guard"

# Safety clamp on the per-frame time delta fed into the absence accumulator. A
# service pause, a camera dropout or a restart must not instantly bank 5 minutes
# of "absence" the moment frames resume.
MAX_EVAL_DT_SECONDS = 30.0

# How many recent quality-gated person crops (largest first) to keep per camera
# so /enroll_guard/ can enrol "whoever is standing at the post right now"
# without re-running any inference.
LIVE_CROP_CACHE = 3
