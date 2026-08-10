"""
Guard-absence business logic.

Fires an alert when no security guard has been present in the guard-post zone for
longer than GUARD_ABSENCE_SECONDS (default 300 = 5 minutes).

Why it is built this way — the "very little guard data" problem
──────────────────────────────────────────────────────────────
There is not enough security-guard imagery to train a "security guard" detector,
so this service never tries. It splits the question in two:

  1. "Is a person there?"  → stock COCO-pretrained YOLO. Zero new data, high
     recall, already reliable.
  2. "Is that person the guard?" → few-shot re-identification: embed the person
     crop and cosine-match it against a small enrolled gallery (model/gallery.py).
     Enrolment is an ops action, not a training run, so 10-30 crops per guard is
     enough. The scarce data becomes a gallery instead of a training set.

Four things make this accurate without more data:

  * **Track-level voting.** A track's guard/not-guard label is the MEDIAN of its
    rolling per-frame similarity scores, never a single frame. Single-frame re-ID
    is noisy under pose change, blur and occlusion; a median over a handful of
    frames is dramatically steadier. Biggest available win, costs nothing.
  * **Asymmetric tuning.** This service alerts on the ABSENCE of evidence, so a
    failure to recognise a guard who IS present becomes a false alarm, while an
    over-permissive match merely delays a true alert. Hence a deliberately loose
    match threshold and a low detector confidence — recall over precision.
  * **The absence accumulator is the smoothing.** Absence is integrated over
    time rather than decided per frame, so a three-second occlusion adds three
    seconds to a 300-second budget instead of flipping a state machine. No
    separate debounce window is needed, and any confirmed presence zeroes it.
  * **Crop-quality gating.** Crops too small, too squashed or too blurred for the
    re-ID net are skipped rather than embedded badly — that is where most false
    mismatches come from.

Frozen contract — do NOT change the signature or the return shape:
    perform_prediction(camera_id, frame, conf, models, resources, CLASS_NAME_FOR_ALERT)
        -> (give_event, alert_main_message, bbox_output, image_selection_bbox, new_frame)
"""
import time
import traceback
from collections import deque
from statistics import median

import cv2
import numpy as np

from service_skeleton.http_server import CAMERAS_DATA

from helper.constants import (
    PERSON_CLASS_INDEX,
    PRESENCE_MODE,
    GUARD_ABSENCE_SECONDS,
    GUARD_ABSENCE_REPEAT_SECONDS,
    GUARD_MATCH_THRESHOLD,
    TRACK_SCORE_WINDOW,
    TRACK_VOTE_MIN_SAMPLES,
    TRACK_STATE_TTL_SECONDS,
    TRACK_STATE_MAX_ENTRIES,
    MIN_CROP_WIDTH,
    MIN_CROP_HEIGHT,
    MIN_CROP_SHARPNESS,
    CROP_ASPECT_MIN,
    CROP_ASPECT_MAX,
    GUARD_AUTO_ENROLL,
    AUTO_ENROLL_MIN_SCORE,
    AUTO_ENROLL_COOLDOWN_SECONDS,
    GALLERY_MAX_PER_CAMERA,
    MAX_EVAL_DT_SECONDS,
    LIVE_CROP_CACHE,
    CLASS_LEVEL_DETAILS,
)
from toolbox.detect import detect
from toolbox.zones import build_zones, inside_any
from toolbox.tracking import new_tracker, update

# Importing this registers /enroll_guard/, /guard_gallery/, /clear_guard_gallery/
# and /guard_status/ on the Flask app that main.py later serves. main.py imports
# `service` before it starts the HTTP thread, so the routes are live in time.
import endpoints  # noqa: F401
from toolbox.annotate import plot_bbox, bbox_entry

# Sourced from constants, never hardcoded here — main.py reads it back per box.
GUARD_CLASS = CLASS_LEVEL_DETAILS["classes_data"][0]
GUARD_CLASS_NAME = GUARD_CLASS["class_name"]
GUARD_AI_CLASS_NAME = GUARD_CLASS["ai_class_name"]

COLOR_GUARD = (0, 200, 0)
COLOR_UNVERIFIED = (0, 200, 255)
COLOR_OTHER = (150, 150, 150)
COLOR_ABSENT = (0, 0, 255)
COLOR_ZONE = (255, 200, 0)


def _clamp_box(x1, y1, x2, y2, width, height):
    """Detector boxes can run off-frame; crops must not."""
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def _crop_is_usable(crop, min_w, min_h, aspect_min, aspect_max, min_sharpness):
    """
    Reject crops the re-ID net cannot embed meaningfully.

    Re-ID models are trained on upright full-body pedestrian crops. Feeding them
    a 20px sliver or a motion-blurred smear produces a confident-looking vector
    that matches nothing in particular — the main source of false mismatches.
    """
    if crop is None or crop.size == 0:
        return False
    height, width = crop.shape[:2]
    if width < min_w or height < min_h:
        return False
    aspect = height / float(width)
    if aspect < aspect_min or aspect > aspect_max:
        return False
    if min_sharpness > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < min_sharpness:
            return False
    return True


def _track_label(scores, threshold, min_samples):
    """Median-vote a track into guard / other / unverified."""
    if len(scores) < min_samples:
        return "unverified"
    return "guard" if median(scores) >= threshold else "other"


def _prune_track_state(track_state, now, ttl, max_entries):
    for track_id in [k for k, v in track_state.items() if now - v["last_seen"] > ttl]:
        del track_state[track_id]
    while len(track_state) > max_entries:
        del track_state[next(iter(track_state))]


def _zone_bounds(polygon_points, width, height):
    """Bounding rect of the configured zones, or the whole frame if none."""
    if not polygon_points:
        return 0, 0, width - 1, height - 1
    xs = [int(p[0]) for poly in polygon_points for p in poly]
    ys = [int(p[1]) for poly in polygon_points for p in poly]
    return _clamp_box(min(xs), min(ys), max(xs), max(ys), width, height)


def perform_prediction(camera_id, frame, conf, models, resources, CLASS_NAME_FOR_ALERT):
    try:
        cam = CAMERAS_DATA[camera_id]
        now = time.time()
        height, width = frame.shape[0], frame.shape[1]

        # ── backend-manageable tunables, re-read every frame ─────────────────
        env_vars = cam.get("env_vars", {}) or {}
        presence_mode = str(env_vars.get("PRESENCE_MODE", PRESENCE_MODE)).strip().lower()
        absence_seconds = float(env_vars.get("GUARD_ABSENCE_SECONDS", GUARD_ABSENCE_SECONDS))
        repeat_seconds = float(env_vars.get("GUARD_ABSENCE_REPEAT_SECONDS", GUARD_ABSENCE_REPEAT_SECONDS))
        match_threshold = float(env_vars.get("GUARD_MATCH_THRESHOLD", GUARD_MATCH_THRESHOLD))
        score_window = int(env_vars.get("TRACK_SCORE_WINDOW", TRACK_SCORE_WINDOW))
        vote_min_samples = int(env_vars.get("TRACK_VOTE_MIN_SAMPLES", TRACK_VOTE_MIN_SAMPLES))
        min_crop_width = int(env_vars.get("MIN_CROP_WIDTH", MIN_CROP_WIDTH))
        min_crop_height = int(env_vars.get("MIN_CROP_HEIGHT", MIN_CROP_HEIGHT))
        min_sharpness = float(env_vars.get("MIN_CROP_SHARPNESS", MIN_CROP_SHARPNESS))
        aspect_min = float(env_vars.get("CROP_ASPECT_MIN", CROP_ASPECT_MIN))
        aspect_max = float(env_vars.get("CROP_ASPECT_MAX", CROP_ASPECT_MAX))
        auto_enroll = int(env_vars.get("GUARD_AUTO_ENROLL", GUARD_AUTO_ENROLL))
        auto_enroll_min_score = float(env_vars.get("AUTO_ENROLL_MIN_SCORE", AUTO_ENROLL_MIN_SCORE))
        auto_enroll_cooldown = float(
            env_vars.get("AUTO_ENROLL_COOLDOWN_SECONDS", AUTO_ENROLL_COOLDOWN_SECONDS)
        )
        gallery_cap = int(env_vars.get("GALLERY_MAX_PER_CAMERA", GALLERY_MAX_PER_CAMERA))
        max_dt = float(env_vars.get("MAX_EVAL_DT_SECONDS", MAX_EVAL_DT_SECONDS))
        live_crop_cache = int(env_vars.get("LIVE_CROP_CACHE", LIVE_CROP_CACHE))

        gallery = resources
        # Guard recognition is only meaningful with an embedder AND templates for
        # THIS camera. Otherwise every mode degrades to "any person present",
        # so an unenrolled camera can never alert nonstop.
        reid_active = bool(gallery is not None and gallery.enabled and gallery.count(camera_id) > 0)
        if presence_mode not in ("any_person", "reid_guard", "hybrid") or not reid_active:
            effective_mode = "any_person"
        else:
            effective_mode = presence_mode

        new_frame = frame.copy()

        # ── zone ────────────────────────────────────────────────────────────
        zones, polygon_points = build_zones(cam.get("extra_fields", {}) or {}, frame.shape)
        for points in polygon_points:
            cv2.polylines(new_frame, [np.array(points, dtype="int32")], True, COLOR_ZONE, 2)

        # ── 1. detect people ────────────────────────────────────────────────
        classes = None if PERSON_CLASS_INDEX is None else [PERSON_CLASS_INDEX]
        boxes = detect(models["person"], frame, conf, classes=classes)

        in_zone = []
        for box in boxes:
            x1, y1, x2, y2 = _clamp_box(box[0], box[1], box[2], box[3], width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            if zones:
                inside, _label = inside_any(zones, x1, y1, x2, y2)
                if not inside:
                    continue
            in_zone.append((x1, y1, x2, y2))

        # ── 2. track (SORT must be updated every frame, even with 0 boxes) ───
        if cam.get("tracker") is None:
            cam["tracker"] = new_tracker(max_age=60, min_hits=0, iou_threshold=0.2)
        tracks = update(cam["tracker"], in_zone)

        track_state = cam.setdefault("guard_track_state", {})

        # ── 3. embed + match each tracked person, vote at track level ────────
        # Embedding runs whenever an embedder exists, even in any_person mode, so
        # the live-crop cache is always warm and /enroll_guard/ works from day one
        # without waiting for a mode switch.
        embed_enabled = bool(gallery is not None and gallery.enabled)
        live_crops = []
        annotated = []
        guard_present = False
        person_present = len(tracks) > 0
        best_auto = None  # (score, vector, identity) candidate for auto-enrolment

        for row in tracks:
            x1, y1, x2, y2 = _clamp_box(row[0], row[1], row[2], row[3], width, height)
            track_id = int(row[4])
            if x2 <= x1 or y2 <= y1:
                continue

            state = track_state.get(track_id)
            if state is None or not isinstance(state.get("scores"), deque):
                state = {"scores": deque(maxlen=score_window), "label": "unverified", "last_seen": now}
                track_state[track_id] = state
            state["last_seen"] = now

            vector = None
            if embed_enabled:
                crop = frame[y1:y2, x1:x2]
                if _crop_is_usable(
                    crop, min_crop_width, min_crop_height, aspect_min, aspect_max, min_sharpness
                ):
                    vector = gallery.embed(crop)
                    if vector is not None:
                        live_crops.append(
                            {
                                "crop": crop.copy(),
                                "vector": vector,
                                "box": (x1, y1, x2, y2),
                                "area": (x2 - x1) * (y2 - y1),
                                "ts": now,
                            }
                        )
                        identity, score = gallery.match(camera_id, vector)
                        if identity is not None:
                            state["scores"].append(score)
                            state["identity"] = identity

            state["label"] = _track_label(state["scores"], match_threshold, vote_min_samples)
            if state["label"] == "guard":
                guard_present = True
                if state["scores"]:
                    voted = float(median(state["scores"]))
                    if vector is not None and (best_auto is None or voted > best_auto[0]):
                        best_auto = (voted, vector, state.get("identity"))

            annotated.append((x1, y1, x2, y2, track_id, state["label"]))

        _prune_track_state(track_state, now, TRACK_STATE_TTL_SECONDS, TRACK_STATE_MAX_ENTRIES)

        if live_crops:
            live_crops.sort(key=lambda c: c["area"], reverse=True)
            cam["guard_live_crops"] = live_crops[:max(1, live_crop_cache)]

        # ── 4. resolve presence for this frame ──────────────────────────────
        # present -> zero the absence budget; hold -> freeze it; else -> accrue.
        if effective_mode == "any_person":
            present, hold = person_present, False
        elif effective_mode == "reid_guard":
            present, hold = guard_present, False
        else:  # hybrid
            present = guard_present
            hold = (not guard_present) and person_present

        last_eval = cam.get("guard_last_eval_ts")
        delta = 0.0 if last_eval is None else min(max(now - last_eval, 0.0), max_dt)
        cam["guard_last_eval_ts"] = now

        if present:
            cam["guard_absence_accum"] = 0.0
            cam["guard_last_alert_ts"] = None
        elif not hold:
            cam["guard_absence_accum"] = float(cam.get("guard_absence_accum", 0.0)) + delta
        absence = float(cam.get("guard_absence_accum", 0.0))

        # ── 5. bounded auto-enrolment: let the gallery adapt on its own ──────
        if (
            auto_enroll
            and reid_active
            and best_auto is not None
            and best_auto[0] >= auto_enroll_min_score
            and gallery.count(camera_id) < gallery_cap
            and now - float(cam.get("guard_last_auto_enroll_ts", 0.0)) >= auto_enroll_cooldown
        ):
            identity = best_auto[2] or GUARD_CLASS_NAME
            if gallery.add(camera_id, best_auto[1], identity, pinned=False):
                cam["guard_last_auto_enroll_ts"] = now

        # ── 6. alert decision ───────────────────────────────────────────────
        give_event = False
        alert_main_message = []
        bbox_output = []
        image_selection_bbox = polygon_points

        if absence >= absence_seconds:
            last_alert = cam.get("guard_last_alert_ts")
            if last_alert is None or (repeat_seconds > 0 and now - float(last_alert) >= repeat_seconds):
                give_event = True
                cam["guard_last_alert_ts"] = now

        if give_event:
            # An absence alert has no object to box, so the guard-post zone (or
            # the whole frame when no zone is configured) carries the class.
            # bbox_output must be non-empty: main.py builds alert.classes from it,
            # and an empty list would fail backend class validation silently.
            zx1, zy1, zx2, zy2 = _zone_bounds(polygon_points, width, height)
            bbox_output.append(
                bbox_entry(GUARD_CLASS_NAME, zx1, zy1, zx2, zy2, ai_class_name=GUARD_AI_CLASS_NAME)
            )
            alert_main_message = [GUARD_CLASS_NAME]
            new_frame = plot_bbox(new_frame, (zx1, zy1), (zx2, zy2), COLOR_ABSENT, 3)

        # ── 7. annotate ─────────────────────────────────────────────────────
        for x1, y1, x2, y2, track_id, label in annotated:
            color = {
                "guard": COLOR_GUARD,
                "unverified": COLOR_UNVERIFIED,
                "other": COLOR_OTHER,
            }.get(label, COLOR_OTHER)
            new_frame = plot_bbox(new_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                new_frame, f"#{track_id} {label}", (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        status = "PRESENT" if present else ("HOLD" if hold else "ABSENT")
        cv2.putText(
            new_frame,
            f"{effective_mode} | {status} | absent {int(absence)}s / {int(absence_seconds)}s",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLOR_ABSENT if absence >= absence_seconds else COLOR_GUARD, 2, cv2.LINE_AA,
        )

        return give_event, alert_main_message, bbox_output, image_selection_bbox, new_frame
    except Exception:
        traceback.print_exc()
        return False, [], [], [], frame
