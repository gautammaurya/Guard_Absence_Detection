"""
Request-driven surface: guard enrolment, gallery inspection and live status.

This is how the "very little guard data" problem is actually solved in the field
— enrolling a guard is an ops action over HTTP, not a training run. Point the
camera at the guard, POST /enroll_guard/, and the post is protected in seconds.

Routes register on the Flask `app` from service_skeleton at import time. service.py
imports this module, and main.py imports service.py before starting the HTTP
thread, so everything below is live by the time the server accepts connections.
main.py itself is untouched.

Endpoints
─────────
POST /enroll_guard/
    {"camera_id": 12, "identity": "guard_a"}                      # from live view
    {"camera_id": 12, "identity": "guard_a", "image_b64": "...",   # from an image
     "bbox": [x1, y1, x2, y2]}                                     # bbox optional
    Also accepts multipart form-data with an `image` file part.
    "from live view" enrols the largest recent quality-gated person crop that
    service.py cached for that camera — no extra inference, no bbox needed.

GET  /guard_gallery/?camera_id=12          template counts per identity
POST /clear_guard_gallery/                 {"camera_id": 12, "identity": "guard_a"}
GET  /guard_status/?camera_id=12           live presence / absence-timer state
"""
import base64
import os
import time
import traceback

import cv2
import numpy as np
from flask import jsonify, request

from service_skeleton.http_server import app, CAMERAS_DATA

from helper.constants import (
    GALLERY_SEED_DIR,
    GUARD_IDENTITY_DEFAULT,
    PRESENCE_MODE,
    GUARD_ABSENCE_SECONDS,
)
from model import gallery as gallery_module


def _payload():
    """Accept JSON, form-encoded or query-string parameters interchangeably."""
    data = {}
    if request.is_json:
        data.update(request.get_json(silent=True) or {})
    data.update({k: v for k, v in request.form.items() if k not in data})
    data.update({k: v for k, v in request.args.items() if k not in data})
    return data


def _decode_image(data):
    """Pull a BGR image out of a multipart upload or a base64 field."""
    upload = request.files.get("image")
    if upload is not None:
        buffer = np.frombuffer(upload.read(), dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    encoded = data.get("image_b64")
    if encoded:
        if "," in encoded[:64]:  # tolerate a data: URL prefix
            encoded = encoded.split(",", 1)[1]
        buffer = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return None


def _persist_seed_crop(gallery, camera_id, identity, crop):
    """
    Write the enrolled crop under the seed directory.

    Templates outlive the .npz: if the re-ID model is ever swapped, every pinned
    identity can be re-embedded from these images instead of re-enrolled by hand.
    The relative path is recorded as already-seeded so a restart doesn't add a
    duplicate template for the same file.
    """
    try:
        cam = str(camera_id)
        directory = os.path.join(GALLERY_SEED_DIR, cam, identity)
        os.makedirs(directory, exist_ok=True)
        name = f"{int(time.time() * 1000)}.jpg"
        cv2.imwrite(os.path.join(directory, name), crop)
        gallery.seeded_files.add(os.path.join(cam, identity, name))
        gallery.save()
        return os.path.join(directory, name)
    except Exception:
        traceback.print_exc()
        return None


@app.route("/enroll_guard/", methods=["POST"])
def enroll_guard():
    try:
        gallery = gallery_module.GALLERY
        if gallery is None:
            return jsonify({"status": "error", "message": "gallery not initialised yet"}), 503
        if not gallery.enabled:
            return jsonify({
                "status": "error",
                "message": "re-ID embedder unavailable — drop the OpenVINO IR pair into "
                           "resources/weights/ and restart the service",
            }), 503

        data = _payload()
        camera_id = data.get("camera_id")
        if camera_id in (None, ""):
            return jsonify({"status": "error", "message": "camera_id is required"}), 400
        identity = str(data.get("identity") or GUARD_IDENTITY_DEFAULT)

        image = _decode_image(data)
        enrolled = []

        if image is not None:
            bbox = data.get("bbox")
            if isinstance(bbox, str):
                bbox = [int(float(v)) for v in bbox.replace("[", "").replace("]", "").split(",")]
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            else:
                crop = image  # already a person crop
            vector = gallery.embed(crop)
            if vector is None:
                return jsonify({"status": "error", "message": "could not embed the supplied image"}), 400
            gallery.add(camera_id, vector, identity, pinned=True)
            enrolled.append({"source": "image", "saved_as": _persist_seed_crop(gallery, camera_id, identity, crop)})
        else:
            # Enrol whoever is standing at the post right now, using the crops
            # service.py already embedded on the last processed frame.
            cam = CAMERAS_DATA.get(camera_id) or CAMERAS_DATA.get(str(camera_id))
            if cam is None:
                return jsonify({"status": "error", "message": f"unknown camera_id {camera_id}"}), 404
            crops = cam.get("guard_live_crops") or []
            if not crops:
                return jsonify({
                    "status": "error",
                    "message": "no recent person crop cached for this camera — make sure a person is "
                               "visible in the guard-post zone and large enough to pass the crop-quality gate",
                }), 409
            limit = int(data.get("count") or 1)
            for entry in crops[:max(1, limit)]:
                if gallery.add(camera_id, entry["vector"], identity, pinned=True):
                    enrolled.append({
                        "source": "live",
                        "box": list(entry["box"]),
                        "saved_as": _persist_seed_crop(gallery, camera_id, identity, entry["crop"]),
                    })

        return jsonify({
            "status": "success",
            "camera_id": camera_id,
            "identity": identity,
            "enrolled": len(enrolled),
            "details": enrolled,
            "gallery_size": gallery.count(camera_id),
            "hint": "set PRESENCE_MODE=hybrid in this camera's env_vars once enrolment looks right",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/guard_gallery/", methods=["GET"])
def guard_gallery():
    try:
        gallery = gallery_module.GALLERY
        if gallery is None:
            return jsonify({"status": "error", "message": "gallery not initialised yet"}), 503
        camera_id = request.args.get("camera_id")
        return jsonify({
            "status": "success",
            "reid_available": gallery.enabled,
            "gallery": gallery.info(camera_id if camera_id else None),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/clear_guard_gallery/", methods=["POST"])
def clear_guard_gallery():
    try:
        gallery = gallery_module.GALLERY
        if gallery is None:
            return jsonify({"status": "error", "message": "gallery not initialised yet"}), 503
        data = _payload()
        camera_id = data.get("camera_id")
        if camera_id in (None, ""):
            return jsonify({"status": "error", "message": "camera_id is required"}), 400
        removed = gallery.clear(camera_id, data.get("identity") or None)
        return jsonify({"status": "success", "removed": removed, "gallery_size": gallery.count(camera_id)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/guard_status/", methods=["GET"])
def guard_status():
    """Live absence-timer state — the first thing to check when tuning."""
    try:
        gallery = gallery_module.GALLERY
        camera_id = request.args.get("camera_id")
        cameras = CAMERAS_DATA
        if camera_id:
            cam = cameras.get(camera_id) or cameras.get(str(camera_id))
            try:
                cam = cam if cam is not None else cameras.get(int(camera_id))
            except (TypeError, ValueError):
                pass
            if cam is None:
                return jsonify({"status": "error", "message": f"unknown camera_id {camera_id}"}), 404
            selected = {camera_id: cam}
        else:
            selected = dict(cameras)

        now = time.time()
        out = {}
        for key, cam in selected.items():
            env_vars = cam.get("env_vars", {}) or {}
            labels = [state.get("label") for state in (cam.get("guard_track_state") or {}).values()]
            out[str(key)] = {
                "presence_mode": str(env_vars.get("PRESENCE_MODE", PRESENCE_MODE)),
                "absence_seconds_threshold": float(
                    env_vars.get("GUARD_ABSENCE_SECONDS", GUARD_ABSENCE_SECONDS)
                ),
                "absence_accumulated_seconds": round(float(cam.get("guard_absence_accum", 0.0)), 1),
                "tracks": {
                    "guard": labels.count("guard"),
                    "unverified": labels.count("unverified"),
                    "other": labels.count("other"),
                },
                "seconds_since_last_eval": (
                    None if cam.get("guard_last_eval_ts") is None
                    else round(now - float(cam["guard_last_eval_ts"]), 1)
                ),
                "seconds_since_last_alert": (
                    None if cam.get("guard_last_alert_ts") is None
                    else round(now - float(cam["guard_last_alert_ts"]), 1)
                ),
                "gallery_size": 0 if gallery is None else gallery.count(key),
            }
        return jsonify({
            "status": "success",
            "reid_available": bool(gallery is not None and gallery.enabled),
            "cameras": out,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
