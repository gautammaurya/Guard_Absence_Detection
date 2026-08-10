"""
Business logic for the service — THIS is the primary file the factory fills.

perform_prediction runs once per (non-skipped) frame for one camera. Compose the
outputs of the models in `models` (and optional `resources`) and decide whether
to fire an event. Build the alert payload with the toolbox helpers.

Frozen contract — do NOT change the signature or the return shape:
    perform_prediction(camera_id, frame, conf, models, resources, CLASS_NAME_FOR_ALERT)
        -> (give_event, alert_main_message, bbox_output, image_selection_bbox, new_frame)

    give_event            : bool  — True fires an alert this frame
    alert_main_message    : list[str] — detected class name(s) for this alert; becomes
                             alert.message_details as-is (main.py no longer wraps it in a
                             list). Service-controlled — usually the distinct classes
                             present in bbox_output, but the service decides.
    bbox_output           : list  — [{'top_left','top_right','bottom_left','bottom_right','class'}]
    image_selection_bbox  : list  — zone polygons drawn on the alert image (or [])
    new_frame             : ndarray — annotated frame (or the original `frame`)

This stub is a VALID, RUNNING service: it detects nothing and never alerts.
Replace the body to implement real logic.
"""
import traceback

from service_skeleton.http_server import CAMERAS_DATA


def perform_prediction(camera_id, frame, conf, models, resources, CLASS_NAME_FOR_ALERT):
    try:
        cam = CAMERAS_DATA[camera_id]

        # ── TODO: replace this body with the service logic ──────────────────────
        # Toolbox verbs available (import what you need):
        #   from toolbox.detect import detect            # run a model -> boxes
        #   from toolbox.zones import build_zones, inside_any
        #   from toolbox.tracking import new_tracker, update
        #   from toolbox.associate import associate      # e.g. person <-> PPE
        #   from toolbox.reid import preprocess, cosine   # embeddings / dedup
        #   from toolbox.search import VectorIndex        # FAISS (opt-in)
        #   from toolbox.annotate import plot_bbox
        #
        # Example (single detector + zone entry):
        #   boxes = detect(models["primary"], frame, conf)
        #   zones = build_zones(cam["extra_fields"], frame.shape)
        #   ...
        # ────────────────────────────────────────────────────────────────────────

        give_event = False
        alert_main_message = []
        bbox_output = []
        image_selection_bbox = []
        new_frame = frame

        return give_event, alert_main_message, bbox_output, image_selection_bbox, new_frame
    except Exception:
        traceback.print_exc()
        return False, [], [], [], frame
