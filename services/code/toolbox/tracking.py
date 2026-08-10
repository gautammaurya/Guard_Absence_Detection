"""
SORT tracker helpers. `Sort` is provided by the service_skeleton package.

A tracker is a 2-tuple: (Sort instance, {tracker_id: state_bool}). Store it in the
per-camera state (see generate_ai_variables) so IDs stay stable across frames.
"""
import numpy as np
from service_skeleton.http_server import Sort


def new_tracker(max_age=100, min_hits=0, iou_threshold=0.1):
    """Create a fresh (Sort, id_state) tuple for one camera."""
    return (Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold), {})


def update(tracker, boxes):
    """
    Update the tracker with this frame's detections.

    SORT's own contract requires update() to be called every frame, even with
    zero detections (passing an empty (0, 5) array) — otherwise its internal
    frame/age bookkeeping stalls and stale tracks never expire via max_age.

    Args:
        tracker : the (Sort, id_state) tuple.
        boxes   : list of (x1, y1, x2, y2, ...) detections (may be empty).
    Returns:
        numpy array of [x1, y1, x2, y2, tracker_id] rows (possibly empty).
    """
    if not boxes:
        dets = np.empty((0, 5))
    else:
        dets = np.array([[b[0], b[1], b[2], b[3], 1.0] for b in boxes])
    return tracker[0].update(dets)


def prune(tracker, max_ids=500):
    """Bound memory by dropping the oldest tracked-id state entries."""
    state = tracker[1]
    while len(state) > max_ids:
        del state[next(iter(state))]
