"""Frame annotation helpers."""
import cv2


def plot_bbox(frame, pt1, pt2, color=(0, 255, 0), thickness=2):
    """Draw a rectangle on the frame and return it."""
    return cv2.rectangle(frame, pt1, pt2, color, thickness, 8)


def bbox_entry(text, x1, y1, x2, y2, ai_class_name=None):
    """
    Build the standard bbox_output dict used in alert payloads.

    ai_class_name is optional and, when given, is read back by main.py to
    populate alert.classes[]['ai_class_name'] for this box's class — sourced
    from the service's own CLASS_LEVEL_DETAILS (helper/constants.py), not
    hardcoded in the frozen main.py. Omit it and main.py falls back to
    "object" (the original single-class stub behaviour).
    """
    entry = {
        'top_left': (x1, y1),
        'top_right': (x2, y1),
        'bottom_left': (x1, y2),
        'bottom_right': (x2, y2),
        'class': text,
    }
    if ai_class_name is not None:
        entry['ai_class_name'] = ai_class_name
    return entry
