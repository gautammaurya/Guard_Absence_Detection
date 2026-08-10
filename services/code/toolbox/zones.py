"""
Zone polygon construction + containment checks.

Zone coordinates from the UI use a fixed canvas; they are scaled to actual frame
pixels here. Two zone types come from a camera's `extra_fields`:
  squares_selection      — rectangles, each carrying a text label
  image_selection_params — freehand polygons
"""
import traceback
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

UI_CANVAS_WIDTH = 1090
UI_CANVAS_HEIGHT = 452


def _create_polygon(selection, scalex, scaley, is_square):
    if not is_square:
        return [(int(p["x"] * scalex), int(p["y"] * scaley)) for p in selection]
    x, y = selection["x"] * scalex, selection["y"] * scaley
    w, h = selection["width"] * scalex, selection["height"] * scaley
    x2, y2 = int(x + w), int(y + h)
    return [(x, y), (x2, y), (x2, y2), (x, y2)]


def build_zones(extra_fields, frame_shape):
    """
    Build all configured zones once per frame.
    Returns (zones, polygon_points) where:
        zones          = [(Polygon, label), ...] for containment checks
        polygon_points = [[(x,y), ...], ...]     raw points for drawing
    """
    try:
        h_org, w_org = frame_shape[0], frame_shape[1]
        scalex = w_org / UI_CANVAS_WIDTH
        scaley = h_org / UI_CANVAS_HEIGHT

        zones, polygon_points = [], []
        for sel in extra_fields.get("squares_selection", []):
            pts = _create_polygon(sel, scalex, scaley, True)
            zones.append((Polygon(pts), sel.get("text", "")))
            polygon_points.append(pts)
        for sel in extra_fields.get("image_selection_params", []):
            pts = _create_polygon(sel, scalex, scaley, False)
            zones.append((Polygon(pts), ""))
            polygon_points.append(pts)
        return zones, polygon_points
    except Exception as e:
        traceback.print_exc()
        print(f"Error building zones: {e}")
        return [], []


def inside_any(zones, x1, y1, x2, y2):
    """
    Test a box against all zones using three probe points (mid, 3/4, bottom-center).
    Returns (inside: bool, label: str).
    """
    x_mid = int(x1 + (x2 - x1) / 2)
    probes = [
        Point(x_mid, int(y1 + (y2 - y1) / 2)),
        Point(x_mid, int(y1 + (y2 - y1) * 3 / 4)),
        Point(x_mid, y2),
    ]
    for polygon, label in zones:
        if any(polygon.contains(p) for p in probes):
            return True, label
    return False, ""
