"""
Spatial association between two sets of boxes — e.g. matching PPE items to the
person wearing them, or a plate to a vehicle.
"""


def iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2, ...) boxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def associate(primary, secondary, iou_threshold=0.1):
    """
    For each primary box, collect the secondary boxes that overlap it.

    Returns: [(primary_box, [matched_secondary_boxes]), ...]
    Useful for compliance-style logic ("does this person have a helmet?").
    """
    result = []
    for p in primary:
        matches = [s for s in secondary if iou(p, s) >= iou_threshold]
        result.append((p, matches))
    return result
