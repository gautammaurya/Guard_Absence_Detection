"""Generic detection helper — run a YOLO model and return plain box tuples."""
import traceback


def detect(model, frame, conf, classes=None, imgsz=1280):
    """
    Run inference and return a list of (x1, y1, x2, y2, cls) integer tuples.

    Args:
        model  : a loaded YOLO model from the registry.
        frame  : BGR image (numpy array).
        conf   : confidence threshold.
        classes: optional list of class indices to keep (None = all).
        imgsz  : inference resolution.
    """
    try:
        kwargs = {"verbose": False, "imgsz": imgsz, "conf": conf, "device": 0}
        if classes is not None:
            kwargs["classes"] = classes
        raw = model.predict(frame, **kwargs)[0].boxes.cpu().numpy()

        boxes = []
        for b in raw:
            x, y, w, h = b.xywh[0]
            x1, y1 = int(x - w / 2), int(y - h / 2)
            boxes.append((x1, y1, int(x1 + w), int(y1 + h), int(b.cls[0])))
        return boxes
    except Exception as e:
        traceback.print_exc()
        print(f"Error during detection: {e}")
        return []
