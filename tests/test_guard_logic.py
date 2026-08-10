"""Smoke test: gallery persistence + absence accumulator, with deps stubbed."""
import os, sys, types, tempfile, shutil
import numpy as np

CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "services", "code")
sys.path.insert(0, CODE)
os.chdir(tempfile.mkdtemp())

# ── stub openvino (gallery -> toolbox.reid imports it) ───────────────────────
ov = types.ModuleType("openvino"); ov.Core = lambda: None
sys.modules["openvino"] = ov

# ── stub flask (endpoints.py) ────────────────────────────────────────────────
flask = types.ModuleType("flask")
flask.jsonify = lambda *a, **k: (a, k)
class _Req:
    is_json = False; form = {}; args = {}; files = {}
    def get_json(self, silent=False): return {}
flask.request = _Req()
sys.modules["flask"] = flask

# ── stub service_skeleton.http_server ───────────────────────────────────────
class FakeSort:
    """Order-stable stand-in for SORT: one persistent id per detection slot."""
    def __init__(self, **kw): pass
    def update(self, dets):
        if len(dets) == 0: return np.empty((0, 5))
        return np.array([[d[0], d[1], d[2], d[3], i + 1] for i, d in enumerate(dets)])
class FakeApp:
    def route(self, *a, **k): return lambda f: f
dt = types.ModuleType("data_transform"); dt.MassagerFactory = object
sys.modules["data_transform"] = dt

# ── stub shapely with a REAL ray-casting polygon so zone tests are honest ────
class _Point:
    def __init__(self, x, y): self.x, self.y = float(x), float(y)
class _Polygon:
    def __init__(self, pts): self.pts = [(float(a), float(b)) for a, b in pts]
    def contains(self, p):
        inside, n = False, len(self.pts)
        for i in range(n):
            x1, y1 = self.pts[i]; x2, y2 = self.pts[(i + 1) % n]
            if (y1 > p.y) != (y2 > p.y):
                if p.x < x1 + (p.y - y1) / (y2 - y1) * (x2 - x1):
                    inside = not inside
        return inside
_sh = types.ModuleType("shapely")
_shg = types.ModuleType("shapely.geometry"); _shg.Point = _Point
_shgp = types.ModuleType("shapely.geometry.polygon"); _shgp.Polygon = _Polygon
sys.modules["shapely"] = _sh
sys.modules["shapely.geometry"] = _shg
sys.modules["shapely.geometry.polygon"] = _shgp

ss = types.ModuleType("service_skeleton"); hs = types.ModuleType("service_skeleton.http_server")
hs.CAMERAS_DATA = {}; hs.Sort = FakeSort; hs.app = FakeApp()
hs.CAMERAS = {}
hs.get_config_json = lambda p: {"service_name": "Guard Absence Detection"}
hs.success_info = lambda *a, **k: None
hs.danger_info = lambda *a, **k: None
hs.send_version_and_classes_details = lambda *a, **k: None
hs.set_parameter_with_servce_name = lambda *a, **k: {}
hs.get_class_mapping_list = lambda *a, **k: {}
hs.get_current_function_name = lambda *a, **k: "fn"
hs.DJANGO_API_ENDPOINT = "http://stub"
hs.GET_CLASS_MAPPING_DETAILS_URL = "http://stub"
hs.RETRAINING_DICT = {}
class _AIVars:
    value_generator = None
hs.AI_VARIABLES = _AIVars()
hs.__all__ = [n for n in vars(hs) if not n.startswith("__")]
sys.modules["service_skeleton"] = ss; sys.modules["service_skeleton.http_server"] = hs

from model.gallery import GuardGallery
import model.gallery as gmod

# ── fake re-ID model: deterministic embedding from mean pixel colour ─────────
class FakeReid:
    height, width = 256, 128
    def predict(self, blob):
        seed = int(abs(blob.mean()) * 1000) % 9973
        rng = np.random.default_rng(seed)
        return rng.normal(size=(1, 256)).astype("float32")

print("── gallery ──")
g = GuardGallery(FakeReid(), "resources/gallery/guard_gallery.npz",
                 seed_dir="resources/gallery/seed", max_per_camera=4)
assert g.enabled
v1 = g.embed(np.full((300, 120, 3), 40, dtype=np.uint8))
v2 = g.embed(np.full((300, 120, 3), 90, dtype=np.uint8))
assert v1 is not None and abs(np.linalg.norm(v1) - 1.0) < 1e-5, "not L2-normalised"
g.add(7, v1, "guard_a", pinned=True)
assert g.count(7) == 1 and g.count(9) == 0, "per-camera separation broken"
ident, score = g.match(7, v1)
assert ident == "guard_a" and score > 0.99, (ident, score)
ident2, score2 = g.match(7, v2)
assert score2 < 0.5, f"unrelated vector scored {score2}"
assert g.match(9, v1) == (None, 0.0), "empty camera must not match"

# eviction keeps pinned, drops oldest un-pinned
for i in range(6):
    g.add(7, g.embed(np.full((300, 120, 3), 100 + i * 10, dtype=np.uint8)), "auto", pinned=False)
assert g.count(7) == 4, g.count(7)
info = g.info(7)["7"]
assert info["identities"]["guard_a"]["pinned"] == 1, "pinned template was evicted!"
print("  eviction respects pinning:", info)

# persistence round-trip
g.save()
g2 = GuardGallery(FakeReid(), "resources/gallery/guard_gallery.npz")
assert g2.load()
assert g2.count(7) == 4, g2.count(7)
assert g2.match(7, v1)[0] == "guard_a", "identity lost across save/load"
print("  save/load round-trip OK, dim =", g2.dim)

# disk seeding, idempotent
import cv2
for n in range(2):
    d = "resources/gallery/seed/7/guard_b"; os.makedirs(d, exist_ok=True)
    cv2.imwrite(f"{d}/{n}.jpg", np.full((300, 120, 3), 200 + n, dtype=np.uint8))
g3 = GuardGallery(FakeReid(), "resources/gallery/g2.npz", seed_dir="resources/gallery/seed", max_per_camera=64)
assert g3.seed_from_disk() == 2
assert g3.seed_from_disk() == 0, "re-seeding duplicated templates"
assert g3.clear(7, "guard_b") == 2 and g3.count(7) == 0
print("  seeding idempotent + clear OK")

# ── absence accumulator via the real perform_prediction ─────────────────────
print("── perform_prediction ──")
gmod.GALLERY = g
import service

# Pull the REAL generate_ai_variables body out of train_apis.py without importing
# the module (its import runs _init_service(), which calls Django).
import ast, traceback as _tb
_src = open(os.path.join(CODE, "helper/train_apis.py")).read()
_fn = [n for n in ast.parse(_src).body
       if isinstance(n, ast.FunctionDef) and n.name == "generate_ai_variables"][0]
_ns = {"traceback": _tb}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), "train_apis", "exec"), _ns)
make_state = _ns["generate_ai_variables"]
_state = make_state()
for required in ("skip_counter", "count_check_dict_in", "tracker", "guard_absence_accum",
                 "guard_last_eval_ts", "guard_last_alert_ts"):
    assert required in _state, f"generate_ai_variables missing {required}"
print("  generate_ai_variables keys:", sorted(_state))

CAM = 7
hs.CAMERAS_DATA[CAM] = make_state()
hs.CAMERAS_DATA[CAM]["extra_fields"] = {}
hs.CAMERAS_DATA[CAM]["env_vars"] = {"GUARD_ABSENCE_SECONDS": 10, "GUARD_ABSENCE_REPEAT_SECONDS": 5}

# Textured frame so crops clear the Laplacian sharpness gate and really do get
# embedded + matched, rather than being skipped as blurry.
frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
PERSON = [(500, 200, 620, 560, 0)]
service.detect = lambda model, f, c, classes=None, imgsz=1280: service._TEST_BOXES

def step(boxes, t):
    service._TEST_BOXES = boxes
    service.time.time = lambda: t
    return service.perform_prediction(CAM, frame, 0.45, {"person": None}, g, {})

# first frame with a person: establishes the clock, no absence
ev, msg, bb, _, _ = step(PERSON, 1000.0)
assert ev is False and msg == [] and bb == []
# person leaves; absence accrues but stays under the 10s threshold
ev, *_ = step([], 1005.0)
assert ev is False, "alerted before threshold"
assert 4.9 < hs.CAMERAS_DATA[CAM]["guard_absence_accum"] < 5.1
# crosses the threshold -> alert, with a class-name list and a non-empty bbox
ev, msg, bb, zones, _ = step([], 1012.0)
assert ev is True, "no alert after threshold"
assert msg == ["Guard Absent"] and isinstance(msg, list), msg
assert len(bb) == 1 and bb[0]["ai_class_name"] == "guard_absent", bb
print("  fired at threshold:", msg, bb[0]["class"], bb[0]["ai_class_name"])
# immediately after: no duplicate alert inside the repeat window
ev, *_ = step([], 1013.0)
assert ev is False, "duplicate alert inside repeat window"
# still absent past the repeat window -> re-alert
ev, *_ = step([], 1020.0)
assert ev is True, "no re-alert after repeat window"
# guard returns -> timer and alert state reset
ev, *_ = step(PERSON, 1021.0)
assert ev is False and hs.CAMERAS_DATA[CAM]["guard_absence_accum"] == 0.0
assert hs.CAMERAS_DATA[CAM]["guard_last_alert_ts"] is None
print("  presence zeroes the accumulator and re-arms")
# long stall (camera offline 1 hour) must be clamped, not banked
step([], 1021.0 + 3600)
banked = hs.CAMERAS_DATA[CAM]["guard_absence_accum"]
assert banked <= 30.1, f"stall banked {banked}s of absence"
print(f"  1h stall clamped to {banked}s (MAX_EVAL_DT_SECONDS)")

# hybrid mode: unrecognised person HOLDS the timer instead of resetting it
hs.CAMERAS_DATA[CAM] = make_state()
hs.CAMERAS_DATA[CAM]["extra_fields"] = {}
hs.CAMERAS_DATA[CAM]["env_vars"] = {"PRESENCE_MODE": "hybrid", "GUARD_ABSENCE_SECONDS": 10}
step([], 2000.0)
step([], 2008.0)                      # empty: accrue to 8s
before = hs.CAMERAS_DATA[CAM]["guard_absence_accum"]
ev, *_ = step(PERSON, 2015.0)         # unknown person: hold
held = hs.CAMERAS_DATA[CAM]["guard_absence_accum"]
assert abs(held - before) < 0.01, f"hybrid did not hold: {before} -> {held}"
assert ev is False
print(f"  hybrid holds timer on unrecognised person ({before:.1f}s -> {held:.1f}s)")

# zone filtering: a person OUTSIDE the guard post must not count as presence.
# UI canvas is 1090x452, frame is 1280x720 -> zone spans x[400,700] y[150,600].
hs.CAMERAS_DATA[CAM] = make_state()
hs.CAMERAS_DATA[CAM]["extra_fields"] = {"squares_selection": [
    {"x": 400 / (1280 / 1090), "y": 150 / (720 / 452),
     "width": 300 / (1280 / 1090), "height": 450 / (720 / 452), "text": "post"}]}
hs.CAMERAS_DATA[CAM]["env_vars"] = {"GUARD_ABSENCE_SECONDS": 10}
OUTSIDE = [(900, 200, 1000, 560, 0)]
step(PERSON, 3000.0)
assert hs.CAMERAS_DATA[CAM]["guard_absence_accum"] == 0.0
step(PERSON, 3005.0)                       # inside the post: no absence banked
assert hs.CAMERAS_DATA[CAM]["guard_absence_accum"] == 0.0, "in-zone person ignored"
ev, msg, bb, zone_out, _ = step(OUTSIDE, 3011.0)   # outside: absence accrues
assert hs.CAMERAS_DATA[CAM]["guard_absence_accum"] > 5.9, "out-of-zone person counted as present"
assert ev is False
ev, msg, bb, zone_out, _ = step(OUTSIDE, 3016.0)   # crosses threshold
assert ev is True and msg == ["Guard Absent"]
assert len(zone_out) == 1 and len(zone_out[0]) == 4, zone_out
zx1, zy1 = bb[0]["top_left"]; zx2, zy2 = bb[0]["bottom_right"]
assert 395 < zx1 < 405 and 695 < zx2 < 705, f"alert box is not the zone: {bb[0]}"
print(f"  zone filtering OK; alert box = zone bounds ({zx1},{zy1})-({zx2},{zy2})")

# error path must still honour the 5-tuple contract
out = service.perform_prediction(999999, frame, 0.45, {}, g, {})   # unknown camera
assert out[0] is False and out[1] == [] and out[2] == [] and out[3] == [] and len(out) == 5
print("  error path returns (False, [], [], [], frame)")

print("\nALL CHECKS PASSED")
