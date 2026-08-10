#!/usr/bin/env python3
"""
Ingest a folder of guard crops into the reference gallery — and tell you whether
they are actually good enough to recognise the guard with.

This is the intended way to use pre-existing cropped guard images. It does not
train anything: each crop becomes one pinned template, matched at runtime by
cosine similarity.

    /usr/bin/python3.10 tools/seed_gallery.py \
        --images /path/to/guard_crops --camera-id 12 --identity guard_a

Run it from the repo root. It needs no GPU (the re-ID net is CPU/OpenVINO), so
none of the cuDNN workarounds in the README apply here.

What it does
────────────
1. Applies the SAME crop-quality gate service.py uses at runtime, so what this
   tool accepts is exactly what the live service would accept. Rejections are
   reported per file with the reason.
2. Copies accepted crops into resources/gallery/seed/<camera_id>/<identity>/ and
   enrols them through the service's own GuardGallery.seed_from_disk(), so the
   result is byte-identical to what a service restart would produce (and is not
   duplicated on the next restart).
3. Reports two things that decide whether the gallery is usable at all:
     * intra-identity similarity — are these crops really the same person, and
       consistent enough to match a live frame?
     * separability vs. impostors — do OTHER people score below your guard?
   Then it suggests a GUARD_MATCH_THRESHOLD from the observed margin.

Why the report matters: a gallery that looks fine but doesn't separate will not
crash anything. It will silently produce false "Guard Absent" alerts, because a
guard who fails to match reads as an empty post. Better to find that out here.

--dry-run checks and reports without writing anything.
"""
import argparse
import importlib.util
import os
import shutil
import sys

import cv2
import numpy as np

CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "services", "code")
sys.path.insert(0, CODE)


def _load_constants():
    """
    Load helper/constants.py directly from its path.

    Deliberately NOT `from helper.constants import ...`: importing the `helper`
    package runs helper/__init__.py -> train_apis -> _init_service(), which calls
    the Django backend at import time. This tool is an offline utility and must
    not need a reachable backend (or a working service_skeleton) to inspect crops.
    """
    spec = importlib.util.spec_from_file_location("_gac_constants", os.path.join(CODE, "helper", "constants.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load_constants()
GALLERY_PATH = C.GALLERY_PATH
GALLERY_SEED_DIR = C.GALLERY_SEED_DIR
GALLERY_MAX_PER_CAMERA = C.GALLERY_MAX_PER_CAMERA
REID_WEIGHTS = C.REID_WEIGHTS
MIN_CROP_WIDTH = C.MIN_CROP_WIDTH
MIN_CROP_HEIGHT = C.MIN_CROP_HEIGHT
MIN_CROP_SHARPNESS = C.MIN_CROP_SHARPNESS
CROP_ASPECT_MIN = C.CROP_ASPECT_MIN
CROP_ASPECT_MAX = C.CROP_ASPECT_MAX
GUARD_MATCH_THRESHOLD = C.GUARD_MATCH_THRESHOLD

from model.gallery import GuardGallery  # noqa: E402
from model.models import LOADERS  # noqa: E402

SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
# Fallback negatives that ship with ultralytics. These are a SMOKE TEST only —
# strangers in unrelated photos, nothing like the people who actually walk past
# your guard post. Always prefer --impostors pointing at crops of other people
# from the SAME camera: that is the real false-match risk, and same-camera
# lighting/viewpoint makes impostors score far higher than these do.
DEFAULT_IMPOSTOR_IMAGES = [
    "/home/dnw/.local/lib/python3.10/site-packages/ultralytics/assets/bus.jpg",
    "/home/dnw/.local/lib/python3.10/site-packages/ultralytics/assets/zidane.jpg",
]


def inspect(path):
    """Return (image, verdict, note) applying service.py's runtime crop gate."""
    image = cv2.imread(path)
    if image is None or image.size == 0:
        return None, "reject", "unreadable image"
    h, w = image.shape[:2]
    if w < MIN_CROP_WIDTH or h < MIN_CROP_HEIGHT:
        return image, "reject", f"{w}x{h} too small (need >= {MIN_CROP_WIDTH}x{MIN_CROP_HEIGHT})"
    aspect = h / float(w)
    if aspect < CROP_ASPECT_MIN:
        hint = " — looks like a head/face crop, not a full body" if aspect < 1.05 else ""
        return image, "reject", f"aspect {aspect:.2f} too wide (need {CROP_ASPECT_MIN}-{CROP_ASPECT_MAX}){hint}"
    if aspect > CROP_ASPECT_MAX:
        return image, "reject", f"aspect {aspect:.2f} too tall (need {CROP_ASPECT_MIN}-{CROP_ASPECT_MAX})"
    sharpness = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if MIN_CROP_SHARPNESS > 0 and sharpness < MIN_CROP_SHARPNESS:
        return image, "reject", f"too blurry (sharpness {sharpness:.0f} < {MIN_CROP_SHARPNESS:.0f})"
    return image, "accept", f"{w}x{h}  aspect {aspect:.2f}  sharpness {sharpness:.0f}"


def impostor_vectors(gallery, paths):
    """Embed whole impostor images plus, for multi-person photos, each person."""
    vectors = []
    detector = None
    for path in paths:
        if not os.path.exists(path):
            continue
        image = cv2.imread(path)
        if image is None:
            continue
        if detector is None:
            try:
                detector = (LOADERS["yolo"](C.VERSION_DATA["version"]["weights"]), C.PERSON_CLASS_INDEX)
            except Exception:
                detector = (None, None)
        model, class_index = detector
        boxes = []
        if model is not None:
            try:
                from toolbox.detect import detect
                boxes = detect(model, image, 0.45, classes=None if class_index is None else [class_index])
            except Exception:
                boxes = []
        crops = [image[b[1]:b[3], b[0]:b[2]] for b in boxes] or [image]
        for crop in crops:
            if crop.size == 0:
                continue
            vector = gallery.embed(crop)
            if vector is not None:
                vectors.append(vector)
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="folder of cropped guard images")
    parser.add_argument("--camera-id", required=True, help="camera these crops came from — templates are per camera")
    parser.add_argument("--identity", default="guard_a", help="label for this guard (default: guard_a)")
    parser.add_argument("--impostors", help="optional folder of OTHER people, to measure separability")
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        sys.exit(f"not a directory: {args.images}")
    if not os.path.exists(REID_WEIGHTS):
        sys.exit(f"re-ID weights missing: {REID_WEIGHTS}")

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(SUFFIXES))
    if not files:
        sys.exit(f"no images ({', '.join(SUFFIXES)}) in {args.images}")

    print(f"\nInspecting {len(files)} crop(s) for identity '{args.identity}' on camera {args.camera_id}")
    print("(gate is identical to the one service.py applies to live frames)\n")
    accepted, rejected = [], []
    for name in files:
        _image, verdict, note = inspect(os.path.join(args.images, name))
        mark = "OK  " if verdict == "accept" else "SKIP"
        print(f"  [{mark}] {name:<40} {note}")
        (accepted if verdict == "accept" else rejected).append(name)

    print(f"\n  accepted {len(accepted)} / {len(files)}   rejected {len(rejected)}")
    if not accepted:
        sys.exit("\nNothing usable. If most rejections are 'aspect too wide', these are probably\n"
                 "face/head crops — the 0277 model is a FULL-BODY person re-ID net. Either supply\n"
                 "full-body crops, or use a face-recognition model instead (see ../FACE_RECOGNITION).")
    if len(accepted) < 5:
        print("  note: fewer than 5 templates is thin — 10-30 per guard per camera is the target.")

    reid = LOADERS["openvino"](REID_WEIGHTS)
    gallery = GuardGallery(reid, GALLERY_PATH, seed_dir=GALLERY_SEED_DIR,
                           max_per_camera=GALLERY_MAX_PER_CAMERA)
    gallery.load()

    if args.dry_run:
        print("\n--dry-run: embedding for the report, writing nothing.")
        vectors = []
        for name in accepted:
            vector = gallery.embed(cv2.imread(os.path.join(args.images, name)))
            if vector is not None:
                vectors.append(vector)
    else:
        # Copy into the seed layout, then enrol via the service's own seeding path
        # so the result matches a restart exactly and is never double-counted.
        destination = os.path.join(GALLERY_SEED_DIR, str(args.camera_id), args.identity)
        os.makedirs(destination, exist_ok=True)
        for name in accepted:
            target = os.path.join(destination, name)
            if not os.path.exists(target):
                shutil.copy2(os.path.join(args.images, name), target)
        added = gallery.seed_from_disk()
        print(f"\n  enrolled {added} new pinned template(s) -> {GALLERY_PATH}")
        print(f"  crops copied to {destination}")
        entry = gallery.cams.get(str(args.camera_id), {"vectors": np.empty((0, 1)), "identities": []})
        vectors = [entry["vectors"][i] for i, name in enumerate(entry["identities"]) if name == args.identity]
        print(f"  gallery now holds {gallery.count(args.camera_id)} template(s) for camera {args.camera_id}")

    if len(vectors) < 2:
        print("\nNeed 2+ templates to report consistency.")
        return

    matrix = np.vstack(vectors)
    sims = matrix @ matrix.T
    pairs = sims[np.triu_indices(len(vectors), k=1)]
    print("\nConsistency — are these crops the same person, seen consistently?")
    print(f"  intra-identity cosine: min {pairs.min():.3f}  median {np.median(pairs):.3f}  max {pairs.max():.3f}")
    if pairs.min() < 0.3:
        print("  ! some pairs barely resemble each other — check for mislabelled crops,")
        print("    or crops of a different person / heavy occlusion.")

    impostor_paths = DEFAULT_IMPOSTOR_IMAGES
    source = "built-in generic negatives"
    if args.impostors and os.path.isdir(args.impostors):
        impostor_paths = [os.path.join(args.impostors, f) for f in sorted(os.listdir(args.impostors))
                          if f.lower().endswith(SUFFIXES)]
        source = args.impostors
    negatives = impostor_vectors(gallery, impostor_paths)
    if not negatives:
        print("\nNo impostor crops available — cannot measure separability.")
        return
    if source == "built-in generic negatives":
        print("\n  NOTE: using built-in fallback negatives (strangers in unrelated photos).")
        print("  This is a smoke test, not a real measurement. Re-run with --impostors pointing")
        print("  at crops of OTHER people from this same camera for a number you can trust.")
        print("  Also make sure your guard is NOT among the impostors — a contaminated negative")
        print("  set shows up here as a near-1.0 impostor score.")

    # Worst case that matters: the single highest score any impostor crop gets
    # against any template in the gallery — that's what would false-match at runtime.
    best_impostor = float(np.max(np.vstack(negatives) @ matrix.T))
    print(f"\nSeparability — do OTHER people score below your guard? ({len(negatives)} impostor crops, {source})")
    print(f"  best impostor score vs this gallery: {best_impostor:.3f}")

    worst_guard = float(pairs.min())
    print(f"\n  worst guard-to-guard {worst_guard:.3f}   vs   best impostor {best_impostor:.3f}")
    if worst_guard <= best_impostor:
        print("  ! OVERLAP: an impostor scores as high as your own crops do against each other.")
        print("    This gallery will misfire. Add more varied crops of the guard (different times")
        print(f"    of day, poses, distances) and keep PRESENCE_MODE=any_person until it separates.")
    else:
        suggested = round((worst_guard + best_impostor) / 2.0, 2)
        print(f"  suggested GUARD_MATCH_THRESHOLD: {suggested}   (current default {GUARD_MATCH_THRESHOLD})")
        print(f"  margin {worst_guard - best_impostor:.3f} — set it via this camera's env_vars, no redeploy needed.")

    print("\nNext: set PRESENCE_MODE=hybrid in camera "
          f"{args.camera_id}'s env_vars, then watch /guard_status/?camera_id={args.camera_id}\n")


if __name__ == "__main__":
    main()
