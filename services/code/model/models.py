import traceback
from ultralytics import YOLO


# ── Model loaders: one per framework, keyed by the "type" in constants.MODELS ──

def _load_yolo(weights):
    return YOLO(weights, task="detect").to("cuda")


def _load_openvino(weights):
    # OpenVINO models (e.g. re-id / embedding nets) load via the toolbox wrapper.
    from toolbox.reid import OpenVinoModel
    return OpenVinoModel(weights)


# Add support for a new framework by adding one entry here — nothing else changes.
LOADERS = {
    "yolo": _load_yolo,
    "openvino": _load_openvino,
}


def initialise_models(version_details):
    """
    Load every model declared in helper.constants.MODELS into a {name: model}
    registry. Model count is arbitrary; main.py treats the result as opaque and
    passes it straight to perform_prediction.
    """
    try:
        from helper.constants import MODELS
        models = {}
        for spec in MODELS:
            loader = LOADERS.get(spec["type"])
            if loader is None:
                raise ValueError(f"No loader registered for model type '{spec['type']}'")
            weights = spec.get("weights")
            if weights is None:  # use the backend-managed active version
                weights = version_details["version"]["weights"]
            models[spec["name"]] = loader(weights)
        return models
    except Exception as e:
        traceback.print_exc()
        print(f"Error initialising models: {e}")
        return {}


def initialise_resources():
    """
    Build the guard identity gallery — this service's one non-model resource.

    It owns the OpenVINO re-ID embedder rather than declaring it in
    constants.MODELS, because the /enroll_guard/ endpoint needs to embed crops
    from the Flask thread, where main.py's `models` registry isn't reachable.
    Loading it here keeps exactly one handle serving both callers.

    Missing re-ID weights are NOT fatal: the gallery comes back disabled and
    service.py falls back to PRESENCE_MODE "any_person", so the service still
    does something useful before anyone has dropped the IR files in.
    """
    try:
        import os
        from helper.constants import (
            REID_WEIGHTS, GALLERY_PATH, GALLERY_SEED_DIR, GALLERY_MAX_PER_CAMERA,
        )
        from model import gallery as gallery_module
        from model.gallery import GuardGallery

        reid_model = None
        if os.path.exists(REID_WEIGHTS):
            try:
                reid_model = LOADERS["openvino"](REID_WEIGHTS)
                print(f"Re-ID embedder loaded from {REID_WEIGHTS}")
            except Exception:
                traceback.print_exc()
                print(f"Failed to load re-ID embedder from {REID_WEIGHTS}")
        else:
            print(
                f"Re-ID weights not found at {REID_WEIGHTS} — guard recognition "
                "disabled, running in any_person presence mode."
            )

        guard_gallery = GuardGallery(
            reid_model=reid_model,
            path=GALLERY_PATH,
            seed_dir=GALLERY_SEED_DIR,
            max_per_camera=GALLERY_MAX_PER_CAMERA,
        )
        guard_gallery.load()
        seeded = guard_gallery.seed_from_disk()
        if seeded:
            print(f"Seeded {seeded} guard template(s) from {GALLERY_SEED_DIR}")

        # Expose to the HTTP enrolment endpoints, which can't see main.py's locals.
        gallery_module.GALLERY = guard_gallery
        return guard_gallery
    except Exception as e:
        traceback.print_exc()
        print(f"Error initialising resources: {e}")
        return None
