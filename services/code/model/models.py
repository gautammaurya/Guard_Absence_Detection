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
    Optional external resources shared across frames — a FAISS index, an identity
    gallery, a DB handle, etc. Return None if the service needs none. Override the
    body per service; main.py threads the return value into perform_prediction.
    """
    return None
