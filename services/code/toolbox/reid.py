"""
OpenVINO embedding / re-identification helper.

Wraps a compiled OpenVINO model (e.g. person-reidentification-*) to produce
feature vectors, plus cosine similarity for de-duplication or matching.
Register such a model in constants.MODELS with type "openvino".
"""
import traceback
import cv2
import numpy as np
import openvino as ov

_core = ov.Core()


class OpenVinoModel:
    """Generic OpenVINO compiled-model wrapper."""

    def __init__(self, model_path, batchsize=-1, device="AUTO"):
        self.model = _core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.height = self.input_layer.shape[2]
        self.width = self.input_layer.shape[3]

        for layer in self.model.inputs:
            shape = layer.partial_shape
            shape[0] = batchsize
            self.model.reshape({layer: shape})
        self.compiled = _core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled.output(0)

    def predict(self, x):
        return self.compiled(x)[self.output_layer]


def preprocess(frame, height, width):
    """Resize + CHW + batch dim for OpenVINO input."""
    try:
        img = cv2.resize(frame, (width, height)).transpose((2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)
    except Exception:
        traceback.print_exc()
        return None


def cosine(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
