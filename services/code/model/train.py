import traceback
from datetime import datetime

import torch
from ultralytics import YOLO


def train(training_params, version_details, epochs):
    """Fine-tune the trainable model from the current checkpoint and return new version data."""
    try:
        model_checkpoint = training_params["model_checkpoint"]
        output_file = training_params["output_file"]
        save_model_path = training_params["save_model_path"]

        model = YOLO(model_checkpoint)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_params = {
            'data': output_file,
            'epochs': epochs,
            'batch': 16,
            'imgsz': 640,
            'name': 'custom_model',
            'device': device,
        }

        model.train(**train_params)
        model.save(save_model_path)

        try:
            version_name = f"V{version_details['version_count'] + 1}"
        except Exception:
            version_name = 'V2'

        return {
            'version_name': version_name,
            'version': {
                'weights': save_model_path,
                'datetime': str(datetime.now()),
            },
        }
    except Exception as e:
        traceback.print_exc()
        print(f"Error during training: {e}")
        return None
