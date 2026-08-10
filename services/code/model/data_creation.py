"""
Data transformation for the training pipeline. Common for all YOLO models.
Override create_training_data if a service trains a non-YOLO model.
"""
import os
import yaml
import traceback
from data_transform import MassagerFactory


def create_yolo_config_yaml(train_path, val_path, num_classes, class_names, output_file):
    """Write a YOLO dataset config YAML."""
    try:
        data = {
            'train': train_path,
            'val': val_path,
            'nc': num_classes,
            'names': class_names,
        }
        with open(output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"YAML configuration file created at {output_file}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error creating YAML configuration file: {e}")


def update_training_data(training_data):
    try:
        new_training_data = []
        for i in training_data:
            inside = []
            for j in i['bbox']:
                inside2_dict = j
                inside2_dict['bbox'] = list(j['bbox'])
                inside.append(inside2_dict)
            new_training_data.append({'image': f"/opt{i['image']}", 'bbox': inside})
        return new_training_data
    except Exception as e:
        traceback.print_exc()
        print(f"Error updating training data: {e}")
        return training_data


def create_training_data(training_data, CLASS_NAME_MASTER, SERVICE_NAME, version_details, NUMBER_OF_CLASSES, CLASSES_NAMES):
    try:
        training_data = update_training_data(training_data)
        massager_object = MassagerFactory.get_massager('yolo')
        other_params = {
            'class_dictionary': {i['class_name']: i['ai_class_index'] for i in CLASS_NAME_MASTER}
        }

        main_dir, train_path, validation_path, test_path, split = massager_object.data_folder_creation(SERVICE_NAME)
        massager_object.register_data(training_data, split, train_path, validation_path, test_path, other_params)
        massager_object.transform()

        output_file = os.path.join(main_dir, "data.yaml")
        model_checkpoint = version_details['version']['weights']
        save_model_path = os.path.join(main_dir, "best_model.pt")
        create_yolo_config_yaml(train_path, validation_path, NUMBER_OF_CLASSES, CLASSES_NAMES, output_file)

        return {
            "model_checkpoint": model_checkpoint,
            "output_file": output_file,
            "save_model_path": save_model_path,
        }
    except Exception as e:
        traceback.print_exc()
        print(f"Error creating training data: {e}")
        return {}
