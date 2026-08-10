import os
import traceback
from sys import path
from pathlib import Path
from multiprocessing import Process

# service_skeleton provides all URL constants, API helpers, Sort, app, AI_VARIABLES, etc.
from service_skeleton.http_server import *

from model.train import train
from model.data_creation import create_training_data
from .constants import VERSION_DATA, CLASS_LEVEL_DETAILS


# ── path & service identity ────────────────────────────────────────────────────

path.append(str(Path(__file__).resolve().parents[2]))

config_json_path = os.path.join("resources", "config", "config.json")
SERVICE_NAME = get_config_json(config_json_path)["service_name"]
success_info(f"SERVICE_NAME: {SERVICE_NAME}")


# ── bootstrap: register service config with backend ───────────────────────────

def _init_service():
    """Run once at import time to wire up service metadata and class config."""
    global ALERT_SEND_URL, EVERY_N_FRAME, CLASS_NAME_MASTER, EVENT_MESSAGE
    global CLASS_NAME_FOR_ALERT, NUMBER_OF_CLASSES, CLASSES_NAMES

    send_version_and_classes_details(config_json_path, DJANGO_API_ENDPOINT)
    service_config = set_parameter_with_servce_name(DJANGO_API_ENDPOINT, SERVICE_NAME)

    ALERT_SEND_URL = f"{DJANGO_API_ENDPOINT}/service_app/create_alert/"
    EVERY_N_FRAME = int(service_config.get('every_n_frame', 5))

    # Resolve class mapping from backend; fall back to local defaults. The
    # backend's response shape isn't guaranteed to match what we expect (it
    # may be a dict missing 'classes_data', or classes_data items missing the
    # keys the dict-comprehension below requires) — validate before trusting
    # it, rather than letting a shape mismatch surface as a bare KeyError.
    CLASS_NAME_MASTER = CLASS_LEVEL_DETAILS['classes_data']
    EVENT_MESSAGE = CLASS_LEVEL_DETAILS['event_message']
    try:
        class_level_details = get_class_mapping_list(SERVICE_NAME, GET_CLASS_MAPPING_DETAILS_URL)
        if isinstance(class_level_details, dict):
            candidate = class_level_details.get('classes_data')
            if candidate and all('ai_class_index' in c and 'class_name' in c for c in candidate):
                CLASS_NAME_MASTER = candidate
                EVENT_MESSAGE = class_level_details.get('event_message', EVENT_MESSAGE)
            else:
                danger_info(
                    "get_class_mapping_list returned classes_data in an unexpected "
                    "shape (missing ai_class_index/class_name) — using local "
                    "CLASS_LEVEL_DETAILS instead.",
                    {"SERVICE_NAME": SERVICE_NAME, "received": class_level_details},
                )
    except Exception:
        pass

    CLASS_NAME_FOR_ALERT = {i['ai_class_index']: i['class_name'] for i in CLASS_NAME_MASTER}
    class_dictionary = {i['class_name']: i['ai_class_index'] for i in CLASS_NAME_MASTER}
    NUMBER_OF_CLASSES = len(class_dictionary)
    CLASSES_NAMES = list(class_dictionary.keys())


# Declare module-level names so they exist before _init_service populates them.
ALERT_SEND_URL = None
EVERY_N_FRAME = 5
CLASS_NAME_MASTER = CLASS_LEVEL_DETAILS['classes_data']
EVENT_MESSAGE = CLASS_LEVEL_DETAILS['event_message']
CLASS_NAME_FOR_ALERT = {}
NUMBER_OF_CLASSES = 0
CLASSES_NAMES = []
EPOCHS = 1

_init_service()


# ── per-camera AI state ────────────────────────────────────────────────────────

def generate_ai_variables():
    """
    Per-camera state, created when a camera is registered.

    'skip_counter' and 'count_check_dict_in' are required by main.py. Everything
    below is guard-absence state, all of it per camera because absence is a
    per-post question and re-ID templates are per-viewpoint:

      tracker                  SORT instance giving people stable IDs so a
                               guard/not-guard vote can accumulate over a track
                               instead of being re-decided every frame.
      guard_track_state        tracker_id -> {scores deque, label, last_seen,
                               identity}; the rolling median of `scores` is what
                               labels a track.
      guard_absence_accum      seconds of confirmed absence banked so far. This
                               accumulator IS the debounce: brief occlusions add
                               a few seconds to a 300s budget rather than
                               resetting a state machine. Zeroed by any
                               confirmed presence.
      guard_last_eval_ts       wall-clock of the previous evaluation. Absence is
                               measured in seconds, never in frames — frame rate
                               varies per camera and INFERENCE_SKIP drops frames.
                               None means "first frame": establish the clock
                               without banking any absence, so a restart can't
                               instantly alert.
      guard_last_alert_ts      supports the re-alert cadence; reset to None when
                               presence resumes so the next episode alerts again.
      guard_last_auto_enroll_ts  cooldown for bounded gallery self-adaptation.
      guard_live_crops         most recent quality-gated person crops + their
                               embeddings, so POST /enroll_guard/ can enrol
                               whoever is at the post right now with no extra
                               inference.
    """
    try:
        from toolbox.tracking import new_tracker

        return {
            "skip_counter": 0,
            "count_check_dict_in": {},
            "tracker": new_tracker(max_age=60, min_hits=0, iou_threshold=0.2),
            "guard_track_state": {},
            "guard_absence_accum": 0.0,
            "guard_last_eval_ts": None,
            "guard_last_alert_ts": None,
            "guard_last_auto_enroll_ts": 0.0,
            "guard_live_crops": [],
        }
    except Exception as e:
        traceback.print_exc()
        print(f"Error in generating ai variables: {e}")


AI_VARIABLES.value_generator = generate_ai_variables


# ── training pipeline ──────────────────────────────────────────────────────────

def start_training_process():
    try:
        version_details = get_active_service_version_json(
            service_name=SERVICE_NAME,
            GET_ACTIVE_SERVICE_VERSION_URL=GET_ACTIVE_SERVICE_VERSION_URL,
        )
        if not len(version_details):
            add_update_new_service_version(
                service_name=SERVICE_NAME,
                version_data=VERSION_DATA,
                ADD_UPDATE_SERVICE_VERSION_URL=ADD_UPDATE_SERVICE_VERSION_URL,
            )
            version_details = VERSION_DATA

        training_data = get_training_data(
            service_name=SERVICE_NAME,
            data_count=100000,
            GET_TRANING_DATA_URL=GET_TRANING_DATA_URL,
        )
        training_params = create_training_data(
            training_data, CLASS_NAME_MASTER, SERVICE_NAME,
            version_details, NUMBER_OF_CLASSES, CLASSES_NAMES,
        )
        if not training_params:
            danger_info("Training data creation failed, aborting")
            return

        new_version_data = train(training_params, version_details, EPOCHS)
        if new_version_data:
            add_update_new_service_version(
                service_name=SERVICE_NAME,
                version_data=new_version_data,
                ADD_UPDATE_SERVICE_VERSION_URL=ADD_UPDATE_SERVICE_VERSION_URL,
            )
            success_info("Training complete, new version registered")
        else:
            danger_info("Training failed, no version registered")
    except Exception as e:
        traceback.print_exc()
        danger_info(f"Training process failed: {e}")


@app.route('/trigger_training_process/', methods=['GET'])
def trigger_training_process():
    try:
        global RETRAINING_DICT
        training_process = Process(target=start_training_process)
        RETRAINING_DICT['process_obj'] = training_process
        RETRAINING_DICT['process_obj'].start()
        success_info(f"{get_current_function_name()} Operation Successful")
    except Exception as e:
        danger_info(f"Not able to {get_current_function_name()} error is {e} {traceback.format_exc()}")

    return "Got your task"
