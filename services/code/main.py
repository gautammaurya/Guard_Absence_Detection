import time
time.sleep(2)  # let dependent services (Django, RabbitMQ) come up before we call them

import traceback
from collections import Counter
from threading import Thread
from time import sleep

# helper import runs service bootstrap (service_skeleton wiring, class config, etc.)
# and re-exports: SERVICE_NAME, ALERT_SEND_URL, EVENT_MESSAGE, CLASS_NAME_FOR_ALERT,
# run_http_server, app, CAMERAS, CAMERAS_DATA, Alert, create_connection_with_rabbit_mq,
# consume_frame, QUEUE_NAME, get_all_service_cameras, get_class_config_data_from_backend,
# get_active_service_version_json, add_update_new_service_version, success_info, danger_info,
# and all URL/endpoint constants from service_skeleton.
from helper import *
from helper.constants import VERSION_DATA, UPDATED_CONFIDENCE, INFERENCE_SKIP

from model.models import initialise_models, initialise_resources
from service import perform_prediction


# ─────────────────────────────────────────────────────────────────────────────
# FROZEN ORCHESTRATION LOOP.
# This file is the stable core of every service and must NOT be edited by the
# factory. It only ever talks to service-specific code through four entry points:
#   initialise_models(version_details) -> models      (opaque registry, any N)
#   initialise_resources()             -> resources    (optional; index/gallery/None)
#   perform_prediction(camera_id, frame, conf, models, resources, CLASS_NAME_FOR_ALERT)
#       -> (give_event, alert_main_message, bbox_output, image_selection_bbox, new_frame)
#   generate_ai_variables()            -> dict         (per-camera state; via service_skeleton)
# As long as those honour their signatures, this loop runs any service unchanged.
# ─────────────────────────────────────────────────────────────────────────────

skip_counter = 0


if __name__ == "__main__":

    try:
        # Resolve active model version; register defaults if absent.
        try:
            version_details = get_active_service_version_json(
                service_name=SERVICE_NAME,
                GET_ACTIVE_SERVICE_VERSION_URL=GET_ACTIVE_SERVICE_VERSION_URL,
            )
            if not len(version_details):
                version_details = VERSION_DATA
                add_update_new_service_version(
                    service_name=SERVICE_NAME,
                    version_data=VERSION_DATA,
                    ADD_UPDATE_SERVICE_VERSION_URL=ADD_UPDATE_SERVICE_VERSION_URL,
                )
                success_info("Version Details registered")
        except Exception:
            version_details = VERSION_DATA
            danger_info("Using Local Version Details, Update the latest server_skeleton package")

        version_details = VERSION_DATA
        success_info(f"VERSION DETAILS: {version_details}")

        models = initialise_models(version_details)
        resources = initialise_resources()

        http_server = Thread(target=run_http_server, args=(app,))
        http_server.start()

        get_all_service_cameras(DJANGO_API_ENDPOINT, SERVICE_NAME)
        CLASS_CONFIG_DICT = get_class_config_data_from_backend(GET_CLASS_CONFIG_URL, SERVICE_NAME)

        CONNECTION, CHANNEL = create_connection_with_rabbit_mq()

        while True:

            try:

                if len(CAMERAS) == 0:
                    sleep(3)
                    continue

                try:
                    rmq_message = consume_frame(CHANNEL, QUEUE_NAME)

                    camera_id = rmq_message.get("camera_id", None)
                    frame = rmq_message.get("frame", None)
                    sequence_number = rmq_message.get("sequence_number", 1)

                    if CONNECTION.is_closed:
                        print("Retrying to connect")
                        CONNECTION, CHANNEL = create_connection_with_rabbit_mq()
                        sleep(3)

                    if camera_id is None:
                        continue

                    if frame is None:
                        continue

                    if camera_id not in CAMERAS_DATA:
                        continue

                    cam_pipeline = CAMERAS_DATA[camera_id]
                    cam_pipeline['frame'] = frame

                    if not cam_pipeline.get("active_status", True):
                        continue

                    conf = cam_pipeline["extra_fields"].get("confidence_score", UPDATED_CONFIDENCE)

                    skip_counter += 1
                    if skip_counter % INFERENCE_SKIP != 0:
                        continue

                    give_event, alert_main_message, bbox_output, image_selection_bbox, new_frame = \
                        perform_prediction(camera_id, frame, conf, models, resources, CLASS_NAME_FOR_ALERT)

                    if len(cam_pipeline["count_check_dict_in"]) > 150:
                        oldest = next(iter(cam_pipeline["count_check_dict_in"]))
                        del cam_pipeline["count_check_dict_in"][oldest]

                    if give_event:
                        print("Event Detected")
                        h_org, w_org, _ = frame.shape
                        service_id = cam_pipeline['service_id']
                        user_id = cam_pipeline['user_id']
                        camera_name = CAMERAS[camera_id]['camera_name']
                        service_name = cam_pipeline['service_name']

                        cam_pipeline['frame'] = new_frame

                        alert = Alert(
                            EVENT_MESSAGE,
                            camera_id=camera_id,
                            service_id=service_id,
                            user_id=user_id,
                            camera_name=camera_name,
                            service_name=service_name,
                            ALERT_SEND_URL=ALERT_SEND_URL,
                            CHANNEL=CHANNEL,
                            sequence_number=sequence_number,
                            consume_frame_data=rmq_message,
                            CLASS_CONFIG_DICT=CLASS_CONFIG_DICT,
                        )

                        res_bbox = {
                            'bbox': bbox_output,
                            'image_size': [h_org, w_org],
                            'image_selection_bbox': image_selection_bbox,
                        }
                        alert.bbox = [res_bbox]

                        class_count_dict = Counter([i['class'] for i in bbox_output])
                        ai_class_name_by_class = {i['class']: i.get('ai_class_name', 'object') for i in bbox_output}
                        alert.classes = [
                            {
                                "class_name": cls,
                                "ai_class_name": ai_class_name_by_class.get(cls, 'object'),
                                "value": True,
                                "count": cnt,
                            }
                            for cls, cnt in class_count_dict.items()
                        ]
                        alert.message_details = alert_main_message
                        alert.send(frame)

                    cam_pipeline["skip_counter"] += 1
                    if cam_pipeline["skip_counter"] >= 100000:
                        cam_pipeline["skip_counter"] = 0

                except Exception as e:
                    print(traceback.format_exc())
                    CONNECTION, CHANNEL = create_connection_with_rabbit_mq()
                    sleep(3)
                    continue

            except Exception:
                print(traceback.format_exc())
                danger_info("Program Exited...")

    except Exception as e:
        print(traceback.format_exc())
        danger_info(f"unable to run the {SERVICE_NAME} program due to {e}")
