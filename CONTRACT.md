# Service Contract

This is the interface the frozen core (`services/code/main.py`) depends on. A
generated service **must honour every signature below**. The factory fills the
*bodies*; it must never change these *signatures*, and must never edit `main.py`.

## Entry points

### 1. `model/models.py` — `initialise_models(version_details) -> dict`
Load every model declared in `helper/constants.py::MODELS` into a `{name: model}`
registry. Count is arbitrary. `main.py` treats the result as opaque.

### 2. `model/models.py` — `initialise_resources() -> Any | None`
Optional shared resources (FAISS index, identity gallery, DB handle). Return
`None` if unused. Threaded into `perform_prediction` unchanged.

### 3. `service.py` — `perform_prediction(camera_id, frame, conf, models, resources, CLASS_NAME_FOR_ALERT)`
The business logic. Runs once per non-skipped frame. `main.py` drops any RabbitMQ
message whose `camera_id` is `None` before this is ever called (`continue`d in
the frame loop) — `camera_id` is guaranteed non-`None` inside `perform_prediction`
and in `generate_ai_variables`/`CAMERAS_DATA` lookups keyed by it. Don't add a
defensive `camera_id is None` check inside the service; it can't happen here.
**Return exactly:**

| Position | Name | Type | Meaning |
|---|---|---|---|
| 0 | `give_event` | `bool` | `True` fires an alert this frame |
| 1 | `alert_main_message` | `list[str]` | Detected class name(s) for this alert — becomes `alert.message_details` as-is (`main.py` does not wrap it). Service-controlled: usually the distinct classes present in `bbox_output`, but the service decides what goes in it. |
| 2 | `bbox_output` | `list[dict]` | Boxes: keys `top_left, top_right, bottom_left, bottom_right, class`, optional `ai_class_name` |
| 3 | `image_selection_bbox` | `list` | Zone polygons for the alert image (or `[]`) |
| 4 | `new_frame` | `ndarray` | Annotated frame (or the original `frame`) |

On error, return `(False, [], [], [], frame)`.

### `alert_main_message` is a class-name list, not free text

`main.py` assigns it straight to `alert.message_details` (no wrapping). Earlier
versions of this contract treated position 1 as a single free-text headline
string wrapped in a list by `main.py` (`alert.message_details = [alert_main_message]`)
— that's been superseded. The service now owns the full list content, e.g.
`["Loitering Detected"]`, or `["no_vest", "no_helmet"]` for a multi-class
service where more than one violation fires in the same frame. Do not return
a prose sentence here.

### `bbox_output[]['ai_class_name']` — how per-class `ai_class_name` reaches the alert

`toolbox/annotate.py::bbox_entry(text, x1, y1, x2, y2, ai_class_name=None)` takes
an optional `ai_class_name`. `main.py` reads it back per box (`i.get('ai_class_name',
'object')`) when building `alert.classes[]`, instead of hardcoding a literal.
Omit it and a box's `ai_class_name` defaults to `"object"` — the original
single-class stub behaviour. A service with a real class should source the
value from its own `helper/constants.py::CLASS_LEVEL_DETAILS` (not a literal
in `service.py`) and pass it through `bbox_entry(...)`, e.g.:

```python
CLASS_AI_NAME = CLASS_LEVEL_DETAILS["classes_data"][0]["ai_class_name"]
...
bbox_output.append(bbox_entry("My Class Name", x1, y1, x2, y2, ai_class_name=CLASS_AI_NAME))
```

This keeps the *decision* of what `ai_class_name` is in `service.py`/`constants.py`
— `main.py` only plumbs through whatever the service attached, it doesn't decide.

### Backend-manageable parameters (thresholds, durations, etc.) — `env_vars`

Per-camera config already flows through `CAMERAS_DATA[camera_id]` two ways:
`cam["extra_fields"]` (zone polygons, `confidence_score` — UI-driven camera
setup) and `cam["env_vars"]` — a dict the backend can push arbitrary named
overrides through, keyed by the same name as the `constants.py` fallback.
`main.py` never touches `env_vars`; `service.py` reads it directly, every
frame, so a backend-side change takes effect without a service restart (no
resolve-once-at-import step, unlike `EVERY_N_FRAME`).

For any service-specific tunable (a duration, a distance, a count) ops should
be able to retune without a redeploy:

1. Define the fallback default in `helper/constants.py`, named exactly what
   the `env_vars` key will be (e.g. `LOITER_THRESHOLD_SECONDS = 60`).
2. In `service.py`, inside `perform_prediction`, read it fresh each call:
   ```python
   env_vars = cam.get("env_vars", {})
   loiter_threshold_seconds = int(env_vars.get("LOITER_THRESHOLD_SECONDS", LOITER_THRESHOLD_SECONDS))
   ```
3. Use the local resolved variable in the logic, not the raw imported
   constant — the constant is only ever the fallback.

Don't wire this through `helper/train_apis.py::_init_service()` /
`service_config` — that path resolves once at process start and doesn't
apply per-camera; `env_vars` is the mechanism that's actually live per-frame
and per-camera.

### 4. `helper/train_apis.py` — `generate_ai_variables() -> dict`
Per-camera state, created when a camera registers. **Must include** `skip_counter`
(int) and `count_check_dict_in` (dict) — `main.py` uses both. Add service-specific
state (trackers, caches) as needed.

## Config surfaces the factory fills

| File | What to fill |
|---|---|
| `helper/constants.py` | `MODELS[]`, `VERSION_DATA`, `CLASS_LEVEL_DETAILS`, `EVENT_MESSAGE` |
| `resources/config/config.json` | `service_name`, `parent_service`, `classes`, `default_confidence` |
| `service.py` | `perform_prediction` body (the recipe) |
| `model/models.py` | extra loaders / `initialise_resources` body if needed |
| `requirements.txt` | per-model / per-resource deps (e.g. uncomment faiss) |

### `service_name` / `parent_service` — required, not placeholder-fillable

These two fields in `config.json` are the join key with Django, not decorative
metadata. **A brand-new `service_name` must be manually pre-registered in
Django admin/backend before the service is first run** — `_init_service()`
(`send_version_and_classes_details`, `set_parameter_with_servce_name`) does not
create the service record, it only works against one that already exists.
The skill must treat `service_name` and `parent_service` as required,
non-skippable clarifying questions, and the generated README must tell the
user to register them in Django *before* running the service.

### Class definitions — must match the backend's registered schema

`CLASS_LEVEL_DETAILS` classes (`class_name`, `ai_class_name`, `ai_class_index`,
`value_type`) are synced to Django via `send_version_and_classes_details` and
read back via `get_class_mapping_list`. At alert time, `Alert.send()` validates
the `classes` payload against the backend's stored schema for this service —
`value` must match `value_type`, and there's a `count_type` check too (resolved
elsewhere in the backend flow; do not add `count_type` to this config — it is
already handled and not something a generated service needs to declare). A
mismatch between what the service declares and what Django has registered
causes the alert to silently fail validation (caught, printed, never sent) —
not a crash, so this is easy to miss.

### Alert delivery — RabbitMQ, not HTTP

`Alert.send(frame)` (called from the frozen `main.py`) delivers over RabbitMQ —
it msgpack-packs `{meta, frame}` and publishes to a fixed exchange
(`ALERT_EXCHANGE_NAME`, "Alert_Queue_Exchange"). The `ALERT_SEND_URL` built in
`main.py` and passed into `Alert(...)` is still constructed (the frozen loop
does this for every service) but is currently **vestigial** — the old HTTP
POST path in `service_skeleton` is dead code. Don't rely on `ALERT_SEND_URL`
doing anything at runtime.

## Frozen — never edit

`services/code/main.py`, and the `service_skeleton` wiring in `helper/train_apis.py`
(everything except the `generate_ai_variables` body and the `MODELS`/class config).

## Toolbox (reusable verbs — import as needed in `service.py`)

`toolbox/detect.py` · `tracking.py` · `zones.py` · `associate.py` · `reid.py` ·
`annotate.py` · `search.py` (FAISS, opt-in).

## Running the service

Mirrors the base-service README's "Running the service" section.

Directly:

```
sudo <python> services/code/main.py
```

Docker (build from the repo root, after the surfaces above are filled — the
frozen `Dockerfile` copies the whole tree in; `APP_VERSION` is a build arg,
default `1.0.0`):

```
docker build -t <service_name>:<tag> --build-arg APP_VERSION=<version> .
docker run -d --name <service_name> --gpus all \
  -v $(pwd)/resources:/workspace/resources \
  -v $(pwd)/logs:/workspace/logs \
  <service_name>:<tag>
```
