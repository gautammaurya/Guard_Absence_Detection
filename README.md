# base-service

A base repository for GPU AI services that consume video frames from RabbitMQ,
run one or more models, and fire alerts to the Django backend. Clone it and fill
a handful of files to produce a new service — the orchestration core never changes.

**This repo is the single source for new services.** The template previously
lived as a branch inside the Intrusion repo; that is retired. Start every new
service from here — not from an Intrusion branch, and not by copying an
already-generated service:

```
git clone -b base-service-template \
  https://github.com/datenandwissenpvtltd/nwarch_ai_baseAiService.git <new-service>
```

Automated path: run `/create-service` (the skill in `.claude/skills/`, also
installed at `~/.claude/skills/` so it works from any directory).

## The idea

`services/code/main.py` is a **frozen** loop: consume frame → skip logic → run
`perform_prediction` → fire alert. It talks to service-specific code only through
the interface in [`CONTRACT.md`](./CONTRACT.md). To build a new service you fill
the *bodies* behind that contract; you never touch `main.py`.

Model count is **not** a design axis — declare any number of models in
`helper/constants.py::MODELS` and compose them however you like in `service.py`.

## Layout

```
services/code/
  main.py            🔒 FROZEN orchestration loop — never edit
  service.py         ✏️  perform_prediction body = the business logic (the recipe)
  model/
    models.py        ✏️  initialise_models (registry) + initialise_resources
    train.py         🔒 training loop
    data_creation.py 🔒 YOLO dataset massaging
  helper/
    constants.py     ✏️  MODELS[], VERSION_DATA, CLASS_LEVEL_DETAILS
    train_apis.py    🔒 skeleton wiring + ✏️ generate_ai_variables body
  toolbox/           🔒 reusable verbs — import what you need
    detect  tracking  zones  associate  reid  annotate  search(FAISS, opt-in)
resources/config/config.json  ✏️  service_name, classes, confidence
Dockerfile  requirements.txt   🔒 (params: image tag, per-model deps)
```

## Creating a new service (manual)

1. Clone this repo into the new service's directory (command above), then
   `rm -rf .git .claude && git init` so it gets its own history and doesn't
   carry the generator skill or push back to this template.
2. Declare models in `helper/constants.py::MODELS` and set `CLASS_LEVEL_DETAILS`.
3. Write `perform_prediction` in `service.py`, composing `toolbox` verbs.
4. Add any per-camera state to `generate_ai_variables`.
5. Set `service_name` / `parent_service` / `classes` in `resources/config/config.json`.
6. Drop weights into `resources/weights/`; add per-model deps to `requirements.txt`.
7. **Register `service_name` under `parent_service` in Django admin/backend.**
   This is a manual, one-time step — the service does not self-register on
   first run. Skipping it means `_init_service()` fails to fully initialize
   (see `CONTRACT.md`).
8. Run it — directly or via Docker (see below).

The stub as shipped is a **valid running service** that detects nothing and never
alerts — a safe starting point you can verify boots before adding logic.

## Running the service

### Directly

```
sudo <python> services/code/main.py
```

### Docker

Build from the repo root — do this after filling the surfaces above, since the
`Dockerfile` copies the whole tree into the image:

```
docker build -t <service_name>:<tag> .
```

`APP_VERSION` is a build arg (default `1.0.0`) baked into the image as
`$APP_VERSION`:

```
docker build -t <service_name>:<tag> --build-arg APP_VERSION=<version> .
```

Run with GPU access (the base image is `nvidia/cuda`) and `resources/` and
`logs/` mounted so weights/config edits and logs persist outside the
container:

```
docker run -d --name <service_name> --gpus all \
  -v $(pwd)/resources:/workspace/resources \
  -v $(pwd)/logs:/workspace/logs \
  <service_name>:<tag>
```

## Recipe examples

| Service | MODELS | `perform_prediction` recipe |
|---|---|---|
| Intrusion | detector (+ reid) | `detect → track → zones.inside_any → entry transition` |
| PPE | person + ppe | `detect ×2 → associate → compliance check` |
| Face recognition | detector + recognizer | `detect → reid.embed → search.query → match`; enroll via HTTP |

## Requirements

Pulls two private wheels from `dwnwcloud.in` (`service_skeleton`, `data_transform`)
— must be reachable at install time. See `requirements.txt`.
