# Guard Absence

Watches a guard-post zone and raises an alert when **no security guard has been
present for longer than 5 minutes** (configurable per camera). It consumes frames
from RabbitMQ and delivers alerts the same way every other service does — the
frozen `services/code/main.py` orchestration loop is untouched.

The hard part of this service was never the timer; it was *"is that the guard?"*
with almost no guard training data. See
[How it works around having very little guard data](#how-it-works-around-having-very-little-guard-data).

| | |
|---|---|
| Django `service_name` | `Guard Absence` |
| Django `parent_service` | `Intrusion` |
| Alert class | `Guard Absent` / `guard_absent` (index 0, `str`) |
| Docker image / container | `guard-absence-detection` |
| Default confidence | `0.45` |

## How it works around having very little guard data

The service **never trains a "security guard" class.** A detector fine-tuned on a
few dozen guard images would be worse at finding people than the stock detector
already is. Instead the question is split in two:

1. **"Is a person there?"** — stock COCO-pretrained YOLO. No new data, high
   recall, already reliable.
2. **"Is that person the guard?"** — few-shot re-identification. The person crop
   is embedded with an OpenVINO re-ID net and cosine-matched against a small
   **enrolled gallery**. Enrolment is an ops action over HTTP, not a training
   run, so 10–30 crops per guard per camera is plenty. The scarce data becomes a
   gallery instead of a training set.

Five design choices carry the accuracy, none of which need more data:

- **Track-level voting.** A track's guard/not-guard label is the **median of its
  rolling per-frame similarity scores** (`TRACK_SCORE_WINDOW`, min
  `TRACK_VOTE_MIN_SAMPLES` samples), never a single frame. Single-frame re-ID is
  noisy under pose change, motion blur and partial occlusion; a median over a
  handful of frames is dramatically steadier. This is the single biggest win
  available and it costs nothing.
- **Asymmetric tuning — recall over precision.** This service alerts on the
  *absence* of evidence, so **failing to recognise a guard who is present becomes
  a false alarm**, while an over-permissive match merely delays a true alert.
  Hence `GUARD_MATCH_THRESHOLD` 0.55 (loose, not 0.8) and detector confidence
  0.45 instead of the usual 0.7.
- **The absence accumulator is the debounce.** Absence is *integrated over
  wall-clock seconds*, not decided per frame, so a three-second occlusion adds
  three seconds to a 300-second budget instead of flipping a state machine. Any
  confirmed presence zeroes it. No separate smoothing window is needed. Time is
  measured in seconds, never frames — frame rate varies per camera and
  `INFERENCE_SKIP` drops frames.
- **Crop-quality gating.** Crops too small, too squashed or too blurred for the
  re-ID net are skipped rather than embedded badly — that is where most false
  mismatches come from.
- **Per-camera galleries, max-over-templates, pinned vs. auto.** Embeddings are
  viewpoint- and lighting-specific, so each camera keeps its own templates.
  Scoring uses the single best-matching template (not a centroid), so appearance
  variation across the gallery helps instead of averaging itself away.
  Human-enrolled templates are **pinned** and never evicted; bounded
  auto-enrolment appends very confident matches so the gallery adapts to
  lighting, shifts and seasons without being able to drift away from ground
  truth.

### Presence modes — start permissive, tighten later

`PRESENCE_MODE` is read per camera from `env_vars` on every frame, so you can
change it from the backend with **no redeploy and no restart**:

| Mode | "Guard is present" means | When to use |
|---|---|---|
| `any_person` **(default)** | any person inside the zone | **Day one.** Needs no gallery, cannot false-alarm from re-ID error, immediately useful. |
| `hybrid` | a gallery match resets the timer; an **unrecognised** person *holds* it (neither resets nor accrues) | Once a gallery is enrolled and proven. The recommended steady state. |
| `reid_guard` | only a gallery match counts | Strictest. Every re-ID miss is a false absence alert. |

A camera whose gallery is still empty **automatically degrades to
`any_person`**, whatever the mode says — an unenrolled camera must never alert
nonstop. The same happens service-wide if the re-ID weights are missing.

## Models

| Registry key | File | Purpose |
|---|---|---|
| `models["person"]` | `resources/weights/yolov8m.pt` | Person detection (COCO class 0). Declared in `constants.MODELS`, backend-version-managed, retrainable via `/trigger_training_process/`. |
| *(gallery-owned)* | `resources/weights/person-reidentification-retail-0287.xml` + `.bin` | 256-d appearance embedding for guard identity. |

The re-ID embedder is deliberately **not** in `constants.MODELS`: it is owned by
the gallery built in `initialise_resources()`, because it must also serve the
`/enroll_guard/` endpoint, which runs on the Flask thread where main.py's
`models` registry isn't reachable. One handle, two callers.

## The recipe

```
detect(models["person"], conf, classes=[0])      toolbox/detect.py
  └─ inside_any(build_zones(extra_fields))       toolbox/zones.py    keep in-zone people only
      └─ update(tracker, boxes)                  toolbox/tracking.py stable IDs (every frame, even empty)
          └─ crop-quality gate                   service.py          size / aspect / Laplacian sharpness
              └─ gallery.embed(crop)             model/gallery.py    OpenVINO 256-d, L2-normalised
                  └─ gallery.match(camera_id)    model/gallery.py    max cosine over that camera's templates
                      └─ median vote per track   service.py          -> guard | other | unverified
                          └─ absence accumulator service.py          present: zero | hold: freeze | absent: += dt
                              └─ bbox_entry(...) toolbox/annotate.py
```

**Alert fires** when `guard_absence_accum >= GUARD_ABSENCE_SECONDS` (300s). It
re-fires every `GUARD_ABSENCE_REPEAT_SECONDS` (600s) while the post is still
empty, and re-arms as soon as presence resumes. Because an absence alert has no
object to box, `bbox_output` carries the **guard-post zone bounds** (or the whole
frame when no zone is configured) as the `Guard Absent` class —
`bbox_output` must be non-empty or `main.py` would build an empty
`alert.classes` and the backend would silently reject the alert.

## Enrolling guards (HTTP)

Routes register on the existing Flask app from `endpoints.py`; `main.py` is not
touched. Default port is whatever `service_skeleton`'s `run_http_server` uses.

**Enrol whoever is standing at the post right now** — no bbox, no extra
inference; it uses the crops `service.py` already embedded on the last processed
frame:

```bash
curl -X POST http://<host>:<port>/enroll_guard/ \
  -H 'Content-Type: application/json' \
  -d '{"camera_id": 12, "identity": "guard_a"}'
```

**Enrol from an image** (a person crop, or a full frame plus `bbox`):

```bash
curl -X POST http://<host>:<port>/enroll_guard/ -F 'image=@guard_a.jpg' -F 'camera_id=12' -F 'identity=guard_a'
```

**Cold-start from disk** — drop images at
`resources/gallery/seed/<camera_id>/<identity>/*.jpg` and restart. They are
embedded and pinned at startup; already-seeded files are never re-added, so
adding more images later only adds the new ones.

Every enrolment also writes its crop into that seed directory, so if the re-ID
model is ever swapped, all pinned identities can be re-embedded from disk instead
of re-enrolled by hand.

**Inspect and manage:**

```bash
curl 'http://<host>:<port>/guard_gallery/?camera_id=12'
curl 'http://<host>:<port>/guard_status/?camera_id=12'     # live absence timer — check this first when tuning
curl -X POST http://<host>:<port>/clear_guard_gallery/ -H 'Content-Type: application/json' \
     -d '{"camera_id": 12, "identity": "guard_a"}'
```

The gallery persists to `resources/gallery/guard_gallery.npz` and survives
restarts. It is git-ignored: enrolled crops of real guards are personal data.

## Tunables (per camera, via `env_vars`)

All of these are re-read **every frame** from `CAMERAS_DATA[camera_id]["env_vars"]`,
falling back to the same-named constant in `helper/constants.py`. Backend changes
take effect with no restart.

| `env_vars` key | Default | Meaning |
|---|---|---|
| `PRESENCE_MODE` | `any_person` | `any_person` / `hybrid` / `reid_guard` |
| `GUARD_ABSENCE_SECONDS` | `300` | absence before the alert fires |
| `GUARD_ABSENCE_REPEAT_SECONDS` | `600` | re-alert cadence while still absent; `0` = alert once |
| `GUARD_MATCH_THRESHOLD` | `0.55` | cosine gate. **False alarms → lower it. Passers-by counted as the guard → raise it.** |
| `TRACK_SCORE_WINDOW` / `TRACK_VOTE_MIN_SAMPLES` | `15` / `4` | track-level median vote |
| `MIN_CROP_WIDTH` / `MIN_CROP_HEIGHT` | `48` / `96` | crop-quality gate |
| `MIN_CROP_SHARPNESS` | `25.0` | Laplacian variance; `0` disables the blur check |
| `CROP_ASPECT_MIN` / `CROP_ASPECT_MAX` | `1.2` / `4.5` | height/width sanity |
| `GUARD_AUTO_ENROLL` | `1` | `0` disables gallery self-adaptation |
| `AUTO_ENROLL_MIN_SCORE` | `0.78` | confidence required to auto-enrol |
| `AUTO_ENROLL_COOLDOWN_SECONDS` | `120` | throttle between auto-enrolments |
| `GALLERY_MAX_PER_CAMERA` | `64` | cap; oldest **un-pinned** templates evicted first |
| `MAX_EVAL_DT_SECONDS` | `30` | clamp so a camera dropout or restart can't instantly bank 5 minutes of absence |
| `LIVE_CROP_CACHE` | `3` | recent crops kept for live enrolment |

Zone polygons and `confidence_score` come from `extra_fields` (UI camera setup),
not `env_vars`.

## Before the first run — checklist

- [ ] **Register `Guard Absence` under parent service `Intrusion` in Django
      admin/backend.** This does **not** happen automatically — nothing in this
      codebase creates the service record, and `_init_service()` only works
      against one that already exists.
- [ ] **Register the class** `Guard Absent` / `guard_absent` / index `0` /
      `value_type` `str` against that `service_name`. A mismatch between this and
      `resources/config/config.json` makes alerts **fail validation silently** —
      caught, printed, never delivered. No crash, so it's easy to miss.
- [ ] **Drop in the person detector:** `resources/weights/yolov8m.pt` (or point
      `VERSION_DATA` / the backend's active version at whatever you use).
- [ ] **Drop in the re-ID weights:** both
      `resources/weights/person-reidentification-retail-0287.xml` **and** `.bin`
      (OpenVINO Open Model Zoo). *Optional to start* — without them the service
      runs fine in `any_person` mode and logs a warning.
- [ ] **Draw the guard-post zone** in the camera UI. With no zone configured the
      **whole frame** is treated as the post, which is usually not what you want.
- [ ] **Enrol each guard per camera**, then flip that camera's `PRESENCE_MODE` to
      `hybrid`. Until then every camera behaves as `any_person`.

Nothing in `service.py` is left as a TODO — the logic is complete for all three
presence modes.

### Status

Code-complete and logic-verified: the gallery (per-camera isolation,
max-over-templates matching, pinned-safe eviction, `.npz` round-trip, idempotent
disk seeding) and `perform_prediction` (threshold firing, class-name list and
non-empty bbox, repeat-window suppression, re-arming on presence, dropout clamp,
hybrid hold, zone filtering, error-path 5-tuple) were exercised against stubbed
`service_skeleton`/OpenVINO and all pass. `main.py` is byte-identical to the
base-service template.

**Not yet done:** no run against live RabbitMQ frames, real weights or a real
Django backend — the checklist above is what stands between this and a first
deployment. Match thresholds in particular should be tuned per site from
`/guard_status/` output.

## Running the service

### Directly

```
sudo <python> services/code/main.py
```

### Docker

Build from the repo root (`APP_VERSION` is a build arg, default `1.0.0`):

```
docker build -t guard-absence-detection:1.0.0 --build-arg APP_VERSION=1.0.0 .
```

Run with GPU access, mounting `resources/` and `logs/` so weights, config and the
enrolled gallery persist outside the container:

```
docker run -d --name guard-absence-detection --gpus all \
  -v $(pwd)/resources:/workspace/resources \
  -v $(pwd)/logs:/workspace/logs \
  guard-absence-detection:1.0.0
```

Mounting `resources/` is **required**, not optional: the enrolled gallery lives at
`resources/gallery/guard_gallery.npz` and would be lost on every container
replacement otherwise.

## Layout

```
services/code/
  main.py            🔒 FROZEN orchestration loop — byte-identical to base-service
  service.py         ✏️  perform_prediction: the guard-absence recipe
  endpoints.py       ✏️  enrolment / gallery / status HTTP routes
  model/
    gallery.py       ✏️  few-shot guard template store (per-camera, pinned, persisted)
    models.py        ✏️  initialise_resources builds the gallery + owns the re-ID net
  helper/
    constants.py     ✏️  MODELS, classes, and every env_vars-overridable default
    train_apis.py    🔒 skeleton wiring + ✏️ generate_ai_variables (per-camera state)
  toolbox/           🔒 detect · zones · tracking · reid · annotate (used as-is)
resources/config/config.json  ✏️  service_name / parent_service / classes
resources/gallery/            ⚙️  runtime: enrolled templates + seed crops (git-ignored)
```

See [`CONTRACT.md`](./CONTRACT.md) for the interface `main.py` depends on.
