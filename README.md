# Guard Absence Detection

Watches a guard-post zone and raises an alert when **no security guard has been
present for longer than 5 minutes** (configurable per camera). It consumes frames
from RabbitMQ and delivers alerts the same way every other service does — the
frozen `services/code/main.py` orchestration loop is untouched.

The hard part of this service was never the timer; it was *"is that the guard?"*
with almost no guard training data. See
[How it works around having very little guard data](#how-it-works-around-having-very-little-guard-data).

| | |
|---|---|
| Django `service_name` | `Guard Absence Detection` |
| Django `parent_service` | `Guard Absence Detection` |
| Alert class | `Guard Absent` / `guard_absent` (index 0, `str`) |
| Docker image / container | `guard_absence_detection` |
| Default confidence | `0.45` |

### Replaces the July 2026 attempt

An earlier version of this service lives, untouched, at
`../Guard_Absence_Detection_OLD_20260728/`. It took the direct route: **one
custom-trained YOLO "guard" class**, a 2-minute threshold, no zone filtering and
no re-identification. That approach is exactly what the scarce guard data can't
support — which is why this rewrite exists. Nothing was deleted; the old tree is
there if you need to compare or recover anything from it.

Carried forward from it: the `Guard Absence Detection` service name and the
`Guard Absent` / `guard_absent` class, so the Django registration stays valid.
Changed: generic person detection + few-shot re-ID instead of a trained guard
class, a 5-minute default, zone filtering, and per-camera backend-tunable
parameters.

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

**Neither model is guard-specific.** There is no "security guard" model to train
or drop in — both files are stock, both are already used by sibling services, and
guard identity lives entirely in the enrolled gallery.

| Registry key | File | Purpose |
|---|---|---|
| `models["person"]` | `resources/weights/yolo11s.pt` | Stock COCO person detection (class 0). Same detector `Intrusion_Detection` uses. Declared in `constants.MODELS`, backend-version-managed. |
| *(gallery-owned)* | `resources/weights/person-reidentification-retail-0277.xml` + `.bin` | 256-d appearance embedding for guard identity. Fleet standard — also shipped by `Intrusion_Detection`, `PPE_Detection`, `PEOPLE_COUNTING`, `Wild_Animal_Detection`, `Common_Pipeline_Optimised`. |

Both are already present in `resources/weights/`, copied from
`../Intrusion_Detection/resources/weights/`. Verified working on this machine
(RTX 4070): `yolo11s` found 4/4 people in `bus.jpg` and 2/2 in `zidane.jpg`, and
`0277` embedded every crop to 256-d with a **1.000 self-match vs. a 0.312
best-impostor score** — a wide margin under the default 0.55 gate.

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

- [ ] **Confirm `Guard Absence Detection` is registered as a service (and as its
      own parent service) in Django admin/backend.** This name is carried over
      from the July 2026 attempt (archived at
      `../Guard_Absence_Detection_OLD_20260728/`), so it may already exist — if
      it does, there is nothing to do here. If it does **not**, register it: this
      never happens automatically, nothing in this codebase creates the service
      record, and `_init_service()` only works against one that already exists.
- [ ] **Register the class** `Guard Absent` / `guard_absent` / index `0` /
      `value_type` `str` against that `service_name`. A mismatch between this and
      `resources/config/config.json` makes alerts **fail validation silently** —
      caught, printed, never delivered. No crash, so it's easy to miss.
- [x] ~~Weights~~ — **done.** `yolo11s.pt` and the `0277` IR pair are already in
      `resources/weights/`, both verified loading and running on GPU. There is no
      guard-trained model to add.
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

Real weights are in place and verified on GPU (see [Models](#models)).

**Not yet done:** no run against live RabbitMQ frames or a real Django backend —
the checklist above is what stands between this and a first deployment. Match
thresholds should be tuned per site from `/guard_status/` output.

## Running the service

### Directly

The interpreter must be `/usr/bin/python3.10` — that's the only one on this box
with `service_skeleton`, `data_transform` and `openvino` installed (miniconda's
`python3` has none of them).

```
sudo /usr/bin/python3.10 services/code/main.py
```

> **cuDNN collision on this machine.** Any CUDA convolution **segfaults** under
> `/usr/bin/python3.10`: torch 2.3.0+cu118 was built against pip cuDNN 8.7.0 but
> loads the system `/lib/x86_64-linux-gnu/libcudnn.so.8` (8.9.7) instead, because
> both a `cu11` and a `cu12` NVIDIA stack are pip-installed. `torch.matmul` and
> `torchvision.ops.nms` on CUDA still work, so it looks healthy until the first
> convolution — then the process dies with no traceback. This affects **every**
> YOLO service run directly on this box, not just this one. Force the matching
> library first:
>
> ```
> export LD_LIBRARY_PATH=/home/dnw/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
> ```
>
> Verify with `python3.10 -c "import torch; print(torch.backends.cudnn.version())"`
> — it must print `8700`, not `8907`. Docker is unaffected (the image installs its
> own matched torch/torchvision), so the container path needs none of this.

### Docker

Build from the repo root (`APP_VERSION` is a build arg, default `1.0.0`):

```
docker build -t guard_absence_detection:1.0.0 --build-arg APP_VERSION=1.0.0 .
```

Run with GPU access, mounting `resources/` and `logs/` so weights, config and the
enrolled gallery persist outside the container:

```
docker run -d --name guard_absence_detection --gpus all \
  -v $(pwd)/resources:/workspace/resources \
  -v $(pwd)/logs:/workspace/logs \
  guard_absence_detection:1.0.0
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
