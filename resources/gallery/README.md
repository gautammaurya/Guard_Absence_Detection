# resources/gallery/ — guard reference templates

This folder is the service's **reference gallery**: the cropped guard images and
the embeddings derived from them. Nothing here is trained; each crop is one
template, matched at runtime by cosine similarity.

Everything except this README is git-ignored — guard crops are personal data, and
the embeddings are runtime state.

## Layout

```
resources/gallery/
  guard_gallery.npz          embeddings + identities, written automatically
  seed/
    <camera_id>/             Django's numeric camera id
      <identity>/            your label for this guard, e.g. guard_a
        *.jpg                the cropped guard images
```

`<camera_id>` matters: re-ID embeddings are viewpoint- and lighting-specific, so
each camera keeps its own templates. Crops from one camera still work on another,
just with lower scores.

## Putting crops in

**Preferred** — from wherever your crops already live, run:

```bash
export LD_LIBRARY_PATH=/home/dnw/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
/usr/bin/python3.10 tools/seed_gallery.py \
    --images /path/to/guard_crops --camera-id 12 --identity guard_a \
    --impostors /path/to/other_people_from_the_same_camera
```

It copies accepted crops into the layout above, enrols them, and reports whether
the gallery actually separates your guard from other people. See the repo README's
"Enrolling guards" section.

**Manual equivalent** — create `seed/<camera_id>/<identity>/` yourself, drop the
`.jpg`s in, and restart the service. They're embedded and pinned at startup.
Already-seeded files are never re-added, so adding more later only adds the new
ones. You skip the quality gate and the separability report this way.

**While running** — `POST /enroll_guard/` with `{"camera_id": 12, "identity":
"guard_a"}` enrols whoever is at the post right now; the crop is saved into
`seed/` too, so it survives a gallery rebuild.

## What the crops need to be

- **Full-body or three-quarter body**, roughly 2:1 portrait. `person-reidentification-retail-0277`
  is a full-body person model — a square face/head crop is rejected. Face-based
  identification needs a different model (see `../../../FACE_RECOGNITION`).
- **At least 48x96 px**, aspect between 1.2 and 4.5, not motion-blurred. Same gate
  the live service applies to frames.
- **10-30 per guard per camera, varied** — different times of day, poses,
  distances. Twenty near-identical frames from one clip count as roughly one.

## Housekeeping

The gallery lives under `resources/` because that's the Docker-mounted volume, so
it survives container replacement. Inspect it with
`GET /guard_gallery/?camera_id=12`, reset a guard with
`POST /clear_guard_gallery/`. Deleting `guard_gallery.npz` is safe — a restart
re-embeds everything under `seed/` from scratch.
