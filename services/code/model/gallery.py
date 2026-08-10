"""
Few-shot guard identity gallery — the answer to "I have very little guard data".

Nothing here is trained. A guard is represented by a handful of re-ID embedding
vectors ("templates") enrolled from crops of that guard, and recognition is a
cosine lookup against them. Ten to thirty crops per guard per camera is plenty,
which turns an unusably small training set into a perfectly adequate gallery.

Design points that matter for accuracy:

* **Per-camera galleries.** Re-ID embeddings are viewpoint- and lighting-
  specific. A vector enrolled on camera A is a poor template for camera B, so
  each camera keeps its own set and never borrows from another.
* **Max-over-templates, not mean.** Scoring against the single best-matching
  template tolerates pose and appearance variation far better than collapsing
  a guard down to one centroid vector.
* **Pinned vs. un-pinned.** Human-enrolled and disk-seeded templates are pinned
  and never evicted. Auto-enrolled ones are un-pinned and evicted oldest-first
  past the cap, so self-adaptation can never drift the gallery away from the
  ground truth a human provided.
* **No FAISS.** Galleries are tens of vectors per camera; a single numpy matmul
  beats an index, and this keeps per-camera separation, deletion and pinning
  that toolbox/search.py's flat VectorIndex doesn't offer.

The module also exposes GALLERY, a process-wide singleton set by
initialise_resources(), because the /enroll_guard/ endpoint (endpoints.py) runs
on the Flask thread where main.py's `resources` handle isn't reachable.
"""
import json
import os
import threading
import traceback

import cv2
import numpy as np

from toolbox.reid import preprocess

# Process-wide handle, set by model.models.initialise_resources().
GALLERY = None

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class GuardGallery:
    """Per-camera store of enrolled guard embedding templates."""

    def __init__(self, reid_model, path, seed_dir=None, max_per_camera=64):
        self.model = reid_model          # None when the re-ID weights are absent
        self.path = path
        self.seed_dir = seed_dir
        self.max_per_camera = max_per_camera
        self.lock = threading.RLock()
        # camera_id(str) -> {"vectors": (N, D) float32, "identities": [str], "pinned": [bool]}
        self.cams = {}
        self.seeded_files = set()
        self.dim = None

    # ── availability ────────────────────────────────────────────────────────

    @property
    def enabled(self):
        """True when embeddings can actually be produced."""
        return self.model is not None

    def count(self, camera_id):
        with self.lock:
            entry = self.cams.get(str(camera_id))
            return 0 if entry is None else len(entry["identities"])

    def info(self, camera_id=None):
        """Human-readable summary for the /guard_gallery/ endpoint."""
        with self.lock:
            cams = self.cams if camera_id is None else {
                str(camera_id): self.cams.get(str(camera_id), {"identities": [], "pinned": []})
            }
            out = {}
            for cam, entry in cams.items():
                per_identity = {}
                for identity, pinned in zip(entry["identities"], entry["pinned"]):
                    slot = per_identity.setdefault(identity, {"pinned": 0, "auto": 0})
                    slot["pinned" if pinned else "auto"] += 1
                out[cam] = {"total": len(entry["identities"]), "identities": per_identity}
            return out

    # ── embedding ───────────────────────────────────────────────────────────

    def embed(self, crop):
        """Embed a BGR person crop into an L2-normalised vector, or None."""
        if self.model is None or crop is None or crop.size == 0:
            return None
        try:
            blob = preprocess(crop, self.model.height, self.model.width)
            if blob is None:
                return None
            vec = np.asarray(self.model.predict(blob)).reshape(-1).astype("float32")
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                return None
            vec /= norm
            if self.dim is None:
                self.dim = vec.shape[0]
            elif vec.shape[0] != self.dim:
                return None
            return vec
        except Exception:
            traceback.print_exc()
            return None

    # ── matching ────────────────────────────────────────────────────────────

    def match(self, camera_id, vector):
        """
        Best template match for this camera.

        Returns (identity, score) — (None, 0.0) when the camera has no templates.
        Score is max cosine similarity over templates, not a mean, so appearance
        variation across the gallery helps instead of averaging itself away.
        """
        if vector is None:
            return None, 0.0
        with self.lock:
            entry = self.cams.get(str(camera_id))
            if entry is None or len(entry["identities"]) == 0:
                return None, 0.0
            if entry["vectors"].shape[1] != vector.shape[0]:
                return None, 0.0
            scores = entry["vectors"] @ vector
            best = int(np.argmax(scores))
            return entry["identities"][best], float(scores[best])

    # ── enrolment ───────────────────────────────────────────────────────────

    def add(self, camera_id, vector, identity, pinned=True, persist=True):
        """Append one template. Evicts oldest un-pinned past max_per_camera."""
        if vector is None:
            return False
        cam = str(camera_id)
        with self.lock:
            entry = self.cams.get(cam)
            if entry is None or len(entry["identities"]) == 0:
                entry = {
                    "vectors": vector.reshape(1, -1).copy(),
                    "identities": [identity],
                    "pinned": [bool(pinned)],
                }
                self.cams[cam] = entry
            else:
                if entry["vectors"].shape[1] != vector.shape[0]:
                    return False
                entry["vectors"] = np.vstack([entry["vectors"], vector.reshape(1, -1)])
                entry["identities"].append(identity)
                entry["pinned"].append(bool(pinned))
                self._evict(entry)
            if persist:
                self._save_unlocked()
            return True

    def _evict(self, entry):
        """Drop oldest un-pinned templates until the cap is respected."""
        while len(entry["identities"]) > self.max_per_camera:
            victim = next((i for i, p in enumerate(entry["pinned"]) if not p), None)
            if victim is None:
                return  # everything is pinned — a human said so, so keep it all
            entry["vectors"] = np.delete(entry["vectors"], victim, axis=0)
            del entry["identities"][victim]
            del entry["pinned"][victim]

    def clear(self, camera_id, identity=None):
        """Remove all templates for a camera, or just one identity's."""
        cam = str(camera_id)
        with self.lock:
            entry = self.cams.get(cam)
            if entry is None:
                return 0
            if identity is None:
                removed = len(entry["identities"])
                self.cams.pop(cam, None)
            else:
                keep = [i for i, name in enumerate(entry["identities"]) if name != identity]
                removed = len(entry["identities"]) - len(keep)
                if keep:
                    entry["vectors"] = entry["vectors"][keep]
                    entry["identities"] = [entry["identities"][i] for i in keep]
                    entry["pinned"] = [entry["pinned"][i] for i in keep]
                else:
                    self.cams.pop(cam, None)
            self._save_unlocked()
            return removed

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self):
        with self.lock:
            self._save_unlocked()

    def _save_unlocked(self):
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            meta = {
                "cams": {
                    cam: {"identities": e["identities"], "pinned": e["pinned"]}
                    for cam, e in self.cams.items()
                },
                "seeded_files": sorted(self.seeded_files),
            }
            arrays = {f"vec::{cam}": e["vectors"] for cam, e in self.cams.items()}
            np.savez(self.path, meta=np.array(json.dumps(meta)), **arrays)
            return True
        except Exception:
            traceback.print_exc()
            return False

    def load(self):
        """Restore a previously saved gallery. Silent no-op when absent."""
        with self.lock:
            if not os.path.exists(self.path):
                return False
            try:
                data = np.load(self.path, allow_pickle=False)
                meta = json.loads(str(data["meta"]))
                self.seeded_files = set(meta.get("seeded_files", []))
                self.cams = {}
                for cam, entry in meta.get("cams", {}).items():
                    key = f"vec::{cam}"
                    if key not in data:
                        continue
                    vectors = np.asarray(data[key], dtype="float32")
                    identities = list(entry.get("identities", []))
                    pinned = list(entry.get("pinned", []))
                    if not (len(identities) == len(pinned) == vectors.shape[0]):
                        continue  # corrupt/partial entry — drop rather than trust
                    self.cams[cam] = {
                        "vectors": vectors,
                        "identities": identities,
                        "pinned": pinned,
                    }
                    if self.dim is None and vectors.size:
                        self.dim = int(vectors.shape[1])
                return True
            except Exception:
                traceback.print_exc()
                return False

    # ── cold-start seeding ──────────────────────────────────────────────────

    def seed_from_disk(self):
        """
        Enrol every image under <seed_dir>/<camera_id>/<identity>/*.jpg as a
        pinned template. Already-seeded paths are skipped, so dropping new
        images in and restarting only adds the new ones.
        """
        if not self.enabled or not self.seed_dir or not os.path.isdir(self.seed_dir):
            return 0
        added = 0
        try:
            for cam in sorted(os.listdir(self.seed_dir)):
                cam_dir = os.path.join(self.seed_dir, cam)
                if not os.path.isdir(cam_dir):
                    continue
                for identity in sorted(os.listdir(cam_dir)):
                    identity_dir = os.path.join(cam_dir, identity)
                    if not os.path.isdir(identity_dir):
                        continue
                    for name in sorted(os.listdir(identity_dir)):
                        if not name.lower().endswith(_IMAGE_SUFFIXES):
                            continue
                        rel = os.path.join(cam, identity, name)
                        if rel in self.seeded_files:
                            continue
                        image = cv2.imread(os.path.join(identity_dir, name))
                        vector = self.embed(image)
                        if vector is None:
                            continue
                        if self.add(cam, vector, identity, pinned=True, persist=False):
                            self.seeded_files.add(rel)
                            added += 1
            if added:
                self.save()
        except Exception:
            traceback.print_exc()
        return added
