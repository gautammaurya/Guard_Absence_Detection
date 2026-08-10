"""
Optional vector-search resource (FAISS).

Only used by services that opt in via initialise_resources() in model/models.py.
Requires `faiss-cpu` (or `faiss-gpu`) in requirements.txt — not installed by default.

Typical use (face recognition, ANPR, image similarity):
    index = VectorIndex(dim=256)
    index.add(embedding, identity="alice")     # enrollment
    matches = index.query(embedding, k=1)       # [(identity, score)]
"""
import traceback

try:
    import faiss
    import numpy as np
except ImportError:  # keep the base importable even without faiss installed
    faiss = None


class VectorIndex:
    """Thin wrapper over a FAISS flat index + a parallel identity list."""

    def __init__(self, dim, metric="cosine"):
        if faiss is None:
            raise ImportError("faiss is not installed; add faiss-cpu to requirements.txt")
        self.dim = dim
        self.metric = metric
        self.index = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
        self.ids = []

    def add(self, vector, identity):
        self.index.add(self._prep(vector))
        self.ids.append(identity)

    def query(self, vector, k=1):
        if self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(self._prep(vector), k)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i != -1]

    def _prep(self, vector):
        v = np.asarray(vector, dtype="float32").reshape(1, -1)
        if self.metric == "cosine":
            faiss.normalize_L2(v)
        return v
