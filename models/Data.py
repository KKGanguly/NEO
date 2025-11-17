import os
import pickle
import hashlib
import numpy as np
from sklearn.neighbors import KDTree


class Data:
    def __init__(self, rows, cache_dir=".cache_kdtree", use_kdtree=True):
        self.rows = rows
        self.cache_dir = cache_dir
        self.use_kdtree = use_kdtree

        os.makedirs(cache_dir, exist_ok=True)

        # ---------------------------------------------------------
        # 1. Auto-compute min/max per column   (NEW)
        # ---------------------------------------------------------
        self.min_vals, self.max_vals = self._compute_min_max()

        # ---------------------------------------------------------
        # 2. Load KD-tree cache if available
        # ---------------------------------------------------------
        if self.use_kdtree:
            if not self._load_cache():
                # Compute numeric vectors only once
                self._compute_vectors_for_kdtree()
                self._build_kdtree()
                self._save_cache()

    # -------------------------------------------------------------
    # AUTO-COMPUTE MIN / MAX FROM DATASET
    # -------------------------------------------------------------
    def _compute_min_max(self):
        """Compute per-column min/max ignoring '?' and string values."""
        cols = len(self.rows[0])
        mins = []
        maxs = []

        for col in range(cols):
            vals = []
            for row in self.rows:
                v = row[col]
                if isinstance(v, (int, float)):  # numeric only
                    vals.append(v)

            if len(vals) == 0:
                # No numeric values — define safe defaults
                mins.append(0)
                maxs.append(1)
            else:
                mins.append(min(vals))
                maxs.append(max(vals))

        return mins, maxs

    # -------------------------------------------------------------
    # ORIGINAL SEMANTICS: normalize / dist / xdist
    # -------------------------------------------------------------
    def normalize(self, value, feature_index):
        if value == "?":
            return "?"

        min_val = self.min_vals[feature_index]
        max_val = self.max_vals[feature_index]

        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

    def dist(self, a, b, index):
        if a == "?" and b == "?":
            return 1

        # categorical string distance
        if isinstance(a, str) and isinstance(b, str) and a != "?":
            return 0 if a == b else 1

        # numeric normalization
        a = self.normalize(a, index)
        b = self.normalize(b, index)

        # missing numeric → opposite side of midpoint
        if a == "?":
            a = 1 if b < 0.5 else 0
        if b == "?":
            b = 1 if a < 0.5 else 0

        return abs(a - b)

    def xdist(self, p1, p2, p=2):
        d = 0
        n = len(p1)
        for idx, (a, b) in enumerate(zip(p1, p2)):
            d += abs(self.dist(a, b, idx)) ** p
        return (d / n) ** (1 / p)

    def nearestRow_bruteforce(self, target_row):
        best_dist = float("inf")
        best = None
        for row in self.rows:
            if row is target_row:
                continue
            d = self.xdist(target_row, row)
            if d < best_dist:
                best_dist = d
                best = row
        return best

    # -------------------------------------------------------------
    # KD-TREE CACHE & BUILDING
    # -------------------------------------------------------------
    def _dataset_hash(self):
        h = hashlib.md5()
        for row in self.rows:
            h.update(str(row).encode())
        return h.hexdigest()

    def _cache_path(self):
        return os.path.join(self.cache_dir, f"kdt_{self._dataset_hash()}.pkl")

    def _load_cache(self):
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.vectors = data["vectors"]
            self.kdtree = data["kdtree"]
            return True
        except Exception:
            return False

    def _save_cache(self):
        path = self._cache_path()
        if os.path.exists(path):
            return  # do not overwrite
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "kdtree": self.kdtree}, f)

    def _encode_value_for_kdtree(self, v, idx):
        if isinstance(v, str) and v != "?":
            return (hash(v) % 10000) / 10000
        if v == "?":
            return 0.5
        mn, mx = self.min_vals[idx], self.max_vals[idx]
        return (v - mn) / (mx - mn) if mx > mn else 0.0

    def _compute_vectors_for_kdtree(self):
        self.vectors = np.array([
            [self._encode_value_for_kdtree(v, idx)
             for idx, v in enumerate(row)]
            for row in self.rows
        ], dtype=float)

    def _build_kdtree(self):
        self.kdtree = KDTree(self.vectors, leaf_size=40)

    # -------------------------------------------------------------
    # FAST NEAREST NEIGHBOR
    # -------------------------------------------------------------
    def nearestRow(self, target_row):
        if not self.use_kdtree or self.kdtree is None:
            return self.nearestRow_bruteforce(target_row)

        vec = np.array([[
            self._encode_value_for_kdtree(v, idx)
            for idx, v in enumerate(target_row)
        ]], dtype=float)

        d, idxs = self.kdtree.query(vec, k=1)
        return self.rows[idxs[0][0]]
