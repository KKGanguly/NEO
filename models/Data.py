import os
import pickle
import hashlib
import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd

class Data:
    def __init__(self, rows, column_types, cache_dir=".cache_kdtree", use_kdtree=True):
        """
        rows: List of lists/tuples containing ALREADY ENCODED values
        column_types: Dict mapping column names to types ('date', 'numeric', 'categorical')
        """
        self.rows = rows
        self.column_types = column_types
        self.cache_dir = cache_dir
        self.use_kdtree = use_kdtree
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.col_types = list(column_types.values())
        self.n_cols = len(self.col_types)

        self.min_vals, self.max_vals = self._compute_min_max()
        self.norm_scale = [
            (1.0 / (mx - mn)) if mx > mn else 0.0
            for mn, mx in zip(self.min_vals, self.max_vals)
        ]

        self._dist_cache = {}
        
        # KD-tree loading/building
        if self.use_kdtree:
            if not self._load_cache():
                self._compute_vectors_for_kdtree()
                self._build_kdtree()
                self._save_cache()
    
    def _compute_min_max(self):
        mins = []
        maxs = []
        col_types_list = list(self.column_types.values())
        
        for idx, col_type in enumerate(col_types_list):
            if col_type in ['numeric']:
                vals = [row[idx] for row in self.rows 
                       if isinstance(row[idx], (int, float)) and row[idx] != "?"]
                if vals:
                    mins.append(min(vals))
                    maxs.append(max(vals))
                else:
                    mins.append(0)
                    maxs.append(1)
            else:
                mins.append(0)
                maxs.append(1)
        return mins, maxs
    
    def normalize(self, value, feature_index):
        if value == "?" or pd.isna(value):
            return "?"
        mn, mx = self.min_vals[feature_index], self.max_vals[feature_index]
        if mx > mn:
            return (value - mn) / (mx - mn)
        else:
            return 0.0
    
    def dist(self, a, b, index):
        col_types_list = list(self.column_types.values())
        coltype = col_types_list[index]
        
        if (a == "?" or pd.isna(a)) and (b == "?" or pd.isna(b)):
            return 1
        
        if coltype in ["categorical", "date"]:
            return 0.0 if a == b else 1.0
        
        # Numeric/date columns: normalize and compute distance
        a_norm = self.normalize(a, index)
        b_norm = self.normalize(b, index)
        
        if a_norm == "?":
            a_norm = 1 if b_norm < 0.5 else 0
        if b_norm == "?":
            b_norm = 1 if a_norm < 0.5 else 0
            
        return abs(a_norm - b_norm)
    
    def xdist(self, r1, r2):
        """
        EXACT same semantics as before:
        - categorical/date: 0 if same else 1
        - numeric: normalized L2
        - missing handling unchanged
        """

        key = (id(r1), id(r2))
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached

        total = 0.0
        n = self.n_cols

        col_types = self.col_types
        min_vals = self.min_vals
        norm_scale = self.norm_scale

        for i in range(n):
            a = r1[i]
            b = r2[i]
            t = col_types[i]

            # both missing
            if a == "?" and b == "?":
                d = 1.0

            # categorical / date
            elif t == "categorical" or t == "date":
                d = 0.0 if a == b else 1.0

            # numeric
            else:
                if a == "?":
                    b = (b - min_vals[i]) * norm_scale[i]
                    a = 1.0 if b < 0.5 else 0.0
                elif b == "?":
                    a = (a - min_vals[i]) * norm_scale[i]
                    b = 1.0 if a < 0.5 else 0.0
                else:
                    a = (a - min_vals[i]) * norm_scale[i]
                    b = (b - min_vals[i]) * norm_scale[i]

                d = abs(a - b)

            total += d * d

        dist = (total / n) ** 0.5

        # symmetric memoization
        self._dist_cache[key] = dist
        self._dist_cache[(key[1], key[0])] = dist

        return dist
    
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
            return
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "kdtree": self.kdtree}, f)
    
    def _encode_value_for_kdtree(self, v, idx):
        col_types_list = list(self.column_types.values())
        coltype = col_types_list[idx]
        
        if v == "?" or pd.isna(v):
            return 0.5
        
        if coltype in ["categorical", "date"]:
            # simple hash -> [0,1]
            return (hash(str(v)) % 10000) / 10000
        
        # Numeric/date: normalize (already encoded as float)
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
    
    # ---------- public ----------
    def nearestRow(self, target_row):
        if not self.use_kdtree or getattr(self, "kdtree", None) is None:
            return self.nearestRow_bruteforce(target_row)
        vec = np.array([[
            self._encode_value_for_kdtree(v, idx)
            for idx, v in enumerate(target_row)
        ]], dtype=float)
        _, idxs = self.kdtree.query(vec, k=1)
        return self.rows[idxs[0][0]]