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
        
        # Compute min/max for numeric/date columns
        self.min_vals, self.max_vals = self._compute_min_max()
        
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
            if col_type in ['numeric', 'date']:
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
        
        if coltype == 'categorical':
            return 0 if a == b else 1
        
        # Numeric/date columns: normalize and compute distance
        a_norm = self.normalize(a, index)
        b_norm = self.normalize(b, index)
        
        if a_norm == "?":
            a_norm = 1 if b_norm < 0.5 else 0
        if b_norm == "?":
            b_norm = 1 if a_norm < 0.5 else 0
            
        return abs(a_norm - b_norm)
    
    def xdist(self, p1, p2, p=2):
        total = 0
        n = len(p1)
        for idx, (a, b) in enumerate(zip(p1, p2)):
            total += abs(self.dist(a, b, idx)) ** p
        return (total / n) ** (1 / p)
    
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
        
        if coltype == 'categorical':
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
    
    def nearestRow(self, target_row):
        if not self.use_kdtree or self.kdtree is None:
            return self.nearestRow_bruteforce(target_row)
        vec = np.array([[
            self._encode_value_for_kdtree(v, idx)
            for idx, v in enumerate(target_row)
        ]], dtype=float)
        _, idxs = self.kdtree.query(vec, k=1)
        return self.rows[idxs[0][0]]