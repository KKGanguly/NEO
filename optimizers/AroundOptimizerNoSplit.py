import numpy as np
import random
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data


class AroundOptimizer(BaseOptimizer):
    """
    FAST + FAITHFUL AROUND optimizer.

    - Implements the original Lua AROUND algorithm
    - Uses exact symbolic distance (xdist)
    - Optimized for Python (loop structure, indexing, locals)
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        # Same data source as other optimizers
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # Distance helper
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        random.seed(seed)
        np.random.seed(seed)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        return v

    def _row_to_hp(self, row):
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    # -----------------------------
    # FAST AROUND (Lua-faithful)
    # -----------------------------
    def _around(self, rows, budget, sample_size):
        rows = rows
        n = len(rows)
        if n == 0:
            return []

        # Store indices, not rows (FASTER)
        chosen_idx = [random.randrange(n)]

        # Local bindings (huge speedup in Python)
        xdist = self.nn.xdist
        randrange = random.randrange
        random_random = random.random

        for _ in range(1, budget):
            total = 0.0
            cand_idx = []
            cand_w = []

            # Sample candidates WITH replacement (Lua-style)
            for _ in range(min(sample_size, n)):
                i = randrange(n)
                row = rows[i]

                # distance to closest chosen
                dmin = float("inf")
                for j in chosen_idx:
                    d = xdist(row, rows[j])
                    if d < dmin:
                        dmin = d

                d2 = dmin * dmin
                if d2 > 0:
                    cand_idx.append(i)
                    cand_w.append(d2)
                    total += d2

            if total <= 0:
                break

            # Weighted random choice
            r = random_random() * total
            acc = 0.0
            for i, w in zip(cand_idx, cand_w):
                acc += w
                if acc >= r:
                    chosen_idx.append(i)
                    break
            else:
                chosen_idx.append(cand_idx[-1])

        return [rows[i] for i in chosen_idx]

    # -----------------------------
    # OPTIMIZE
    # -----------------------------
    def optimize(self):
        budget = self.config["n_trials"]
        sample_size = min(32, len(self.X_df))

        print(f"=== Running AROUND (FAST, faithful), budget={budget} ===")

        all_rows = self.X_df.values.tolist()

        # 1. Diverse sampling
        selected_rows = self._around(
            rows=all_rows,
            budget=budget,
            sample_size=sample_size,
        )

        # 2. Evaluate
        self.logging_util.start_logging()

        best_hp = None
        best_score = -float("inf")

        for row in selected_rows:
            hp = self._row_to_hp(row)
            score = self.model_wrapper.run_model(hp)
            fitness = 1 - score

            self.logging_util.log(hp, fitness, 1)

            if score > best_score:
                best_score = score
                best_hp = hp

        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value = best_score

        print(f"Best config = {self.best_config}")
        print(f"Best score  = {self.best_value}")

        return self.best_config, self.best_value
