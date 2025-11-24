import numpy as np
import random
from math import sqrt
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data


class AroundOptimizer(BaseOptimizer):
    """
    Pure AROUND optimizer:
    - Selects a diverse set of rows from the FULL dataset (no train/test split)
    - Evaluates the true model on selected rows
    - Chooses the best observed configuration
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        # FULL encoded dataframe
        self.X_df = self.model_wrapper.X_encoded
        self.columns = list(self.X_df.columns)

        # KD-tree for nearest-row mapping
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

    # -----------------------------
    # Helper: clean numpy values
    # -----------------------------
    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)): return int(v)
        if isinstance(v, (np.floating, np.float64)): return float(v)
        return v

    def _row_dict(self, row_list):
        return {col: self._clean(v) for col, v in zip(self.columns, row_list)}

    def _vec(self, row):
        return np.array(row, dtype=float)

    # -----------------------------
    # AROUND sampler (diversity)
    # -----------------------------
    def _around(self, rows, budget, sample_size=32):
        rows = list(rows)
        if len(rows) == 0:
            return []

        chosen = [random.choice(rows)]
        chosen_vecs = [self._vec(chosen[0])]
        remaining = list(rows)
        remaining.remove(chosen[0])

        for _ in range(1, budget):
            if not remaining:
                break

            # Candidate subset
            k = min(sample_size, len(remaining))
            candidates = random.sample(remaining, k)

            best_dist = -1
            best_candidate = None

            for c in candidates:
                c_vec = self._vec(c)
                # distance to nearest selected
                d = min(np.linalg.norm(c_vec - z) for z in chosen_vecs)
                if d > best_dist:
                    best_dist = d
                    best_candidate = c

            chosen.append(best_candidate)
            chosen_vecs.append(self._vec(best_candidate))
            remaining.remove(best_candidate)

        return chosen

    # -----------------------------
    # OPTIMIZE over full dataset
    # -----------------------------
    def optimize(self):
        budget = self.config["n_trials"]
        print(f"=== Running AROUND optimizer on FULL data, budget={budget} ===")

        all_rows = self.X_df.values.tolist()

        # 1. Choose diverse rows
        selected_rows = self._around(all_rows, budget, sample_size=64)

        # 2. Evaluate selected rows
        self.logging_util.start_logging()

        best_hp = None
        best_fitness = float("inf")

        for r in selected_rows:
            hp = self._row_dict(self.nn.nearestRow(r))
            score = self.model_wrapper.run_model(hp)
            fitness = 1 - score
            self.logging_util.log(hp, fitness, 1)

            if fitness < best_fitness:
                best_fitness = fitness
                best_hp = hp

        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value = best_fitness

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")

        return self.best_config, self.best_value
