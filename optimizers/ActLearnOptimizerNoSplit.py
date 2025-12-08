import tempfile
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import active_learning.src.bl as bl
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data


class ActLearnOptimizer(BaseOptimizer):

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        # Encoded X table
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # KD-tree for nearest row lookup
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

        # Build BL CSV
        self._bl_csv_path = self._make_bl_csv()

    # ---------------------------
    # Cleaning helpers
    # ---------------------------
    def _clean(self, v):
        import numpy as np
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        return v

    def _row_to_dict(self, row):
        return {col: self._clean(v) for col, v in zip(self.columns, row)}

    # ---------------------------
    # Build BL CSV
    # ---------------------------
    def _make_bl_csv(self):
        X = self.model_wrapper.X.copy()
        Y = self.model_wrapper.y.copy()

        # Check Y suffixes
        for col in Y.columns:
            if not (col.endswith("+") or col.endswith("-")):
                raise ValueError(
                    f"Y column '{col}' must end with + or -"
                )

        df = pd.concat([X, Y], axis=1)

        new_cols = []
        for col in df.columns:
            if col in Y.columns:
                new_cols.append(col)
            else:
                new_cols.append(col if col.endswith("X") else col + "X")

        df.columns = new_cols

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix="_bl.csv",
            mode="w"
        )
        df.to_csv(tmp.name, index=False)
        return tmp.name

    # ---------------------------
    # Main optimization
    # ---------------------------
    def optimize(self):

        n_trials = self.config["n_trials"]
        print(f"=== Running ACTLEARN optimizer (budget={n_trials}) ===")

        # Load BL CSV dataset
        bl_data = bl.Data(bl.csv(self._bl_csv_path))

        if len(bl_data.cols.y) == 0:
            raise ValueError("BL did not detect any Y objective columns.")

        # NEW API: Stop budget follows the SMAC style
        bl.the.budget = n_trials

        # Run Active Learning
        # (new API: bl.actLearn returns a learner object with .best property)
        learner = bl.actLearn(bl_data, shuffle=True)

        # Get best row (new API)
        best_row_raw = learner.best  # matches bl.first(learner.best.rows)

        # Extract only X columns (strip Y)
        clean_row = best_row_raw[:len(self.columns)]

        # Map to nearest actual row in table
        encoded_row = self.nn.nearestRow(clean_row)
        hp = self._row_to_dict(encoded_row)

        # Evaluate using TRUE model
        self.logging_util.start_logging()
        score = self.model_wrapper.run_model(hp)
        fitness = 1 - score
        self.logging_util.log(hp, fitness, 1)
        self.logging_util.stop_logging()

        self.best_config = hp
        self.best_value = fitness

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")

        return self.best_config, self.best_value
